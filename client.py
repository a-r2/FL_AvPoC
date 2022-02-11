from settings import *
from constants import *
from nn_algorithm import *
from utils import *

import flwr as fl
from tensorflow import keras 
import tensorflow as tf
import numpy as np

class FLClient(fl.client.NumPyClient):
    """
    Client class for federated learning
    """

    def __init__(self, client_ind, client_dir, model, train_dataset, test_dataset):
        self.client_ind                            = client_ind
        self.client_str                            = "CLIENT " + str(client_ind)
        self.client_dir                            = client_dir
        self.model                                 = model #neural network
        self.x_train, self.y_train, self.SNR_train = train_dataset
        self.x_test, self.y_test, self.SNR_test    = test_dataset
        self.history                               = list() #list of dicts containing fit history
        # Fit callbacks
        self.fit_callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.client_dir, MODEL_BEST_STR),
                save_best_only=True,
                monitor="loss",
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=20,
                min_lr=0.0001,
            ),
            keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=50,
            ),
        ]

    def get_parameters(self):
        """
        Gets local model parameters
        """
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """
        Fitting and evaluation

        Fits the model on the local training dataset and evaluates the model on the testing dataset

        Args:

            parameters: model's weights (list)
            config: parameters that configure the fitting or/and evaluation process (dict)
        
        Returns:

            parameters_prime: new model's weights (list)
            num_examples_train: number of training samples (int)
            results: fitting history (history)
        """
        rnd = config["rnd"] #get current FL round
        self.model.set_weights(parameters) #update local model parameters from global model
        # Modify learning rate
        self.model.optimizer.lr = FL_LR[rnd - 1]
        # Fit the model on training dataset
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size       = TRAIN_BATCH_SIZE,
            epochs           = FL_EPOCHS[rnd - 1],
            callbacks        = self.fit_callbacks,
        )
        # Return updated model parameters (weights) and results (loss and accuracy)
        parameters_prime   = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results            = {
            "loss": history.history["loss"][0],
            "categorical_accuracy": history.history["categorical_accuracy"][0]
        }
        # Evaluate on test dataset
        loss_test, accuracy_test = self.model.evaluate(self.x_test, self.y_test, batch_size=TRAIN_BATCH_SIZE)
        # Save results
        loss_fit     = history.history["loss"]
        accuracy_fit = history.history["categorical_accuracy"]
        self.history.append({}) #append new empty directory
        self.history[-1].update(loss_fit = loss_fit)
        self.history[-1].update(accuracy_fit = accuracy_fit)
        self.history[-1].update(loss_test = loss_test)
        self.history[-1].update(accuracy_test = accuracy_test)
        np.save(os.path.join(self.client_dir, FL_CLIENTHIST_STR), self.history) #save fit history
        # Print round results
        results_kwargs = dict()
        results_kwargs.update(loss_fit = loss_fit)
        results_kwargs.update(loss_test = loss_test)
        results_kwargs.update(accuracy_fit = accuracy_fit)
        results_kwargs.update(accuracy_test = accuracy_test)
        print_fiteval(self.client_str, rnd, **results_kwargs)
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """
        Overriden by FL strategy's eval_fn in server.py
        """
        pass

def run_client(client_ind : int, clients_dir : str, dataset : list) -> None:
    """
    FL client flow
    """
    # Create client directory
    client_dir = os.path.join(clients_dir, str(client_ind))
    os.mkdir(client_dir)
    # Client string constant
    CLIENT_STR = "CLIENT " + str(client_ind)
    # Load and compile model
    model = createModel()  
    model.compile(
                    optimizer = TRAIN_OPTIMIZER,
                    loss      = TRAIN_LOSS,
                    metrics   = TRAIN_METRICS,
    )
    # Get datasets arrays
    train_dataset = dataset[0]
    test_dataset  = dataset[1]
    # Reformat output samples
    reformat_train_output(train_dataset)
    reformat_test_output(test_dataset)
    # Print shapes
    shapes_kwargs = dict()
    shapes_kwargs.update(train_dataset = train_dataset)
    shapes_kwargs.update(test_dataset = test_dataset)
    print_shape(CLIENT_STR, **shapes_kwargs)
    # Load FL client
    client = FLClient(client_ind, client_dir, model, train_dataset, test_dataset) #create client
    # Start federated learning
    fl.client.start_numpy_client(FL_SOCKET, client=client) #start client
