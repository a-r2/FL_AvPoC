from settings import *
from constants import *
from nn_algorithm import *
from utils import *

import flwr as fl
from tensorflow import keras 
import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

FLServerHist = list() #list to store dicts which contains evaluation results

def select_fl_strategy(server_dir : str, model : keras.Model, history : list, test_dataset : list):
    """
    Federated learning strategy

    Selects strategy, as defined by `FL_STRATEGY`

    Args:

        server_dir: server directory where server-side simulation results are stored (str)
        model: neural network model (keras.Model)
        history: simulation history where server-side evaluation results over the test dataset are stored (list)
        test_dataset: test input, output and SNR (x_test, y_test, SNR_test) (list)

    Returns:

        Federated learning strategy
    """
    if FL_STRATEGY == "FastAndSlow":
        strategy = fl.server.strategy.FastAndSlow(
            fraction_fit          = FL_FRACT_FIT,
            fraction_eval         = FL_FRACT_EVAL,
            min_fit_clients       = FL_CLIENTS_NUM,
            min_eval_clients      = FL_CLIENTS_NUM,
            min_available_clients = FL_CLIENTS_NUM,
            eval_fn               = get_eval_fn(server_dir, model, history, test_dataset),
            on_fit_config_fn      = fit_config,
            initial_parameters    = fl.common.weights_to_parameters(model.get_weights()),
        )
    elif FL_STRATEGY == "FedAdagrad":
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit          = FL_FRACT_FIT,
            fraction_eval         = FL_FRACT_EVAL,
            min_fit_clients       = FL_CLIENTS_NUM,
            min_eval_clients      = FL_CLIENTS_NUM,
            min_available_clients = FL_CLIENTS_NUM,
            eval_fn               = get_eval_fn(server_dir, model, history, test_dataset),
            on_fit_config_fn      = fit_config,
            initial_parameters    = fl.common.weights_to_parameters(model.get_weights()),
        )
    elif FL_STRATEGY == "FedAvg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit          = FL_FRACT_FIT,
            fraction_eval         = FL_FRACT_EVAL,
            min_fit_clients       = FL_CLIENTS_NUM,
            min_eval_clients      = FL_CLIENTS_NUM,
            min_available_clients = FL_CLIENTS_NUM,
            eval_fn               = get_eval_fn(server_dir, model, history, test_dataset),
            on_fit_config_fn      = fit_config,
            initial_parameters    = fl.common.weights_to_parameters(model.get_weights()),
        )
    elif FL_STRATEGY == "FedAdam":
        strategy = fl.server.strategy.FedAdam(
            fraction_fit          = FL_FRACT_FIT,
            fraction_eval         = FL_FRACT_EVAL,
            min_fit_clients       = FL_CLIENTS_NUM,
            min_eval_clients      = FL_CLIENTS_NUM,
            min_available_clients = FL_CLIENTS_NUM,
            eval_fn               = get_eval_fn(server_dir, model, history, test_dataset),
            on_fit_config_fn      = fit_config,
            initial_parameters    = fl.common.weights_to_parameters(model.get_weights()),
        )
    elif FL_STRATEGY == "FedYogi":
        strategy = fl.server.strategy.FedYogi(
            fraction_fit          = FL_FRACT_FIT,
            fraction_eval         = FL_FRACT_EVAL,
            min_fit_clients       = FL_CLIENTS_NUM,
            min_eval_clients      = FL_CLIENTS_NUM,
            min_available_clients = FL_CLIENTS_NUM,
            eval_fn               = get_eval_fn(server_dir, model, history, test_dataset),
            on_fit_config_fn      = fit_config,
            initial_parameters    = fl.common.weights_to_parameters(model.get_weights()),
        )
    elif FL_STRATEGY == "QFedAvg":
        strategy = fl.server.strategy.QFedAvg(
            fraction_fit          = FL_FRACT_FIT,
            fraction_eval         = FL_FRACT_EVAL,
            min_fit_clients       = FL_CLIENTS_NUM,
            min_eval_clients      = FL_CLIENTS_NUM,
            min_available_clients = FL_CLIENTS_NUM,
            eval_fn               = get_eval_fn(server_dir, model, history, test_dataset),
            on_fit_config_fn      = fit_config,
            initial_parameters    = fl.common.weights_to_parameters(model.get_weights()),
        )
    return strategy

def fit_config(rnd : int):
    """
    Client-side configuration for local fitting

    Sends configuration (round number) to FL clients

    Args:

        rnd: federated learning round (int)

    Returns:

        Fitting configuration (dict)
    """
    return {"rnd" : rnd}

def get_eval_fn(server_dir : str, model : keras.Model, history : list, test_dataset : list):
    """
    Server-side evaluation function

    Defines function to evaluate the aggregated model

    Args:

        server_dir: server directory where server-side simulation results are stored (str)
        model: neural network model (keras.Model)
        history: simulation history where server-side evaluation results over the test dataset are stored (list)
        test_dataset: test input, output and SNR (x_test, y_test, SNR_test) (list)

    Returns:

        Function for evaluating the aggregated model on the test dataset

    """
    # Get test dataset
    x_test, y_test, SNR_test = test_dataset
    # Define server-side evaluation function
    def evaluate(
        weights : fl.common.Weights,
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Evaluate new aggregated model
        model.set_weights(weights) #update model with the latest parameters
        loss_test, accuracy_test = model.evaluate(x_test, y_test) #evaluate the new model
        # Save results in history
        history.append({}) #append new empty dictionary
        history[-1].update(loss_test = loss_test)
        history[-1].update(accuracy_test = accuracy_test)
        np.save(os.path.join(server_dir, FL_SERVERHIST_STR), history) #save history
        rnd = int(len(history) - 1) #FL round
        # Print round results
        results_kwargs = dict()
        results_kwargs.update(loss_test = loss_test)
        results_kwargs.update(accuracy_test = accuracy_test)
        print_fiteval("SERVER", rnd, **results_kwargs)
        return loss_test, {"accuracy": accuracy_test}
    return evaluate

def run_server(server_dir : str, test_dataset : list, model_weights : fl.common.Weights) -> None:
    """
    FL server flow
    """
    # Load and compile model
    model = createModel()  
    model.compile(
                    optimizer = TRAIN_OPTIMIZER,
                    loss      = TRAIN_LOSS,
                    metrics   = TRAIN_METRICS,
    )
    if model_weights:
        model.set_weights(model_weights)
    # Reformat output samples
    reformat_test_output(test_dataset)
    # Print shapes
    shapes_kwargs = dict()
    shapes_kwargs.update(test_dataset = test_dataset)
    print_shape("SERVER", **shapes_kwargs)
    # Load FL strategy
    strategy = select_fl_strategy(server_dir, model, FLServerHist, test_dataset) #select FL strategy
    # Start federated learning
    #fl.server.grpc_server.grpc_server.grpc.insecure_channel(FL_SOCKET, options=(("grpc.enable_http_proxy","0"),))
    fl.server.start_server(FL_SOCKET, config={"num_rounds": FL_ROUNDS_NUM}, strategy=strategy) #start server
