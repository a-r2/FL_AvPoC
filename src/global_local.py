from settings import *
from constants import *
from nn_algorithm import *
from utils import *

def run_global_local(proc_type : str, global_local_dir : str, dataset : list):
    """
    Global and local training and evaluation

    Trains and evaluates a global or local model, as defined by `proc_type`, on the complete training dataset and on client 0's training dataset respectively, and evaluates the model on the complete testing dataset

    Args:

        dataset: type of model (GLOBAL/LOCAL) (str)
        global_local_dir: global/local directory (str)
        dataset: train and test datasets (list)
    """
    if proc_type == "global":
        PROC_TYPE_STR = "GLOBAL"
        GLOCHIST_STR  = GLOBALHIST_STR
    elif proc_type == "local":
        PROC_TYPE_STR = "LOCAL"
        GLOCHIST_STR  = LOCALHIST_STR
    # Load and compile model
    model = createModel()  
    model.compile(
                    optimizer = TRAIN_OPTIMIZER,
                    loss      = TRAIN_LOSS,
                    metrics   = TRAIN_METRICS,
    )
    # Fit callbacks
    fit_callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(global_local_dir, MODEL_BEST_STR),
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
    # Extract datasets
    train_dataset = dataset[0]
    test_dataset  = dataset[1]
    # Reformat output samples
    reformat_train_output(train_dataset)
    reformat_test_output(test_dataset)
    # Extract train samples
    x_train   = train_dataset[0]
    y_train   = train_dataset[1]
    SNR_train = train_dataset[2]
    # Extract test samples
    x_test   = test_dataset[0]
    y_test   = test_dataset[1]
    SNR_test = test_dataset[2]
    # Print shapes
    shapes_kwargs = dict()
    shapes_kwargs.update(train_dataset = train_dataset)
    shapes_kwargs.update(test_dataset = test_dataset)
    print_shape(PROC_TYPE_STR, **shapes_kwargs)
    # Initialize history
    global_local_history = list() #initialize as list
    # Loop
    for rnd in FL_ROUNDS:
        # Modify learning rate
        model.optimizer.lr = FL_LR[rnd - 1]
        # Fit the model on training dataset
        history = model.fit(
            x_train,
            y_train,
            batch_size       = TRAIN_BATCH_SIZE,
            epochs           = FL_EPOCHS[rnd - 1],
            callbacks        = fit_callbacks,
        )
        # Evaluate on test dataset
        loss_test, accuracy_test = model.evaluate(x_test, y_test, batch_size=TRAIN_BATCH_SIZE)
        # Save results
        loss_fit     = history.history["loss"]
        accuracy_fit = history.history["categorical_accuracy"]
        global_local_history.append({}) #append empty dictionary
        global_local_history[-1].update(loss_fit = loss_fit)
        global_local_history[-1].update(accuracy_fit = accuracy_fit)
        global_local_history[-1].update(loss_test = loss_test)
        global_local_history[-1].update(accuracy_test = accuracy_test)
        np.save(os.path.join(global_local_dir, GLOCHIST_STR), global_local_history) #save fit history
        # Print round results
        results_kwargs = dict()
        results_kwargs.update(loss_fit = loss_fit)
        results_kwargs.update(loss_test = loss_test)
        results_kwargs.update(accuracy_fit = accuracy_fit)
        results_kwargs.update(accuracy_test = accuracy_test)
        print_fiteval(PROC_TYPE_STR, rnd, **results_kwargs)
