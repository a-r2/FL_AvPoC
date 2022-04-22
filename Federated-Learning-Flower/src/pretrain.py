from settings import *
from constants import *
from nn_algorithm import *
from utils import *

def run_pretrain(dataset : list) -> list:
    """
    Pretraining

    Trains the model taking a partition from `dataset`'s training dataset (0%/25%/50%/75% as defined by `PRETRAIN`) and evaluates the model on `dataset`'s testing dataset (the complete simulation's testing dataset)

    Args:

        dataset: training and testing datasets (list)

    Returns:

        Resulting weights of pretrained model (list)
    """
    # Load and compile model
    model = createModel()  
    model.compile(
                    optimizer = PRETRAIN_OPTIMIZER,
                    loss      = PRETRAIN_LOSS,
                    metrics   = PRETRAIN_METRICS,
    )
    # Copy datasets
    train_dataset = dataset[0].copy()
    test_dataset  = dataset[1].copy()
    # Get pretrain partition
    x_train   = train_dataset[0]
    y_train   = train_dataset[1]
    SNR_train = train_dataset[2]
    if PRETRAIN == "LOW":
        part_to = int(0.25 * TRAIN_SAMP_NUM)
    elif PRETRAIN == "MID":
        part_to = int(0.5 * TRAIN_SAMP_NUM)
    elif PRETRAIN == "HIGH":
        part_to = int(0.75 * TRAIN_SAMP_NUM)
    x_train   = x_train[:part_to]
    y_train   = y_train[:part_to]
    SNR_train = SNR_train[:part_to]
    y_train   = y_train.astype("int")
    train_dataset = [x_train, y_train, SNR_train]
    # Reformat output samples
    reformat_train_output(train_dataset)
    reformat_test_output(test_dataset)
    # Get train samples
    x_train   = train_dataset[0]
    y_train   = train_dataset[1]
    SNR_train = train_dataset[2]
    # Get test samples
    x_test   = test_dataset[0]
    y_test   = test_dataset[1]
    SNR_test = test_dataset[2]
    # Print shapes
    shapes_kwargs = dict()
    shapes_kwargs.update(train_dataset = train_dataset)
    shapes_kwargs.update(test_dataset = test_dataset)
    print_shape("PRETRAIN", **shapes_kwargs)
    # Fit the model on train dataset
    history = model.fit(
        x_train,
        y_train,
        batch_size       = PRETRAIN_BATCH_SIZE,
        epochs           = PRETRAIN_EPOCHS,
        validation_split = PRETRAIN_VAL_SPLIT,
    )
    # Get results
    loss_test, accuracy_test = model.evaluate(x_test, y_test, batch_size=TRAIN_BATCH_SIZE) #evaluate fitted model on test dataset
    loss_fit                 = history.history["loss"]
    accuracy_fit             = history.history["categorical_accuracy"]
    loss_val                 = history.history["val_loss"]
    accuracy_val             = history.history["val_categorical_accuracy"]
    # Print results
    print("\n")
    print("PRETRAIN:\n")
    print("Loss (fit): " + str(loss_fit))
    print("Loss (val): " + str(loss_val))
    print("Loss (test): " + str(loss_test))
    print("Accuracy (fit): " + str(accuracy_fit))
    print("Accuracy (val): " + str(accuracy_val))
    print("Accuracy (test): " + str(accuracy_test))
    print("\n")
    return model.get_weights()
