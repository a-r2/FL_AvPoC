from settings import *
from constants import *

from tensorflow import keras 

# Neural network model
def createModel() -> keras.Model:
    """
    Timeseries classifier

    Creates a 4-layers convolutional neural network in order to classify timeseries 

    Returns:

        Neural network model (keras.Model)
    """
    # Input layer
    input_layer  = keras.layers.Input(MODEL_X_SHAPE)        
    # First convolutional layer
    conv1        = keras.layers.Conv1DTranspose(filters=16, kernel_size=3, padding="same")(input_layer)
    conv1        = keras.layers.BatchNormalization()(conv1)
    conv1        = keras.layers.ReLU()(conv1)
    # Second convolutional layer
    conv2        = keras.layers.Conv1D(filters=16, kernel_size=3, padding="same")(conv1)
    conv2        = keras.layers.BatchNormalization()(conv2)
    conv2        = keras.layers.ReLU()(conv2)
    # Third convolutional layer
    conv3        = keras.layers.Conv1D(filters=16, kernel_size=3, padding="same")(conv2)
    conv3        = keras.layers.BatchNormalization()(conv3)
    conv3        = keras.layers.ReLU()(conv3)
    # Fourth convolutional layer
    conv4        = keras.layers.Conv1D(filters=16, kernel_size=3, padding="same")(conv3)
    conv4        = keras.layers.BatchNormalization()(conv4)
    conv4        = keras.layers.ReLU()(conv4)
    # Pooling Layer
    gap          = keras.layers.GlobalAveragePooling1D()(conv4)
    # Fully connected layer - Softmax classifier
    output_layer = keras.layers.Dense(MODEL_Y_NUM, activation="softmax")(gap)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)    
