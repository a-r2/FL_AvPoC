import h5py
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

#Script parameters
PLOT          = True
PLOT_SAMP_IND = 0

EVAL           = False
EVAL_MODEL_DIR = '1'

TRAIN            = False
TRAIN_SAMP_NUM   = 1000
TRAIN_TEST_FRACT = 0.9 #% over dataset
TRAIN_VAL_FRACT  = 0.2 #% over test samples
TRAIN_EPOCHS     = 10
TRAIN_BATCH_SIZE = 32
TRAIN_OPTIMIZER  = "adam"
TRAIN_LOSS       = "categorical_crossentropy"
TRAIN_METRICS    = ["categorical_accuracy"]

DIR_MODELS_STR    = "trained_models"
FILE_MODEL_STR    = "best_model.h5"
HISTORY_MODEL_STR = "train_history.npy"

#Plots parameters
mpl.rcParams['axes.labelsize']    = 14
mpl.rcParams['axes.titlesize']    = 16
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.titlesize']  = 16
mpl.rcParams['legend.fontsize']   = 14
mpl.rcParams['lines.linestyle']   = '-'
mpl.rcParams['lines.linewidth']   = 1
mpl.rcParams['lines.marker']      = '.'
mpl.rcParams['lines.markersize']  = 2
mpl.rcParams['xtick.labelsize']   = 12
mpl.rcParams['ytick.labelsize']   = 12

#Dataset files
dataset_file = "2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
classes_file = "2018.01/classes.txt"

#Get labels (modulation classes)
with open(classes_file, "r") as f:
    lines = str()
    for line in f:
        line  = line.rstrip() #remove newline
        lines += line
f.close()
exec(lines) #create classes list
classes = np.array(classes) #convert to numpy array

#Get dataset (IQ samples, ground truth classes and SNR)
with h5py.File(dataset_file, "r") as f:
    #Get h5df keys
    h5df_keys = list(f.keys())
    IQ_key    = h5df_keys[0]
    GT_key    = h5df_keys[1]
    SNR_key   = h5df_keys[2]
    #Get h5df data
    IQ  = f[IQ_key][:TRAIN_SAMP_NUM] #IQ samples
    GT  = f[GT_key][:TRAIN_SAMP_NUM] #ground truth classes
    SNR = f[SNR_key][:TRAIN_SAMP_NUM] #SNR
f.close()

num_classes   = len(classes)
traintest_ind = int(TRAIN_TEST_FRACT * TRAIN_SAMP_NUM)

if PLOT:
    #Plot IQ sample
    GT_label = classes[np.nonzero(GT[PLOT_SAMP_IND,:])][0]
    fig = plt.figure('Dataset sample #' + str(PLOT_SAMP_IND))
    fullwidth, fullheight = fig.canvas.manager.window.winfo_screenwidth(), fig.canvas.manager.window.winfo_screenheight()
    fig.canvas.manager.window.wm_geometry('%dx%d+0+0' % (fullwidth, fullheight))
    plt.suptitle(GT_label)
    plt.plot(IQ[PLOT_SAMP_IND,:,0])
    plt.plot(IQ[PLOT_SAMP_IND,:,1])
    plt.xlabel('Samples [-]')
    plt.ylabel('In-phase [-] | Quadrature [-]')
    plt.legend(["I", "Q"], loc="best")
    plt.grid()
    plt.autoscale(enable=True, tight=True)
    plt.show()

if TRAIN:
    #Create directory
    if not os.path.exists(DIR_MODELS_STR):
        os.mkdir(DIR_MODELS_STR)
    dirs_list = os.listdir(DIR_MODELS_STR)
    if not dirs_list:
        model_dir = os.path.join(DIR_MODELS_STR,"1") 
    else:
        last_ind  = int(dirs_list[-1])
        new_ind   = last_ind + 1
        model_dir = os.path.join(DIR_MODELS_STR, str(new_ind))
    os.mkdir(model_dir)
    #Preprocessing
    x_train = IQ[:traintest_ind]
    y_train = GT[:traintest_ind]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 2))
    idx     = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]
    #Construct model
    input_shape  = x_train.shape[1:]
    input_layer  = keras.layers.Input(input_shape)
    norm_layer   = keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", gamma_initializer="ones", beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(input_layer)
    conv1        = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(norm_layer)
    conv1        = keras.layers.BatchNormalization()(conv1)
    conv1        = keras.layers.ReLU()(conv1)
    conv2        = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2        = keras.layers.BatchNormalization()(conv2)
    conv2        = keras.layers.ReLU()(conv2)
    conv3        = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3        = keras.layers.BatchNormalization()(conv3)
    conv3        = keras.layers.ReLU()(conv3)
    gap          = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
    model        = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    #Train model
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, FILE_MODEL_STR),
            save_best_only=True,
            monitor="val_loss",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            min_lr=0.0001,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=50,
            verbose=1,
        ),
    ]
    model.compile(
        optimizer=TRAIN_OPTIMIZER,
        loss=TRAIN_LOSS,
        metrics=TRAIN_METRICS,
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=TRAIN_BATCH_SIZE,
        epochs=TRAIN_EPOCHS,
        callbacks=callbacks,
        validation_split=TRAIN_VAL_FRACT,
        verbose=1,
    )
    np.save(os.path.join(model_dir, HISTORY_MODEL_STR), history.history)

if EVAL:
    #Preprocessing
    x_test = IQ[traintest_ind:]
    y_test = GT[traintest_ind:]
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 2))
    #Load model
    try:
        assert(model)
    except:
        model_path = os.path.join(DIR_MODELS_STR, EVAL_MODEL_DIR, FILE_MODEL_STR)
        model      = keras.models.load_model(model_path)
    #Load train history
    try:
        assert(history)
    except:
        history_path = os.path.join(DIR_MODELS_STR, EVAL_MODEL_DIR, HISTORY_MODEL_STR)
        history = np.load(history_path, allow_pickle='TRUE').item()
    #Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(": ".join("Test accuracy", test_acc))
    print(": ".join("Test loss", test_loss))
    #Plot evaluation
    for metric in TRAIN_METRICS:
        fig = plt.figure('Model evaluation')
        fullwidth, fullheight = fig.canvas.manager.window.winfo_screenwidth(), fig.canvas.manager.window.winfo_screenheight()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.ylabel(metric + " [-]", fontsize="large")
        plt.xlabel("Epoch [-]", fontsize="large")
        plt.legend(["Train", "Eval"], loc="best")
        plt.grid()
        plt.autoscale(enable=True, tight=True)
        plt.show()
