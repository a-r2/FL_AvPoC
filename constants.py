from settings import *

import numpy as np

# Directory/file constants
CSVFILE_STR       = "settings.csv" #settings
FL_NIIDIST_STR    = "niid_distribution.npz" #definition of non-iid distribution | do not forget ".npz"
FL_CLIENTHIST_STR = "client_history.npy" #client history (fit, val, test) | do not forget ".npy"
FL_SERVERHIST_STR = "server_history.npy" #server history (test) | do not forget ".npy"
GLOBALHIST_STR    = "global_history.npy" #global history (fit, val, test) | do not forget ".npy"
LOCALHIST_STR     = "local_history.npy" #local history (fit, val, test) | do not forget ".npy"
MODEL_BEST_STR    = "best_model.h5" #best local model | do not forget ".h5"
RESBASEDIR_STR    = "trained_models" #base directory where simulations results are stored

# Neural network constants
MODEL_X_SHAPE = (1024, 2) #input shape of neural network
MODEL_Y_NUM   = 8 #output length of neural network
MODEL_Y       = tuple(range(MODEL_Y_NUM)) #possible outputs
MODEL_SNR     = tuple(range(0,22,2)) #possible SNR
MODEL_SNR_NUM = len(MODEL_SNR) #number of possible SNR

# Federated learning constants
FL_CLIENTS = np.arange(FL_CLIENTS_NUM) #possible clients
FL_ROUNDS  = np.arange(1, FL_ROUNDS_NUM + 1)
FL_EPOCHS  = (TRAIN_EPOCHS_INIT + np.frompyfunc(lambda x : (FL_EPOCHINCR * x) ** FL_EPOCHORD, 1, 1)(np.repeat(FL_ROUNDS[:FL_ROUNDS_NUM // FL_EPOCHREP] - 1, FL_EPOCHREP))).astype(int)
FL_LR      = TRAIN_LR_INIT - np.frompyfunc(lambda x : (FL_LRDECR * x) ** FL_LRORD, 1, 1)(np.repeat(FL_ROUNDS[:FL_ROUNDS_NUM // FL_LREP] - 1, FL_LREP))

# Print
SEP_STR = "############################################################################" #separator to better visualize prints from this program
