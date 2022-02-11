# Dataset parameters
TRAIN_DATASET_FILE = "../Datasets/DataSet_train.npz" #train dataset path
TEST_DATASET_FILE  = "../Datasets/DataSet_test.npz" #test dataset path

# Pretraining parameters
PRETRAIN            = "NONE" #"NONE", "LOW", "MID", "HIGH"
PRETRAIN_BATCH_SIZE = 32 #batch size
PRETRAIN_EPOCHS     = 1 #number of epochs
PRETRAIN_LOSS       = "categorical_crossentropy" #loss type
PRETRAIN_METRICS    = ["categorical_accuracy"] #pretrain metric
PRETRAIN_OPTIMIZER  = "adam" #pretrain optimizer

# Train global
TRAIN_GLOBAL = True #False, True

# Train local 
TRAIN_LOCAL = True #False, True

# Training parameters
TRAIN_SAMP_NUM    = 6000 #number of train samples (max = 350448)
TRAIN_BATCH_SIZE  = 32 #batch size
TRAIN_EPOCHS_INIT = 1 #number of epochs
TRAIN_LR_INIT     = 1e-3 #learning rate
TRAIN_LOSS        = "categorical_crossentropy" #loss type
TRAIN_METRICS     = ["categorical_accuracy"] #train metric
TRAIN_OPTIMIZER   = "adam" #train optimizer

# Test parameters
TEST_SAMP_NUM = 2000 #number of test samples (max = 10000)

# Federated learning parameters
FL_CLIENTS_NUM       = 2 #number of clients
FL_CLASSOVERLAP_NIID = "NONE" #"NONE", "LOW", "MID", "HIGH"
FL_DISTYPE           = "iid" #"iid", "non-iid"
FL_SNROVERLAP_NIID   = "NONE" #"NONE", "LOW", "MID", "HIGH"
FL_FRACT_EVAL        = 1 #percentage of clients needed to eval global model
FL_FRACT_FIT         = 1 #percentage of clients needed to fit global model
FL_ROUNDS_NUM        = 3 #number of rounds
FL_SOCKET            = "127.0.0.1:8080" #FL socket (ip:port)
FL_STRATEGY          = "FedAvg" #"FastAndSlow", "FedAdagrad", "FedAvg", "FedAdam", "FedYogi", "QFedAvg"
FL_EPOCHINCR         = 0 #increment of epochs per round [0, Inf)
FL_EPOCHORD          = 1 #polynomial order of epochs increment per round [0, Inf)
FL_EPOCHREP          = 1 #repeat epochs for FL_LREP rounds [1, FL_ROUNDS_NUM]
FL_LRDECR            = 0 #decrement of learning rate per round
FL_LRORD             = 1 #polynomial order of learning rate decrement per round [0, Inf)
FL_LREP              = 1 #repeat learning rate for FL_LREP rounds [1, FL_ROUNDS_NUM]
