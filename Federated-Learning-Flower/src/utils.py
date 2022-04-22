from settings import *
from constants import *

import numpy as np
import os

def one_hot(array : np.ndarray) -> np.ndarray:
    """
    One-hot encode

    Transforms the classifier's output from single-index format to one-dimensional array format

    Args:
    
        array: classifier's output(s) as a single element by integers ranging between [0, MODEL_Y_NUM - 1] (numpy array)

    Returns:

        Classifier's output(s) as an one-dimensional array of length MODEL_Y_NUM where the element corresponding to the classified output is 1 and all other elements are zero (numpy array)
    """
    array_1 = np.squeeze(np.eye(MODEL_Y_NUM)[array.reshape(-1)])
    return array_1

def norm_ts(x : np.ndarray) -> np.ndarray:
    """
    Timeseries normalization

    Normalizes timeseries within the range [-1, 1]

    Args:

        x: timeseries (numpy array)

    Returns:

        Normalized timeseries (numpy array)

    """
    x_max  = np.max(x)
    x_min  = np.min(x)
    x_norm = 2 * (x - 0.5 * (x_max + x_min)) / (x_max - x_min)
    return x_norm

def create_dirs():
    """
    Directories creation

    Create directories that store simulations results, i.e.: base directory, server directory, clients directory, global model directory and local model directory

    Returns:

       base_dir: simulation's base directory path (str)
       server_dir: server's directory path (str)
       clients_dir: clients' directory path (str)
       global_dir: global model's directory path (str)
       local_dir: local model's directory path (str)
    """
    # Check existence of results directory
    if not os.path.exists(RESBASEDIR_STR):
        os.mkdir(RESBASEDIR_STR)
    # Find last results directory
    dirs_list = os.listdir(RESBASEDIR_STR)
    if not dirs_list:
        base_dir = os.path.join(RESBASEDIR_STR,"1") 
    else:
        dirs_list_int = list(map(int, dirs_list))
        dirs_list_int.sort()
        last_ind = dirs_list_int[-1]
        new_ind  = last_ind + 1
        base_dir = os.path.join(RESBASEDIR_STR, str(new_ind))
    server_dir  = os.path.join(base_dir, "server")
    clients_dir = os.path.join(base_dir, "clients")
    global_dir  = os.path.join(base_dir, "global")
    local_dir   = os.path.join(base_dir, "local")
    # Create base, server and clients directories
    os.mkdir(base_dir) #store server and clients directories
    os.mkdir(server_dir) #store server-side results
    os.mkdir(clients_dir) #store clients-side results
    os.mkdir(global_dir) #store global model results
    os.mkdir(local_dir) #store local model results
    return base_dir, server_dir, clients_dir, global_dir, local_dir

def load_train_iid_partition(client : int, train_dataset : list) -> list:
    """
    Loading iid partition for training

    Loads a iid partition from the selected train dataset

    Args:

        client: index representing the client (from 0 to FL_CLIENTS_NUM) (int)
        train_dataset: train input, output and SNR (x_train, y_train, SNR_train) (list)

    Returns:

        Selected partition from train input, output and SNR (x_train_iid, y_train_iid, SNR_train_iid) (list)
    """
    assert client in FL_CLIENTS
    # Extract train samples
    x_train   = train_dataset[0]
    y_train   = train_dataset[1]
    SNR_train = train_dataset[2]
    # Partition indices
    part_from = int(client * TRAIN_SAMP_NUM / FL_CLIENTS_NUM)
    part_to   = int((client + 1) * TRAIN_SAMP_NUM / FL_CLIENTS_NUM)
    # Get partitions
    x_train_iid   = x_train[part_from : part_to]
    y_train_iid   = y_train[part_from : part_to]
    SNR_train_iid = SNR_train[part_from : part_to]
    # Cast y_train_iid from float to int
    y_train_iid = y_train_iid.astype("int")
    # Client's train iid dataset
    client_train_dataset = [x_train_iid, y_train_iid, SNR_train_iid]
    return client_train_dataset

def load_train_niid_partition(client : int, train_dataset : list, niid_dist : tuple) -> list:
    """
    Loading non-iid partition for training

    Loads a non-iid partition from the selected train dataset

    Args:

        client: index representing the client (from 0 to FL_CLIENTS_NUM) (int)
        train_dataset: train input, output and SNR (x_train, y_train, SNR_train) (list)
        niid_dist: selected non-iid distribution for the simulation (tuple)

    Returns:

        Selected partition from train input, output and SNR (x_train_niid, y_train_niid, SNR_train_niid) (list)
    """
    assert client in FL_CLIENTS
    # Extract train samples
    x_train   = train_dataset[0]
    y_train   = train_dataset[1]
    SNR_train = train_dataset[2]
    # Load non-iid distribution
    client_modclass_snr      = niid_dist[0]
    client_modclass_snr_part = niid_dist[1]
    # Extract client's train dataset
    x_train_niid, y_train_niid, SNR_train_niid = extract_train_niid_data(client, train_dataset, niid_dist)
    # Client's train non-iid dataset
    client_train_dataset = [x_train_niid, y_train_niid, SNR_train_niid]
    return client_train_dataset

def get_niid_distribution(clients_dir : str) -> tuple:
    """
    Non-iid distribution

    Defines non-iid distribution of the train dataset among clients

    Args:

        clients_dir: clients base directory path (str)

    Returns:

        Non-iid distribution for the simulation (tuple)        
    """
    # Lists
    clients_modclass    = list(list() for _ in MODEL_Y) #list of clients per modclass
    modclasses_client   = list(list() for _ in FL_CLIENTS) #list of modclasses per client
    modclasses_dyn      = list(MODEL_Y).copy() #dynamic list
    client_modclass_snr = np.frompyfunc(list, 0, 1)(np.empty((FL_CLIENTS_NUM, MODEL_Y_NUM), dtype=object)) #array of SNR list indexed by client and modclass
    # Randomly distribute modclasses among clients
    if FL_CLIENTS_NUM < MODEL_Y_NUM:
        while modclasses_dyn:
            for client in FL_CLIENTS:
                modclass_choice = np.random.choice(modclasses_dyn)    
                modclasses_client[client].append(modclass_choice)
                modclasses_dyn.remove(modclass_choice)
                if not modclasses_dyn:
                    break
    else:
        for client in FL_CLIENTS:
            modclass_choice = np.random.choice(modclasses_dyn)    
            modclasses_client[client].append(modclass_choice)
            modclasses_dyn.remove(modclass_choice)
            if not modclasses_dyn:
                modclasses_dyn = list(MODEL_Y).copy()
    # Overlap modclasses
    if FL_CLASSOVERLAP_NIID != "NONE":
        if FL_CLASSOVERLAP_NIID == "LOW": #25%
            overlap_modclasses_num = int(0.25 * MODEL_Y_NUM)
        elif FL_CLASSOVERLAP_NIID == "MID": #50%
            overlap_modclasses_num = int(0.5 * MODEL_Y_NUM)
        elif FL_CLASSOVERLAP_NIID == "HIGH": #75%
            overlap_modclasses_num = int(0.75 * MODEL_Y_NUM)
        overlap_modclasses = np.random.choice(MODEL_Y, overlap_modclasses_num, replace=False)
        for client in FL_CLIENTS:
            modclasses_client[client].extend(overlap_modclasses)
    for client in FL_CLIENTS:
        modclasses_client[client]= np.unique(modclasses_client[client]).tolist()
    # Randomly distribute SNR among clients
    for modclass in MODEL_Y:
        for client in FL_CLIENTS:
            if modclass in modclasses_client[client]:
                clients_modclass[modclass].append(client)
    for modclass in MODEL_Y:
        clients_modclass_num = len(clients_modclass[modclass])
        if clients_modclass_num > 1:
            if clients_modclass_num > MODEL_Y_NUM:
                snr_choices = np.random.choice(MODEL_SNR[1:-1], clients_modclass_num - 1, replace=True)
            else:
                snr_choices = np.random.choice(MODEL_SNR[1:-1], clients_modclass_num - 1, replace=False)
            snr_choices     = np.sort(snr_choices).tolist()
            snr_choices_num = len(snr_choices)
            if snr_choices_num == 1:
                client1 = clients_modclass[modclass][0]
                client2 = clients_modclass[modclass][-1]
                snr     = snr_choices[0]
                snr_ind = MODEL_SNR.index(snr)
                client_modclass_snr[client1][modclass].extend(MODEL_SNR[:snr_ind] if snr_ind > 0 else [0])
                client_modclass_snr[client2][modclass].extend(MODEL_SNR[snr_ind:])
            else:
                prev_snr     = snr_choices[0]
                prev_snr_ind = 0
                for i in range(snr_choices_num):
                    cur_snr     = snr_choices[i]
                    cur_snr_ind = MODEL_SNR.index(cur_snr)
                    client      = clients_modclass[modclass][i]
                    if i == 0:
                        client_modclass_snr[client][modclass].extend(MODEL_SNR[:cur_snr_ind + 1])
                    else:
                        if cur_snr == prev_snr:
                            client_modclass_snr[client][modclass].extend(MODEL_SNR[prev_snr_ind:cur_snr_ind + 1])

                        else:
                            client_modclass_snr[client][modclass].extend(MODEL_SNR[prev_snr_ind + 1:cur_snr_ind + 1])
                    prev_snr     = cur_snr
                    prev_snr_ind = cur_snr_ind
                client                                = clients_modclass[modclass][-1]
                last_snr                              = snr_choices[-1]
                last_snr_ind                          = MODEL_SNR.index(last_snr)
                client_modclass_snr[client][modclass].extend(MODEL_SNR[last_snr_ind + 1:])
        else:
            for client in clients_modclass[modclass]:
                client_modclass_snr[client][modclass] = list(MODEL_SNR)
    # Overlap SNR for clients sharing modclasses
    if FL_SNROVERLAP_NIID != "NONE":
        if FL_SNROVERLAP_NIID == "LOW": #25%
            overlap_snr_num = int(0.25 * MODEL_SNR_NUM)
        elif FL_SNROVERLAP_NIID == "MID": #50%
            overlap_snr_num = int(0.5 * MODEL_SNR_NUM)
        elif FL_SNROVERLAP_NIID == "HIGH": #75%
            overlap_snr_num = int(0.75 * MODEL_SNR_NUM)
        for client in FL_CLIENTS:
            for modclass in MODEL_Y:
                snr_list = client_modclass_snr[client][modclass]
                if snr_list:
                    if not ((MODEL_SNR[0] in snr_list) and (MODEL_SNR[-1] in snr_list)):
                        if (MODEL_SNR[0] in snr_list) and (MODEL_SNR[-1] not in snr_list):
                            max_snr      = snr_list[-1]
                            max_snr_ind  = MODEL_SNR.index(max_snr)
                            ext_snr_list = MODEL_SNR[max_snr_ind + 1:max_snr_ind + overlap_snr_num + 1]
                        elif (MODEL_SNR[0] not in snr_list) and (MODEL_SNR[-1] in snr_list):
                            min_snr      = snr_list[0]
                            min_snr_ind  = MODEL_SNR.index(min_snr)
                            ext_snr_list = MODEL_SNR[min_snr_ind - overlap_snr_num:min_snr_ind]
                        else:
                            max_snr      = snr_list[-1]
                            max_snr_ind  = MODEL_SNR.index(max_snr)
                            min_snr      = snr_list[0]
                            min_snr_ind  = MODEL_SNR.index(min_snr)
                            ext_snr_list = MODEL_SNR[max(0, min_snr_ind - (overlap_snr_num // 2)):min(MODEL_SNR_NUM, max_snr_ind + (overlap_snr_num // 2) + 1)]
                        client_modclass_snr[client][modclass].extend(ext_snr_list)
                        client_modclass_snr[client][modclass] = np.unique(client_modclass_snr[client][modclass]).tolist()
    # Part dataset
    client_modclass_snr_part = np.frompyfunc(tuple, 0, 1)(np.empty((FL_CLIENTS_NUM, MODEL_Y_NUM, MODEL_SNR_NUM), dtype=object))
    for modclass in MODEL_Y:
        clients_modclass_list = clients_modclass[modclass]
        if clients_modclass_list:
            snr_aggregate = list()
            for client in clients_modclass_list:
                snr_aggregate.extend(client_modclass_snr[client][modclass])
            snr_aggregate.sort()
            snr_count = np.zeros(MODEL_SNR_NUM)
            for snr in snr_aggregate:
                if snr in MODEL_SNR:
                    snr_ind = MODEL_SNR.index(snr)
                    snr_count[snr_ind] += 1
            snr_client = client_modclass_snr[:, modclass] #snr lists per client
            clients_in_snr_func = np.frompyfunc(lambda snr: [snr in snr_client[client] for client in FL_CLIENTS], 1, 1)
            FL_CLIENTS_NP       = np.array(FL_CLIENTS)
            for client in clients_modclass_list:
                snr_list   = client_modclass_snr[client, modclass] #snr list of current client
                for snr in snr_list:
                    if snr in snr_list:
                        snr_found_in_client = np.array(clients_in_snr_func(snr)) #
                        clients_in_snr      = FL_CLIENTS_NP[snr_found_in_client] #list of clients sharing modclass and snr
                        clients_in_snr_num  = len(clients_in_snr)
                        client_snr_ind      = np.where(clients_in_snr == client)
                        client_snr_ind      = int(client_snr_ind[0]) #from numpy array to int
                        snr_ind             = MODEL_SNR.index(snr)
                        client_modclass_snr_part[client, modclass, snr_ind] = (client_snr_ind, int(snr_count[snr_ind])) #(index of the client sharing same modclass and snr, number of clients sharing same modclass and snr)
    np.savez(os.path.join(clients_dir, FL_NIIDIST_STR), client_modclass_snr = client_modclass_snr, client_modclass_snr_part = client_modclass_snr_part) #save non-iid distribution in clients directory
    niid_dist = (client_modclass_snr, client_modclass_snr_part)
    return niid_dist

def extract_train_niid_data(client : int, train_dataset : list, niid_dist : tuple) -> list:
    """
    Train non-iid data

    Extracts data from train dataset as defined by the non-iid distribution

    Args:

        client: index representing the client (from 0 to FL_CLIENTS_NUM) (int)
        train_dataset: train input, output and SNR (x_train, y_train, SNR_train) (list)
        niid_dist: selected non-iid distribution for the simulation (tuple)

    Returns:

        Selected partition from train input, output and SNR (x_train_niid, y_train_niid, SNR_train_niid) (list)
    """
    # Extract train samples
    x_train   = train_dataset[0]
    y_train   = train_dataset[1]
    SNR_train = train_dataset[2]
    # Extract non-iid distribution definition of the client
    modclass_snr      = niid_dist[0][client]
    modclass_snr_part = niid_dist[1][client]
    # Extract partition indexes for the client
    modclasses_list = list(range(MODEL_Y_NUM))
    x_train_niid    = np.array([], dtype=float)
    y_train_niid    = np.array([], dtype=int)
    SNR_train_niid  = np.array([], dtype=int)
    extract_ind     = np.array([])
    for modclass in modclasses_list:
        snr_list = modclass_snr[modclass]
        if len(snr_list) > 0:
            for snr in snr_list:
                snr_ind   = list(MODEL_SNR).index(snr)
                ind_parts = modclass_snr_part[modclass][snr_ind]
                if len(ind_parts) > 0:
                    part_ind          = ind_parts[0]
                    part_fract        = ind_parts[1]
                    poss_modclass_ind = np.where(y_train == modclass)[0] #all possible indexes by modclass
                    poss_snr_ind      = np.where(SNR_train == snr)[0] #all possible indexes by SNR
                    poss_ind          = [ind for ind in poss_modclass_ind if ind in poss_snr_ind] #all possible indexes by modclass and SNR
                    poss_ind_num      = len(poss_ind) #number of all possible indexes by modclass and SNR
                    if part_fract == 1:
                        extract_ind = poss_ind
                    else:
                        from_poss_ind = int(part_ind * poss_ind_num / part_fract)
                        to_poss_ind   = int((part_ind + 1) * poss_ind_num / part_fract)
                        extract_ind   = poss_ind[from_poss_ind : to_poss_ind]
                    x_train_niid   = np.append(x_train_niid, x_train[extract_ind])
                    y_train_niid   = np.append(y_train_niid, y_train[extract_ind])
                    SNR_train_niid = np.append(SNR_train_niid, SNR_train[extract_ind])
    x_train_niid_len = len(x_train_niid)
    x_train_niid_1   = np.reshape(x_train_niid, (int(x_train_niid_len / (MODEL_X_SHAPE[0] * MODEL_X_SHAPE[1])), MODEL_X_SHAPE[0], MODEL_X_SHAPE[1]))
    return x_train_niid_1, y_train_niid, SNR_train_niid

def load_partial_dataset() -> list:
    """
    Loading train and test partitions

    Loads part of the train and test datasets as defined by TRAIN_SAMP_NUM and TEST_SAMP_NUM

    IMPORTANT: source train and test datasets must be independent and identically distributed (iid)
    
    Returns:

        Selected partition from train and test input, output and SNR (train_dataset, test_dataset) (list)

    """
    # Make TensorFlow logs less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Load complete train dataset
    npzfile_train = np.load(TRAIN_DATASET_FILE)
    X_train       = npzfile_train['X'] #(samples_num, sample_size, I_Q) = (350448, 1024, 2)
    Y_train       = npzfile_train['Y'] #(samples_num, sample_size, I_Q) = (350448,)
    SNR_train     = npzfile_train['SNR'] #(samples_num,) = (350448,)
    # Load complete test dataset
    npzfile_test   = np.load(TEST_DATASET_FILE)
    X_test         = npzfile_test['X_test'] #(samples_num, sample_size, I_Q) = (10000, 1024, 2)
    Y_test         = npzfile_test['Y_test'] #(samples_num, sample_size, I_Q) = (10000,)
    SNR_test       = npzfile_test['SNR_test'] #(samples_num,) = (10000,)
    # Get selected (partial) train dataset
    x_train   = X_train[:TRAIN_SAMP_NUM]
    y_train   = Y_train[:TRAIN_SAMP_NUM]
    SNR_train = SNR_train[:TRAIN_SAMP_NUM]
    # Get selected (partial) test dataset
    x_test   = X_test[:TEST_SAMP_NUM]
    y_test   = Y_test[:TEST_SAMP_NUM]
    SNR_test = SNR_test[:TEST_SAMP_NUM]
    # Train dataset
    train_dataset = [x_train, y_train, SNR_train]
    # Test dataset
    test_dataset = [x_test, y_test, SNR_test]
    # Dataset
    dataset = [train_dataset, test_dataset]
    return dataset

def reformat_train_output(train_dataset: list):
    """
    Reformat train output

    Apply one-hot encode to train output samples

    Args:

        train_dataset: train input, output and SNR (x_train, y_train, SNR_train) (list)
    """
    # Extract train output samples
    y_train = train_dataset[1]
    # Reformat output labels
    y_train_1 = one_hot(y_train)
    # Replace output samples
    train_dataset[1] = y_train_1

def reformat_test_output(test_dataset: list):
    """
    Reformat test output

    Apply one-hot encode to test output samples

    Args:

        test_dataset: test input, output and SNR (x_test, y_test, SNR_test) (list)
    """
    # Extract test output samples
    y_test = test_dataset[1]
    # Reformat output labels
    y_test_1 = one_hot(y_test)
    # Replace output samples
    test_dataset[1] = y_test_1

def print_assert_fl_settings():
    """
    FL settings printing
    
    Prints FL-related settings
    """
    data_str = list()
    data_str.append("".join(("FL_CLIENTS: ", str(FL_CLIENTS), "\n", "FL_ROUNDS: ", str(FL_ROUNDS), "\n", "FL_EPOCHS: ", str(FL_EPOCHS), "\n", "FL_LR: ", str(FL_LR))))
    data_str          = "".join(data_str)
    fl_settings_print = "".join(("\n", SEP_STR, "\n\n", data_str, "\n\n", SEP_STR, "\n"))
    print(fl_settings_print)
    try:
        assert len(FL_CLIENTS) > 1
        assert len(FL_ROUNDS) > 1
        assert len(FL_EPOCHS) > 1
        assert len(FL_LR) > 1
    except:
        print("INVALID SETTINGS\n")
        quit()

def print_shape(type_str : str, **kwargs):
    """
    Dataset shapes printing
    
    Prints datasets' shapes

    Args:

        type_str: print title (str)
        kwargs: training and testing datasets (dict)
    """
    data_str = list()
    for key, value in kwargs.items():
        if key == "train_dataset":
            x_train     = value[0]
            y_train     = value[1]
            SNR_train   = value[2]
            data_str.append("".join(("(", type_str, ") x_train: ", str(x_train.shape), "\n(", type_str, ") y_train: ", str(y_train.shape), "\n(", type_str, ") SNR_train: ", str(SNR_train.shape), "\n")))
        elif key == "test_dataset":
            x_test      = value[0]
            y_test      = value[1]
            SNR_test    = value[2]
            data_str.append("".join(("(", type_str, ") x_test: ", str(x_test.shape), "\n(", type_str, ") y_test: ", str(y_test.shape), "\n(", type_str, ") SNR_test: ", str(SNR_test.shape), "\n")))
    data_str    = "".join(data_str)
    shape_print = "".join(("\n", SEP_STR, "\n\n", data_str, "\n", SEP_STR, "\n"))
    print(shape_print)

def print_fiteval(type_str : str, rnd : int, **kwargs):
    """
    Fitting and evaluation printing
    
    Prints fitting and evaluation results

    Args:

        type_str: print title (str)
        rnd: FL rounds (int)
        kwargs: fitting and evaluation results (dict)
    """
    data_str = list()
    for key, value in kwargs.items():
        if key == "loss_fit":
            loss_fit = value
            data_str.append("".join(("Loss (fit): ", str(loss_fit), "\n")))
        elif key == "loss_test":
            loss_test = value
            data_str.append("".join(("Loss (test): ", str(loss_test), "\n")))
        elif key == "accuracy_fit":
            accuracy_fit = value
            data_str.append("".join(("Accuracy (fit): ", str(accuracy_fit), "\n")))
        elif key == "accuracy_test":
            accuracy_test = value
            data_str.append("".join(("Accuracy (test): ", str(accuracy_test), "\n")))
    data_str      = "".join(data_str)
    fiteval_print = "".join(("\n", SEP_STR, "\n\n", type_str, " | ROUND ", str(rnd), ":\n\n", data_str, "\n", SEP_STR, "\n"))
    print(fiteval_print)
