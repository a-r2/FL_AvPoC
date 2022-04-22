from settings import *
from utils import *

import numpy as np

# Local train dataset
npzfile = np.load(TRAIN_DATASET_FILE)
X       = npzfile['X'] #(samples_num, sample_size, I_Q) = (350448, 1024, 2)
Y       = npzfile['Y'] #(samples_num, sample_size, I_Q) = (350448,)
SNR     = npzfile['SNR'] #(samples_num,) = (350448,)
classes = npzfile['classes'] #(classes_num,) = (8,)

# Preprocessing train dataset 
x_train, y_train, SNR = X[:TRAIN_SAMP_NUM], Y[:TRAIN_SAMP_NUM], SNR[:TRAIN_SAMP_NUM]

x_train_cum     = np.array([])
y_train_cum     = np.array([])
SNR_cum         = np.array([])
niid_dist_tuple = get_niid_distribution(".")
for client in range(FL_CLIENTS_NUM):
    x_train_niid, y_train_niid, SNR_niid = extract_train_niid_data(client, (x_train, y_train, SNR), niid_dist_tuple)
    x_train_cum                          = np.append(x_train_cum, x_train_niid)
    y_train_cum                          = np.append(y_train_cum, y_train_niid)
    SNR_cum                              = np.append(SNR_cum, SNR_niid)
x_train_cum_len = len(x_train_cum)
x_train_cum     = np.reshape(x_train_cum, (int(x_train_cum_len / (MODEL_X_SHAPE[0] * MODEL_X_SHAPE[1])), MODEL_X_SHAPE[0], MODEL_X_SHAPE[1]))
print(": ".join(("client_modclass_snr", str(niid_dist_tuple[0]))))
print(": ".join(("client_modclass_snr_part", str(niid_dist_tuple[1]))))
print("\n\n".join(("x_train_cum shape:", str(x_train_cum.shape), "")))
print("\n\n".join(("y_train_cum shape:", str(y_train_cum.shape), "")))
print("\n\n".join(("SNR_cum shape:", str(SNR_cum.shape), "")))
print("\n\n".join(("x_train len diff:", str(len(x_train_cum) - len(x_train)), "")))
print("\n\n".join(("y_train len diff:", str(len(y_train_cum) - len(y_train)), "")))
print("\n\n".join(("SNR len diff:", str(len(SNR_cum) - len(SNR)), "")))
x_train_diff = np.setdiff1d(x_train, x_train_cum)
y_train_diff = np.setdiff1d(y_train, y_train_cum)
SNR_diff     = np.setdiff1d(SNR, SNR_cum)
print("\n\n".join(("x_train diff:", str(x_train_diff), "")))
print("\n\n".join(("y_train diff:", str(y_train_diff), "")))
print("\n\n".join(("SNR diff:", str(SNR_diff), "")))
