from constants import *

import csv
import os

def save_settings(base_dir):
    """
    Settings saving

    Stores simulation settings in a CSV file

    Args:

        base_dir: simulation's base directory path (str)
    """
    CSV_PATH = os.path.join(base_dir, CSVFILE_STR)
    with open(CSV_PATH, "w", newline="") as csvfile:
       csvwriter = csv.writer(csvfile, delimiter = " ") 
       csvwriter.writerow(["PRETRAIN", PRETRAIN])
       csvwriter.writerow(["PRETRAIN_BATCH SIZE", str(PRETRAIN_BATCH_SIZE)])
       csvwriter.writerow(["PRETRAIN_EPOCHS", str(PRETRAIN_EPOCHS)])
       csvwriter.writerow(["PRETRAIN_LOSS", PRETRAIN_LOSS])
       csvwriter.writerow(["PRETRAIN_METRICS", *PRETRAIN_METRICS])
       csvwriter.writerow(["PRETRAIN_OPTIMIZER", PRETRAIN_OPTIMIZER])
       csvwriter.writerow(["TRAIN_GLOBAL", str(TRAIN_GLOBAL)])
       csvwriter.writerow(["TRAIN_LOCAL", str(TRAIN_LOCAL)])
       csvwriter.writerow(["TRAIN_SAMP_NUM", str(TRAIN_SAMP_NUM)])
       csvwriter.writerow(["TRAIN_BATCH_SIZE", str(TRAIN_BATCH_SIZE)])
       csvwriter.writerow(["TRAIN_EPOCHS_INIT", str(TRAIN_EPOCHS_INIT)])
       csvwriter.writerow(["TRAIN_LR_INIT", str(TRAIN_LR_INIT)])
       csvwriter.writerow(["TRAIN_LOSS", TRAIN_LOSS])
       csvwriter.writerow(["TRAIN_METRICS", *TRAIN_METRICS])
       csvwriter.writerow(["TRAIN_OPTIMIZER", TRAIN_OPTIMIZER])
       csvwriter.writerow(["TRAIN_SAMP_NUM", str(TRAIN_SAMP_NUM)])
       csvwriter.writerow(["FL_CLIENTS_NUM", str(FL_CLIENTS_NUM)])
       csvwriter.writerow(["FL_CLASSOVERLAP_NIID", FL_CLASSOVERLAP_NIID])
       csvwriter.writerow(["FL_DISTYPE", FL_DISTYPE])
       csvwriter.writerow(["FL_SNROVERLAP_NIID", FL_SNROVERLAP_NIID])
       csvwriter.writerow(["FL_FRACT_EVAL", str(FL_FRACT_EVAL)])
       csvwriter.writerow(["FL_FRACT_FIT", str(FL_FRACT_FIT)])
       csvwriter.writerow(["FL_ROUNDS_NUM", str(FL_ROUNDS_NUM)])
       csvwriter.writerow(["FL_SOCKET", FL_SOCKET])
       csvwriter.writerow(["FL_STRATEGY", FL_STRATEGY])
       csvwriter.writerow(["FL_EPOCHINCR", str(FL_EPOCHINCR)])
       csvwriter.writerow(["FL_EPOCHORD", str(FL_EPOCHORD)])
       csvwriter.writerow(["FL_EPOCHREP", str(FL_EPOCHREP)])
       csvwriter.writerow(["FL_LRDECR", str(FL_LRDECR)])
       csvwriter.writerow(["FL_LRORD", str(FL_LRORD)])
       csvwriter.writerow(["FL_LREP", str(FL_LREP)])
