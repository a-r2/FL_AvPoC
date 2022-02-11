from settings import *
from csv_logging import *

from pretrain import *
from client import *
from server import *
from global_local import *

import logging
import multiprocessing as mp
import sys

if __name__ == "__main__":
    # Print and assert FL settings
    print_assert_fl_settings()
    # Create directories
    base_dir, server_dir, clients_dir, global_dir, local_dir = create_dirs()
    # Save simulation settings
    save_settings(base_dir)
    # Load partial train dataset
    dataset       = load_partial_dataset()
    train_dataset = dataset[0]
    test_dataset  = dataset[1]
    # Pretrain, if applicable
    if PRETRAIN == "NONE":
        model_weights = None
    elif PRETRAIN in ["LOW", "MID", "HIGH"]:
        model_weights = run_pretrain(dataset) #create and pretrain model
    # Generate non-iid distribution, if applicable
    if FL_DISTYPE == "non-iid":
        niid_dist = get_niid_distribution(clients_dir)
    # Create server process
    server_proc   = mp.Process(target=run_server, name="SERVER", args=(server_dir, test_dataset, model_weights)) #spawn server process
    # Create clients processes
    clients_procs = []
    for client in FL_CLIENTS:
        # Load datasets
        if FL_DISTYPE == "iid":
            client_train_dataset = load_train_iid_partition(client, train_dataset) #fit + val
        elif FL_DISTYPE == "non-iid":
            client_train_dataset = load_train_niid_partition(client, train_dataset, niid_dist) #fit + val
        # Construct client dataset
        client_dataset = [client_train_dataset, test_dataset]
        # Spawn client subprocess
        client_proc = mp.Process(target=run_client, name="CLIENT " + str(client), args=(client, clients_dir, client_dataset)) #spawn client process
        # Append client process to clients processes lists
        clients_procs.append(client_proc)
        if TRAIN_LOCAL:
            if client == 0:
                # Create local model process for client 0
                local_proc = mp.Process(target=run_global_local, name="LOCAL", args=("local", local_dir, client_dataset)) #spawn local model train process
    # Create conventional model process
    if TRAIN_GLOBAL:
        global_proc = mp.Process(target=run_global_local, name="GLOBAL", args=("global", global_dir, dataset)) #spawn conventional model train process
    # Aggregate all processes
    procs = [server_proc, *clients_procs]
    if TRAIN_GLOBAL:
        procs.append(global_proc)
    if TRAIN_LOCAL:
        procs.append(local_proc)
    # Start processes
    for proc in procs:
        print("\n" + proc.name + " PROCESS STARTED\n")
        proc.start()
    # Main loop
    while True:
        try:
            for proc in procs:
                if not proc.is_alive(): #check if process terminated
                    print("\n" + proc.name + " PROCESS TERMINATED\n")
                    procs.remove(proc)
            if not procs:
                print("\nEXITING\n")
                sys.exit(0)
        except KeyboardInterrupt:
            for proc in procs:
                print("\n" + proc.name + " PROCESS KILLED\n")
                proc.kill()
            print("\nEXITING\n")
            sys.exit(0)
