from constants import *

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os

# Program parameters
RESULTS_PATH = "trained_models/1/" #folder that contains `clients` and `server` directories
PLOT_TYPE    = "accuracy_fit" #"loss_fit", "accuracy_fit", "loss_test", "accuracy_test"

# Plots parameters
mpl.rcParams['axes.labelsize']    = 14
mpl.rcParams['axes.titlesize']    = 16
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.titlesize']  = 16
mpl.rcParams['legend.fontsize']   = 14
mpl.rcParams['lines.linestyle']   = '-'
mpl.rcParams['lines.linewidth']   = 2
mpl.rcParams['lines.marker']      = '.'
mpl.rcParams['lines.markersize']  = 8
mpl.rcParams['xtick.labelsize']   = 12
mpl.rcParams['ytick.labelsize']   = 12

LINES       = ["solid", "dotted", "dashed", "dashdot"]
MARKERS     = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "X", "D"]
LINES_NUM   = len(LINES)
MARKERS_NUM = len(MARKERS)

def select_random_line_marker(current_lines_markers : list):
    """
    Line and marker selection

    Selects a random line-marker combination

    Args:

        current_lines_markers: already selected line-marker combinations (list)
    """
    line_choice   = np.random.choice(LINES)
    marker_choice = np.random.choice(MARKERS)
    while (line_choice, marker_choice) in current_lines_markers:
        line_choice   = np.random.choice(LINES)
        marker_choice = np.random.choice(MARKERS)
    new_line_marker = (line_choice, marker_choice)
    current_lines_markers.append(new_line_marker)
    return new_line_marker

# Get history data
dirs_list    = next(os.walk(RESULTS_PATH))[1] #list directories in RESULTS_PATH
clients_path = os.path.join(RESULTS_PATH, "clients")
server_path  = os.path.join(RESULTS_PATH, "server")
global_path  = os.path.join(RESULTS_PATH, "global")
local_path   = os.path.join(RESULTS_PATH, "local")
if "clients" in dirs_list:
    CLIENTS_NUM = int(len(next(os.walk(clients_path))[1]))
CLIENTS           = np.arange(CLIENTS_NUM)
clients_histories = list()
for client in CLIENTS:
    client_path = os.path.join(clients_path, str(client))
    clients_histories.append(np.load(os.path.join(client_path, FL_CLIENTHIST_STR), allow_pickle=True)) #load client history
server_history = np.load(os.path.join(server_path, FL_SERVERHIST_STR), allow_pickle=True) #load server history
global_history = np.load(os.path.join(global_path, GLOBALHIST_STR), allow_pickle=True) #load global model history
local_history  = np.load(os.path.join(local_path, LOCALHIST_STR), allow_pickle=True) #load local model history

FL_ROUNDS_NUM = len(server_history) - 1
FL_ROUNDS     = np.arange(1, FL_ROUNDS_NUM + 1)
if PLOT_TYPE in ["loss_fit", "accuracy_fit"]:
    EPOCHS_FL_ROUND = np.array([len(clients_histories[0][fl_round - 1][PLOT_TYPE]) for fl_round in FL_ROUNDS]) #number of epochs per FL round
elif PLOT_TYPE in ["loss_test", "accuracy_test"]:
    EPOCHS_FL_ROUND = np.array([1 for fl_round in FL_ROUNDS]) #number of epochs per FL round
EPOCHS_NUM      = int(np.sum(EPOCHS_FL_ROUND)) #number of total epochs (not including initial server-side round)
EPOCHS          = np.arange(1, EPOCHS_NUM + 1)
line_marker     = list()
legend_list     = list()
if PLOT_TYPE in ["loss_fit", "accuracy_fit"]:
    # Clients in legend
    if CLIENTS_NUM <= 8:
        clients_legend = CLIENTS
    else:
        clients_legend = list(map(int, np.floor(np.linspace(0, CLIENTS_NUM, 10))))
    # Extract history data
    clients_data = np.frompyfunc(list, 0, 1)(np.empty(CLIENTS_NUM, dtype=object))
    global_data = np.array([])
    local_data  = np.array([])
    for client in CLIENTS:
        for fl_round in FL_ROUNDS:
            clients_data[client] = np.append(clients_data[client], clients_histories[client][fl_round - 1][PLOT_TYPE]) #client data
    for fl_round in FL_ROUNDS:
        global_data = np.append(global_data, global_history[fl_round - 1][PLOT_TYPE]) #global model data
        local_data  = np.append(local_data, local_history[fl_round - 1][PLOT_TYPE]) #local model data
    # Format clients data as matrix
    clients_data_mat = np.NaN * np.ones((CLIENTS_NUM, FL_ROUNDS_NUM))
    for client in CLIENTS:
        for fl_round in FL_ROUNDS:
            clients_data_mat[client, fl_round - 1] = clients_data[client][fl_round - 1]
    # Plot
    fig = plt.figure()
    last_line, last_marker = select_random_line_marker(line_marker)
    clients_min = np.min(clients_data_mat, axis = 0)
    clients_max = np.max(clients_data_mat, axis = 0)
    plt.plot(EPOCHS, clients_min, linestyle=last_line, marker=last_marker) #client min plot
    legend_list.append("Clients min")
    last_line, last_marker = select_random_line_marker(line_marker)
    plt.plot(EPOCHS, clients_max, linestyle=last_line, marker=last_marker) #client max plot
    legend_list.append("Clients max")
    last_line, last_marker = select_random_line_marker(line_marker)
    plt.plot(EPOCHS, global_data, linestyle=last_line, marker=last_marker) #global model plot
    legend_list.append("Global")
    last_line, last_marker = select_random_line_marker(line_marker)
    plt.plot(EPOCHS, local_data, linestyle=last_line, marker=last_marker) #local model plot
    legend_list.append("Local")
    plt.xlabel("Epoch [-]")
    plt.ylabel(PLOT_TYPE + " [-]")
    plt.legend(legend_list, loc="best")
    plt.grid()
    plt.autoscale(enable=True, tight=True)
    plt.show()
elif PLOT_TYPE in ["loss_test", "accuracy_test"]:
    # Extract history data
    server_data      = np.array([])
    global_data      = np.array([])
    local_data       = np.array([])
    SERVER_FL_ROUNDS = np.arange(FL_ROUNDS_NUM + 1)
    for fl_round in SERVER_FL_ROUNDS:
        server_data = np.append(server_data, server_history[fl_round][PLOT_TYPE]) #server data by FL round
    for fl_round in FL_ROUNDS:
        global_data = np.append(global_data, global_history[fl_round - 1][PLOT_TYPE]) #global model data
        local_data  = np.append(local_data, local_history[fl_round - 1][PLOT_TYPE]) #local model data
    # Plot
    fig = plt.figure()
    last_line, last_marker = select_random_line_marker(line_marker)
    plt.plot(server_data, linestyle=last_line, marker=last_marker) #server plot
    legend_list.append("Server")
    last_line, last_marker = select_random_line_marker(line_marker)
    plt.plot(FL_ROUNDS, global_data, linestyle=last_line, marker=last_marker) #global model plot
    legend_list.append("Global")
    last_line, last_marker = select_random_line_marker(line_marker)
    plt.plot(FL_ROUNDS, local_data, linestyle=last_line, marker=last_marker) #local model plot
    legend_list.append("Local")
    plt.xlabel("Round [-]")
    plt.ylabel(PLOT_TYPE + " [-]")
    plt.legend(legend_list, loc="best")
    plt.grid()
    plt.autoscale(enable=True, tight=True)
    plt.show()
