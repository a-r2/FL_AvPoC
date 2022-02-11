import subprocess
import os

if __name__ == "__main__":
    # Parameters
    SETTINGS_PATH = "settings_loop" #directory storing settings files
    SETTINGS_TO   = "~/'Federated Learning'/Code/settings.py" #settings file holding the configuration of the simulation
    CP_PATH       = "cp" #copy | Linux: "cp" / Windows: "copy"
    PYTHON_PATH   = "~/'Federated Learning'/venv/bin/python" #python | Linux: "~/'Federated Learning'/venv/bin/python" / Windows: "python"
    SIM_PATH      = "run.py" #simulation
    # Check existence of settings directory
    if not os.path.exists(SETTINGS_PATH):
        error_msg = "\nNo `" + SETTINGS_PATH + "` directory was found\n"
        print(error_msg)
        quit()
    # Find settings files
    files_list = os.listdir(SETTINGS_PATH)
    if not files_list:
        error_msg = "\nNo settings files were found in settings directory\n"
        print(error_msg)
        quit()
    # Loop
    for i in range(len(files_list)):
        settings_file = files_list[i]
        settings_from = os.path.join(SETTINGS_PATH, settings_file)
        cp_cmd     = " ".join((CP_PATH, settings_from, SETTINGS_TO))
        python_cmd = " ".join((PYTHON_PATH, SIM_PATH))
        subprocess.run(cp_cmd, shell = True)
        subprocess.run(python_cmd, shell = True)
