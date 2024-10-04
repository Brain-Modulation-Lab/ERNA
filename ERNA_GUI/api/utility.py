import numpy as np
import os
from pathlib import Path
import hydra
from hydra import initialize,compose

current_script = Path(__file__)
BASE_DIR = current_script.parent.parent.parent
# get run_details

def load_config(config_path = os.path.join('../conf/device'), config_name="argus"):
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear() 

    # Initialize Hydra and set the configuration path
    initialize(config_path=config_path, version_base=None)
    
    # Compose the configuration object by loading `config.yaml`
    cfg = compose(config_name=config_name)
    return cfg



# general function to get attribute instance of class
def fetch_attr(ERNA, attr):
    return getattr(ERNA,attr, None)
# --------------------------------------------
# get annotation table about metadata

def fetch_annot(ERNA, which = None, id=None, task=None):
    if which is not None:
        annot = fetch_attr(ERNA, which)  # Fetch the IBI annotations
        # Filter based on `ids`
        if id is not None:
            if isinstance(id, int):
                id = [id]  # Convert single integer to list
            annot = annot[annot['Run'].isin(id)]  # Filter by ids
        
        # Filter based on `task`
        if task is not None:
            annot = annot[annot['Task'] == task]  # Filter by task
        
        return annot
    else: 
        Warning("Type a well-defined annot table")
    


def fetch_runschema(ERNA):
    run_annot = fetch_annot(ERNA,"run_annot")
    runschema = {'stimScan': [], 'ampRamp': []}

    # Group rows by 'Task' and extract runs for 'stimScan' and 'ampRamp'
    grouped = run_annot.groupby('Task')['Run'].apply(list).to_dict()

    # Fill the runschema with available groups
    for task in ['stimScan', 'ampRamp']:
        if task in grouped:
            runschema[task] = grouped[task]

    return runschema
        






# --------------------------------------------
# get store data
def fetch_store(ERNA):
    return fetch_attr(ERNA, 'store')
# --------------------------------------------
# fetch ibi data in a matrix (# bursts x # time points)
def fetch_IBImatrix(data_run):
    erna = data_run['data']['data_erna']

    # Initialize lists to collect data
    stim_vec = []
    bursts = []
    ibi_matrix_list = []
    time_list = []

    for i, (burst, burst_data) in enumerate(erna.items()):
        ibi = burst_data['ibi']['erna']
        time = (burst_data['ibi']['time'] - burst) * 1000

        # Append IBI data to the list
        ibi_matrix_list.append(ibi)
        stim_vec.append(burst_data['ibi']['stim_amp'])  # Assuming stim_amp is scalar
        bursts.append(i + 1)
        time_list.append(time)  # Accumulate time data

    # Convert the list of rows into a NumPy array
    ibi_matrix = np.array(ibi_matrix_list).T
    time_values = np.mean(np.array(time_list), axis=0)  # Average time if necessary
    return ibi_matrix, bursts, time_values, stim_vec
# --------------------------------------------    
# fetch ipi data in a matrix (# burst x # pulses x # time points)
def fetch_IPImatrix(data_run):
    erna = data_run['data']['data_erna']

    # Initialize lists to collect data
    stim_vec = []
    bursts = []
    ipi_matrix_list = []
    time_matrix_list = []
    npulses_inburst = []

    # Iterate through bursts and filter only those with 10 pulses
    for i, (burst_id, burst_data) in enumerate(erna.items()):
        burst_ipis = []
        ipi_times = []
        
        # Collect IPIs and times for this burst
        for pulse_id, ipi_data in burst_data['ipi'].items():
            ipi = ipi_data['erna']  # Shape (116,)
            time = (ipi_data['time'] - pulse_id)* 1000  # Time in milliseconds relative to first pulse
            # Collect the IPI and time for the current pulse
            burst_ipis.append(ipi)  # Append the IPI array for this pulse
            ipi_times.append(time)  # Append the corresponding time data
        
        # Check if the burst has exactly 10 pulses
        if len(burst_ipis) == 10:
            # Append the burst data to the lists
            bursts.append(i + 1)  # Track burst index (1-based)
            ipi_matrix_list.append(np.array(burst_ipis))  # Convert to array for consistent shape
            time_matrix_list.append(np.array(ipi_times))
            stim_vec.append(burst_data['ibi']['stim_amp'])  # Assuming stim_amp is scalar
            npulses_inburst.append(len(burst_ipis))  # Should be 10 for every entry here

    # Convert lists to NumPy arrays (or leave as lists if they are variable in size)
    ipi_matrix = ipi_matrix_list  # List of arrays
    time_matrix = time_matrix_list  # List of arrays


    return ipi_matrix, time_matrix, bursts, npulses_inburst, stim_vec

# fetch  run data
def fetch_run(ERNA, id=None, task=None):
    run_annot = fetch_annot(ERNA,"run_annot")
    store = fetch_store(ERNA)

    # Fetch data based on IDs
    if id is not None:
        if isinstance(id, int):
            id = [id]
        data_run = {}
        for single_id in id:
            if single_id in run_annot['Run'].values:
                # Filter the store data based on the single_id
                data_run[single_id] = store.get(single_id, 'Data not available for this ID')
            else:
                data_run[single_id] = 'ID not found in run annotations.'
        return data_run

    # Fetch data based on task
    elif task is not None:
        data_run = {}
        for idx, row in run_annot.iterrows():
            if row['Task'] == task:
                single_id = row['Run']
                if single_id in store:
                    data_run[single_id] = store[single_id]
                else:
                    data_run[single_id] = 'Data not available for this ID.'
        return data_run

    # If neither id nor task is provided
    else:
        return 'No id or task provided.'
# --------------------------------------------        








