import numpy as np
import os
import json
import pandas as pd
import warnings
from typing import Dict, Any
import streamlit as st
import yaml
from hydra import initialize,compose
import hydra
import platform

from ERNA_GUI.utils.handle_errors import (
    DataFileNotFoundError,AnnotationFileNotFoundError,PostProcessError
)

from ERNA_GUI.utils.handle_files import (
    load_table, load_file
)

# setup paths
@st.cache_data
def setup_path(PARTICIPANT_ID: str, device: str) -> Dict[str,Any]:

    base_path = get_base_path()

    PARTICIPANT_DATA_PATH = os.path.join(base_path, "sourcedata", f"sub-{PARTICIPANT_ID}", "ses-intraop", device)
    PARTICIPANT_ANNOT_PATH = os.path.join(base_path, "derivatives", f"sub-{PARTICIPANT_ID}", "annot")
    PARTICIPANT_SAVE_PATH = os.path.join(base_path, "derivatives", f"sub-{PARTICIPANT_ID}", "erna")
    PARTICIPANT_FIGURE_PATH = os.path.join(PARTICIPANT_SAVE_PATH,"figures")
   
    


    PARTICIPANT = {
        'id': PARTICIPANT_ID,
        'data_path': PARTICIPANT_DATA_PATH,
        'annot_path': PARTICIPANT_ANNOT_PATH,
        'save_path': PARTICIPANT_SAVE_PATH,
        'figure_path': PARTICIPANT_FIGURE_PATH
    }    
    return PARTICIPANT


def get_base_path():
    """Determine the base path dynamically based on the operating system."""
    system = platform.system()
    
    if system == "Windows":
        # Windows path
        return r"Z:\ERNA"
    elif system == "Darwin":  # macOS
        # macOS path
        return "/Volumes/Nexus4/ERNA"
    else:
        raise ValueError(f"Unsupported operating system: {system}")



def load_config(config_path = os.path.join('../conf'), config_name="config", overrides = None):
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear() 

    # Initialize Hydra and set the configuration path
    initialize(config_path=config_path, version_base=None)
    
    if overrides is not None:
        # Compose the configuration object by loading `config.yaml`
        cfg = compose(config_name=config_name)
    else:
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg




@st.cache_data
def load_erna_annots(annot_path, participant_id, annot_keys="raw", sep='\t') -> Dict[str, pd.DataFrame]:
    """
    Loads the specified annotations for a participant.
    
    :param PARTICIPANT: The participant dictionary containing the participant's information.
    :param annot_keys: List of annotation types to load. If None, all annotations are loaded.
    :param sep: The separator used for reading the annotation files. Default is tab.
    :return: A dictionary with annotation names as keys and the corresponding DataFrames as values.
    """
    annot_paths = setup_annot_paths(annot_path,participant_id)
    erna_annots = {}

    # Default annotation keys, if none are specified
    if annot_keys is None:
        annot_keys = ['run_annot', 'stimScan_annot', 'ampRamp_annot', 'channels_annot', 'coords_annot', 'IPI_annot','IBI_annot']
    elif annot_keys == "raw":
        annot_keys = ['run_annot', 'stimScan_annot', 'ampRamp_annot', 'channels_annot', 'coords_annot']
    elif annot_keys == "preproc":
        annot_keys = ['IPI_annot','IBI_annot']
        
    if sep == '\t':
        ftype = 'tsv'
    else:
        ftype = 'csv'

    for key in annot_keys:
        if key in annot_paths:
            erna_annots[key] = load_table(annot_paths[key],ftype = ftype ,verbose=False)
        else:
            raise(AnnotationFileNotFoundError(f"Warning: Annotation {key} not found in the provided participant's data."))
    
    return erna_annots


def setup_annot_paths(annot_folder,participant_id) -> Dict[str, str]:
    """
    Sets up the paths to the annotation files based on the participant information.
    
    :param PARTICIPANT: The participant dictionary containing the participant's information.
    :return: A dictionary mapping annotation names to their file paths.
    """
    
    annot_paths = {
        'run_annot': os.path.join(annot_folder, f"subj-{participant_id}_ses-intraop_runs.tsv"),
        'stimScan_annot': os.path.join(annot_folder, f"subj-{participant_id}_ses-intraop_task-stimScan_annot.tsv"),
        'ampRamp_annot': os.path.join(annot_folder, f"subj-{participant_id}_ses-intraop_task-ampRamp_annot.tsv"),
        'channels_annot': os.path.join(annot_folder, f"subj-{participant_id}_ses-intraop_channels.tsv"),
        'coords_annot': os.path.join(annot_folder, f"subj-{participant_id}_ses-intraop_electrodes.tsv"),
        'IPI_annot': os.path.join(annot_folder, f"subj-{participant_id}_sess-intraop_IPI.tsv"),
        'IBI_annot': os.path.join(annot_folder, f"subj-{participant_id}_sess-intraop_IBI.tsv"),
    }
    return annot_paths


# loading functions for erna results 
# ToDo: need to fix this function
def load_erna_results(PARTICIPANT: dict, data_keys: str = 'erna') -> Dict[str, Any]:
    
    
    """
    Load ERNA data from a JSON file for a given participant.

    Parameters:
    - PARTICIPANT_ID: str, ID of the participant.

    Returns:
    - A dictionary containing the data from the JSON file, or None if an error occurs.
    """
    data_paths = setup_data_paths(PARTICIPANT['save_path'],PARTICIPANT['id'])
    filename = data_paths[data_keys]
    
    data = load_file(fpath = filename, ftype= 'json', verbose = True)
    return data



'''
def load_ERNAfiles_results2annot(PARTICIPANT: dict, which_annot: str = 'both') -> Any:
    """
    Load ERNA files and return annotation data.

    Parameters:
    - PARTICIPANT_ID: str, ID of the participant.
    - which_annot: str, type of annotation to load ('ipi', 'ibi', or 'both').

    Returns:
    - A tuple of DataFrames (IPI_annot, IBI_annot) if which_annot is 'both'.
    - A single DataFrame (IPI_annot) if which_annot is 'ipi'.
    - A single DataFrame (IBI_annot) if which_annot is 'ibi'.
    - None if which_annot is invalid.
    """
    annots = setup_annot_paths(PARTICIPANT)
    IPI_filename = annots['IPI_annot']
    IBI_filename = annots["IBI_annot"]

    
    if which_annot == 'both':
        try:
            IPI_annot = pd.read_csv(IPI_filename, sep='\t')
            IBI_annot = pd.read_csv(IBI_filename, sep='\t')
            return IPI_annot, IBI_annot
        except FileNotFoundError as e:
            raise(DataFileNotFoundError(f"An unexpected error occurred: {e}"))
    elif which_annot == 'ipi':
        try:
            IPI_annot = pd.read_csv(IPI_filename, sep='\t')
            return IPI_annot
        except FileNotFoundError as e:
            raise(DataFileNotFoundError(f"An unexpected error occurred: {e}"))
    elif which_annot == 'ibi':
        try:
            IBI_annot = pd.read_csv(IBI_filename, sep='\t')
            return IBI_annot
        except FileNotFoundError as e:
            raise(DataFileNotFoundError(f"An unexpected error occurred: {e}"))
    else:
        raise(DataFileNotFoundError('Wrong call of load_ERNAfiles_results2annot: which_annot can be only "ipi", "ibi" or "both".'))
'''


# saving functions for erna results
def convert_to_serializable(data):
    """Recursively convert numpy arrays to lists."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_serializable(item) for item in data)
    elif isinstance(data, set):
        return list(convert_to_serializable(item) for item in data)
    return data




    

'''
def store_ERNAfiles_results2params(ERNA):
    params_filename = )
    with open(params_filename, 'w') as json_file:
        json.dump(convert_to_serializable(ERNA.params), json_file, indent=4)   
    print(f"All params saved to {params_filename}") 
'''

def setup_data_paths(save_path, participant_id) -> dict:
    data_paths = {
        'raw': os.path.join(save_path, f"subj-{participant_id}_sess-intraop_data-raw.json"),
        'erna': os.path.join(save_path, f"subj-{participant_id}_sess-intraop_data-erna.json"),
        'params':os.path.join(save_path, f"subj-{participant_id}_sess-intraop_params.json")
    }
    return data_paths

    
    

'''
  def store_results(self, ERNA, data_keys = "erna"):
        # Convert the store dictionary to a JSON-compatible format
        data_paths = setup_data_paths(ERNA.participant_save,ERNA.participant_id)
        
        if not isinstance(data_keys, list): data_keys = [data_keys] 

        for _,data_key in enumerate(data_keys):
            filename = data_paths[data_key]    
            store_for_json = {}
            if data_key == 'params':
                store_for_json = convert_to_serializable(ERNA.params)
            else:
                for key, value in ERNA.store.items():
                        if data_key == "raw":
                            data_to_save = {k: v for k, v in value['data'].items() if k != 'data_cont'}
                        elif data_key == "erna":
                            data_to_save = {k: v for k, v in value['data'].items() if k != 'data_erna'}
                    
                        store_for_json[key] = {
                            'file': value['file'],
                            'run_details': value['run_details'].to_dict(orient='records'),  # Convert DataFrame to dict
                            'stimChannel': convert_to_serializable(value['stimChannel']),
                            'data': convert_to_serializable(data_to_save)  # Convert to JSON-compatible format
                        }
            # Save the entire store to a JSON file
            with open(filename, 'w') as json_file:
                json.dump(store_for_json, json_file, indent=4)
            print(f"All ERNA {data_key} saved to {filename}")


def store_ERNAfiles_results2annot(ERNA):
    annot_paths = setup_annot_paths(ERNA.participant_annot, ERNA.participant_id)
    IPI_filename = annot_paths['IPI_annot']
    ERNA.IPI_annot.to_csv(IPI_filename, sep='\t', index=False)
    IBI_filename = annot_paths['IBI_annot']
    ERNA.IBI_annot.to_csv(IBI_filename, sep='\t', index=False)    
    print(f"All features saved to {IPI_filename} and {IBI_filename}")            
'''