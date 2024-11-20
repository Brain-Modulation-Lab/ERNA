import numpy as np
import os
import json
import pandas as pd
import warnings
from typing import Dict, Any
import streamlit as st


# setup paths
@st.cache_data
def setup_path(PARTICIPANT_ID: str, device: str) -> Dict[str,Any]:
    PARTICIPANT_DATA_PATH = f"/Volumes/Nexus4/ERNA/sourcedata/sub-{PARTICIPANT_ID}/ses-intraop/{device}"
    PARTICIPANT_ANNOT_PATH = f"/Volumes/Nexus4/ERNA/derivatives/sub-{PARTICIPANT_ID}/annot"
    PARTICIPANT_SAVE_PATH = f"/Volumes/Nexus4/ERNA/derivatives/sub-{PARTICIPANT_ID}/erna"
    PARTICIPANT_FIGURE_PATH = PARTICIPANT_SAVE_PATH + "/figures"
   

    PARTICIPANT = {
        'id': PARTICIPANT_ID,
        'data_path': PARTICIPANT_DATA_PATH,
        'annot_path': PARTICIPANT_ANNOT_PATH,
        'save_path': PARTICIPANT_SAVE_PATH,
        'figure_path': PARTICIPANT_FIGURE_PATH
    }    
    return PARTICIPANT

# loading functions for erna results 
# ToDo: need to fix this function
def load_ERNAfiles_results2data(PARTICIPANT_ID: str) -> Dict[str, Any]:
    """
    Load ERNA data from a JSON file for a given participant.

    Parameters:
    - PARTICIPANT_ID: str, ID of the participant.

    Returns:
    - A dictionary containing the data from the JSON file, or None if an error occurs.
    """
    PARTICIPANT = setup_path(PARTICIPANT_ID)
    filename = os.path.join(PARTICIPANT['save_path'], f"subj-{PARTICIPANT['id']}_sess-intraop_data-erna.json")

    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        warnings.warn(f"File not found: {filename}", UserWarning)
        return None
    except json.JSONDecodeError:
        warnings.warn(f"Error decoding JSON in file: {filename}", UserWarning)
        return None
    except Exception as e:
        warnings.warn(f"An unexpected error occurred: {e}", UserWarning)
        return None



def load_ERNAfiles_results2annot(PARTICIPANT_ID: str, which_annot: str = 'both') -> Any:
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
    PARTICIPANT = setup_path(PARTICIPANT_ID)
    IPI_filename = os.path.join(PARTICIPANT['annot_path'], f"subj-{PARTICIPANT['id']}_sess-intraop_IPI.tsv")
    IBI_filename = os.path.join(PARTICIPANT['annot_path'], f"subj-{PARTICIPANT['id']}_sess-intraop_IBI.tsv")

    
    if which_annot == 'both':
        try:
            IPI_annot = pd.read_csv(IPI_filename, sep='\t')
            IBI_annot = pd.read_csv(IBI_filename, sep='\t')
            return IPI_annot, IBI_annot
        except FileNotFoundError as e:
            warnings.warn(f"File not found: {e}", UserWarning)
            return None
    elif which_annot == 'ipi':
        try:
            IPI_annot = pd.read_csv(IPI_filename, sep='\t')
            return IPI_annot
        except FileNotFoundError as e:
            warnings.warn(f"File not found: {e}", UserWarning)
            return None
    elif which_annot == 'ibi':
        try:
            IBI_annot = pd.read_csv(IBI_filename, sep='\t')
            return IBI_annot
        except FileNotFoundError as e:
            warnings.warn(f"File not found: {e}", UserWarning)
            return None
    else:
        warnings.warn('Wrong call of load_ERNAfiles_results2annot: which_annot can be only "ipi", "ibi" or "both".', UserWarning)
        return None



# saving functions for erna results
def convert_to_serializable(data):
    """Recursively convert numpy arrays to lists."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    return data

def store_ERNAfiles_results2data(ERNA, which_data):
    # Convert the store dictionary to a JSON-compatible format

    if which_data == "raw":
        store_filename = os.path.join(ERNA.participant_save, f"subj-{ERNA.participant_id}_sess-intraop_data-raw.json")
    elif which_data == "erna":
        store_filename = os.path.join(ERNA.participant_save, f"subj-{ERNA.participant_id}_sess-intraop_data-erna.json")

    
    store_for_json = {}
    for key, value in ERNA.store.items():
        if which_data == "raw":
            data_to_save = {k: v for k, v in value['data'].items() if k != 'data_cont'}
        elif which_data == "erna":
            data_to_save = {k: v for k, v in value['data'].items() if k != 'data_erna'}
    
        store_for_json[key] = {
            'file': value['file'],
            'run_details': value['run_details'].to_dict(orient='records'),  # Convert DataFrame to dict
            'stimChannel': convert_to_serializable(value['stimChannel']),
            'data': convert_to_serializable(data_to_save)  # Convert to JSON-compatible format
        }

    # Save the entire store to a JSON file
    with open(store_filename, 'w') as json_file:
        json.dump(store_for_json, json_file, indent=4)
    print(f"All ERNA data saved to {store_filename}")
    
def store_ERNAfiles_results2annot(ERNA):
    IPI_filename = os.path.join(ERNA.participant_annot, f"subj-{ERNA.participant_id}_sess-intraop_IPI.tsv")
    ERNA.IPI_annot.to_csv(IPI_filename, sep='\t', index=False)
    IBI_filename = os.path.join(ERNA.participant_annot, f"subj-{ERNA.participant_id}_sess-intraop_IBI.tsv")
    ERNA.IBI_annot.to_csv(IBI_filename, sep='\t', index=False)    
    print(f"All features saved to {IPI_filename} and {IBI_filename}")