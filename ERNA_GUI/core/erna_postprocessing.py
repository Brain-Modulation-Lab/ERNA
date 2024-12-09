import os
from ERNA_GUI.core.utilityw import fetch_runschema
from ERNA_GUI.io.general import (
    load_erna_annots, setup_path, load_config, load_erna_results
)
from ERNA_GUI.utils.handle_errors import (
    DataFileNotFoundError,AnnotationFileNotFoundError,PostProcessError, ImplementationError
)
from copy import deepcopy

from ERNA_GUI.utils.progress_bar import ProgressBar
from ERNA_GUI.utils.handle_files import load_file, save_object, delete_object
from ERNA_GUI.utils.handle_ftypes import check_task_validity, check_python_list
from ERNA_GUI.utils.preproc_utils import fetch_task_data

# handle cache [temporary here] ------------------------------

# cache temporary
from typing import Dict
import pickle

# Persistent cache file path


def load_persistent_cache(CACHE_FILE):
    """Load the data cache from a file, if it exists."""
    if os.path.exists(CACHE_FILE):
        data_cache = load_file(CACHE_FILE,'pkl',verbose = True)
    else:
        data_cache = None
    return data_cache

def save_persistent_cache(CACHE_FILE, data_cache):
    """Save the data cache to a file."""
    save_object(data_cache, CACHE_FILE, ask_before_overwrite= False, convert_numpy_to_python= True, verbose = True)
  
        
def clear_persistent_cache(CACHE_FILE):
    """Deletes the persistent cache file, if it exists."""
    if os.path.exists(CACHE_FILE):
        delete_object(CACHE_FILE, verbose = True)
    else:
        print("No cache file found to clear.")

def setup_cache_path(save_path,  CACHE_FILENAME = "erna_data_cache.pkl"):
    CACHE_FILEPATH = os.path.join(save_path, CACHE_FILENAME)
    return CACHE_FILEPATH

# ------------------------------------------------------------


# parent class ------------------

class PostProcess:
    def __init__(self, participant: dict, valid_tasks) -> None:
        required_keys = {'id', 'data_path', 'save_path', 'annot_path'}
        if not required_keys.issubset(participant.keys()):
            raise ValueError(f"Participant data must include {required_keys}")
        self._participant = participant
        self._participant_id: str = participant['id']
        self._participant_raw: str = participant['data_path']
        self._participant_save: str = participant['save_path']
        self._participant_annot: str = participant['annot_path']
        
        # Load annotations and data
        self._annotations: dict | None = self._load_annots()
        self._data: dict | None = self._load_data()
        self._params: dict | None = self._load_params()
        self._valid_tasks = valid_tasks
        


    def _load_annots(self) -> dict:
        if not os.path.exists(self._participant_annot):
            raise AnnotationFileNotFoundError(
                f"Annotation folder not found for participant {self._participant_id} at {self._participant_annot}"
            )
        
        
        annotations = load_erna_annots(self._participant_annot, self._participant_id)
        for key in annotations:
            print(f"Loaded succesfully {key}")
        return annotations

    def _load_data(self) -> dict:
        if not os.path.exists(self._participant_raw):
            raise DataFileNotFoundError(
                f"Data folder not found for participant {self._participant_id} at {self._participant_save}"
            )
        
        # handle cache   
        CACHE_FILEPATH  = setup_cache_path(self._participant_save) 
        _data_cache = load_persistent_cache(CACHE_FILEPATH)   
        if _data_cache is None:
            print("No cache available")
            data = load_erna_results(self.participant, 'erna')
            save_persistent_cache(CACHE_FILEPATH, data)
        else:
            # cache available
            print("Using cache information...")
            data = deepcopy(_data_cache)
        return data
    
    def _load_params(self) -> dict:
        if not os.path.exists(self._participant_raw):
            raise DataFileNotFoundError(
                f"Data folder not found for participant {self._participant_id} at {self._participant_save}"
            )
        return load_erna_results(self._participant, data_keys='params')



    
    def load_data_task(self, which_tasks = ["stimScan", "ampRamp"]):
        return fetch_task_data(self._data, self._annotations['run_annot'],  which_tasks, self._valid_tasks)
        

    # Properties ----------------------
    
    @property
    def participant(self) -> str:
        return self._participant
    
    @property
    def annotations(self) -> dict:
        return self._annotations
    
    @property
    def params(self) -> dict:
        return self._params
    
cfg = load_config()

def load_preprocessed_run(PARTICIPANT, which_tasks = ["stimScan", "ampRamp"]) -> PostProcess: 
    valid_tasks = cfg.device.TASKS
    results = PostProcess(PARTICIPANT, valid_tasks=valid_tasks) 
    # get task data
    task_data = results.load_data_task(which_tasks = which_tasks)
    return task_data


def load_preprocessed_runs(PARTICIPANTS_ID) -> PostProcess:
    which_tasks = ["stimScan", "ampRamp"]
    PARTICIPANTS_ID = check_python_list(PARTICIPANTS_ID)
    db = {}
    for PARTICIPANT_ID in PARTICIPANTS_ID:
        PARTICIPANT = setup_path(PARTICIPANT_ID, device= "argus")
        db[PARTICIPANT_ID] = load_preprocessed_run(PARTICIPANT, which_tasks)
    return db


if __name__ == "__main__":
    PARTICIPANTS_ID = ["DM2001","DM2002"] # manually set it
    db = load_preprocessed_runs(PARTICIPANTS_ID)
    print(db)