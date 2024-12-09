import os
import numpy as np
import pandas as pd
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import logging



from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import streamlit as st
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go


from ERNA_GUI.utils.preproc_utils import (
    remove_zerostim, preprocess_erna, extract_pulses, extract_erna_slice,
    flatten_ipi_features, compute_IBI_features, compute_IPI_features, 
    fetch_run_task, fetch_run_details, fetch_stimCh_coordinates, fetch_IBImatrix
)

from ERNA_GUI.utils.progress_bar import ProgressBar

from ERNA_GUI.io.argus import (
    load_erna_file, load_selected_file,
)

from ERNA_GUI.io.general import (
    load_erna_annots, setup_path, setup_annot_paths, setup_data_paths
)

from ERNA_GUI.utils.handle_files  import (
    save_dict
)

from ERNA_GUI.utils.handle_ftypes import (
    check_python_list
)

from abc import ABC, abstractmethod


class PreProcess(ABC):
        
    def __init__(self, participant_id, participant_raw, 
                 participant_save,participant_annot, 
                 file_obj, params):
        self.participant_id = participant_id      
        self.participant_raw = participant_raw
        self.participant_save = participant_save
        self.participant_annot = participant_annot
       
        self.file_obj = file_obj
        self.params = params
        #self.run_annot = run_annot
        #self.stimScan_annot = stimScan_annot
        #self.ampRamp_annot = ampRamp_annot

        
        
    def get_task(self, run_annot):
        task = fetch_run_task(run_annot, self.run)
        return task

    def get_run_details(self,stimScan_annot,ampRamp_annot, run_annot):
        return fetch_run_details({'ampRamp': ampRamp_annot, 'stimScan' : stimScan_annot}, run_annot, self.run)
        
    def get_stimCh(self, run_details, channels_annot, coords_annot):
        return fetch_stimCh_coordinates( run_details, channels_annot, coords_annot)
    
    @abstractmethod
    def extract_erna_data():
       pass
   
    @abstractmethod
    def preprocess_erna():
       pass

    @abstractmethod
    def get_pulse_locs():
       pass

    @abstractmethod
    def get_erna_slice():
       pass
   
    @abstractmethod
    def get_erna_features():
       pass   
   
    @abstractmethod
    def set_windows():
       pass
      
class ERNAFileHandler(PreProcess):
    def __init__(self, participant_id, participant_raw, 
                 participant_save,participant_annot, 
                 file_obj, params) -> None: 
        super().__init__(
            participant_id=participant_id,
            participant_raw=participant_raw,
            participant_save=participant_save,
            participant_annot=participant_annot,
            file_obj=file_obj,
            params=params,
        )
  
        #self.run_annot = run_annot
        #self.stimScan_annot = stimScan_annot
        #self.ampRamp_annot = ampRamp_annot

        self.data = {}
        self.is_preprocessed = False
        
    
    # inherited methods ------------    
    def get_task(self, run_annot):
        return super().get_task(run_annot)
        
    def get_run_details(self,stimScan_annot,ampRamp_annot, run_annot):
        return super().get_run_details(stimScan_annot,ampRamp_annot, run_annot)
        
    def get_stimCh(self, run_details, channels_annot, coords_annot):
        return super().get_stimCh(run_details, channels_annot, coords_annot)      
    # ----------------------------        
    
                
    def extract_erna_data(self):
        try:
            mat_contents = load_selected_file(os.path.join(self.participant_raw, self.file_obj))
            self.run = mat_contents["run"]
            self.data = {
                'time': mat_contents["time"],
                'erna_cont': mat_contents["data_cont"],
                'timebursts': mat_contents["timebursts"],
                'stim_amp': mat_contents["stim_amp"]
            }
            self.data['timebursts'], self.data['stim_amp'] = remove_zerostim(self.data['timebursts'], self.data['stim_amp'])
            print(f"Extracted ERNA data for {self.file_obj}")
        except Exception as e:
            raise ValueError(f"Error extracting ERNA data from file {self.file_obj}: {e}")

    def preprocess_erna(self):
        if not self.is_preprocessed:
            try:
                sig = preprocess_erna(
                    self.data['erna_cont'],
                    self.params['flow_cut'],
                    self.params['fhigh_cut'],
                    self.params['order_low'],
                    self.params['order_high'],
                    self.params['srate']
                )
                if self.params['flip_polarity']:
                    sig = -sig
                self.is_preprocessed = True
                self.data['erna_cont'] = sig
                print(f"Preprocessed ERNA data for {self.file_obj}")
            except Exception as e:
                raise ValueError(f"Error preprocessing ERNA data: {e}")

    def get_pulse_locs(self):
        self.preprocess_erna()
        try:
            self.data['pulse_locs'], _, _ = extract_pulses(
                self.data['erna_cont'], self.data['timebursts'], self.data['time'],
                self.params['win_left'], self.params['win_right'], self.params['thr_pulses'],
                self.params['stim_freq'], self.params['srate']
            )
            print(f"Extracted pulse locations for {self.file_obj}")
        except Exception as e:
            raise ValueError(f"Error extracting pulse locations: {e}")

    def get_erna_slice(self):
        try:
            self.data['data_erna'] = extract_erna_slice(
                self.data['erna_cont'], self.data['pulse_locs'], self.data['time'], self.data['stim_amp'],
                self.params['ipi_left'], self.params['ipi_right'], self.params['ibi_left'], self.params['ibi_right'],
                self.params['which_detrend']
            )
            
            # get recommended polarity
            test_slice = {'data': self.data}
            ibi_matrix, _, time_values, _ = fetch_IBImatrix(test_slice)
 
            average_ibi = np.mean(ibi_matrix, axis=1)[np.array(time_values) >= 3.1]

            polarity = self.params["flip_polarity"]
            if abs(np.min(average_ibi)) >= abs(max(average_ibi)) and np.argmin(average_ibi) <= np.argmax(average_ibi):
                print(f"Abs min [{abs(min(average_ibi))} [index:{np.argmax(min(average_ibi))}] >= {abs(max(average_ibi))} [index:{np.argmax(max(average_ibi))}]]: adjust polarity {polarity} -> {not polarity}")
                self.params["flip_polarity"] = not self.params["flip_polarity"]
                self.data['erna_cont'] = -self.data['erna_cont']
                # delete and redo
                del self.data['data_erna']
                self.data['data_erna'] = extract_erna_slice(
                    self.data['erna_cont'], self.data['pulse_locs'], self.data['time'], self.data['stim_amp'],
                    self.params['ipi_left'], self.params['ipi_right'], self.params['ibi_left'], self.params['ibi_right'],
                    self.params['which_detrend']
                )
            else:
                print(f"polarity: {polarity} OK")
                
            
            print(f"Extracted ERNA slice for {self.file_obj}")
        except Exception as e:
            raise ValueError(f"Error extracting ERNA slice: {e}")

    def get_erna_features(self):
        try:
            erna_slice = self.data['data_erna']

            #self.params['mindist'] = np.round(self.params['mindist_prop'] / self.params['stim_freq'] * self.params['srate'])
            print(' ------------# self params on srate' )
            print(self.params['win_ipi_max'])
            if self.params['window_max_depends_on_srate']:
                self.params['win_ipi_max'] = 1/self.params['stim_freq']*1000 - 1
                if self.params['win_ipi_max'] > 10: # do not wait for more tha 15 ms to find a peak...
                    self.params['win_ipi_max'] = 10

            print(self.params['win_ipi_max'])

            # any time
            IPI_features = compute_IPI_features(erna_slice, self.params)
            
            IBI_features = compute_IBI_features(erna_slice, self.params)
            print("ERNA features computed")
            return IPI_features, IBI_features
        except Exception as e:
            raise ValueError(f"Error computing ERNA features: {e}")

    def set_windows(self):
        try:
            print(self.params)
            stim_freq = self.params['stim_freq']
            self.params['win_left'] = -int(np.floor(self.params['NPULSES_PER_BURST'] / stim_freq * self.params['srate']))
            self.params['win_right'] = int(np.floor(0.030 * self.params['srate']))
            self.params['ipi_left'] = int(np.floor(0.0013 * self.params['srate']))
            self.params['ipi_right'] = int(np.floor((1 / stim_freq - 0.0018) * self.params['srate']))
            self.params['ibi_left'] = int(np.floor(0.0013 * self.params['srate']))
            self.params['ibi_right'] = int(np.floor(self.params['IBI_WINDOW'][str(stim_freq)] * self.params['srate']))
            print("Windows set for ERNA processing")
        except Exception as e:
            raise ValueError(f"Error setting windows: {e}")




class ERNAPlatformRequest:
    def __init__(self, participant, srate=25000, NPULSES_PER_BURST=10,
                 flow_cut=1000, fhigh_cut=2, order_low=4, order_high=4,
                 flip_polarity=True, which_detrend='linear',
                 IBI_WINDOW=None, thr_pulses=50,
                 peak_prom_ipi = 10, win_ipi_min=2.5, win_ipi_max = None, window_max_depends_on_srate = True, num_mad_outlier = 4,
                 peak_prom=5, peak_width=0.5, win_ibi_min = 2.75, win_ibi_max = 5.5, mindist=2, minpeaks=2, npeaks=8,
                 write_flag =True, ask_before_overwrite = True ):

        if IBI_WINDOW is None:
            IBI_WINDOW = {'25': 0.025, '65': 0.025, '130': 0.025, '180': 0.025}

        self.participant_id = participant['id']
        self.participant_raw = participant['data_path']
        self.participant_annot = participant['annot_path']
        self.participant_save = participant['save_path']
        self.participant_figure = participant["figure_path"]
        self.store = {}
        self.IPI_annot = pd.DataFrame()
        self.IBI_annot = pd.DataFrame()
        self._progress_bar = None
        


        self.params = {
            'srate': srate,
            'NPULSES_PER_BURST': NPULSES_PER_BURST,
            'flow_cut': flow_cut,
            'fhigh_cut': fhigh_cut,
            'order_low': order_low,
            'order_high': order_high,
            'flip_polarity': flip_polarity,
            'which_detrend': which_detrend,
            'IBI_WINDOW': dict(IBI_WINDOW),
            'thr_pulses': thr_pulses,
            'peak_prom_ipi': peak_prom_ipi,
            'win_ipi_min':win_ipi_min,
            'win_ipi_max':win_ipi_max,
            'window_max_depends_on_srate':window_max_depends_on_srate,
            'num_mad_outlier':num_mad_outlier,
            'peak_prom': peak_prom,
            'peak_width': np.round(peak_width * srate / 1000),
            'win_ibi_min': win_ibi_min,
            'win_ibi_max': win_ibi_max,
            'mindist_prop': np.round(mindist * srate / 1000),
            'minpeaks': minpeaks,
            'npeaks': npeaks,
            'write_flag': write_flag,
            'ask_before_overwrite' : ask_before_overwrite

        }
        
        print("List of parameters:")
        print(json.dumps(self.params, indent = 4))

        self.get_annots()
        #self.set_windows()

    def get_annots(self):
        try:
            erna_annots = load_erna_annots(self.participant_annot, self.participant_id, annot_keys = "raw")
            self.run_annot = erna_annots['run_annot']
            self.stimScan_annot = erna_annots["stimScan_annot"]
            self.ampRamp_annot = erna_annots["ampRamp_annot"]
            self.channels_annot = erna_annots["channels_annot"]
            self.coords_annot = erna_annots["coords_annot"]            
            print(f"Annotations loaded for participant {self.participant_id}")
        except Exception as e:
            raise ValueError(f"Error loading annotations: {e}")

    def get_erna_files(self):
        try:
            mat_files = load_erna_file(self.participant_raw)
            print(f"ERNA files loaded: {mat_files}")
            return mat_files
        except Exception as e:
            raise ValueError(f"Error loading ERNA files: {e}")

    def get_erna_data(self, mat_file ) -> ERNAFileHandler:
        try:
            file_handler = ERNAFileHandler(self.participant_id,
                self.participant_raw, self.participant_save, self.participant_annot, mat_file, self.params
            )
            file_handler.extract_erna_data()
            run_details = file_handler.get_run_details(self.stimScan_annot, self.ampRamp_annot, self.run_annot)
            print("run_det")
            print(run_details)
            if run_details is not None:
                
                file_handler.params['stim_freq'] = run_details['Freq'].values[0]
            file_handler.set_windows()

            file_handler.get_pulse_locs()
            file_handler.get_erna_slice()
            return file_handler, run_details
        except Exception as e:
            raise ValueError(f"Error getting ERNA data from file {mat_file}: {e}")

    def process_pipeline(self, mat_file):
        try:
            
            data, run_details = self.get_erna_data(mat_file)
            IPI_features, IBI_features = data.get_erna_features()
            stimChannel = data.get_stimCh(run_details, self.channels_annot, self.coords_annot)
            
            # add fields
            #data.IPI_features = IPI_features
            #data.IBI_features = IBI_features
            data.run_details = run_details
            data.stimChannel = stimChannel
            
            return data, IPI_features, IBI_features
        except Exception as e:
            raise ValueError(f"Error in processing pipeline for file {mat_file}: {e}")

    def store_params(self, filename):
        store_for_json = self.params
        save_dict(store_for_json,filename,'json',ask_before_overwrite=self.params['ask_before_overwrite'],convert_numpy_to_python=True, verbose=True)

    def store_erna(self, filename):
        store_for_json = {}
        for key, value in self.store.items():
                data_to_save = {k: v for k, v in value['data'].items() if k != 'data_erna'}
                store_for_json[key] = {
                    'file': value['file'],
                    'run_details': value['run_details'].to_dict(orient='records'),  # Convert DataFrame to dict
                    'stimChannel': value['stimChannel'],
                    'data': data_to_save  # Convert to JSON-compatible format
                }    
        save_dict(store_for_json,filename,'json',ask_before_overwrite=self.params['ask_before_overwrite'],convert_numpy_to_python=True, verbose=True)

        
    def store_raw(self, filename):
        store_for_json = {}
        for key, value in self.store.items():
                data_to_save = {k: v for k, v in value['data'].items() if k != 'data_cont'}
                store_for_json[key] = {
                    'file': value['file'],
                    'run_details': value['run_details'].to_dict(orient='records'),  # Convert DataFrame to dict
                    'stimChannel': value['stimChannel'],
                    'data': data_to_save  # Convert to JSON-compatible format
                }    
        save_dict(store_for_json,filename,'json',ask_before_overwrite=self.params['ask_before_overwrite'],convert_numpy_to_python=True, verbose=True)

    def store_annot(self, filename):
        save_dict(self.IPI_annot,filename['IPI_annot'],'tsv',ask_before_overwrite=self.params['ask_before_overwrite'],convert_numpy_to_python=False, verbose=True)
        save_dict(self.IBI_annot,filename['IBI_annot'],'tsv',ask_before_overwrite=self.params['ask_before_overwrite'],convert_numpy_to_python=False, verbose=True)
   

        



    def store_results(self, data_keys: list):
        # Convert the store dictionary to a JSON-compatible format
        data_paths = setup_data_paths(self.participant_save,self.participant_id)

        annot_paths = setup_annot_paths(self.participant_annot, self.participant_id)

        if not isinstance(data_keys, list): data_keys = [data_keys] 
        for _,data_key in enumerate(data_keys):
            if data_key == 'params':
                self.store_params(data_paths[data_key] )
            elif data_key == 'annots':
                self.store_annot(annot_paths)
            elif data_key == 'raw':
                self.store_raw(data_paths[data_key] )
            elif data_key == 'erna':
                self.store_erna(data_paths[data_key] )
            # Save the entire store to a JSON file
            print(f"All ERNA {data_key} saved.")


def process_ERNAfile(ERNA, mat_file):
    try:
        ERNAfile, IPI_features, IBI_features = ERNA.process_pipeline(mat_file)


        # Update internal database with relevant data
        ERNA.store[ERNAfile.run] = {
            'file': mat_file,
            'run_details': ERNAfile.run_details,
            'stimChannel': ERNAfile.stimChannel,
            'data': ERNAfile.data  # Save full data if needed
        }

        
        # Create DataFrame from features
        ipi_df = pd.DataFrame.from_dict(flatten_ipi_features(IPI_features))
        ibi_df = pd.DataFrame.from_dict(IBI_features).transpose()

        # Combine features and run details
        #combined_df = pd.concat([ipi_df, ibi_df], axis=1)
        ipi_df['Run'] = ERNAfile.run_details['Run'].values[0]  # Assuming 'Run' is a column
        ibi_df['Run'] = ERNAfile.run_details['Run'].values[0]  # Assuming 'Run' is a column

        # Append to the cumulative DataFrame
        ERNA.IPI_annot = pd.concat([ERNA.IPI_annot, ipi_df], ignore_index=True)
        ERNA.IBI_annot = pd.concat([ERNA.IBI_annot, ibi_df], ignore_index=True)
  
        #
        #ERNAfile.dict_to_tsv(pd.DataFrame.from_dict(flatten_ipi_features(IPI_features)), ERNAfile.bids_notation('_IPI.tsv', 'annot'))
        #ERNAfile.dict_to_tsv(pd.DataFrame.from_dict(IBI_features).transpose(), ERNAfile.bids_notation('_IBI.tsv', 'annot'))
    except Exception as e:
        print(f"Failed to process {mat_file}: {e}")

def process_ERNAfilesbatch(ERNAs):
    """
    Processes a single ERNA instance or a list of ERNA instances.

    Args:
        ERNAs: An instance of ERNA or a list of ERNA instances.
    """
    
    ERNAs = check_python_list(ERNAs) # Wrap a single instance into a list for uniform processing.


    for ERNA in ERNAs:
        print(f"Processing participant: {ERNA.participant_id}")
        
        # Get the mat files associated with this ERNA instance.
        mat_files = ERNA.get_erna_files()

        # Initialize a progress bar for this instance.
        ERNA._progress_bar = ProgressBar(
            n_steps=len(mat_files),
            title=f"Preprocessing ERNA in {ERNA.participant_id}"
        )

        # Process files one by one.
        for i, mat_file in enumerate(mat_files):
            print(f"Start file {mat_file}: {i + 1} - {len(mat_files)}")
            process_ERNAfile(ERNA, mat_file)
            print("OK!")
            print(f"Completed file {mat_file}: {i + 1} - {len(mat_files)}")
            
            if ERNA._progress_bar is not None:
                ERNA._progress_bar.update_progress()

        # Close the progress bar for this instance.
        if ERNA._progress_bar is not None:
            ERNA._progress_bar.close()

        # If the write flag is set, store results for this instance.
        if ERNA.params.get('write_flag', False):
            print(f"Start saving for participant: {ERNA.participant_id}")
            ERNA.store_results(data_keys=["annots", "params", "erna"])
            
        print(f"Completed processing for participant: {ERNA.participant_id}\n")










    
    

















'''
# Function to generate distinctive colors
def linspecer(N):
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    return colors

def get_mesh_from_nifti(nifti_file = "resources/STN_thr05.nii.gz"):
    img = nib.load(nifti_file)
    data = img.get_fdata()
    verts, faces, _, _ = measure.marching_cubes(data, level=0.5)
    affine = img.affine
    verts = nib.affines.apply_affine(affine, verts)
    return verts, faces

def plot_3Dmesh(nifti_file = "resources/STN_thr05.nii.gz"):
    verts, faces = get_mesh_from_nifti(nifti_file)
    x, y, z = zip(*verts)
    i, j, k = zip(*faces)
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
    x=x,
    y=y,
    z=z,
    i=i,
    j=j,
    k=k,
    color='orange',
    opacity=0.2,
    name='Subthalamic Nucleus'
    ))
    # Set plot layout
    fig.update_layout(
        scene=dict(
            xaxis_title='MNI X [mm]',
            yaxis_title='MNI Y [mm]',
            zaxis_title='MNI Z [mm]',
            aspectmode='cube'  # Ensure equal scaling on axes

        ),
    )
    return fig



def create_spheres(coords, radius=1):
    d = np.pi / 32  # Increment for theta and phi

    theta, phi = np.mgrid[0:np.pi + d:d, 0:2 * np.pi + d:d]
    # Convert to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi) + coords[0]
    y = radius * np.sin(theta) * np.sin(phi) + coords[1]
    z = radius * np.cos(theta) + coords[2]

    return x, y, z


@st.cache_data
def show_recordings(fig, coords, radius = 1):
    # Plot the recording location
    for coord in coords:
        x,y,z = create_spheres(coord, radius)
        if coord[0] > 0:
            x  = - x # flip from RH to LH
        fig.add_trace(go.Mesh3d(
            x=x.flatten(), 
            y=y.flatten(), 
            z=z.flatten(), color='red', opacity=0.50, alphahull=0
            )
        )
    fig.update_traces(lighting=dict(ambient=0.5, specular=1.0))
    
    return fig
    
'''

