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
from ERNA_GUI.io.general import (
    load_config
)
from ERNA_GUI.utils.handle_ftypes import check_task_validity, check_python_list

# leave here for now
cfg = load_config()


# fetch data task
def fetch_task_data(results, run_annot, which_tasks, valid_tasks = cfg.device.TASKS) -> dict:
    # Ensure `which_tasks` is a proper list.
    which_tasks = check_python_list(which_tasks)

    data_run = {}

    for task in which_tasks:
        # Check task validity.
        #task = check_task_validity(task, valid_tasks)
        print(task)
        
        # Ensure the task key exists in the dictionary.
   
        # Iterate through the run annotations for the task.
        for _, row in run_annot.iterrows():
            if row['Task'] == task:
                single_id = str(row['Run'])  # Fetch the unique Run ID.
                
                # Ensure sub-dictionary for the run ID exists.
                #if single_id not in data_run[task]:
                #    data_run[task][single_id] = {}


                # Fetch data for the Run ID, or add a message if unavailable.
                if single_id in results:

                    #data_run[task][single_id]['Run_details'] = results[single_id]['run_details']
                    #data_run[task][single_id]['stimChannels'] = results[single_id]['stimChannels']
                    #data_run[task][single_id]['data'] = results[single_id]['data']
                    data_run = unpack_results(data_run, results[single_id], single_id, task)
                
                else:
                    raise FileNotFoundError(f'Data not available for Run ID {single_id} under Task {task}')
    
    return data_run             
                    

def unpack_results(data_run, result, run, task):
   # Initialize the required keys if they do not exist
    keys_to_initialize = [
        'run', 'run_details', 'task', 'stimChannels', 
        'time_gtc', 'data_cont', 'data_erna', 
        'timebursts', 'stim_amp'
    ]

    for key in keys_to_initialize:
        data_run.setdefault(key, [])
        
    # Append the relevant data
    data_run['run'].append(run)
    data_run['task'].append(task)
    data_run['run_details'].append(result.get('run_details', None))
    data_run['stimChannels'].append(result.get('stimChannels', None))
    data_run['time_gtc'].append(result['data'].get('time', None))
    data_run['data_cont'].append(result['data'].get('erna_cont', None))
    data_run['data_erna'].append(result['data'].get('erna_data', None))
    data_run['timebursts'].append(result['data'].get('timebursts', None))
    data_run['stim_amp'].append(result['data'].get('stim_amp', None))
    
    return data_run


@st.cache_data
def fetch_run_task(run_annot, run):
    task = run_annot.loc[run_annot['Run'] == run, 'Task'].values[0]
    return task



@st.cache_data
def fetch_run_details(task_annots, run_annot, run, task = None):
    """
    Fetch details for a specific run from annotations based on the task in the run_annot.

    Args:
        annotations (dict): Dictionary containing annotation dataframes for different tasks.
        run_annot (DataFrame): Dataframe with task annotations for each run.
        run (str): The run identifier.

    Returns:
        DataFrame or None: The details of the run, or None if no details are found.
    """
    try:
        if task is None:
            task = fetch_run_task(run_annot, run)  # Fetch the task for this run

        if task in task_annots:
            run_details = task_annots[task][task_annots[task]['Run'] == run]
        else:
            run_details = None
        print(f"Run details fetched for run {run}")
        return run_details
    except KeyError as e:
        raise ValueError(f"Task '{task}' not found in annotations: {e}")
    except Exception as e:
        raise ValueError(f"Error getting run details for run {run}: {e}")
    
    




@st.cache_data
def fetch_stimCh_coordinates( run_details, channels_annot, coords_annot):
    stimCh = run_details['StimC'].values[0]
    ## Get the name where channel matches StimC
    #channelStim_name = channels_annot.loc[channels_annot['channel'] == stimCh, 'name'].values[0]
    
    # Get the name where channel matches StimC
    # Split stimCh if it contains multiple channels
    channels_to_query = stimCh.split('-')

    # Fetch names corresponding to all channels
    channel_stims = channels_annot.loc[channels_annot['channel'].isin(channels_to_query), 'name'].values
    
    # If you need them as a list or a concatenated string
    channelStim = list(channel_stims)  # as a list
    channelStim_string = '-'.join(channel_stims)  # as a string
    
    coords_mni = []
    coords_native = []
    
    for name in channelStim:
        coords_native.append(coords_annot.loc[coords_annot['name'] == name, ['native_x', 'native_y', 'native_z']].values[0])# Append the coordinates            
        coords_mni.append(coords_annot.loc[coords_annot['name'] == name, ['mni_x', 'mni_y', 'mni_z']].values[0])# Append the coordinates
    coords_native = np.array(coords_native)
    coords_mni = np.array(coords_mni)
    
    
    channelStim = {
        'name' : channelStim_string,
        'coords' : {
            'native'  : coords_native,
            'mni' :   coords_mni
        }  
    }
    return channelStim
        
        

@st.cache_data
def preprocess_erna(sig, flow_cut=5000, fhigh_cut=2, order_low=4, order_high=4, srate=25000):
    sos_low = signal.butter(order_low, flow_cut, 'low', output='sos', fs=srate)
    sig = signal.sosfilt(sos_low, sig)
    sos_high = signal.butter(order_high, fhigh_cut, 'high', output='sos', fs=srate)
    sig = signal.sosfilt(sos_high, sig)
    return sig

@st.cache_data
def extract_pulses(erna, timebursts, time, win_left, win_right, thr_pulses, stim_freq, srate):
    pulse_locs, erna_slice, time_erna_slice = {}, {}, {}

    for i in timebursts:
        erna_slice[i], idx_win = slice_erna(erna, time, i, win_left, win_right)
        peaks, _ = find_peaks(np.diff(erna_slice[i]), height=thr_pulses, distance=np.round((1 / stim_freq) / 2 * srate))
        time_current = time[np.arange(idx_win[0], idx_win[1] + 1)]
        if len(peaks) > 0:
            pulse_locs[i] = time_current[peaks + 1]
        else:
            pulse_locs[i] = np.array([i]) #if no peak identified at all, put at least the last logged out by the device
        
        time_erna_slice[i] = time[np.arange(idx_win[0], idx_win[1] + 1)]
    return pulse_locs, erna_slice, time_erna_slice

@st.cache_data
def extract_erna_slice(erna, pulse_locs, time, stim_amp,ipi_left, ipi_right, ibi_left, ibi_right, which_detrend = 'linear'):
    detrend_dict = {"linear":linear_func,
                     "quadratic":quad_func   
    }
    erna_slice = {}
    for stim_amp_i, (burst_i,(burst, pulses)) in zip(stim_amp,enumerate(pulse_locs.items())):
        if pulses.any():
            # Ensure the burst key and sub-dictionaries exist
            if burst not in erna_slice:
                erna_slice[burst] = {'ibi': {}, 'ipi': {}}
            
            # Extract IBI slice
            erna_slice[burst]['ibi']['burst_id'] = burst_i            
            erna_slice[burst]['ibi']['erna'], burst_idx_win = slice_erna(erna, time, burst, ibi_left, ibi_right)
            erna_slice[burst]['ibi']['time'] = time[np.arange(burst_idx_win[0], burst_idx_win[1] + 1)]
            erna_slice[burst]['ibi']['stim_amp'] = stim_amp_i
           
           
            if which_detrend != "None":
                
                erna_slice[burst]['ibi']['erna'] = detrend_func(erna_slice[burst]['ibi']['erna'], erna_slice[burst]['ibi']['time'], detrend_dict[which_detrend])
                
            for pulse_i, pulse in enumerate(pulses):
                # Ensure the pulse key and sub-dictionary exist
                if pulse not in erna_slice[burst]['ipi']:
                    erna_slice[burst]['ipi'][pulse] = {}
                
                # Extract IPI slice
                erna_slice[burst]['ipi'][pulse]['pulse_id'] = pulse_i    
                erna_slice[burst]['ipi'][pulse]['burst_id'] = burst_i                              
                erna_slice[burst]['ipi'][pulse]['erna'], pulse_idx_win = slice_erna(erna, time, pulse, ipi_left, ipi_right)
                erna_slice[burst]['ipi'][pulse]['time'] = time[np.arange(pulse_idx_win[0], pulse_idx_win[1] + 1)]
                erna_slice[burst]['ipi'][pulse]['stim_amp'] = stim_amp_i
               
                if which_detrend != "None":
                    erna_slice[burst]['ipi'][pulse]['erna'] = detrend_func(erna_slice[burst]['ipi'][pulse]['erna'], erna_slice[burst]['ipi'][pulse]['time'], detrend_dict[which_detrend])  
    return erna_slice 
    

def slice_erna(erna, time, evt, win_left, win_right): # don't cache this for efficiency
    idx_win = get_idx_win(time, evt, win_left, win_right)
    tmp = erna[np.arange(idx_win[0], idx_win[1] + 1)]
    erna_slice = tmp.reshape(1, -(win_left) + (win_right) + 1).flatten()
    return erna_slice, idx_win

def get_idx_win(time, evt, win_left, win_right): # don't cache this for efficiency
    return np.argwhere(time == evt)[0] + (win_left, win_right)


def detrend_func(y, x, func):
    popt, _ = curve_fit(func, x, y)
    func_fit = func(x, *popt)  
    yd = y - func_fit
    return yd


def linear_func(x, a, b):
    return a * x + b

def quad_func(x, a, b,c):
    return a * x**2 + b*x + c


def remove_zerostim(timebursts, stim_amp):         
    # remove 0 mA
    timebursts = timebursts[np.where(stim_amp > 0)[0]]   
    stim_amp = stim_amp[np.where(stim_amp > 0)[0]]   
    return timebursts, stim_amp




@st.cache_data
def compute_IPI_features(erna, thr_ipi_params):
    peak_prom = thr_ipi_params['peak_prom_ipi']
    win_ipi_min = thr_ipi_params['win_ipi_min']
    win_ipi_max = thr_ipi_params['win_ipi_max']
    num_mad_outlier = thr_ipi_params["num_mad_outlier"]
    
    
    IPI_features = {}
    all_amplitudes = []
    all_latencies = []
    burst_pulse_map = []
    
    # Collect amplitude and latency data for all pulses
    for burst, burst_data in erna.items():
        if burst not in IPI_features:
            IPI_features[burst] = {}        
        for i, (pulse, ipi_data) in enumerate(burst_data['ipi'].items()):
            ipi = ipi_data['erna']
            time = ipi_data['time'] - pulse
            mask = (time >= win_ipi_min / 1000) & (time <= win_ipi_max / 1000)
            ipi_win = ipi[mask]
            time_win = time[mask]

            if ipi_win.any():
                loc_peak = np.argmax(ipi_win)
               
                loc_trough = np.argmin(ipi_win)
                peak = ipi_win[loc_peak]
                trough = ipi_win[loc_trough]
                lat = time_win[loc_peak]
                amplitude = abs(peak - trough)
                
                if amplitude >= peak_prom and lat <= 0.005:            
                    # Store features in temp structure
                    IPI_features[burst][pulse] = {
                        'pulse_id': ipi_data['pulse_id'],
                        'burst_id': ipi_data['burst_id'],
                        'stim_amp': ipi_data['stim_amp'],
                        'peak': peak,
                        'trough': trough,
                        'amplitude': amplitude,
                        'latency': lat * 1000  # convert to ms
                    }
                    
                    # Collect values for MAD outlier detection
                    all_amplitudes.append(amplitude)
                    all_latencies.append(lat * 1000)  # convert to ms
                    burst_pulse_map.append((burst, pulse))

    # Convert to numpy arrays for efficient operations
    all_amplitudes = np.array(all_amplitudes)
    all_latencies = np.array(all_latencies)

    # Find outliers using MAD
    amplitude_outliers = mad_based_outlier(all_amplitudes, threshold=num_mad_outlier)
    latency_outliers = mad_based_outlier(all_latencies, threshold=num_mad_outlier)

    # Remove outliers
    for i, (burst, pulse) in enumerate(burst_pulse_map):
        if amplitude_outliers[i] or latency_outliers[i]:
            del IPI_features[burst][pulse]


    return IPI_features
    
# remvoe evertyhing threshold times outside the mad
def mad_based_outlier(points, threshold=4):
    """
    Returns a boolean array where True indicates that the corresponding value is an outlier.
    """
    median = np.median(points)
    diff = np.abs(points - median)
    mad = np.median(diff)
    
    if mad == 0:  # To avoid division by zero
        return np.zeros(len(points), dtype=bool)
    
    modified_z_score = 0.6745 * diff / mad # 0.6745 is a scaling factor to link mad with std
    return modified_z_score > threshold
    
      
def flatten_ipi_features(IPI_features):
    flattened_data = []
    for burst, burst_data in IPI_features.items():
        for pulse, pulse_data in burst_data.items():
            flattened_entry = {
                'burst': burst,
                'pulse': pulse,
                'pulse_id': pulse_data['pulse_id'],
                'burst_id': pulse_data['burst_id'],
                'stim_amp': pulse_data['stim_amp'],
                'peak': pulse_data['peak'],
                'trough': pulse_data['trough'],
                'amplitude': pulse_data['amplitude'],
                'latency': pulse_data['latency']
            }
            flattened_data.append(flattened_entry)
    return flattened_data    

@st.cache_data
def compute_IBI_features(erna, thr_ibi_params):
    
    peak_prom = thr_ibi_params['peak_prom']
    peak_width = thr_ibi_params['peak_width']
    win_ibi_min = thr_ibi_params['win_ibi_min']
    win_ibi_max = thr_ibi_params['win_ibi_max']    
    mindist = thr_ibi_params['mindist_prop']
    npeaks = thr_ibi_params['npeaks']
    minpeaks = thr_ibi_params['minpeaks']



    IBI_features = {}
    for burst, burst_data in erna.items():
        if burst not in IBI_features:
            IBI_features[burst] = {}

        ibi = burst_data['ibi']['erna']
        time = (burst_data['ibi']['time'] - burst) * 1000  # Time in milliseconds

        if ibi.any():
            peaks_tmp, _ = find_peaks(ibi, width=peak_width, distance=mindist, prominence=peak_prom)
            troughs_tmp, _ = find_peaks(-ibi, width=peak_width, distance=mindist, prominence=peak_prom)

            if len(peaks_tmp) >= minpeaks and len(troughs_tmp) >= minpeaks:
                if troughs_tmp[0] < peaks_tmp[0]:
                    troughs = troughs_tmp[1:]
                else:
                    troughs = troughs_tmp

                if peaks_tmp[-1] > troughs_tmp[-1]:
                    peaks = peaks_tmp[:-1]
                else:
                    peaks = peaks_tmp

                npeaks_id = len(peaks)

                if npeaks_id > npeaks:
                    peaks = peaks[:npeaks]
                    troughs = troughs[:npeaks]

                if npeaks_id >= minpeaks:
                    amplitude = ibi[peaks[0]] - ibi[troughs[0]]
                    latency = time[peaks[0]]
                    frequency = 0.5 / (time[troughs[0]] - latency) * 1000
                    

                    # Apply latency and frequency conditions
                    if win_ibi_min <= latency <= win_ibi_max and frequency > 0:  # Keep the burst only if latency is within [2ms, 15ms] and frequency is positive
                        peaks_list = {}
                        trough_list = {}

                        for i, (peak, trough) in enumerate(zip(peaks, troughs)):
                            if i not in IBI_features[burst]:
                                peaks_list[i] = {}
                                trough_list[i] = {}

                            peaks_list[i]['lat'] = time[peak]
                            peaks_list[i]['amp'] = ibi[peak]
                            trough_list[i]['lat'] = time[trough]
                            trough_list[i]['amp'] = ibi[trough]

                        # Save features
                        IBI_features[burst]['burst_id'] = burst_data['ibi']['burst_id']
                        IBI_features[burst]['stim_amp'] = burst_data['ibi']['stim_amp']
                        IBI_features[burst]['ERNA_flag'] = 1
                        IBI_features[burst]['peaks'] = peaks_list
                        IBI_features[burst]['troughs'] = trough_list
                        IBI_features[burst]['npeaks'] = npeaks_id
                        IBI_features[burst]['amplitude'] = amplitude
                        IBI_features[burst]['latency'] = latency
                        IBI_features[burst]['frequency'] = frequency
                    else:
                        # If the latency or frequency is out of bounds, discard the burst
                        IBI_features[burst]['burst_id'] = burst_data['ibi']['burst_id']
                        IBI_features[burst]['stim_amp'] = burst_data['ibi']['stim_amp']
                        IBI_features[burst]['ERNA_flag'] = 0
                else:
                    IBI_features[burst]['burst_id'] = burst_data['ibi']['burst_id']
                    IBI_features[burst]['stim_amp'] = burst_data['ibi']['stim_amp']
                    IBI_features[burst]['ERNA_flag'] = 0
            else:
                IBI_features[burst]['burst_id'] = burst_data['ibi']['burst_id']
                IBI_features[burst]['stim_amp'] = burst_data['ibi']['stim_amp']
                IBI_features[burst]['ERNA_flag'] = 0

    return IBI_features




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