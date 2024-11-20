import os
import numpy as np
import pandas as pd
import re
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import streamlit as st
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go

from ERNA_GUI.viz.plotting import create_spheres
from ERNA_GUI.io.argus import get_argus_lfp

@st.cache_data
def load_erna_file(folder_path):
    if len(folder_path) > 0:
        mat_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and "_record" in os.path.join(folder_path, f)]
        if mat_files:
            st.sidebar.write(f"{len(mat_files)} txt files found in this participant.")
        else:
            st.write("No txt files found in the uploaded folder.")
    else:
        mat_files = []
    return mat_files

@st.cache_data(persist=True)
def load_selected_file(selected_file, max_fields = 3095):
  
    try:
        df = pd.read_table(selected_file, header=None, skiprows=1, na_values = "not available")
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
        # Try reading with a different method or skip bad lines
        try:
            df = pd.read_csv(selected_file, header=None, skiprows=1, delimiter='\t', na_values = "not available")
        except Exception as e:
            print(f"Failed to read the file using fallback method: {e}")
            ### Loop the data lines

            try:
              
                ### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
                column_names = [i for i in range(0, max_fields)]
                
                ### Read csv
                df = pd.read_csv(selected_file, header=None, skiprows=1, delimiter='\t', na_values = "not available", usecols=column_names)
                
            except Exception as e:
                print(f"An error occurred: {e}")
   
    
    print(df)
    print(get_run_id(os.path.basename(selected_file)))
    run_id = get_run_id(os.path.basename(selected_file))
    data_cont_df, time_vect_cont, stim_amp, timebursts = get_argus_lfp(df, srate=25000) # this is onlt for argus
    

    return {
        "run": run_id,
        "time": time_vect_cont,
        "data_cont": data_cont_df,
        "stim_amp": stim_amp,
        "timebursts": timebursts     
    }

@st.cache_data
def load_erna_annots(annot_files, participant):
    run_annot = pd.read_csv(os.path.join(annot_files, f"subj-{participant}_ses-intraop_runs.tsv"), sep='\t')
    stimScan_annot = pd.read_csv(os.path.join(annot_files, f"subj-{participant}_ses-intraop_task-stimScan_annot.tsv"), sep='\t')
    ampRamp_annot = pd.read_csv(os.path.join(annot_files, f"subj-{participant}_ses-intraop_task-ampRamp_annot.tsv"), sep='\t')
    channels_annot = pd.read_csv(os.path.join(annot_files, f"subj-{participant}_ses-intraop_channels.tsv"), sep='\t')
    coords_annot = pd.read_csv(os.path.join(annot_files, f"subj-{participant}_ses-intraop_electrodes.tsv"), sep='\t')
   
    return run_annot, stimScan_annot, ampRamp_annot, channels_annot, coords_annot



@st.cache_data
def get_run_id(filename):
    pattern = re.compile(r'record(\d+)')
    match = pattern.search(filename)
    return int(match.group(1))

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


def set_cmap(vmin, vmax, cm):
    norm = plt.Normalize(vmin, vmax )
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm) 
    return norm, sm

@st.cache_data
def compute_IPI_features(erna, thr_ipi_params):
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
                
                if amplitude >= 10 and lat <= 0.0056:            
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
    mindist = thr_ibi_params['mindist']
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
                    if 2 <= latency <= 15 and frequency > 0:  # Keep the burst only if latency is within [2ms, 15ms] and frequency is positive
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

'''
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
# Sample data to replace your variables
nRuns = 5
runs_annot = {'Run': np.arange(1, nRuns+1)}
stimScan_annot = {'Run': np.arange(1, nRuns+1)}
ampRamp_annot = {'Run': np.arange(1, nRuns+1)}
IPA = {'timepulses': [np.linspace(0, 10, 100) for _ in range(nRuns)],
       'run_id': np.arange(1, nRuns+1)}


# Generate colors
colors = linspecer(nRuns)



def plot_annot_runs():
# Create figure and layout
fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [1, 1, 1]})

# Plot all events for runs_annot
ax1 = axs[0]
for run_i in range(nRuns):
    run = runs_annot['Run'][run_i]
    ax1.plot(IPA['timepulses'][run_i], run*np.ones(len(IPA['timepulses'][run_i])), linewidth=2, color=colors[IPA['run_id'][run_i]-1])
ax1.set_xlabel('Time GTC [s]')
ax1.set_ylabel('Argus file [id]')

# Plot all events for stimScan_annot
ax2 = axs[1]
for run_i in range(len(stimScan_annot['Run'])):
    run = stimScan_annot['Run'][run_i]
    ax2.plot(IPA['timepulses'][IPA['run_id'] == run][0], run*np.ones(len(IPA['timepulses'][IPA['run_id'] == run][0])), linewidth=2, color=colors[IPA['run_id'][IPA['run_id'] == run][0]-1])
ax2.set_xlabel('Time GTC [s]')
ax2.set_ylabel('StimScan file [id]')

# Plot all events for ampRamp_annot
ax3 = axs[2]
for run_i in range(len(ampRamp_annot['Run'])):
    run = ampRamp_annot['Run'][run_i]
    ax3.plot(IPA['timepulses'][IPA['run_id'] == run][0], run*np.ones(len(IPA['timepulses'][IPA['run_id'] == run][0])), linewidth=2, color=colors[IPA['run_id'][IPA['run_id'] == run][0]-1])
ax3.set_xlabel('Time GTC [s]')
ax3.set_ylabel('AmpRamp file [id]')

# Set common axis range
def set_common_axis_range(axs, axis):
    if axis == 1:
        lims = [min(ax.get_xlim()[0] for ax in axs), max(ax.get_xlim()[1] for ax in axs)]
        for ax in axs:
            ax.set_xlim(lims)
    elif axis == 2:
        lims = [min(ax.get_ylim()[0] for ax in axs), max(ax.get_ylim()[1] for ax in axs)]
        for ax in axs:
            ax.set_ylim(lims)

set_common_axis_range(axs, 1)
set_common_axis_range(axs, 2)

# Save the figure
fig.savefig(f'subj-PATIENT_ses-SESSION_argus.png')

plt.show()

'''


