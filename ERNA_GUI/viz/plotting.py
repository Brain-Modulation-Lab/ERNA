import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import plotly.graph_objects as go
from pathlib import Path
import os
import plotly.io as pio
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
from ERNA_GUI.core.utility import (
    fetch_IBImatrix, fetch_IPImatrix
)

from ERNA_GUI.io.general import load_config


# get base dir of package
current_script = Path(__file__)
BASE_DIR = current_script.parent.parent.parent


import streamlit as st



# Function to generate distinctive colors
def linspecer(N):
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    return colors

def set_cmap(vmin, vmax, cm):
    norm = plt.Normalize(vmin, vmax )
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm) 
    return norm, sm

# show recordings
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

def get_mesh_from_nifti(nifti_file):
    img = nib.load(nifti_file)
    data = img.get_fdata()
    verts, faces, _, _ = measure.marching_cubes(data, level=0.5)
    affine = img.affine
    verts = nib.affines.apply_affine(affine, verts)
    return verts, faces


def plot_DISTALmini(fig = None, coords = None, radius = 1): # this plot STN, GPe and GPi w/ coords if needed
    if fig is None:
        fig = go.Figure()
    fig = plot_3Dmesh(fig, region_input = 'STN')
    fig = plot_3Dmesh(fig, region_input = 'GPi')
    fig = plot_3Dmesh(fig, region_input = 'GPe')
    if coords is not None:
        fig = show_recordings(fig, coords, radius = radius)
    return fig


def plot_3Dmesh(fig = None, region_input = 'STN', atlas = 'DISTAL'):
    if isinstance(region_input, str):
        name_region = region_input
        if name_region == "STN":
            color = 'orange'
            nifti_file = os.path.join(BASE_DIR, f"resources/{atlas}/STN.nii.gz")
        elif name_region == "GPi":
            color = 'blue'
            nifti_file = os.path.join(BASE_DIR, f"resources/{atlas}/GPi.nii.gz")
        elif name_region == "GPe":
            color = 'green'
            nifti_file = os.path.join(BASE_DIR, f"resources/{atlas}/GPe.nii.gz")
        else:
            raise ValueError(f"Unknown region name: {name_region}")
    elif isinstance(region_input, dict):
        nifti_file = region_input.get('nifti_file')
        color = region_input.get('color', 'gray')  # Default color if not provided
        name_region = region_input.get('name', 'Unknown Region')
    else:
        raise TypeError("Input must be a string (region name) or a dictionary with 'nifti_file', 'color', and 'name'.")
    
       
        
    verts, faces = get_mesh_from_nifti(nifti_file)
    x, y, z = zip(*verts)
    i, j, k = zip(*faces)
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Mesh3d(
    x=x,
    y=y,
    z=z,
    i=i,
    j=j,
    k=k,
    color= color,
    opacity=0.2,
    name= name_region
    ))          
    
        
    # Set plot layout
    fig.update_layout(
        scene=dict(
            xaxis_title='MNI X [mm]',
            yaxis_title='MNI Y [mm]',
            zaxis_title='MNI Z [mm]',
            #aspectmode='cube'  # Ensure equal scaling on axes

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
def plot_voltage(time, erna, stim_amp, timebursts, win_size, win_slid):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(time, erna, linewidth=0.4)
    ax2.plot(timebursts, stim_amp, color='m', linewidth=2)
    for x in timebursts:
        ax.axvline(x=x, color='r', linestyle='--', linewidth=0.4)

    if win_size:
        ax.set_xlim((win_slid, win_slid + win_size * 1E-3))
        ax2.set_xlim((win_slid, win_slid + win_size * 1E-3))

    ax.set_xlabel('time [GTC]')
    ax.set_ylabel('Voltage')
    ax2.set_ylabel('Stim. amp [mA]')
    return fig


@st.cache_data
def plot_erna_stimartefact(time_erna_pulses_toplot, erna_pulses_toplot, pulse_locs_timeburst, timeburst):

    fig, ax = plt.subplots()
    ax.plot(time_erna_pulses_toplot, erna_pulses_toplot)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=0.4, alpha = 0.8)
    ax.set_xlabel("Time after burst [ms]")
    ax.set_ylabel('LFP [uV]')


    if erna_pulses_toplot.any():
        ax2 = ax.twinx()
        ax2.plot(time_erna_pulses_toplot[:-1], np.diff(erna_pulses_toplot), color='r', alpha = 0.4)
        ax2.set_ylabel('Derivative Voltage', color='r')
        #col13.text_input('Number of Pulses:', value=len(pulse_locs[timeburst]))
        for x in pulse_locs_timeburst:
            ax.axvline(x=(x - timeburst)*1000, color='g', linestyle='--', linewidth=0.4)

    return fig
 



@st.cache_data
def plot_ipi(erna_slice_timeburst, IPI_features_timeburst, is_ipi_plot_collapsed, is_ipi_features, stim_freq, wideband):
    
    cm = plt.cm.hot
    norm, sm = set_cmap(1, len(erna_slice_timeburst['ipi'].values()), cm)
                
    fig, ax = plt.subplots()
    for i, (pulse, ipi_data) in enumerate(erna_slice_timeburst['ipi'].items()):
        if is_ipi_plot_collapsed:
            plt.plot((ipi_data['time'] - pulse)*1000, ipi_data['erna'], color = cm(norm(i)))      
            ax.set_xlabel('Time after pulse [ms]')
            ax.set_xlim(-0.3,1/stim_freq*1000 + 0.3)  
            ax.axvline(0,linestyle='--',color= 'k')   
            ax.axvline(1/stim_freq*1000,linestyle='--',color= 'k')    
            ax.axvspan(0,wideband, color='g', alpha=0.01)
        
            if is_ipi_features:
                if pulse in IPI_features_timeburst.keys():
                    ax.scatter(IPI_features_timeburst[pulse]['latency'],IPI_features_timeburst[pulse]['peak'], s = IPI_features_timeburst[pulse]['peak'], color =cm(norm(i)), alpha = 0.8)
                    ax.axvline(IPI_features_timeburst[pulse]['latency'],linestyle='--',color =cm(norm(i)), linewidth = 0.8)
    
                
                        
                                
        else:
            plt.plot(ipi_data['time']*1000, ipi_data['erna'], color = cm(norm(i)))
            ax.set_xlabel('Time [ms]')

    
    ax.set_ylabel('Erna [uV]')
    cbar = fig.colorbar(sm, ax= ax)
    cbar.set_label('# pulse')
    return fig

@st.cache_data
def plot_ibi(erna_slice_timeburst, IBI_features_timeburst, timeburst, is_ibi_features):
    fig, ax = plt.subplots()
    
    ibi_data = erna_slice_timeburst['ibi']
    
    # Plot ERNA data
    ax.plot((ibi_data['time'] - timeburst) * 1000, ibi_data['erna'], color='k')
    ax.set_xlabel('Time after burst [ms]')
    ax.axvline(0, linestyle='--', color='k')   
    ax.set_ylabel('Erna [uV]')
    
    # Plot peaks and troughs if IBI features are available and ERNA_flag is 1
    
    if is_ibi_features and IBI_features_timeburst['ERNA_flag'] == 1:
        for _, peak in IBI_features_timeburst['peaks'].items():
            ax.scatter(peak['lat'], peak['amp'], s=abs(peak['amp']), color='r', alpha=1)
        for _, trough in IBI_features_timeburst['troughs'].items():
            ax.scatter(trough['lat'], trough['amp'], s=abs(trough['amp']), color='b', alpha=1)  
    

        
        
        # Plot ERNA data
        ax.plot((ibi_data['time'] - timeburst) * 1000, ibi_data['erna'], color='k')
        ax.set_xlabel('Time after burst [ms]')
        ax.axvline(0, linestyle='--', color='k')   
        ax.set_ylabel('Erna [uV]')    
    return fig


@st.cache_data
def plot_features(IBI_features, time, timeburst, plot_type):
    fig, ax = plt.subplots()
    
    if plot_type == 'amplitude':
        ax.set_xlabel('Time burst [ms]')
        ax.set_ylabel('Erna peak-to-trough [uV]')
        for key, value in IBI_features.items():
            if value['ERNA_flag']:
                ax.scatter(key, value['amplitude'], s=abs(value['amplitude']), color='k', alpha=0.4)
        if IBI_features[timeburst]['ERNA_flag'] == 1:
            ax.scatter(timeburst, IBI_features[timeburst]['amplitude'], s=abs(IBI_features[timeburst]['amplitude']), color='r', alpha=1)
            ax.axvline(timeburst, linestyle='--', color='k')
        ax.set_ylim(bottom=0)    

    elif plot_type == 'latency':
        ax.set_xlabel('Time burst [ms]')
        ax.set_ylabel('Erna latency [ms]')
        for key, value in IBI_features.items():
            if value['ERNA_flag']:
                ax.scatter(key, value['latency'], s=abs(value['latency']), color='k', alpha=0.4)
        if IBI_features[timeburst]['ERNA_flag'] == 1:
            ax.scatter(timeburst, IBI_features[timeburst]['latency'], s=abs(IBI_features[timeburst]['latency']), color='r', alpha=1)
            ax.axvline(timeburst, linestyle='--', color='k')
        ax.set_ylim((2,6))    

    elif plot_type == 'frequency':
        ax.set_xlabel('Time burst [ms]')
        ax.set_ylabel('Erna frequency [Hz]')
        for key, value in IBI_features.items():
            if value['ERNA_flag']:
                ax.scatter(key, value['frequency'], s=abs(value['frequency']), color='k', alpha=0.4)
        if IBI_features[timeburst]['ERNA_flag'] == 1:
            ax.scatter(timeburst, IBI_features[timeburst]['frequency'], s=abs(IBI_features[timeburst]['frequency']), color='r', alpha=1)
            ax.axvline(timeburst, linestyle='--', color='k')
        ax.set_ylim(200, 450)    
    
    ax.set_xlim(np.min(time), np.max(time))
    return fig







