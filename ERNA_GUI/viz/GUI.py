import os
import numpy as np
import streamlit as st
import pandas as pd

from ERNA_GUI.core.processing import load_erna_file, load_selected_file, load_erna_annots
from ERNA_GUI.core.processing import preprocess_erna, extract_pulses, extract_erna_slice
from ERNA_GUI.core.processing import compute_IPI_features, compute_IBI_features
from ERNA_GUI.core.processing import set_cmap, remove_zerostim, flatten_ipi_features
from ERNA_GUI.viz.plotting import plot_3Dmesh, show_recordings
from ERNA_GUI.io.general import setup_path
import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
# get base dir of package
# = Path(__file__)
#BASE_DIR = current_script.parent.parent.parent
#sys.path.append(BASE_DIR)

hydra.core.global_hydra.GlobalHydra.instance().clear()

import subprocess

def launch_streamlit():
    # RESET HYDRA
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    

    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the streamlit_app.py file
    streamlit_app_path = os.path.join(base_dir, "GUI.py")
    # Build the command for Streamlit
    command = ["streamlit", "run", streamlit_app_path]
    # Capture command-line arguments
    overrides = sys.argv[1:]  # Skip the first argument (script name)
    

    # If overrides are provided, append them to the command
    if overrides:
        command.extend(overrides)
    subprocess.run(command, check=True)  # Adjust the path if necessary
    


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    st.set_page_config(page_title="ERNA GUI", layout="wide")
    st.sidebar.header("Load the ERNA file ('*.txt')")
    # initialzie STN mesh
    figSTN = plot_3Dmesh()
    PARTICIPANTS_ID = cfg.device.PARTICIPANTS_ID  # list of aprticipants
    print(PARTICIPANTS_ID)
    
    #participant = st.sidebar.selectbox("Select patient erna folder:", list(PARTICIPANTS_ID))
    #st.sidebar.write(f"Selected participant: {participant}")
    participant = st.sidebar.text_input("Selected participant:", value=cfg.device.PARTICIPANTS_ID)
    
    
    if participant: # any partticipant selected?
        PARTICIPANT = setup_path(participant, device = cfg.device.NAME)

        uploaded_files = PARTICIPANT['data_path']
        annot_files = PARTICIPANT['annot_path']
        run_annot, stimScan_annot, ampRamp_annot, channels_annot, coords_annot = load_erna_annots(annot_files, participant)

        mat_files = load_erna_file(uploaded_files)
        selected_file = st.sidebar.selectbox("Select a TXT file", mat_files)
        srate = st.sidebar.number_input("Sampling rate [Hz]", value=cfg.device.srate)
        print("Default parameters:")
        print(OmegaConf.to_yaml(cfg))  # Or use to_container for a dict representation

        st.divider()
        if selected_file:
            file_obj = selected_file
            
            
            # Reset session state for new file
            st.session_state.erna = None
            st.session_state.original_erna = None
            st.session_state.timepulses = []
            st.session_state.IPI_features = {}
            st.session_state.IBI_features = {}
            
            mat_contents = load_selected_file(os.path.join(uploaded_files, file_obj))
            run = mat_contents["run"]
            time = mat_contents["time"]
            erna_cont = mat_contents["data_cont"]
            timebursts = mat_contents["timebursts"]
            stim_amp = mat_contents["stim_amp"]
            
            
            timebursts, stim_amp = remove_zerostim(timebursts,stim_amp)


            # Update session state with new data
            st.session_state.original_erna = erna_cont
            st.session_state.erna = erna_cont

                

            task = run_annot.loc[run_annot['Run'] == run, 'Task'].values[0]

            if task == "stimScan":
                run_details = stimScan_annot[stimScan_annot['Run'] == run]
            elif task == "ampRamp":
                run_details = ampRamp_annot[ampRamp_annot['Run'] == run]
            else:
                st.error("This is not a valid task")
                return

            stim_freq = run_details['Freq'].values[0]
            # Get the StimC value from run_details
            stimCh = run_details['StimC'].values[0]

            # Get the name where channel matches StimC
            # Split stimCh if it contains multiple channels
            channels_to_query = stimCh.split('-')

            # Fetch names corresponding to all channels
            channel_stims = channels_annot.loc[channels_annot['channel'].isin(channels_to_query), 'name'].values
            
            # If you need them as a list or a concatenated string
            channelStim = list(channel_stims)  # as a list
            channelStim_string = '-'.join(channel_stims)  # as a string
        
    

            #channelStim = channels_annot.loc[channels_annot['channel'] == stimCh, 'name'].values[0]
            
            # Assuming channelStim is a list of names
            coords_rec = []
            for name in channelStim:
                coords_rec.append(coords_annot.loc[coords_annot['name'] == name, ['mni_x', 'mni_y', 'mni_z']].values[0])# Append the coordinates
            coords_rec = np.array(coords_rec)

            #coords_rec = coords_annot.loc[coords_annot['name'] == channelStim, ['mni_x','mni_y','mni_z']].values[0]
            #channelStim = channels_annot['channel'] == stimCh, 'name'].values

            st.title(f"Recording {task} run: {run} [Stim. {channelStim_string} at {stim_freq} Hz]")
            expander_annot = st.expander("See details run")
            expander_annot.table(run_details)


            st.sidebar.divider()
            
            st.sidebar.header("Raw data panel")
            


            preprocess_on = st.sidebar.checkbox("Preprocessing")
        
            expander_filter = st.sidebar.expander("Details preprocessing")
            colexp1, colexp2 = expander_filter.columns(2)
            flow_cut = colexp1.number_input("Freq. Low Pass [Hz]", value=cfg.processing.preprocessing.flow_cut)
            fhigh_cut = colexp2.number_input("Freq. High Pass [Hz]", value=cfg.processing.preprocessing.fhigh_cut)
            order_low = colexp1.number_input("Order Low Pass", value=cfg.processing.preprocessing.order_low)
            order_high = colexp2.number_input("Order High Pass", value=cfg.processing.preprocessing.order_high)    
            
            flip_on =  st.sidebar.checkbox("Flip polarity")

            st.sidebar.divider()
            st.sidebar.header("ERNA analysis")
            st.sidebar.select_slider('Burst to show:', options =range(0,len(timebursts + 1)), value = 50, key = "timeburst")
        
            identiy_pulses_on = st.sidebar.checkbox("Identify pulses")
            expander_polarity = st.sidebar.expander("Details identify pulses")
            thr_pulses = expander_polarity.number_input("Threshold for pulses [uV]", value=cfg.processing.pulse_identification.threshold)
            which_detrend = expander_polarity.selectbox("Which detrend?",["None","linear","quadratic"])

            identiy_ipi_on = st.sidebar.checkbox("Identify ERNA in pulses (IPI)")
            expander_ipi = st.sidebar.expander("Details show ERNA in pulses (IPI)")
            is_ipi_plot_collapsed = expander_ipi.toggle("Collapse IPI plot in burst")
            is_ipi_features = expander_ipi.toggle("Extract IPI features")
            win_ipi_min = expander_ipi.number_input("Window min [ms]", value=cfg.processing.ipi_identification.window_min)
            win_ipi_max = expander_ipi.number_input("Window max [ms]", value=1/stim_freq*1000 - 1.2)
            num_mad_outlier = expander_ipi.number_input("MAD outlier thr [#]", value=cfg.processing.ipi_identification.mad_outlier_threshold)
            
            
            thr_ipi_params = {'win_ipi_min':win_ipi_min,
                            'win_ipi_max':win_ipi_max,
                            "num_mad_outlier": num_mad_outlier
            }
                
            #thr_pulses = expander_polarity.number_input("Threshold for pulses [Voltage]", value=50)
            identiy_ibi_on = st.sidebar.checkbox("Identify ERNA after burst (IBI)")
            expander_ibi = st.sidebar.expander("Details show ERNA after burst (IBI)")
            is_ibi_features = expander_ibi.toggle("Extract IBI features")
            thr_ibi_peakprom = expander_ibi.number_input("Threshold peak prominence [uV]", value=cfg.processing.ibi_identification.peak_prominence)
            thr_ibi_peakwidth = expander_ibi.number_input("Threshold peak width [ms]", value=cfg.processing.ibi_identification.peak_width)
            thr_ibi_mindist = expander_ibi.number_input("Threshold peaks distance [ms]", value= cfg.processing.ibi_identification.mindist_perc/stim_freq*1000) # default 25% IPI
            thr_ibi_minpeaks= expander_ibi.number_input("Threshold # peaks [#]", value= cfg.processing.ibi_identification.min_peaks) 
            thr_ibi_npeaks= expander_ibi.number_input("Maximum # peaks [#]", value= cfg.processing.ibi_identification.npeaks) 
            
            
            thr_ibi_params = {'peak_prom': thr_ibi_peakprom,
                            'peak_width': np.round(thr_ibi_peakwidth*srate/1000),
                            'mindist': np.round(thr_ibi_mindist*srate/1000),
                            'minpeaks' : thr_ibi_minpeaks,                          
                            'npeaks': thr_ibi_npeaks          
            }
            

        
            if preprocess_on:
                st.session_state.erna = preprocess_erna(st.session_state.original_erna, flow_cut=flow_cut, fhigh_cut=fhigh_cut, order_low=order_low, order_high=order_high, srate=srate)
            else:
                st.session_state.erna = st.session_state.original_erna
                
            if flip_on:
                st.session_state.erna = - st.session_state.erna
            
            
            tab11, tab12, tab13 = st.tabs(["Raw Data", "ERNA analysis", "Results"])
            
            with tab11:
                
                col1, col2, col3 = st.columns([2, 2, 1])
                win_slid = col1.slider('Time slider', min_value=int(time[0]), max_value=int(time[-1]), value=44700)
                win_size = col2.number_input("Window size [ms]")
                col4, col5= st.columns([3,2])
        
                fig, ax = plt.subplots()
                ax2 = ax.twinx()
                ax.plot(time, st.session_state.erna, linewidth=0.4)
                ax2.plot(timebursts, stim_amp, color='m', linewidth=2)
                for x in timebursts:
                    ax.axvline(x=x, color='r', linestyle='--', linewidth=0.4)

                if win_size:
                    ax.set_xlim((win_slid, win_slid + win_size * 1E-3))
                    ax2.set_xlim((win_slid, win_slid + win_size * 1E-3))

                ax.set_xlabel('time [GTC]')
                ax.set_ylabel('Voltage')
                ax2.set_ylabel('Stim. amp [mA]')
                col4.pyplot(fig)
                
                # show STN with recordings
                figSTN = show_recordings(figSTN, coords_rec, radius = 0.45)
                col5.plotly_chart(figSTN)
                

            with tab12:
                
                # these are just for plotting and saving time series
                # window for pulses
                win_left = - int(np.floor(cfg.device.NPULSES_PER_BURST / stim_freq * srate))
                win_right = int(np.floor(0.030 * srate))
                
                # window for ipi
                ipi_left = int(np.floor(0.0013 * srate))            
                ipi_right = int(np.floor((1/stim_freq - 0.0018) * srate))            
                
                # window for ibi
                ibi_left = int(np.floor(0.0013 * srate))            
                ibi_right = int(np.floor(cfg.device.IBI_WINDOW[str(stim_freq)] * srate))                 
                # ------------------------------------------------
            
                
                
                    
                st.text("Here we analyze the ERNA recordings to extract the ERNA wave after DBS stimulation (IBI) and between consecutive DBS pulses (IPI).")

                col21, _, _ = st.columns([4, 2, 1])   
                col21.text("There are 3 main steps:")    
                        
                # show pulses here
                pulse_container = col21.container(border=True)
                pulse_container.markdown("**1. Identify pulses in the recordings**")
                
                if identiy_pulses_on:
                    pulse_expander = pulse_container.expander('Show pulses in a burst')
                    #col11, _, _, _ = pulse_expander.columns([2, 1, 1, 3])
                    # get pulses
                    pulse_locs, erna_pulses, time_erna_pulses = extract_pulses(st.session_state.erna, timebursts, time, win_left, win_right,thr_pulses,stim_freq,srate)

                    
                    
                    if st.session_state.timeburst is not None:
                        
                        #col12.text_input('Stim [mA]:', value=stim_amp[timeburst_oi])
                        timeburst = timebursts[st.session_state.timeburst]
                        erna_pulses_toplot = erna_pulses[timeburst]
                        time_erna_pulses_toplot = (time_erna_pulses[timeburst] - timeburst)*1000

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
                            for x in pulse_locs[timeburst]:
                                ax.axvline(x=(x - timeburst)*1000, color='g', linestyle='--', linewidth=0.4)
                        pulse_expander.info(f"Identified # {len(pulse_locs[timeburst])} pulses in the burst # {st.session_state.timeburst}. The current stimulation is {stim_amp[st.session_state.timeburst]} mA.")
                        pulse_expander.pyplot(fig)
                        
                else:
                    pulse_container.warning('Activate Identify pulse checkbox on the left', icon="⚠️")
                col21.divider()     


                # extract IPI and IBI
                if identiy_ipi_on : 
                    erna_slice = extract_erna_slice(st.session_state.erna, pulse_locs, time, stim_amp, ipi_left, ipi_right, ibi_left, ibi_right, which_detrend)
                                        
                # show IPI here
                ipi_container = col21.container(border=True)
                ipi_container.markdown("**2. Identify IPI in the recordings**")                

                if identiy_ipi_on:  
                    if len(pulse_locs[timeburst]) > 0:    
                        ipi_expander = ipi_container.expander('Show IPIs in a burst')   
                        
                        # extract IPI laterncy and amplitude
                        if is_ipi_features:
                            IPI_features = compute_IPI_features(erna_slice, thr_ipi_params = thr_ipi_params)
                            st.session_state.IPI_features = IPI_features
                            
                            
                                
                                        
                        if st.session_state.timeburst is not None:
                            

                            timeburst = timebursts[st.session_state.timeburst]
                        # Plot IPI data
                            cm = plt.cm.hot
                            
                            norm, sm = set_cmap(1, len(erna_slice[timeburst]['ipi'].values()), cm)
                                        
                            fig_ipi, ax = plt.subplots()
                            for i, (pulse, ipi_data) in enumerate(erna_slice[timeburst]['ipi'].items()):
                                if is_ipi_plot_collapsed:
                                    plt.plot((ipi_data['time'] - pulse)*1000, ipi_data['erna'], color = cm(norm(i)))      
                                    ax.set_xlabel('Time after pulse [ms]')
                                    ax.set_xlim(-0.3,1/stim_freq*1000 + 0.3)  
                                    ax.axvline(0,linestyle='--',color= 'k')   
                                    ax.axvline(1/stim_freq*1000,linestyle='--',color= 'k')    
                                    ax.axvspan(0, cfg.device.WIDEBAND[str(stim_freq)]*1000, color='g', alpha=0.01)
                                
                                    if is_ipi_features:
                                        if pulse in IPI_features[timeburst].keys():
                                            ax.scatter(IPI_features[timeburst][pulse]['latency'],IPI_features[timeburst][pulse]['peak'], s = IPI_features[timeburst][pulse]['peak'], color =cm(norm(i)), alpha = 0.8)
                                            ax.axvline(IPI_features[timeburst][pulse]['latency'],linestyle='--',color =cm(norm(i)), linewidth = 0.8)
                            
                                        
                                                
                                                        
                                else:
                                    plt.plot(ipi_data['time']*1000, ipi_data['erna'], color = cm(norm(i)))
                                    ax.set_xlabel('Time [ms]')

                            
                            ax.set_ylabel('Erna [uV]')
                            cbar = fig_ipi.colorbar(sm, ax= ax)
                            cbar.set_label('# pulse')
                            #ax.title(f'Erna slices for Burst {timeburst}')
                            if is_ipi_plot_collapsed:
                                ipi_expander.info(f" {len(pulse_locs[timeburst])} pulses in the burst # {st.session_state.timeburst}. The current stimulation is {stim_amp[st.session_state.timeburst]} mA. The plot is collapsed around pulse onset.")
                            else:
                                ipi_expander.info(f" {len(pulse_locs[timeburst])} pulses in the burst # {st.session_state.timeburst}. The current stimulation is {stim_amp[st.session_state.timeburst]} mA.")

                            #ax.show()
                            ipi_expander.pyplot(fig_ipi)
                    else:
                        ipi_container.warning('Pulses cannot be correctly identified in this burst', icon="⚠️")
                                                                    
                
                else:
                    ipi_container.warning('Activate Identify ERNA in pulses (IPI) checkbox on the left', icon="⚠️")
                col21.divider()  
                
                
                
                # SHOW IBI HERE
                ibi_container = col21.container(border=True)
                ibi_container.markdown("**3. Identify IBI in the recordings**")

                if identiy_ibi_on:         
                    ibi_expander = ibi_container.expander('Show IBIs in a burst')   
                    
                    # extract IBI features
                    if is_ibi_features:
                        IBI_features = compute_IBI_features(erna_slice, thr_ibi_params=thr_ibi_params)
                        st.session_state.IBI_features = IBI_features

                    if st.session_state.timeburst is not None:
                        timeburst = timebursts[st.session_state.timeburst]
                        
                        # Plot IbI data
                        fig_ibi, ax = plt.subplots()
                        
                        ibi_data = erna_slice[timeburst]['ibi']
                        
                        # Plot ERNA data
                        ax.plot((ibi_data['time'] - timeburst) * 1000, ibi_data['erna'], color='k')
                        ax.set_xlabel('Time after burst [ms]')
                        ax.axvline(0, linestyle='--', color='k')   
                        ax.set_ylabel('Erna [uV]')
                        
                        # Plot peaks and troughs if IBI features are available and ERNA_flag is 1
                        
                        if is_ibi_features and IBI_features[timeburst]['ERNA_flag'] == 1:
                            for _, peak in IBI_features[timeburst]['peaks'].items():
                                ax.scatter(peak['lat'], peak['amp'], s=abs(peak['amp']), color='r', alpha=1)
                            for _, trough in IBI_features[timeburst]['troughs'].items():
                                ax.scatter(trough['lat'], trough['amp'], s=abs(trough['amp']), color='b', alpha=1)  
                        

                            
                            
                            # Plot ERNA data
                            ax.plot((ibi_data['time'] - timeburst) * 1000, ibi_data['erna'], color='k')
                            ax.set_xlabel('Time after burst [ms]')
                            ax.axvline(0, linestyle='--', color='k')   
                            ax.set_ylabel('Erna [uV]')                 
                        
                        # Set plot title and display in Streamlit
                        #ax.set_title(f'ERNA slices for Burst {timeburst}')
                        ibi_expander.info(f"Resonant activity after burst #{st.session_state.timeburst}. The current stimulation is {stim_amp[st.session_state.timeburst]} mA.")
                        ibi_expander.pyplot(fig_ibi)
                        
                        if is_ibi_features:    
                            tabAmp,tabLat,tabFreq = ibi_expander.tabs(
                             ['Amplitude', 'Latency', 'Frequency']
                            )   
                            fig_gc, ax = plt.subplots()
                            
                            with tabAmp:
                                # amplitude
                                for key, value in IBI_features.items():
                                    if value['ERNA_flag']:
                                        ax.scatter(key, value['amplitude'], s=abs(value['amplitude']), color = 'k', alpha = 0.4)
                                        #if value['npeaks'] >= 3:
                                        #    print(value)
                                        #    ax.scatter(key, value['peaks'][1]['amp'] - value['troughs'][1]['amp'], s=abs(value['peaks'][1]['amp'] - value['troughs'][1]['amp']), color = 'b', alpha = 0.4)                                   
                                            
                                ax.set_xlabel('Time burst [ms]')
                                ax.set_ylabel('Erna peak-to-trough [uV]')
                                if IBI_features[timeburst]['ERNA_flag'] == 1:
                                    ax.scatter(timeburst, IBI_features[timeburst]['amplitude'], s=abs(IBI_features[timeburst]['amplitude']), color = 'r', alpha = 1)
                                    #if IBI_features[timeburst]['npeaks'] >= 3:                           
                                    #    ax.scatter(timeburst, IBI_features[timeburst]['peaks'][1]['amp'] - IBI_features[timeburst]['troughs'][1]['amp'], s=abs(IBI_features[timeburst]['peaks'][1]['amp'] - IBI_features[timeburst]['troughs'][1]['amp']), color = 'r', alpha = 1)
                                    ax.axvline(timeburst, linestyle='--', color='k')  
                            with tabLat:                            
                                # latency    
                                for key, value in IBI_features.items():
                                    if value['ERNA_flag']:
                                        ax.scatter(key, value['latency'], s=abs(value['latency']), color = 'k', alpha = 0.4)
                                        #if value['npeaks'] >= 3:
                                        #    print(value)
                                        #    ax.scatter(key, value['peaks'][1]['amp'] - value['troughs'][1]['amp'], s=abs(value['peaks'][1]['amp'] - value['troughs'][1]['amp']), color = 'b', alpha = 0.4)                                   
                                            
                                ax.set_xlabel('Time burst [ms]')
                                ax.set_ylabel('Erna latency [ms]')
                                if IBI_features[timeburst]['ERNA_flag'] == 1:
                                    ax.scatter(timeburst, IBI_features[timeburst]['latency'], s=abs(IBI_features[timeburst]['latency']), color = 'r', alpha = 1)
                                    #if IBI_features[timeburst]['npeaks'] >= 3:                           
                                    #    ax.scatter(timeburst, IBI_features[timeburst]['peaks'][1]['amp'] - IBI_features[timeburst]['troughs'][1]['amp'], s=abs(IBI_features[timeburst]['peaks'][1]['amp'] - IBI_features[timeburst]['troughs'][1]['amp']), color = 'r', alpha = 1)
                                    ax.axvline(timeburst, linestyle='--', color='k')  
                            with tabFreq:                                                               
                                # frequency    
                                for key, value in IBI_features.items():
                                    if value['ERNA_flag']:
                                        ax.scatter(key, value['frequency'], s=abs(value['frequency']), color = 'k', alpha = 0.4)
                                        #if value['npeaks'] >= 3:
                                        #    print(value)
                                        #    ax.scatter(key, value['peaks'][1]['amp'] - value['troughs'][1]['amp'], s=abs(value['peaks'][1]['amp'] - value['troughs'][1]['amp']), color = 'b', alpha = 0.4)                                   
                                            
                                ax.set_xlabel('Time burst [ms]')
                                ax.set_ylabel('Erna frequency [Hz]')
                                if IBI_features[timeburst]['ERNA_flag'] == 1:
                                    ax.scatter(timeburst, IBI_features[timeburst]['frequency'], s=abs(IBI_features[timeburst]['frequency']), color = 'r', alpha = 1)
                                    #if IBI_features[timeburst]['npeaks'] >= 3:                           
                                    #    ax.scatter(timeburst, IBI_features[timeburst]['peaks'][1]['amp'] - IBI_features[timeburst]['troughs'][1]['amp'], s=abs(IBI_features[timeburst]['peaks'][1]['amp'] - IBI_features[timeburst]['troughs'][1]['amp']), color = 'r', alpha = 1)
                                    ax.axvline(timeburst, linestyle='--', color='k')                                                                                             
                            ibi_expander.pyplot(fig_gc)                        
                        
                else:
                    ibi_container.warning('Activate Identify ERNA after burst (IBI) checkbox on the left', icon="⚠️")
                col21.divider()              
            

            with tab13: # this is for results 
                st.text(f"Downloading results:")
                IPI_features = st.session_state.IPI_features
                IBI_features = st.session_state.IBI_features
            
                col11, col12,_ = st.columns([1,1,6])
                col21, col22,_ = st.columns([1,1,6])
                
                # save JSON               
                col11.download_button(
                    label="Download IPI JSON",
                    file_name="IPI_results.json",
                    mime="application/json",
                    data=json.dumps(IPI_features)
                )   
                col12.download_button(
                    label="Download IBI JSON",
                    file_name="IBI_results.json",
                    mime="application/json",
                    data=json.dumps(IBI_features)
                )              
                st.divider()
                # save CSV    
                col21.download_button(
                    label="Download IPI CSV",
                    data=pd.DataFrame.from_dict(flatten_ipi_features(IPI_features)).to_csv(index=False) ,
                    file_name="IPI_results.csv",
                    mime="text/csv"
                )             

                col22.download_button(
                    label="Download IBI CSV",
                    data=pd.DataFrame.from_dict(IBI_features).transpose().to_csv(index=False) ,
                    file_name="IBI_results.csv",
                    mime="text/csv"
                )        


if __name__ == "__main__":
    main()
