import os
import numpy as np
import pandas as pd
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from ERNA_GUI.core.processing import (
    load_erna_file, load_selected_file, load_erna_annots,
    preprocess_erna, extract_pulses, extract_erna_slice,
    compute_IPI_features, compute_IBI_features,
    remove_zerostim, flatten_ipi_features
)

from ERNA_GUI.io.general import (
    setup_path,
    store_ERNAfiles_results2annot,
    store_ERNAfiles_results2data
)

from ERNA_GUI.api.utility import (
    fetch_runschema, 
    fetch_run, 
    fetch_annot,
    fetch_IBImatrix
)

from ERNA_GUI.viz.plotting import (
    plot_stimChannelLocation,
    plot_IBImatrix,
    animate_IBIavg,
    plot_IPIavg,
    animate_IPIavg,
    plot_IBIfeatures_dynamics,
    plot_IPIfeatures_dynamics,
    plot_continuous_data
)

import sys

class ERNAFileHandler:
    def __init__(self, participant_id, participant_raw, participant_save,participant_annot, file_obj, params):
        self.participant_id = participant_id      
        self.participant_raw = participant_raw
        self.participant_save = participant_save
        self.participant_annot = participant_annot
       
        self.file_obj = file_obj
        self.params = params
        #self.run_annot = run_annot
        #self.stimScan_annot = stimScan_annot
        #self.ampRamp_annot = ampRamp_annot

        self.data = {}
        self.is_preprocessed = False

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

            self.params['mindist'] = np.round(self.params['mindist_prop'] / self.params['stim_freq'] * self.params['srate'])
            self.params['win_ipi_max'] = 1/self.params['stim_freq']*1000 - 1
            IPI_features = compute_IPI_features(erna_slice, self.params)
            
            IBI_features = compute_IBI_features(erna_slice, self.params)
            print("ERNA features computed")
            return IPI_features, IBI_features
        except Exception as e:
            raise ValueError(f"Error computing ERNA features: {e}")

    def set_windows(self):
        try:
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

    def get_task(self, run_annot):
        run = self.run
        task = run_annot.loc[run_annot['Run'] == run, 'Task'].values[0]
        return task

    def get_run_details(self,stimScan_annot,ampRamp_annot, run_annot):
        try:
            run = self.run
            task = self.get_task(run_annot)

            if task == "stimScan":
                run_details = stimScan_annot[stimScan_annot['Run'] == run]
            elif task == "ampRamp":
                run_details = ampRamp_annot[ampRamp_annot['Run'] == run]
            else:
                run_details = None
            print(f"Run details fetched for run {run}")
            return run_details
        except Exception as e:
            raise ValueError(f"Error getting run details: {e}")
        
    def get_stimCh(self, run_details, channels_annot, coords_annot):
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



class ERNAPlatformRequest:
    def __init__(self, participant, srate=25000, NPULSES_PER_BURST=10,
                 flow_cut=1000, fhigh_cut=2, order_low=4, order_high=4,
                 flip_polarity=True, which_detrend='linear',
                 IBI_WINDOW=None, thr_pulses=50,
                 win_ipi_min=2.5, num_mad_outlier = 4,
                 peak_prom=5, peak_width=0.5, mindist_prop=0.25, minpeaks=2, npeaks=8):

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
            'win_ipi_min':win_ipi_min,
            'num_mad_outlier':num_mad_outlier,
            'peak_prom': peak_prom,
            'peak_width': np.round(peak_width * srate / 1000),
            'mindist_prop': mindist_prop,
            'minpeaks': minpeaks,
            'npeaks': npeaks
        }
        
        print("List of parameters:")
        print(json.dumps(self.params, indent = 4))

        self.get_annots()
        #self.set_windows()

    def get_annots(self):
        try:
            self.run_annot, self.stimScan_annot, self.ampRamp_annot, self.channels_annot, self.coords_annot, = load_erna_annots(self.participant_annot, self.participant_id)
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

    def get_erna_data(self, mat_file):
        try:
            file_handler = ERNAFileHandler(self.participant_id,
                self.participant_raw, self.participant_save, self.participant_annot, mat_file, self.params
            )
            file_handler.extract_erna_data()
            run_details = file_handler.get_run_details(self.stimScan_annot, self.ampRamp_annot, self.run_annot)

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


@hydra.main(version_base=None, config_path="../conf", config_name="config")       
def main(cfg:DictConfig):
    PARTICIPANT_ID = cfg.device.PARTICIPANTS_ID  # list of aprticipants
    logger = logging.getLogger(__name__)
    logger.info("This is a log for the job.")
    logger.info("Job initialization...")
    
    try:
        PARTICIPANT = setup_path(PARTICIPANT_ID, device= cfg.device.NAME)
        
        logger.info("Job started.")
        ERNA = ERNAPlatformRequest(PARTICIPANT, srate = cfg.device.srate, NPULSES_PER_BURST=cfg.device.NPULSES_PER_BURST,
                 flow_cut= cfg.processing.preprocessing.flow_cut, fhigh_cut=cfg.processing.preprocessing.fhigh_cut, 
                 order_low=cfg.processing.preprocessing.order_low, order_high=cfg.processing.preprocessing.order_high,
                 flip_polarity=cfg.processing.preprocessing.flip_polarity, which_detrend=cfg.processing.pulse_identification.detrend,
                 IBI_WINDOW=cfg.device.IBI_WINDOW, thr_pulses=cfg.processing.pulse_identification.threshold,
                 win_ipi_min=cfg.processing.ipi_identification.window_min, num_mad_outlier = cfg.processing.ipi_identification.mad_outlier_threshold,
                 peak_prom=cfg.processing.ibi_identification.peak_prominence, peak_width=cfg.processing.ibi_identification.peak_width,
                 mindist_prop=cfg.processing.ibi_identification.mindist_perc, minpeaks=cfg.processing.ibi_identification.min_peaks,
                 npeaks=cfg.processing.ibi_identification.npeaks)
        
        mat_files = ERNA.get_erna_files()
        # run file-by-file analysis    
        for i,mat_file in enumerate(mat_files):
            print(f"Start file {mat_file}: {i + 1} - {len(mat_files)}")
            process_ERNAfile(ERNA, mat_file)
            print("OK!")   
            print(f"Completed file {mat_file}: {i + 1} - {len(mat_files)}")

        # list of possible runs
        runSchema = fetch_runschema(ERNA)
        print("List of possible runs:")
        print(runSchema)
        
        if cfg.processing.output.store_flag:
            logger.info(f"Annot and data are saved.")
            # store annots and data
            store_ERNAfiles_results2annot(ERNA)
            #store_ERNAfiles_results2data(ERNA,'raw') # this is extremely slow
            store_ERNAfiles_results2data(ERNA,'erna')
            
        if cfg.processing.output.savefig_flag:
            
            for task in ["stimScan","ampRamp"]:
                logger.info(f"Plot are generated for task: {task}")
                print(f"Analyzing task: {task}")

                print("Fetching data...")
                task_data = fetch_run(ERNA,task=task)

                for i, run_id in enumerate(runSchema[task]): 
                    # print(run)
                    #run = runSchema[task][run_id]
                    print(f"Extracting data and annots for run: {run_id} [task: {task}], [{i + 1}-{len(runSchema[task])}, {(i+1)/len(runSchema[task])*100:.2f}%]")
                    run_data = task_data[run_id]

                    IBI_annot = fetch_annot(ERNA, "IBI_annot", id= run_id)
                    IPI_annot = fetch_annot(ERNA, "IPI_annot", id = run_id)

                    print("Run details:")
                    run_details = run_data["run_details"]
                    run_details
                    Burst_mode = run_details['Burst_mode'].values[0]
                    
                    print("Computing raw and location of recordings")
                    plot_continuous_data(run_data, output_path=os.path.join(ERNA.participant_figure, 'subj-' + ERNA.participant_id + '_task-' + task + '_run-0' + str(run_id) + '_plot-continuous-argus'), show = False)
                    plot_stimChannelLocation(run_data,output_path=os.path.join(ERNA.participant_figure, 'subj-' + ERNA.participant_id + '_task-' + task + '_run-0' + str(run_id) + '_plot-3DDISTAL-StimChanLocation'), show = False)
                    
                    print("Computing ERNA signal during the inter-burst interval (IBI)")
                    plot_IBImatrix(run_data,output_path=os.path.join(ERNA.participant_figure, 'subj-' + ERNA.participant_id + '_task-' + task + '_run-0' + str(run_id) + '_plot-erna-IBI'), show= False)
                    animate_IBIavg(run_data,output_path=os.path.join(ERNA.participant_figure, 'subj-' + ERNA.participant_id + '_task-' + task + '_run-0' + str(run_id) + '_animation-erna-IBI'), range_y=[-200,250], show = False)
                    
                    print("Computing ERNA signal during the inter-pulse interval (IPI)")
                    if Burst_mode:
                        plot_IPIavg(run_data, output_path=os.path.join(ERNA.participant_figure, 'subj-' + ERNA.participant_id + '_task-' + task + '_run-0' + str(run_id) + '_plot-erna-IPI'), show= False)
                        animate_IPIavg(run_data,output_path=os.path.join(ERNA.participant_figure, 'subj-' + ERNA.participant_id + '_task-' + task + '_run-0' + str(run_id) + '_animation-erna-IPI'), range_y = [-100, 250], show = False)
                    else:
                        Warning("Burst mode is not active! Skip this step")
                    
                    print("Computing IBI feature dynamics over time")
                    plot_IBIfeatures_dynamics(IBI_annot, ref="time", output_path=os.path.join(ERNA.participant_figure, 'subj-' + ERNA.participant_id + '_task-' + task + '_run-0' + str(run_id) + '_plot-erna-IBI-features'), show= False)
                    #plot_IBIfeatures_dynamics(IBI_annot, ref="charge", stim_freq = run_data['run_details']['Freq'][0])
                        
                    print("Computing IPI feature dynamics over time")
                    if Burst_mode:
                        plot_IPIfeatures_dynamics(IPI_annot, timescale = "pulse",output_path=os.path.join(ERNA.participant_figure, 'subj-' + ERNA.participant_id + '_task-' + task + '_run-0' + str(run_id) + '_plot-erna-IPI-features_timescale-pulse'), show= False)
                        plot_IPIfeatures_dynamics(IPI_annot, timescale = "burst", ref = "time",output_path=os.path.join(ERNA.participant_figure, 'subj-' + ERNA.participant_id + '_task-' + task + '_run-0' + str(run_id) + '_plot-erna-IPI-features_timescale-burst'), show= False)
                    #plot_IPIfeatures_dynamics(IPI_annot, timescale = "burst", ref = "charge", stim_freq = run_data['run_details']['Freq'][0])
                    else:
                        Warning("Burst mode is not active! Skip this step")  
        logger.info(f"Job completed!")                        
    except Exception as e:
        Warning("Select the participant by overridding participant: python ERNA.py PARTICIPANTS_ID=<participant_id>")
        logger.info(f"This job failed:{e}")

        
if __name__ == "__main__":
    main()
    
