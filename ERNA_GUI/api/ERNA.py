import os
import numpy as np
import pandas as pd
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import logging



from ERNA_GUI.core.erna_preprocessing import (
    ERNAPlatformRequest, process_ERNAfilesbatch
)

from ERNA_GUI.io.general import (
    setup_path,

)

from ERNA_GUI.utils.handle_ftypes import (
    check_python_list
)
hydra.core.global_hydra.GlobalHydra.instance().clear()

# preprocess ERNA
@hydra.main(version_base=None, config_path="../conf", config_name="config")       
def ERNAPreprocessing(cfg:DictConfig):
    PARTICIPANTS_ID = cfg.device.PARTICIPANTS_ID  # list of aprticipants
    logger = logging.getLogger(__name__)
    logger.info("This is a log for the job.")
    logger.info("Job initialization...")
    print(PARTICIPANTS_ID)
    
    PARTICIPANTS_ID = check_python_list(PARTICIPANTS_ID)
    print(PARTICIPANTS_ID)
    
    try:
        logger.info(f"Processing participants: {PARTICIPANTS_ID}")

        # Iterate over all participants
        ERNA_instances = []
        for participant_id in PARTICIPANTS_ID:
            print(participant_id)
            PARTICIPANT = setup_path(participant_id, device=cfg.device.NAME)

            ERNA = ERNAPlatformRequest(
                PARTICIPANT,
                srate=cfg.device.srate,
                NPULSES_PER_BURST=cfg.device.NPULSES_PER_BURST,
                flow_cut=cfg.processing.preprocessing.flow_cut,
                fhigh_cut=cfg.processing.preprocessing.fhigh_cut,
                order_low=cfg.processing.preprocessing.order_low,
                order_high=cfg.processing.preprocessing.order_high,
                flip_polarity=cfg.processing.preprocessing.flip_polarity,
                which_detrend=cfg.processing.pulse_identification.detrend,
                IBI_WINDOW=cfg.device.IBI_WINDOW,
                thr_pulses=cfg.processing.pulse_identification.threshold,
                peak_prom_ipi=cfg.processing.ipi_identification.peak_prominence,
                win_ipi_min=cfg.processing.ipi_identification.window_min,
                num_mad_outlier=cfg.processing.ipi_identification.mad_outlier_threshold,
                window_max_depends_on_srate=cfg.processing.ipi_identification.window_max_depends_on_srate,
                win_ipi_max=cfg.processing.ipi_identification.window_max,
                peak_prom=cfg.processing.ibi_identification.peak_prominence,
                peak_width=cfg.processing.ibi_identification.peak_width,
                win_ibi_min=cfg.processing.ibi_identification.window_min,
                win_ibi_max=cfg.processing.ibi_identification.window_max,
                mindist=cfg.processing.ibi_identification.mindist,
                minpeaks=cfg.processing.ibi_identification.min_peaks,
                npeaks=cfg.processing.ibi_identification.npeaks,
                write_flag=cfg.processing.output.write_flag,
                ask_before_overwrite=cfg.processing.output.ask_before_overwrite
            )

            ERNA_instances.append(ERNA)

        # Preprocess all files for all participants
        process_ERNAfilesbatch(ERNA_instances)

        logger.info("Job completed successfully!")

    except Exception as e:
        logger.error(
            "Select the participant by overriding PARTICIPANTS_ID: "
            "python ERNA.py PARTICIPANTS_ID=<participant_id>"
        )
        logger.error(f"This job failed: {e}")
        
        
def ERNAPostprocessing():
    pass


def main():
    ERNAPreprocessing()
    ERNAPostprocessing()

if __name__ == "__main__":
    main()

    
    
    
    '''   
        # store data
        if cfg.processing.output.store_flag:
            logger.info(f"Annot and data are saved.")
            # store annots and data
            store_ERNAfiles_results2annot(ERNA)
            store_ERNAfiles_results2params(ERNA)
            #store_ERNAfiles_results2data(ERNA,'raw') # this is extremely slow
            store_ERNAfiles_results2data(ERNA,'erna')
        
        
        
        runSchema = fetch_runschema(ERNA)
        
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
                        
'''  