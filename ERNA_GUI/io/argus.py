import numpy as np
import os
import json
import pandas as pd
import warnings
from typing import Dict, Any

import streamlit as st
import re


# loading function for input files
@st.cache_data
def get_argus_lfp(df, srate = 25000):
    data = df.iloc[:, 19:].values
    stim_amp = get_stim_amp(df)
    timebursts = get_stim_events(df)

    nbursts, nsamples = data.shape
    data = np.transpose(data)
    time_vect = np.transpose(np.array([t + np.arange(0, nsamples) / srate for t in timebursts]))

    num_overlap = np.round((time_vect[-1, :-1] - time_vect[0, 1:]) * srate).astype(int) + 1
    data_cont, time_vect_cont = [], []
    for pi in range(nbursts - 1):
        if num_overlap[pi] > 0:
            data_cont.append(data[:(data.shape[0] - num_overlap[pi]), pi])
            time_vect_cont.append(time_vect[:(time_vect.shape[0] - num_overlap[pi]), pi])
        elif num_overlap[pi] < 0:
            data_cont.append(data[:, pi])
            data_cont.append(np.full((abs(num_overlap[pi]),), np.nan))
            time_vect_cont.append(time_vect[:, pi])
            time_vect_cont.append(time_vect[-1, pi] + 1 / srate * np.arange(1, abs(num_overlap[pi]) + 1))
        else:
            data_cont.append(data[:, pi])
            time_vect_cont.append(time_vect[:, pi])

    data_cont = np.concatenate(data_cont)
    time_vect_cont = np.concatenate(time_vect_cont)
    data_cont = np.concatenate((data_cont, data[:, -1])).reshape(-1, 1)
    time_vect_cont = np.concatenate((time_vect_cont, time_vect[:, -1])).reshape(-1, 1).flatten()

    data_cont_df = pd.DataFrame(data_cont).interpolate(method='linear').values.flatten()
    return data_cont_df, time_vect_cont, stim_amp, timebursts

@st.cache_data
def get_stim_amp(df):
    return df.iloc[:, 2].values

@st.cache_data
def get_stim_events(df):
    return timestamp2gtc(df.iloc[:, 0].values)

@st.cache_data
def timestamp2gtc(timestamp):
    hrs = timestamp // 10000
    rem1 = timestamp % 10000
    mns = rem1 // 100
    scs = rem1 % 100
    return 60 * 60 * hrs + 60 * mns + scs    
        

@st.cache_data
def get_run_id(filename):
    pattern = re.compile(r'record(\d+)')
    match = pattern.search(filename)
    return int(match.group(1))


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
   
    

    run_id = get_run_id(os.path.basename(selected_file))
    data_cont_df, time_vect_cont, stim_amp, timebursts = get_argus_lfp(df, srate=25000) # this is onlt for argus
    

    return {
        "run": run_id,
        "time": time_vect_cont,
        "data_cont": data_cont_df,
        "stim_amp": stim_amp,
        "timebursts": timebursts     
    }
    
    