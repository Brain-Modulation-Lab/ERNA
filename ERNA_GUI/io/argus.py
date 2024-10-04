import numpy as np
import os
import json
import pandas as pd
import warnings
from typing import Dict, Any

import streamlit as st



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
        

