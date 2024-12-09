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



