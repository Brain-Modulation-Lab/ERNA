# streamlit_app.py

import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime

# Global variables
data_buffer = []
data_metadata = {"last_update": None, "total_data_received": 0}
runinfo = {}
montage = []
ping_check = False

RUNINFO_TOSHOW = ['SUBJECT', 'RUN_ID', 'DIGOUT', 'MONTAGE', 'STIM_HEMI', 'STIMSEQ']
MONTAGE_TOSHOW = 'MONTAGE_TABLE'

# Streamlit app configuration
st.set_page_config(page_title="Real-Time ERNA OR", layout="wide")
st.title("Real-Time ERNA OR Recordings")

# Placeholders for plot, metadata, and table
runinfo_placeholder = st.empty()
montage_placeholder = st.empty()
status_placeholder = st.empty()
metadata_placeholder = st.empty()
plot_placeholder = st.empty()

st.sidebar.title("Streaming check:")
start_streaming = st.sidebar.button("Start streaming")
ping_status_placeholder = st.sidebar.empty()

# Function to update data
def update_data():
    global data_buffer, data_metadata, runinfo, montage, ping_check
    # You can fetch new data from FastAPI server via WebSocket or HTTP request
    if ping_check:
        ping_status_placeholder.success("Ping-test success! Connection is done.")
        
        if runinfo:
            # Display runinfo data as a table
            runinfo_df = pd.DataFrame([runinfo])  # Convert runinfo dictionary to DataFrame
            runinfo_placeholder.dataframe(runinfo_df.set_index(runinfo_df.columns[0]))

        if montage:
            # Display montage data as a table
            with montage_placeholder.expander("Montage Data"):
                if isinstance(montage, list):
                    montage_df = pd.DataFrame(montage)  # Convert montage list to DataFrame
                    st.dataframe(montage_df)  # Display montage inside the expander
                else:
                    st.write("Montage data is not in expected format.")

        if data_buffer:
            # Create DataFrame from buffer
            df = pd.DataFrame(data_buffer, columns=["X", "Y"])

            # Update the plot
            plot_placeholder.line_chart(df, x='X', y='Y')  # Plot the data
            data_buffer.clear()

            # Display metadata
            metadata_placeholder.write(f"Data Size: {len(data_buffer)}")
            metadata_placeholder.write(f"Metadata: {data_metadata}")

            # Indicate active streaming
            status_placeholder.success("Receiving data...")

        else:
            # Indicate no data received
            status_placeholder.info("Waiting for data...")

# Trigger the data fetching when "Start streaming" button is pressed
if start_streaming:
    # You could replace this with fetching data from FastAPI server, e.g., via HTTP requests
    update_data()

# Periodic update using Streamlit's rerun mechanism
st.rerun()
