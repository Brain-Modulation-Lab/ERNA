import asyncio
import websockets
import json
import streamlit as st
import pandas as pd
import threading
import time
from datetime import datetime

from plotting_OR import create_electrode_layout, plot_voltage

# Global buffer to store received data
data_buffer = []
data_metadata = {"last_update": None, "total_data_received": 0}
runinfo = {}
montage = []
ping_check = False

# variables of interest
RUNINFO_TOSHOW = ['SUBJECT','RUN_ID','DIGOUT','MONTAGE','STIM_HEMI','STIMSEQ']
MONTAGE_TOSHOW = 'MONTAGE_TABLE'

# Streamlit app configuration
st.set_page_config(page_title="ERNA OR", layout="wide")
st.title("Real-Time ERNA OR Recordings")

# Placeholders for plot, metadata, and table
runinfo_placeholder = st.empty()
montage_placeholder = st.empty()
status_placeholder = st.empty()
metadata_placeholder = st.empty()



plot_placeholder = st.empty()


st.sidebar.title("Streaming check:")
# start threader
start_streaming = st.sidebar.button("Start streaming")
ping_status_placeholder = st.sidebar.empty()



# Buffer size limit
BUFFER_SIZE = 3000  # Higher than ripple buffer
INACTIVITY_TIMEOUT = 5  # Clear buffer after 5 seconds of inactivity

# session states
if 'MONTAGE_HEMI' not in st.session_state.keys():
    st.session_state.MONTAGE_HEMI = None

if 'PING_TEST' not in st.session_state.keys():
    st.session_state.PING_TEST = False



# WebSocket server handler to accept incoming data
async def handle_connection(websocket, path=None):
    global data_buffer, data_metadata, runinfo, montage, ping_check
    async for message in websocket:

        if message == "ping":
            await websocket.send("pong")
            print("Server received ping!")
            ping_check = True

        else:
            try:
                data = json.loads(message)
                stream = data['stream']
                # Handle runinfo data
                if data['streaminfo'] == 'runinfo':
                    
                    runinfo = {key: stream[key] for key in RUNINFO_TOSHOW if key in stream}
                    montage = stream.get(MONTAGE_TOSHOW, [])

                # Handle rundata (streaming x, y data)
                elif data['streaminfo'] == 'rundata':
                    new_data = list(zip(stream["x"], stream["y"]))  # Create pairs of (x, y)

                    # Append new data to the buffer
                    data_buffer.extend(new_data)

                    # Update metadata
                    data_metadata["last_update"] = time.time()
                    data_metadata["last_update_format"] = datetime.fromtimestamp(data_metadata["last_update"]).strftime("%Y-%m-%d: %H.%M.%S")
                    data_metadata["total_data_received"] += len(new_data)

                # Limit buffer size to the last BUFFER_SIZE data points
                #if len(data_buffer) > BUFFER_SIZE:
                #    data_buffer = data_buffer[-BUFFER_SIZE:]


            except Exception as e:
                print(f"Error processing message: {e}")

# WebSocket server function
async def start_websocket_server():
    server = await websockets.serve(handle_connection, "localhost", 5500)
    await server.wait_closed()

# Function to start the asyncio event loop for the WebSocket server
def start_websocket_server_in_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_websocket_server())

# Streamlit UI for real-time plotting and table
# Start WebSocket server in a separate thread

if start_streaming:
    threading.Thread(target=start_websocket_server_in_thread, daemon=True).start()
 
    ping_status_placeholder.warning("Waiting for ping test!")
    plot_container = st.container()


    # handle ERNA plot
    tabLH, tabRH = plot_container.tabs(["Left Hemisphere", "Right Hemisphere"])   
    with tabLH:
        figLH = create_electrode_layout(label = [["","L3",""],["L2a","L2b","L2c"],["L1a","L1b","L1c"],["","L0",""]])
        #st.plotly_chart(figLH, key = 'LH')

    with tabRH:
        figRH = create_electrode_layout(label = [["","R3",""],["R2a","R2b","R2c"],["R1a","R1b","R1c"],["","R0",""]])
        #st.plotly_chart(figRH, key = 'RH')

    st.sidebar.title("Plot settings:")
    ERNA_yrange = st.sidebar.slider("Adjust voltage range [uV]", -1000, 1000, value = (-200,200))
    ERNA_xrange = st.sidebar.slider("Adjust time range [ms]", -25, 50, value = (-10,30))
    ERNA_winpeakrange = st.sidebar.slider("Adjust window peak [ms]", 1, 20, value = (2, 5))






    # Streamlit update loop
    while True:

        if ping_check:
            ping_status_placeholder.success("Ping-test success! Connection is done.")




            if runinfo:
                # Display runinfo data as a table
                runinfo_df = pd.DataFrame([runinfo])  # Convert runinfo dictionary to DataFrame
                runinfo_placeholder.dataframe(runinfo_df.set_index(runinfo_df.columns[0]))
                #st.session_state.MONTAGE_HEMI = runinfo['STIM_HEMI']

            if montage:
                # Display montage data as a table
                with montage_placeholder.expander("Montage Data"):
                    if isinstance(montage, list):
                        montage_df = pd.DataFrame(montage)  # Convert montage list to DataFrame
                        st.dataframe(montage_df)  # Display montage inside the expander
                    else:
                        st.write("Montage data is not in expected format.")





            if data_buffer:
                # Clear stale data if inactivity timeout is exceeded
                current_time = time.time()
                if current_time - data_metadata.get("last_update", 0) > INACTIVITY_TIMEOUT:
                    data_buffer.clear()
                    status_placeholder.warning("Buffer cleared due to inactivity!")
                    time.sleep(1)
                else:
                    # Create DataFrame from buffer
                    df = pd.DataFrame(data_buffer, columns=["X", "Y"])

                    # Update the plot
                    #with tabLH:
                    plot_placeholder.line_chart(df, x='X', y='Y')  # Plot the data

                    plot_voltage(figLH, df['X'], df['Y'], xlim, ylim, winpeak)
                    plot_voltage(figRH, df['X'], df['Y'], xlim, ylim, winpeak)

                    # clear buffer [to check]
                    data_buffer.clear()

                    # Display metadata
                    metadata_placeholder.write(f"Data Size: {len(data_buffer)}")
                    metadata_placeholder.write(f"Metadata: {data_metadata}")

                    # Indicate active streaming
                    status_placeholder.success("Receiving data...")

            else:
                # Indicate no data received
                status_placeholder.info("Waiting for data...")   

        # Sleep briefly to prevent excaessive CPU usage
        time.sleep(0.1)
