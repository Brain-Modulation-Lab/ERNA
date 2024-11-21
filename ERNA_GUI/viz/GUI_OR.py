import asyncio
import websockets
import json
import streamlit as st
import pandas as pd
import threading
import time
from datetime import datetime

# Global buffer to store received data
data_buffer = []
data_metadata = {"last_update": None, "total_data_received": 0}
runinfo = {}
montage = []

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

start_button = st.button("Start Streaming")
plot_placeholder = st.empty()

# Buffer size limit
BUFFER_SIZE = 3000  # Higher than ripple buffer
INACTIVITY_TIMEOUT = 5  # Clear buffer after 5 seconds of inactivity

# session states
if 'MONTAGE_HEMI' not in st.session_state.keys():
    st.session_state.MONTAGE_HEMI = None

'''
if st.session_state.MONTAGE_HEMI == 'bilateral':
    [tabLH, tabRH] = plot_placeholder.tabs(["Left hemisphere","Right Hemisphere"])
elif st.session_state.MONTAGE_HEMI == 'LH':
    tabLH = plot_placeholder.tabs(["Left hemisphere"])
elif st.session_state.MONTAGE_HEMI == 'RH':
    tabRH = plot_placeholder.tabs(["Left hemisphere"])
'''


# WebSocket server handler to accept incoming data
async def handle_connection(websocket, path=None):
    global data_buffer, data_metadata, runinfo, montage
    async for message in websocket:

        if message == "ping":
            await websocket.send("pong")
            print("Server received ping!")

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
if start_button:
    # Start WebSocket server in a separate thread
    threading.Thread(target=start_websocket_server_in_thread, daemon=True).start()

    # Streamlit update loop
    while True:

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

        # Sleep briefly to prevent excessive CPU usage
        time.sleep(0.1)
