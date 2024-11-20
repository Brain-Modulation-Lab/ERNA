import asyncio
import websockets
import json
import streamlit as st
import pandas as pd
import threading
import time

# WebSocket URI
uri = "ws://localhost:8765"

# Global buffer to store received data
data_buffer = []

# Streamlit app configuration
st.set_page_config(page_title="Real-Time Data Stream", layout="wide")
st.title("Real-Time Data Plot")

# Placeholder for the plot
plot_placeholder = st.empty()

# Buffer size limit
BUFFER_SIZE = 3000

# Function to receive data from the WebSocket server and update the buffer
async def receive_data():
    global data_buffer  # Declare that we are modifying the global variable
    last_update = time.time()  # Throttling updates
    while True:
        try:
            # Update the plot every 100ms (or adjust according to performance)
            if time.time() - last_update > 0.05:
                if data_buffer:
                    # Create DataFrame from buffer
                    df = pd.DataFrame(data_buffer, columns=["X", "Y"])
                    print(df)
                    plot_placeholder.line_chart(df.set_index("X"))  # Plot the data
                    last_update = time.time()

            await asyncio.sleep(0.05)  # Prevent the event loop from blocking
        except Exception as e:
            st.error(f"Error receiving data: {e}")
            break

# WebSocket server handler to accept incoming data
async def handle_connection(websocket, path):
    global data_buffer  # Declare that we are modifying the global variable
    async for message in websocket:
        try:
            data = json.loads(message)
            new_data = list(zip(data['x'], data['y']))  # Create pairs of (x, y)

            # Append new data to the buffer
            data_buffer.extend(new_data)

            # Limit buffer size to the last 500 data points
            if len(data_buffer) > BUFFER_SIZE:
                data_buffer = data_buffer[-BUFFER_SIZE:]
            print(len(data_buffer))
        except Exception as e:
            print(f"Error processing message: {e}")

# WebSocket server function to start listening on a separate thread
def start_server():
    # Start WebSocket server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = loop.run_until_complete(websockets.serve(handle_connection, "localhost", 8765))
    loop.run_until_complete(server.wait_closed())

# Streamlit UI to start receiving data
if st.button("Start Streaming"):
    # Run WebSocket server in a separate thread
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    
    # Run the data receiving and plotting in Streamlit
    asyncio.run(receive_data())
