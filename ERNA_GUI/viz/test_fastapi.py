import streamlit as st
import httpx  # For asynchronous requests
import matplotlib.pyplot as plt
import random
import asyncio
import websockets
import json

# URL for FastAPI endpoint
FASTAPI_URL = "http://localhost:8000/update_data"

# Function to send data to FastAPI asynchronously
async def send_data_to_fastapi(data):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(FASTAPI_URL, json={"data": data})
            return response.json()  # Return the response from FastAPI
    except Exception as e:
        return {"error": str(e)}
    
    # WebSocket URL (FastAPI)
WS_URL = "ws://localhost:8000/ws/ripple"

# Function to fetch real-time data from WebSocket
async def fetch_data_from_ws():
    async with websockets.connect(WS_URL) as websocket:
        while True:
            data = await websocket.recv()
            data = json.loads(data)
            print("connexting data")
            yield data  # Yield data for real-time plotting
            

# Function to generate real-time data (can be any data, for this example random numbers)
def generate_data():
    return {"x": list(range(10)), "y": [random.randint(1, 100) for _ in range(10)]}

# Streamlit app to display data and plot it in real-time
st.title("Real-Time Plot from FastAPI")

# Initialize session_state to store ongoing data
if "run" not in st.session_state:
    st.session_state.run = False

# Toggle to start/stop real-time data updates
if st.button("Start Real-Time Plot"):
    st.session_state.run = True

if st.button("Stop Real-Time Plot"):
    st.session_state.run = False



slider  = st.slider("check_value")

# Placeholder for dynamic plot update

placeholder = st.empty()

placeholder2 = st.empty()

# Asynchronous task to update data and plot it in real-time
async def real_time_plot_nows():
    while st.session_state.run:
        # Generate random data (simulating real-time data stream)
        data = generate_data()

        # Send data to FastAPI asynchronously
        response = await send_data_to_fastapi(data)

        # Show the response from FastAPI
        placeholder.json(response)

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(data["x"], data["y"], label="Random Data", color="blue")
        ax.set_title("Real-Time Data Plot")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()

        # Display plot
        placeholder.pyplot(fig)

        # Add a small delay to simulate real-time plotting
        await asyncio.sleep(2)  # Async sleep for non-blocking behavior
        
        
        
        
# Real-time plot generator function
async def real_time_plot():
    async for data in fetch_data_from_ws():
        
        placeholder2.write(data)
        # Plot the received data
        fig, ax = plt.subplots()
        ax.plot(data["x"], data["y"], label=f"Real-time Data {slider}", color="blue")
        ax.set_title("Real-Time Data Plot")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()

        # Display plot
        placeholder.pyplot(fig)
        


# Run the real-time plot asynchronously
if st.session_state.run:
    asyncio.run(real_time_plot())



    