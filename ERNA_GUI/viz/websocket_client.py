import websockets
import asyncio
import json
# Asynchronous function to send data
async def send_data(uri, data):
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(data)
            print(f"Sent data: {data}")


    except Exception as e:
        print(f"Error sending data: {e}")

# Synchronous wrapper function to send data
def send(uri, data):
    asyncio.run(send_data(uri, data))  # Runs the async function directly
    
    
if __name__ == "__main__":
    send("ws://localhost:8000/ws/ripple",json.dumps({"x": [1, 2 ,3, 4, 5], "y": [3, 4, 5, 6, 7]}))
    