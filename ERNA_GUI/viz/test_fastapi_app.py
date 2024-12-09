from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
import asyncio
import json
import random

# Define the structure of the incoming data (dictionary format)
class DataModel(BaseModel):
    data: dict

app = FastAPI()

# POST endpoint to receive data
@app.post("/update_data")
async def update_data(payload: DataModel):
    # Here you would process the data or update any model/database
    data = payload.data  # This is the dictionary sent from Streamlit
    print(f"Received data: {data}")

    # You can process the data or simply return it
    return {"message": "Data received", "data": data}


@app.websocket("/ws/generate_data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Simulate sending data
        x = list(range(10))
        y = [random.randint(1, 100) for _ in range(10)]
        data = {"x": x, "y": y}
        await websocket.send_json(data)
        await asyncio.sleep(2)  # Simulate delay between data sends
        

@app.websocket("/ws/ripple")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()  # Receive text data
            print(f"Received data: {data}")
            await websocket.send_json(f"Echo: {data}")  # Echo the data back
            print("data sent")
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: {e.code}, {e.reason}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure the WebSocket is closed
        if websocket.client_state == "CONNECTED":
            await websocket.close()
            print("WebSocket closed successfully.")

                
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
