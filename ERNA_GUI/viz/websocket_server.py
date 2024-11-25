import asyncio
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from datetime import datetime
from typing import List, Dict

# Global buffer to store received data
data_buffer: List[Dict] = []
data_metadata = {"last_update": None, "total_data_received": 0}
runinfo = {}
montage = []
ping_check = False

app = FastAPI()

# WebSocket handler to accept incoming data
@app.websocket("/ws")
async def handle_connection(websocket: WebSocket):
    global data_buffer, data_metadata, runinfo, montage, ping_check
    await websocket.accept()

    try:
        while True:
            message = await websocket.receive_text()

           
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    # Run the FastAPI server with Uvicorn
    uvicorn.run(app, host="localhost", port=8000)
