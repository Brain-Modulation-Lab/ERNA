from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")
            # Simulate broadcasting the received data
    except Exception as e:
        print(f"WebSocket connection closed: {e}")