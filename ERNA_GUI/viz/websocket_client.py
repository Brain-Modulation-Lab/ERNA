import websockets
import asyncio

async def send_data(uri, data):
    async with websockets.connect(uri) as websocket:
        await websocket.send(data)