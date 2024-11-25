import threading
import asyncio
import websockets
from flask import Flask
from flask_socketio import SocketIO
import json

app = Flask(__name__)
socketio = SocketIO(app)

# WebSocket handling code
async def handle_connection(websocket, path = None):
    async for message in websocket:
        print(f"Received message: {message}")
        try:
            data = json.loads(message)
            # Process the data here (like adding to buffer, etc.)
        except Exception as e:
            print(f"Error processing message: {e}")

async def start_websocket_server():
    # Start a new asyncio event loop to run the websocket server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = websockets.serve(handle_connection, 'localhost', 5500)
    loop.run_until_complete(server)

# Start WebSocket server in a separate thread
def start_websocket_thread():
    thread = threading.Thread(target=start_websocket_server)
    thread.daemon = True  # Allow the thread to exit when the main app exits
    thread.start()

# Flask route for the home page
@app.route('/')
def home():
    return "WebSocket server is running..."

# Initialize the WebSocket server when the app starts
if __name__ == '__main__':
    start_websocket_thread()  # Start WebSocket server in a separate thread
    socketio.run(app, host='localhost', port=5500, debug=True)
