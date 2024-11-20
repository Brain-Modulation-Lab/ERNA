% MATLAB WebSocket client configuration
uri = 'ws://localhost:8765';

if count(py.sys.path, '/path/to/your/python/module') == 0 
    insert(py.sys.path, int32(0), '/path/to/your/python/module'); 
end

m = py.importlib.import_module('websocket_client'); 
py.importlib.reload(m);

% Send data in real-time
for i = 1:2
    data.x = (100*(i - 1) + 1):0.1:100*2;  % X values (e.g., 10 points per batch)
    data.y = sin(2*pi*i*data.x);  % Y values (random example data)
    jsonData = jsonencode(data);  % Convert to JSON
    m.send(uri, jsonData);  % Send data via WebSocket
    pause(0.1);  % Pause 50ms
end
