import zmq
import json

def get_data_from_server(ticker, port="5555"):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    
    socket.send_string(ticker)
    message = socket.recv().decode('utf-8')
    return json.loads(message)

if __name__ == "__main__":
    ticker = "AAPL"  # Example ticker
    data = get_data_from_server(ticker)
    print(data)
