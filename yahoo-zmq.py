import zmq
import json

def start_server(port="5555"):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    
    while True:
        # Wait for next request from client
        message = socket.recv().decode('utf-8')
        
        # Process request (assume message is the stock ticker)
        try:
            stock_data = get_stock_data(message)
            socket.send_string(json.dumps(stock_data))
        except Exception as e:
            # If there's an error, send an error message
            socket.send_string(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    start_server()
