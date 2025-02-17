import zmq

import json
from typing import List

class StockSubscriber:
    def __init__(self, ports: List[int], symbols: List[str] = None):
        self.context = zmq.Context()
        self.sockets = []
        self.poller = zmq.Poller()
        
        # Create a socket for each port and register with poller
        for port in ports:
            socket = self.context.socket(zmq.SUB)
            socket.connect(f"tcp://localhost:{port}")
            
            # Subscribe to specific symbols or all if none specified
            if symbols:
                for symbol in symbols:
                    socket.setsockopt_string(zmq.SUBSCRIBE, symbol)
            else:
                socket.setsockopt_string(zmq.SUBSCRIBE, "")
                
            self.sockets.append(socket)
            self.poller.register(socket, zmq.POLLIN)

    def receive(self, timeout=1000):
        events = dict(self.poller.poll(timeout))
        messages = []
        
        for socket in self.sockets:
            if socket in events:
                message = socket.recv_string()
                symbol, data = message.split(" ", 1)
                messages.append(json.loads(data))
        
        return messages

    def close(self):
        for socket in self.sockets:
            socket.close()
        self.context.term()

def main():
    # Example usage
    ports = [5555, 5556, 5557]  # Connect to all publishers
    symbols = ["AAPL", "JPM", "TSLA"]  # Optional: specify symbols to subscribe to
    
    subscriber = StockSubscriber(ports, symbols)
    
    try:
        print("Listening for stock updates... Press Ctrl+C to stop")
        while True:
            messages = subscriber.receive()
            for message in messages:
                print(f"Received: {message}")
                
    except KeyboardInterrupt:
        print("\nShutting down subscriber...")
        subscriber.close()

if __name__ == "__main__":
    main() 