import zmq
import yfinance as yf
import json
import time
from queue import Queue
from threading import Thread
from typing import Dict, List

class StockPublisher:
    def __init__(self, port: int, symbols: List[str]):
        self.port = port
        self.symbols = symbols
        self.queue = Queue()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.running = True

    def get_stock_data(self, symbol: str) -> dict:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.info
            return {
                "symbol": symbol,
                "price": data.get("currentPrice", 0),
                "volume": data.get("volume", 0),
                "timestamp": time.time()
            }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def data_collector(self):
        while self.running:
            for symbol in self.symbols:
                data = self.get_stock_data(symbol)
                if data:
                    self.queue.put(data)
            time.sleep(60)  # Update every minute

    def publish(self):
        while self.running:
            if not self.queue.empty():
                data = self.queue.get()
                message = json.dumps(data)
                self.socket.send_string(f"{data['symbol']} {message}")
                print(f"Port {self.port} published: {message}")
            time.sleep(0.1)

    def start(self):
        collector_thread = Thread(target=self.data_collector)
        publisher_thread = Thread(target=self.publish)
        collector_thread.daemon = True
        publisher_thread.daemon = True
        collector_thread.start()
        publisher_thread.start()

    def stop(self):
        self.running = False
        self.socket.close()
        self.context.term()

class PublisherManager:
    def __init__(self):
        self.publishers: Dict[int, StockPublisher] = {}

    def add_publisher(self, port: int, symbols: List[str]):
        if port in self.publishers:
            print(f"Publisher already exists on port {port}")
            return
        publisher = StockPublisher(port, symbols)
        self.publishers[port] = publisher
        publisher.start()
        print(f"Started publisher on port {port} for symbols {symbols}")

    def remove_publisher(self, port: int):
        if port in self.publishers:
            self.publishers[port].stop()
            del self.publishers[port]
            print(f"Removed publisher on port {port}")

    def stop_all(self):
        for publisher in self.publishers.values():
            publisher.stop()
        self.publishers.clear()

def main():
    manager = PublisherManager()
    
    # Example configuration
    publishers_config = {
        5555: ["AAPL", "GOOGL", "MSFT"],  # Tech stocks
        5556: ["JPM", "BAC", "GS"],       # Banking stocks
        5557: ["TSLA", "F", "GM"]         # Auto stocks
    }

    try:
        # Start publishers based on configuration
        for port, symbols in publishers_config.items():
            manager.add_publisher(port, symbols)

        print("Press Ctrl+C to stop all publishers")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down all publishers...")
        manager.stop_all()

if __name__ == "__main__":
    main() 