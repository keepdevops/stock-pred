import zmq
import yfinance as yf
import json
import time
from queue import Queue
from threading import Thread
import tkinter as tk
from tkinter import ttk
from datetime import datetime

def setup_zmq_publisher():
    # Initialize ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")  # Publishing on port 5555
    return socket

def setup_zmq_responder():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5556")  # Request/Reply on port 5556
    return socket

def get_stock_data(ticker_symbol):
    # Get real-time stock data using yfinance
    stock = yf.Ticker(ticker_symbol)
    data = stock.history(period="1d", interval="1m")
    if not data.empty:
        latest_data = data.iloc[-1]
        return {
            "symbol": ticker_symbol,
            "price": float(latest_data["Close"]),
            "volume": int(latest_data["Volume"]),
            "timestamp": str(latest_data.name)
        }
    return None

def data_collector(queue, ticker_symbols):
    while True:
        for symbol in ticker_symbols:
            data = get_stock_data(symbol)
            if data:
                queue.put(data)
        time.sleep(60)  # Wait for 1 minute before next update

class TickerRequestWindow:
    def __init__(self, add_ticker_callback):
        self.root = tk.Tk()
        self.root.title("Stock Ticker Request")
        self.root.geometry("300x150")
        
        self.add_ticker_callback = add_ticker_callback
        
        # Create and pack widgets
        label = ttk.Label(self.root, text="Enter Ticker Symbol:")
        label.pack(pady=10)
        
        self.ticker_entry = ttk.Entry(self.root)
        self.ticker_entry.pack(pady=5)
        
        submit_btn = ttk.Button(self.root, text="Add Ticker", command=self.submit_ticker)
        submit_btn.pack(pady=10)
        
    def submit_ticker(self):
        ticker = self.ticker_entry.upper()
        if ticker:
            self.add_ticker_callback(ticker)
            self.ticker_entry.delete(0, tk.END)
    
    def start(self):
        self.root.mainloop()

class YFinanceZMQPublisher:
    def __init__(self, symbol, port=5555):
        self.symbol = symbol
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.ticker = yf.Ticker(symbol)
        
    def publish_data(self):
        while True:
            try:
                # Get real-time data
                data = self.ticker.history(period='1d', interval='1m')
                if not data.empty:
                    latest = data.iloc[-1]
                    message = {
                        'symbol': self.symbol,
                        'timestamp': datetime.now().isoformat(),
                        'price': latest['Close'],
                        'volume': latest['Volume'],
                        'high': latest['High'],
                        'low': latest['Low'],
                        'open': latest['Open']
                    }
                    
                    # Publish the data
                    self.socket.send_string(f"{self.symbol} {json.dumps(message)}")
                    print(f"Published: {message}")
                
                time.sleep(60)  # Wait for 1 minute before next update
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

# Example subscriber client
class YFinanceZMQSubscriber:
    def __init__(self, symbol, port=5555):
        self.symbol = symbol
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, symbol)
        
    def receive_data(self):
        while True:
            try:
                message = self.socket.recv_string()
                topic, data = message.split(' ', 1)
                data = json.loads(data)
                print(f"Received: {data}")
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

def main():
    pub_socket = setup_zmq_publisher()
    rep_socket = setup_zmq_responder()
    ticker_symbols = ["AAPL", "GOOGL", "MSFT"]  # Example tickers
    data_queue = Queue()
    
    def add_ticker(symbol):
        if symbol not in ticker_symbols:
            ticker_symbols.append(symbol)
            print(f"Added new ticker: {symbol}")
    
    # Start data collector thread
    collector_thread = Thread(target=data_collector, args=(data_queue, ticker_symbols))
    collector_thread.daemon = True
    collector_thread.start()
    
    # Start Tkinter window in a separate thread
    window = TickerRequestWindow(add_ticker)
    window_thread = Thread(target=window.start)
    window_thread.daemon = True
    window_thread.start()
    
    print("Starting ZMQ publisher for stock data...")
    try:
        while True:
            # Handle publish events
            if not data_queue.empty():
                data = data_queue.get()
                message = json.dumps(data)
                pub_socket.send_string(f"{data['symbol']} {message}")
                print(f"Published: {message}")
            
            # Handle ticker requests (non-blocking)
            try:
                request = rep_socket.recv_string(flags=zmq.NOBLOCK)
                rep_socket.send_string(json.dumps({"tickers": ticker_symbols}))
                print(f"Received request for ticker list")
            except zmq.Again:
                pass
                
            time.sleep(0.1)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\nShutting down publisher...")
        pub_socket.close()
        rep_socket.close()

if __name__ == "__main__":
    # Run publisher
    symbol = "AAPL"  # Example stock symbol
    
    # To run as publisher
    publisher = YFinanceZMQPublisher(symbol)
    publisher.publish_data()
    
    # To run as subscriber (in a separate process)
    # subscriber = YFinanceZMQSubscriber(symbol)
    # subscriber.receive_data()
