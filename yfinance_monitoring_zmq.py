import zmq
import time
from datetime import datetime

class IndustryDataMonitor:
    def __init__(self, port=5555):
        # Initialize ZMQ context and subscriber socket
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://localhost:{port}")
        
        # Subscribe to all messages
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        
        print(f"Monitoring industry data updates on port {port}...")

    def monitor(self):
        """Monitor and display data updates"""
        try:
            while True:
                # Receive message with a timeout of 1 second
                try:
                    message = self.subscriber.recv_string(flags=zmq.NOBLOCK)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Parse and display the message
                    if message.startswith("COMPANY_INFO_UPDATE"):
                        _, ticker = message.split()
                        print(f"[{timestamp}] Received company info update for {ticker}")
                    
                    elif message.startswith("PRICE_UPDATE"):
                        _, ticker = message.split()
                        print(f"[{timestamp}] Received price data update for {ticker}")
                    
                    else:
                        print(f"[{timestamp}] Received unknown message: {message}")
                
                except zmq.Again:
                    # No message received, continue waiting
                    time.sleep(0.1)
                    continue
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up ZMQ resources"""
        self.subscriber.close()
        self.context.term()

def main():
    try:
        monitor = IndustryDataMonitor(port=5556)
        monitor.monitor()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 