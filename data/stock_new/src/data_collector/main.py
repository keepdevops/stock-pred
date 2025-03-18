import sys
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
import json
import threading
from queue import Queue

from config.config_manager import ConfigurationManager
from modules.data_loader import DataLoader
from modules.database import DatabaseConnector

class DataCollectorGUI:
    """Standalone GUI for data collection management."""
    
    def __init__(self, root: tk.Tk, config_manager: ConfigurationManager):
        self.root = root
        self.config = config_manager
        self.logger = logging.getLogger("DataCollector")
        
        # Initialize components
        self.db = DatabaseConnector(
            self.config.data_processing.database["path"],
            self.logger
        )
        self.data_loader = DataLoader(self.db, self.config.data_collection, self.logger)
        
        # Setup message queue for thread communication
        self.message_queue = Queue()
        
        self.setup_gui()
        self.setup_periodic_check()
    
    def setup_gui(self):
        """Initialize the GUI components."""
        self.root.title("Stock Market Data Collector")
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Collection Status
        self.setup_status_frame(main_frame)
        
        # Ticker Management
        self.setup_ticker_frame(main_frame)
        
        # Collection Controls
        self.setup_control_frame(main_frame)
        
        # Progress and Logs
        self.setup_progress_frame(main_frame)
    
    def setup_status_frame(self, parent):
        """Setup the status display frame."""
        frame = ttk.LabelFrame(parent, text="Collection Status", padding="5")
        frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Status indicators
        self.historical_status = ttk.Label(frame, text="Historical: Inactive")
        self.historical_status.grid(row=0, column=0, padx=5)
        
        self.realtime_status = ttk.Label(frame, text="Realtime: Inactive")
        self.realtime_status.grid(row=0, column=1, padx=5)
        
        self.last_update = ttk.Label(frame, text="Last Update: Never")
        self.last_update.grid(row=0, column=2, padx=5)
    
    def setup_ticker_frame(self, parent):
        """Setup the ticker management frame."""
        frame = ttk.LabelFrame(parent, text="Ticker Management", padding="5")
        frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Ticker list
        self.ticker_list = tk.Listbox(frame, height=5, selectmode=tk.MULTIPLE)
        self.ticker_list.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        # Load current tickers
        for ticker in self.config.data_collection.tickers:
            self.ticker_list.insert(tk.END, ticker)
        
        # Ticker controls
        ttk.Button(frame, text="Add Ticker", command=self.add_ticker).grid(row=1, column=0)
        ttk.Button(frame, text="Remove Selected", command=self.remove_tickers).grid(row=1, column=1)
    
    def setup_control_frame(self, parent):
        """Setup the collection control frame."""
        frame = ttk.LabelFrame(parent, text="Collection Controls", padding="5")
        frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Historical controls
        ttk.Label(frame, text="Historical:").grid(row=0, column=0)
        ttk.Button(
            frame,
            text="Start Collection",
            command=self.start_historical_collection
        ).grid(row=0, column=1)
        
        # Realtime controls
        ttk.Label(frame, text="Realtime:").grid(row=1, column=0)
        self.realtime_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Enable",
            variable=self.realtime_var,
            command=self.toggle_realtime_collection
        ).grid(row=1, column=1)
    
    def setup_progress_frame(self, parent):
        """Setup the progress and log frame."""
        frame = ttk.LabelFrame(parent, text="Progress and Logs", padding="5")
        frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        
        # Log display
        self.log_text = tk.Text(frame, height=10, width=50)
        self.log_text.grid(row=1, column=0, sticky="ew")
        
        # Scrollbar for logs
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)
    
    def setup_periodic_check(self):
        """Setup periodic check for message queue."""
        self.root.after(100, self.check_message_queue)
    
    def check_message_queue(self):
        """Check for messages from worker threads."""
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.handle_message(message)
        
        self.root.after(100, self.check_message_queue)
    
    def handle_message(self, message):
        """Handle messages from worker threads."""
        msg_type = message.get("type")
        content = message.get("content")
        
        if msg_type == "log":
            self.log_text.insert(tk.END, f"{content}\n")
            self.log_text.see(tk.END)
        elif msg_type == "progress":
            self.progress_var.set(content)
        elif msg_type == "status":
            self.update_status(content)
    
    def update_status(self, status):
        """Update status indicators."""
        if "historical" in status:
            self.historical_status.config(text=f"Historical: {status['historical']}")
        if "realtime" in status:
            self.realtime_status.config(text=f"Realtime: {status['realtime']}")
        if "last_update" in status:
            self.last_update.config(text=f"Last Update: {status['last_update']}")
    
    def add_ticker(self):
        """Add a new ticker to the collection list."""
        from tkinter import simpledialog
        ticker = simpledialog.askstring("Add Ticker", "Enter ticker symbol:")
        if ticker:
            ticker = ticker.upper()
            if ticker not in self.ticker_list.get(0, tk.END):
                self.ticker_list.insert(tk.END, ticker)
                self.save_ticker_changes()
    
    def remove_tickers(self):
        """Remove selected tickers from the collection list."""
        selected = self.ticker_list.curselection()
        for index in reversed(selected):
            self.ticker_list.delete(index)
        self.save_ticker_changes()
    
    def save_ticker_changes(self):
        """Save ticker changes to configuration."""
        tickers = list(self.ticker_list.get(0, tk.END))
        self.config.update_section("data_collection", {"tickers": tickers})
        self.config.save_configuration()
    
    def start_historical_collection(self):
        """Start historical data collection in a separate thread."""
        tickers = list(self.ticker_list.get(0, tk.END))
        if not tickers:
            messagebox.showwarning("Warning", "No tickers selected")
            return
        
        thread = threading.Thread(
            target=self._historical_collection_worker,
            args=(tickers,)
        )
        thread.daemon = True
        thread.start()
    
    def _historical_collection_worker(self, tickers):
        """Worker function for historical data collection."""
        try:
            total = len(tickers)
            for i, ticker in enumerate(tickers, 1):
                self.message_queue.put({
                    "type": "log",
                    "content": f"Collecting historical data for {ticker}..."
                })
                
                self.data_loader.collect_historical_data(ticker)
                
                progress = (i / total) * 100
                self.message_queue.put({
                    "type": "progress",
                    "content": progress
                })
            
            self.message_queue.put({
                "type": "log",
                "content": "Historical data collection completed"
            })
            
        except Exception as e:
            self.message_queue.put({
                "type": "log",
                "content": f"Error in historical collection: {str(e)}"
            })
    
    def toggle_realtime_collection(self):
        """Toggle realtime data collection."""
        if self.realtime_var.get():
            self.start_realtime_collection()
        else:
            self.stop_realtime_collection()
    
    def start_realtime_collection(self):
        """Start realtime data collection."""
        self.realtime_thread = threading.Thread(
            target=self._realtime_collection_worker
        )
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
    
    def stop_realtime_collection(self):
        """Stop realtime data collection."""
        if hasattr(self, 'realtime_thread'):
            self.data_loader.stop_realtime_collection()
            self.realtime_thread.join()
    
    def cleanup(self):
        """Cleanup resources before exit."""
        self.stop_realtime_collection()
        self.db.close()

def main():
    """Application entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/data_collector.log')
        ]
    )
    
    try:
        # Initialize configuration
        config_manager = ConfigurationManager("config/data_collection.json")
        
        # Create and run GUI
        root = tk.Tk()
        app = DataCollectorGUI(root, config_manager)
        root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        messagebox.showerror("Fatal Error", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main() 