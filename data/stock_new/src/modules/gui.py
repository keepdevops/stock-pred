import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import threading
import time

from src.config.config_manager import ConfigurationManager
from src.modules.database import DatabaseConnector
from src.modules.data_loader import DataLoader
from src.modules.stock_ai_agent import StockAIAgent
from src.modules.trading.real_trading_agent import RealTradingAgent
from src.utils.visualization import StockVisualizer
from ..database.nasdaq_database import NasdaqDatabase

class StockGUI:
    """Main GUI interface for the Stock Market Analyzer."""
    
    def __init__(self, root, config_manager, db, data_loader):
        self.root = root
        self.config = config_manager
        self.db = db
        self.data_loader = data_loader
        self.logger = logging.getLogger("GUI")
        
        # Initialize NASDAQ database
        self.nasdaq_db = NasdaqDatabase()
        
        # Store available tickers
        self.available_tickers = []
        self.realtime_running = False
        self.realtime_thread = None
        self.collection_interval = 60  # seconds
        
        self._setup_gui()
        self.load_nasdaq_tickers()
    
    def _setup_gui(self):
        """Setup GUI components."""
        # Main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for ticker selection
        self.left_frame = ttk.Frame(self.main_container)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # Ticker Selection Section
        self.ticker_frame = ttk.LabelFrame(self.left_frame, text="Ticker Selection")
        self.ticker_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Search/Filter
        self.search_frame = ttk.Frame(self.ticker_frame)
        self.search_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.search_frame, text="Filter:").pack(side=tk.LEFT)
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_tickers)
        self.search_entry = ttk.Entry(
            self.search_frame, 
            textvariable=self.search_var
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Ticker Listbox with Scrollbar
        self.listbox_frame = ttk.Frame(self.ticker_frame)
        self.listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.ticker_listbox = tk.Listbox(
            self.listbox_frame,
            selectmode=tk.EXTENDED,
            exportselection=False,
            width=30,
            height=20
        )
        self.ticker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ticker_listbox.bind('<<ListboxSelect>>', self.on_ticker_select)

        scrollbar = ttk.Scrollbar(
            self.listbox_frame,
            orient=tk.VERTICAL,
            command=self.ticker_listbox.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ticker_listbox.config(yscrollcommand=scrollbar.set)

        # Selection info
        self.selection_label = ttk.Label(self.ticker_frame, text="Selected: 0")
        self.selection_label.pack(pady=5)

        # Ticker Details
        self.details_frame = ttk.LabelFrame(self.ticker_frame, text="Ticker Details")
        self.details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.details_text = tk.Text(self.details_frame, height=4, wrap=tk.WORD)
        self.details_text.pack(fill=tk.X, padx=5, pady=5)
        self.details_text.config(state=tk.DISABLED)

        # Right frame for charts and analysis (placeholder)
        self.right_frame = ttk.Frame(self.main_container)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_nasdaq_tickers(self):
        """Load tickers from NASDAQ database."""
        try:
            self.status_var.set("Loading NASDAQ tickers...")
            self.root.update()
            
            self.available_tickers = self.nasdaq_db.get_all_symbols()
            
            if not self.available_tickers:
                raise ValueError("No tickers loaded from NASDAQ database")
            
            self.ticker_listbox.delete(0, tk.END)
            for ticker in self.available_tickers:
                self.ticker_listbox.insert(tk.END, ticker)
            
            self.status_var.set(f"Loaded {len(self.available_tickers)} NASDAQ tickers")
            self.logger.info(f"Successfully loaded {len(self.available_tickers)} NASDAQ tickers")
            
        except Exception as e:
            error_msg = f"Error loading NASDAQ tickers: {e}"
            self.logger.error(error_msg)
            self.status_var.set("Error loading NASDAQ tickers")
            messagebox.showerror("Error", error_msg)

    def filter_tickers(self, *args):
        """Filter tickers based on search text."""
        search_text = self.search_var.get().upper()
        self.ticker_listbox.delete(0, tk.END)
        
        filtered_tickers = [ticker for ticker in self.available_tickers if search_text in ticker.upper()]
        for ticker in filtered_tickers:
            self.ticker_listbox.insert(tk.END, ticker)

    def on_ticker_select(self, event):
        """Handle ticker selection event."""
        try:
            selected = self.get_selected_tickers()
            self.selection_label.config(text=f"Selected: {len(selected)}")
            
            # Show details for the last selected ticker
            if selected:
                last_selected = selected[-1]
                info = self.nasdaq_db.get_symbol_info(last_selected)
                if info:
                    details = (
                        f"Symbol: {info.get('Symbol', 'N/A')}\n"
                        f"Name: {info.get('Name', 'N/A')}\n"
                        f"Sector: {info.get('Sector', 'N/A')}\n"
                        f"Industry: {info.get('Industry', 'N/A')}"
                    )
                    
                    self.details_text.config(state=tk.NORMAL)
                    self.details_text.delete(1.0, tk.END)
                    self.details_text.insert(tk.END, details)
                    self.details_text.config(state=tk.DISABLED)
                    
        except Exception as e:
            self.logger.error(f"Error updating selection: {e}")

    def get_selected_tickers(self):
        """Get list of selected tickers."""
        return [self.ticker_listbox.get(idx) for idx in self.ticker_listbox.curselection()]

    def start_realtime_collection(self):
        """Start real-time data collection."""
        try:
            # Get collection interval
            try:
                self.collection_interval = int(self.interval_var.get())
                if self.collection_interval < 30:
                    raise ValueError("Interval must be at least 30 seconds")
            except ValueError as e:
                messagebox.showerror("Error", str(e))
                self.realtime_var.set(False)
                return

            # Get selected tickers
            selected_tickers = self.get_selected_tickers()
            if not selected_tickers:
                messagebox.showwarning("Warning", "Please select tickers for real-time collection")
                self.realtime_var.set(False)
                return

            self.realtime_running = True
            self.realtime_thread = threading.Thread(
                target=self._realtime_collection_loop,
                args=(selected_tickers,),
                daemon=True
            )
            self.realtime_thread.start()

            self.logger.info("Real-time collection started")
            self.status_var.set("Real-time collection active")
            
        except Exception as e:
            self.logger.error(f"Error starting real-time collection: {e}")
            messagebox.showerror("Error", f"Failed to start real-time collection: {e}")
            self.realtime_var.set(False)

    def stop_realtime_collection(self):
        """Stop real-time data collection."""
        try:
            self.realtime_running = False
            if self.realtime_thread:
                self.realtime_thread.join(timeout=5)
            self.realtime_thread = None
            
            self.logger.info("Real-time collection stopped")
            self.status_var.set("Real-time collection stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping real-time collection: {e}")
            messagebox.showerror("Error", f"Failed to stop real-time collection: {e}")

    def _realtime_collection_loop(self, tickers):
        """Real-time collection loop."""
        while self.realtime_running:
            try:
                for ticker in tickers:
                    if not self.realtime_running:
                        break
                        
                    self.status_var.set(f"Collecting data for {ticker}")
                    self.root.update()
                    
                    df = self.data_loader.collect_historical_data(ticker)
                    if df is not None:
                        self.db.save_stock_data(df, ticker)
                        
                self.status_var.set("Waiting for next collection cycle")
                self.root.update()
                
                # Wait for next interval
                start_time = time.time()
                while self.realtime_running and (time.time() - start_time) < self.collection_interval:
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in real-time collection: {e}")
                if self.realtime_running:
                    self.status_var.set(f"Collection error: {str(e)}")
                    self.root.update()
                    time.sleep(5)

    def toggle_realtime(self):
        """Toggle real-time collection."""
        if self.realtime_var.get():
            self.start_realtime_collection()
        else:
            self.stop_realtime_collection()

    def load_selected_data(self):
        """Load data for selected tickers."""
        selected_tickers = self.get_selected_tickers()
        if not selected_tickers:
            messagebox.showwarning("Warning", "Please select tickers to load")
            return

        try:
            self.status_var.set("Loading selected ticker data...")
            self.root.update()

            # Construct query for selected tickers
            placeholders = ','.join(['?' for _ in selected_tickers])
            query = f"""
                SELECT date, ticker, open, high, low, close, adj_close, volume
                FROM stock_data
                WHERE ticker IN ({placeholders})
                ORDER BY ticker, date
            """

            # Load data
            with self.db.get_connection() as conn:
                df = conn.execute(query, selected_tickers).fetchdf()

            if df.empty:
                raise ValueError("No data found for selected tickers")

            # Process the data (implement your analysis logic here)
            self.process_loaded_data(df)
            
            self.status_var.set(f"Loaded data for {len(selected_tickers)} tickers")
            self.logger.info(f"Successfully loaded data for {len(selected_tickers)} tickers")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.status_var.set("Error loading data")
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def process_loaded_data(self, df):
        """Process the loaded data for analysis."""
        # Implement your data processing and analysis logic here
        # This could include:
        # - Calculating technical indicators
        # - Updating charts
        # - Performing analysis
        # - Updating the GUI with results
        pass 