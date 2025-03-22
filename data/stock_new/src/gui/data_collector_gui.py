import tkinter as tk
from tkinter import ttk, messagebox
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

class DataCollectorGUI:
    def __init__(self, db_manager=None, logger=None):
        self.db_manager = db_manager
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Initializing GUI...")  # Debug log
        
        # Initialize basic attributes
        self.ticker_data = {}
        self.tickers = []
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Stock Data Collector")
        self.root.minsize(800, 600)
        
        self.logger.info("Created main window")  # Debug log
        
        # Initialize variables
        self.setup_variables()
        
        # Create GUI elements
        self.create_gui()
        
        self.logger.info("GUI setup complete")  # Debug log
        
        # Load initial data
        self.load_and_update_tickers()

    def setup_variables(self):
        """Initialize Tkinter variables"""
        self.selected_ticker = tk.StringVar()
        self.exchange_var = tk.StringVar(value="All")
        self.status_var = tk.StringVar(value="Initializing...")
        self.search_var = tk.StringVar()
        self.price_min = tk.StringVar()
        self.price_max = tk.StringVar()

    def create_gui(self):
        """Create the main GUI elements"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Control Frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Buttons
        ttk.Button(control_frame, text="Update Data", command=self.update_data).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Export Data", command=self.export_data).grid(row=0, column=1, padx=5)

        # Filter Frame
        filter_frame = ttk.LabelFrame(main_frame, text="Filters", padding="5")
        filter_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Exchange filter
        ttk.Label(filter_frame, text="Exchange:").grid(row=0, column=0, padx=5)
        self.exchange_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.exchange_var,
            values=["All", "NASDAQ", "NYSE"],
            state='readonly',
            width=15
        )
        self.exchange_combo.grid(row=0, column=1, padx=5)
        self.exchange_combo.bind('<<ComboboxSelected>>', self.apply_filters)

        # Search filter
        ttk.Label(filter_frame, text="Search:").grid(row=1, column=0, padx=5)
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=30)
        search_entry.grid(row=1, column=1, padx=5)
        self.search_var.trace('w', lambda *args: self.apply_filters())

        # Ticker Selection Frame
        selection_frame = ttk.LabelFrame(main_frame, text="Stock Selection", padding="5")
        selection_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        # Ticker list
        self.stock_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.selected_ticker,
            width=50
        )
        self.stock_combo.pack(fill=tk.X, padx=5, pady=5)

        # Status Frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        # Status label
        ttk.Label(status_frame, textvariable=self.status_var).pack(fill=tk.X)

    def load_and_update_tickers(self):
        """Load tickers and update the GUI"""
        try:
            self.status_var.set("Loading tickers...")
            self.root.update()
            
            if self.db_manager:
                df = self.db_manager.get_all_symbols()
                if not df.empty:
                    self.process_ticker_data(df)
                else:
                    self.use_default_tickers()
            else:
                self.use_default_tickers()
                
        except Exception as e:
            self.logger.error(f"Error loading tickers: {e}")
            self.use_default_tickers()

    def process_ticker_data(self, df):
        """Process ticker data from DataFrame"""
        try:
            self.ticker_data = {}
            for _, row in df.iterrows():
                ticker = str(row['symbol']).strip()
                if ticker and ticker.lower() != 'nan':
                    self.ticker_data[ticker] = {
                        'name': str(row.get('name', '')).strip(),
                        'exchange': str(row.get('exchange', '')).strip()
                    }
            
            self.tickers = sorted(list(self.ticker_data.keys()))
            self.update_display()
            self.status_var.set(f"Loaded {len(self.tickers)} tickers")
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            self.use_default_tickers()

    def update_display(self):
        """Update the display with current ticker data"""
        display_values = []
        for ticker in self.tickers:
            info = self.ticker_data[ticker]
            display_text = f"{ticker} - {info['name']}"
            if info['exchange']:
                display_text += f" ({info['exchange']})"
            display_values.append(display_text)
        
        self.stock_combo['values'] = display_values
        if display_values:
            self.stock_combo.set(display_values[0])

    def apply_filters(self, *args):
        """Apply filters to ticker list"""
        try:
            filtered_tickers = self.tickers.copy()
            
            # Exchange filter
            if self.exchange_var.get() != "All":
                filtered_tickers = [
                    ticker for ticker in filtered_tickers
                    if self.ticker_data[ticker]['exchange'] == self.exchange_var.get()
                ]

            # Search filter
            search_term = self.search_var.get().lower()
            if search_term:
                filtered_tickers = [
                    ticker for ticker in filtered_tickers
                    if search_term in ticker.lower() or
                    search_term in self.ticker_data[ticker]['name'].lower()
                ]

            self.update_filtered_display(filtered_tickers)
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {e}")
            self.status_var.set(f"Filter error: {e}")

    def update_filtered_display(self, filtered_tickers):
        """Update display with filtered tickers"""
        display_values = []
        for ticker in sorted(filtered_tickers):
            info = self.ticker_data[ticker]
            display_text = f"{ticker} - {info['name']}"
            if info['exchange']:
                display_text += f" ({info['exchange']})"
            display_values.append(display_text)
        
        self.stock_combo['values'] = display_values
        if display_values:
            self.stock_combo.set(display_values[0])
        self.status_var.set(f"Showing {len(display_values)} of {len(self.tickers)} tickers")

    def use_default_tickers(self):
        """Use default tickers when loading fails"""
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.ticker_data = {
            ticker: {'name': '', 'exchange': 'NASDAQ'}
            for ticker in self.tickers
        }
        self.update_display()
        self.status_var.set("Using default tickers")

    def update_data(self):
        """Update the data"""
        try:
            self.load_and_update_tickers()
        except Exception as e:
            self.logger.error(f"Update failed: {e}")
            self.status_var.set(f"Update error: {e}")

    def export_data(self):
        """Export the data"""
        try:
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export as CSV
            csv_file = export_dir / f"stock_data_{timestamp}.csv"
            df = pd.DataFrame.from_dict(self.ticker_data, orient='index')
            df.to_csv(csv_file)

            self.status_var.set(f"Data exported to {csv_file}")
            messagebox.showinfo("Export Complete", f"Data exported to {csv_file}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            messagebox.showerror("Export Error", str(e))

    def run(self):
        """Start the GUI"""
        self.logger.info("Starting GUI main loop...")  # Debug log
        self.root.mainloop()
