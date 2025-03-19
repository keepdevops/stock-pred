import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime
import yfinance as yf
import pandas as pd
from src.data.nasdaq_symbols import NASDAQ_SYMBOLS
from src.database.database_connector import DataCollector
import polars as pl
import os

class DataCollectorGUI:
    def __init__(self, root, config, db):
        """Initialize the GUI."""
        self.root = root
        self.config = config
        self.db = db
        self.logger = logging.getLogger(__name__)
        
        # Configure root window grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="5 5 5 5")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure main frame grid
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Setup GUI components
        self.setup_gui_components()
        
        # Load NASDAQ symbols automatically
        self.load_nasdaq_symbols()

    def setup_gui_components(self):
        """Setup all GUI components."""
        # Configure main frame grid
        for i in range(3):  # Three main sections
            self.main_frame.grid_rowconfigure(i, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Setup each section
        self.setup_ticker_management()
        self.setup_historical_collection()
        self.setup_realtime_collection()

    def setup_ticker_management(self):
        """Setup ticker management components."""
        # Create frame for ticker management
        ticker_frame = ttk.LabelFrame(self.main_frame, text="Ticker Management", padding="5 5 5 5")
        ticker_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Add search/filter functionality
        filter_frame = ttk.Frame(ticker_frame)
        filter_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        filter_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(filter_frame, text="Filter:").grid(row=0, column=0, padx=(0,5))
        self.filter_var = tk.StringVar()
        self.filter_var.trace('w', self.filter_tickers)
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var)
        self.filter_entry.grid(row=0, column=1, sticky="ew")
        
        # Create list frame
        list_frame = ttk.Frame(ticker_frame)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        ticker_frame.grid_rowconfigure(1, weight=1)
        ticker_frame.grid_columnconfigure(0, weight=1)
        
        # Add selection count label
        self.selection_label = ttk.Label(list_frame, text="Selected: 0")
        self.selection_label.grid(row=0, column=0, sticky="w", padx=5)
        
        # Create scrolled frame for listbox
        self.ticker_list = tk.Listbox(
            list_frame,
            selectmode="extended",
            height=10,
            width=30
        )
        scrollbar_y = ttk.Scrollbar(list_frame, orient="vertical", command=self.ticker_list.yview)
        scrollbar_x = ttk.Scrollbar(list_frame, orient="horizontal", command=self.ticker_list.xview)
        self.ticker_list.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Grid the listbox and scrollbars
        self.ticker_list.grid(row=1, column=0, sticky="nsew")
        scrollbar_y.grid(row=1, column=1, sticky="ns")
        scrollbar_x.grid(row=2, column=0, sticky="ew")
        
        # Configure list frame grid
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection event
        self.ticker_list.bind('<<ListboxSelect>>', self.update_selection_count)
        
        # Create button frame
        button_frame = ttk.Frame(ticker_frame)
        button_frame.grid(row=2, column=0, sticky="ew", pady=5)
        button_frame.grid_columnconfigure(1, weight=1)
        
        # Add buttons
        ttk.Button(button_frame, text="Select All", 
                   command=self.select_all_tickers).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="Clear Selection", 
                   command=self.clear_selection).grid(row=0, column=1, padx=2)
        ttk.Button(button_frame, text="Invert Selection", 
                   command=self.invert_selection).grid(row=0, column=2, padx=2)

    def setup_historical_collection(self):
        """Setup historical data collection components."""
        historical_frame = ttk.LabelFrame(self.main_frame, text="Historical Data Collection", padding="5 5 5 5")
        historical_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure frame grid
        historical_frame.grid_columnconfigure(1, weight=1)
        
        # Date range selection
        ttk.Label(historical_frame, text="Start Date:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.start_date = ttk.Entry(historical_frame)
        self.start_date.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.start_date.insert(0, "2023-03-18")
        
        ttk.Label(historical_frame, text="End Date:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.end_date = ttk.Entry(historical_frame)
        self.end_date.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.end_date.insert(0, "2025-03-18")
        
        # Progress bar and label
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(historical_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.status_label = ttk.Label(historical_frame, text="")
        self.status_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=5)
        
        # Download button
        ttk.Button(historical_frame, text="Download Selected Tickers", 
                   command=self.download_selected_tickers).grid(row=4, column=0, columnspan=2, pady=5)

    def setup_realtime_collection(self):
        """Setup real-time data collection components."""
        realtime_frame = ttk.LabelFrame(self.main_frame, text="Real-time Data Collection", padding="5 5 5 5")
        realtime_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure frame grid
        realtime_frame.grid_columnconfigure(1, weight=1)
        
        # Interval selection
        ttk.Label(realtime_frame, text="Update Interval (seconds):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.interval_entry = ttk.Entry(realtime_frame)
        self.interval_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.interval_entry.insert(0, "60")
        
        # Control buttons frame
        control_frame = ttk.Frame(realtime_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Collection", 
                                      command=self.start_collection)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Collection", 
                                     command=self.stop_collection, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)

    def _clean_symbol(self, symbol: str) -> str:
        """
        Clean a ticker symbol by replacing special characters with standard alternatives.
        
        Conversions:
        ^ -> -P-     (preferred shares)
        / -> -W-     (warrants)
        = -> -U-     (units)
        % -> -R-     (rights)
        & -> -AND-   (combined companies)
        . -> -       (general delimiter)
        + -> -PLUS-  (special classes)
        # -> -H-     (special holdings)
        * -> -S-     (special conditions)
        @ -> -AT-    (special market)
        """
        if not isinstance(symbol, str):
            return str(symbol)
        
        replacements = {
            '^': '-P-',    # Preferred shares
            '/': '-W-',    # Warrants
            '=': '-U-',    # Units
            '%': '-R-',    # Rights
            '&': '-AND-',  # Combined companies
            '.': '-',      # General delimiter
            '+': '-PLUS-', # Special classes
            '#': '-H-',    # Special holdings
            '*': '-S-',    # Special conditions
            '@': '-AT-',   # Special market
            ' ': '-',      # Spaces
        }
        
        cleaned = str(symbol).strip().upper()  # Ensure uppercase
        for char, replacement in replacements.items():
            cleaned = cleaned.replace(char, replacement)
        
        # Remove any duplicate dashes
        while '--' in cleaned:
            cleaned = cleaned.replace('--', '-')
        
        # Remove leading/trailing dashes
        cleaned = cleaned.strip('-')
        
        return cleaned

    def _restore_symbol(self, cleaned_symbol: str) -> str:
        """
        Restore a cleaned symbol back to Yahoo Finance format.
        """
        replacements = {
            '-P-': '^',    # Preferred shares
            '-W-': '/',    # Warrants
            '-U-': '=',    # Units
            '-R-': '%',    # Rights
            '-AND-': '&',  # Combined companies
            '-PLUS-': '+', # Special classes
            '-H-': '#',    # Special holdings
            '-S-': '*',    # Special conditions
            '-AT-': '@',   # Special market
        }
        
        yahoo_symbol = cleaned_symbol
        for replacement, char in replacements.items():
            yahoo_symbol = yahoo_symbol.replace(replacement, char)
        
        # Handle simple dash replacement last to avoid conflicts
        # Only replace single dashes that aren't part of other replacements
        if yahoo_symbol.count('-') == 1:
            yahoo_symbol = yahoo_symbol.replace('-', '.')
        
        return yahoo_symbol

    def find_nasdaq_screener_file(self):
        """Find the most recent NASDAQ screener CSV file in the project directory."""
        try:
            # Get the project root directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            
            # Look for NASDAQ screener files
            nasdaq_files = []
            for root, _, files in os.walk(project_root):
                for file in files:
                    if file.startswith('nasdaq_screener') and file.endswith('.csv'):
                        full_path = os.path.join(root, file)
                        nasdaq_files.append((full_path, os.path.getmtime(full_path)))
            
            if not nasdaq_files:
                raise FileNotFoundError("No NASDAQ screener CSV file found in project directory")
            
            # Get the most recent file
            newest_file = max(nasdaq_files, key=lambda x: x[1])
            self.logger.info(f"Found NASDAQ screener file: {newest_file[0]}")
            return newest_file[0]
            
        except Exception as e:
            self.logger.error(f"Error finding NASDAQ screener file: {str(e)}")
            raise

    def load_nasdaq_symbols(self):
        """Load NASDAQ symbols from CSV file and clean them."""
        try:
            # Find the NASDAQ screener file
            csv_path = self.find_nasdaq_screener_file()
            
            # Read CSV using polars
            self.logger.info(f"Reading CSV from: {csv_path}")
            df = pl.read_csv(csv_path)
            
            # Clean the Symbol column using polars expressions
            df = df.with_columns([
                pl.col("Symbol")
                    .str.strip_chars()  # Use strip_chars instead of strip
                    .cast(pl.Utf8)      # Ensure string type
                    .map_elements(self._clean_symbol)  # Apply cleaning function
                    .alias("Symbol")
            ])
            
            # Convert to unique list of symbols and sort
            symbols = (df.select("Symbol")
                        .unique()
                        .sort("Symbol")
                        .to_series()
                        .to_list())
            
            self.logger.info(f"Loaded {len(symbols)} NASDAQ symbols")
            
            # Log some examples of original and cleaned symbols
            sample_size = min(5, len(df))
            original_df = df.select("Symbol").head(sample_size)
            original_symbols = original_df.to_series().to_list()
            cleaned_symbols = [self._clean_symbol(s) for s in original_symbols]
            
            self.logger.info("Sample symbol conversions:")
            for orig, cleaned in zip(original_symbols, cleaned_symbols):
                if orig != cleaned:
                    self.logger.info(f"  {orig} -> {cleaned}")
                else:
                    self.logger.info(f"  {orig} (unchanged)")
            
            # Clear existing items
            self.ticker_list.delete(0, tk.END)
            
            # Add symbols to listbox
            for symbol in symbols:
                self.ticker_list.insert(tk.END, symbol)
            
            # Update status
            self.status_label.config(text=f"Loaded {len(symbols)} symbols")
            self.root.update()
            
            # Update the filter to show all symbols initially
            self.filter_tickers()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading NASDAQ symbols: {str(e)}")
            return False

    def filter_tickers(self, *args):
        """Filter the ticker list based on search text."""
        search_text = self.filter_var.get().upper()
        self.ticker_list.delete(0, tk.END)
        for symbol in NASDAQ_SYMBOLS:
            if search_text in symbol:
                self.ticker_list.insert(tk.END, symbol)

    def update_selection_count(self, event=None):
        """Update the selection count label."""
        count = len(self.ticker_list.curselection())
        self.selection_label.config(text=f"Selected: {count}")

    def invert_selection(self):
        """Invert the current selection."""
        current_selection = set(self.ticker_list.curselection())
        all_indices = set(range(self.ticker_list.size()))
        new_selection = all_indices - current_selection
        
        self.ticker_list.selection_clear(0, tk.END)
        for idx in new_selection:
            self.ticker_list.selection_set(idx)
        
        self.update_selection_count()

    def select_all_tickers(self):
        """Select all tickers in the list."""
        self.ticker_list.select_set(0, tk.END)
        self.update_selection_count()
        self.logger.info("Selected all tickers")

    def clear_selection(self):
        """Clear the current selection."""
        self.ticker_list.selection_clear(0, tk.END)
        self.update_selection_count()
        self.logger.info("Cleared ticker selection")

    def add_ticker(self):
        """Add a ticker to the list."""
        ticker = self.ticker_entry.get().strip().upper()
        if ticker:
            if ticker not in self.ticker_list.get(0, tk.END):
                self.ticker_list.insert(tk.END, ticker)
                self.save_tickers()
                self.ticker_entry.delete(0, tk.END)
            else:
                messagebox.showwarning("Warning", f"Ticker {ticker} already exists")
        else:
            messagebox.showwarning("Warning", "Please enter a ticker")

    def remove_ticker(self):
        """Remove selected ticker(s) from the list."""
        selection = self.ticker_list.curselection()
        if selection:
            for index in reversed(selection):
                self.ticker_list.delete(index)
            self.save_tickers()
        else:
            messagebox.showwarning("Warning", "Please select ticker(s) to remove")

    def clear_tickers(self):
        """Clear all tickers from the list."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all tickers?"):
            self.ticker_list.delete(0, tk.END)
            self.save_tickers()

    def save_tickers(self):
        """Save tickers to configuration."""
        try:
            tickers = list(self.ticker_list.get(0, tk.END))
            self.config.set('tickers', tickers)
            self.config.save()
        except Exception as e:
            self.logger.error(f"Error saving tickers: {e}")
            messagebox.showerror("Error", f"Failed to save tickers: {e}")

    def get_selected_tickers(self):
        """Get list of selected tickers."""
        selection = self.ticker_list.curselection()
        return [self.ticker_list.get(index) for index in selection] if selection else []

    def start_collection(self):
        """Start real-time data collection."""
        try:
            interval = int(self.interval_entry.get())
            # Add real-time collection logic here
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid interval")

    def stop_collection(self):
        """Stop real-time data collection."""
        # Add stop collection logic here
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def run(self):
        self.root.mainloop() 

    def update_progress(self, processed, total, ticker):
        """Update progress in both progress bar and window title."""
        progress = (processed / total) * 100
        self.progress_var.set(progress)
        self.status_label.config(text=f"Downloading {ticker}...")
        self.root.title(f"Stock Market Data Collector - {processed}/{total} ({progress:.1f}%)")
        self.root.update() 

    def download_selected_tickers(self):
        """Download historical data for all selected tickers."""
        selected_tickers = self.get_selected_tickers()
        if not selected_tickers:
            messagebox.showwarning("Warning", "Please select at least one ticker")
            return

        total_tickers = len(selected_tickers)
        processed_tickers = 0
        successful_tickers = 0
        failed_tickers = []

        try:
            for display_ticker in selected_tickers:
                try:
                    # Convert display ticker back to Yahoo Finance format
                    yahoo_ticker = self._restore_symbol(display_ticker)
                    
                    self.logger.info(f"Attempting to download {display_ticker} (Yahoo: {yahoo_ticker})")
                    
                    # Create Ticker object with Yahoo format
                    stock = yf.Ticker(yahoo_ticker)
                    
                    # Get info to verify ticker is valid
                    self.logger.info(f"Attempting to download {yahoo_ticker}")
                    
                    try:
                        # Download historical data
                        df = stock.history(period="2y")
                        
                        if df.empty:
                            self.logger.warning(f"No data available for {yahoo_ticker}")
                            failed_tickers.append(f"{display_ticker} (no data)")
                            continue
                        
                        # Log the raw data
                        self.logger.info(f"Raw data shape for {yahoo_ticker}: {df.shape}")
                        self.logger.info(f"Raw columns: {df.columns.tolist()}")
                        
                        # Reset index and prepare data
                        df = df.reset_index()
                        
                        # Create new dataframe with required columns
                        new_df = pd.DataFrame({
                            'date': pd.to_datetime(df['Date']).dt.tz_localize(None),
                            'ticker': display_ticker,  # Use the cleaned ticker format
                            'open': df['Open'],
                            'high': df['High'],
                            'low': df['Low'],
                            'close': df['Close'],
                            'adj_close': df['Close'],  # Using Close as adj_close
                            'volume': df['Volume']
                        })
                        
                        # Convert data types
                        new_df['volume'] = new_df['volume'].astype('int64')
                        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close']
                        new_df[numeric_cols] = new_df[numeric_cols].astype('float64')
                        
                        # Remove any rows with NaN values
                        new_df = new_df.dropna()
                        
                        if not new_df.empty:
                            # Save to database
                            rows_saved = self.db.save_ticker_data(new_df)
                            if rows_saved:
                                successful_tickers += 1
                                self.logger.info(f"Successfully saved {len(new_df)} rows for {display_ticker}")
                        else:
                            self.logger.warning(f"No valid data after processing for {display_ticker}")
                            failed_tickers.append(f"{display_ticker} (no data)")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {yahoo_ticker}: {str(e)}")
                        failed_tickers.append(f"{display_ticker} (processing error)")
                    
                except Exception as e:
                    self.logger.error(f"Error downloading {display_ticker}: {str(e)}")
                    failed_tickers.append(f"{display_ticker} (download error)")
                
                finally:
                    # Update progress
                    processed_tickers += 1
                    self.update_progress(processed_tickers, total_tickers, display_ticker)

            # Show completion message
            self.status_label.config(text="Download completed!")
            summary = (f"Download completed!\n"
                      f"Successfully processed: {successful_tickers}\n"
                      f"Failed: {len(failed_tickers)}\n"
                      f"Total: {total_tickers}")
            self.logger.info(summary)
            messagebox.showinfo("Success", summary)

        except Exception as e:
            self.logger.error(f"Error in batch download: {str(e)}")
            messagebox.showerror("Error", f"Download failed: {str(e)}")

        finally:
            # Re-enable download button and reset status
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="")
            self.progress_var.set(0)
            self.root.update()
            self.root.title("Stock Market Data Collector") 

    def reload_nasdaq_symbols(self):
        """Reload NASDAQ symbols from CSV file."""
        try:
            self.reload_button.config(state='disabled')
            self.status_label.config(text="Reloading NASDAQ symbols...")
            self.root.update()

            # Clear existing items and search
            self.ticker_list.delete(0, tk.END)
            self.filter_var.set("")
            
            # Reload symbols
            success = self.load_nasdaq_symbols()
            
            if success:
                messagebox.showinfo("Success", "NASDAQ symbols reloaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to reload NASDAQ symbols")
                
        except Exception as e:
            self.logger.error(f"Error reloading NASDAQ symbols: {str(e)}")
            messagebox.showerror("Error", f"Failed to reload NASDAQ symbols: {str(e)}")
        
        finally:
            self.reload_button.config(state='normal')
            self.status_label.config(text="")
            self.root.update() 