import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import logging
from datetime import datetime, timedelta
import duckdb
from src.data.data_collector import DataCollector  # Adjust import path as needed

class DataCollectorGUI:
    def __init__(self, config):
        """Initialize the GUI."""
        # Initialize all attributes first
        self._realtime_collection_active = False
        self._realtime_collection_worker = None
        self.selected_tickers = set(['AAPL', 'MSFT', 'GOOG', 'TSLA', 'F'])
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create the main window
        self.window = tk.Tk()
        self.window.title("Stock Market Data Collector")
        
        # Initialize collector
        self.collector = DataCollector(config, self.logger)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.window, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.window, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Initialize GUI components
        self.realtime_status_var = tk.StringVar(value="Status: Stopped")
        self.start_button = None
        self.stop_button = None
        
        # Create GUI sections
        self.create_download_frame()
        self.create_progress_frame()
        self.create_realtime_controls()

        # Set up window closing handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_ticker_frame(self):
        """Create the ticker selection frame."""
        ticker_frame = ttk.LabelFrame(self.main_frame, text="Ticker Selection", padding="5")
        ticker_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Default tickers
        self.ticker_vars = {}
        default_tickers = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'F']
        
        for i, ticker in enumerate(default_tickers):
            var = tk.BooleanVar(value=True)
            self.ticker_vars[ticker] = var
            ttk.Checkbutton(
                ticker_frame,
                text=ticker,
                variable=var,
                command=self.update_selected_tickers
            ).grid(row=i//2, column=i%2, padx=5, pady=2, sticky=tk.W)

        # Add ticker entry
        ttk.Label(ticker_frame, text="Add Ticker:").grid(row=len(default_tickers)//2 + 1, column=0, padx=5, pady=2)
        self.new_ticker_entry = ttk.Entry(ticker_frame, width=10)
        self.new_ticker_entry.grid(row=len(default_tickers)//2 + 1, column=1, padx=5, pady=2)
        ttk.Button(
            ticker_frame,
            text="Add",
            command=self.add_ticker
        ).grid(row=len(default_tickers)//2 + 1, column=2, padx=5, pady=2)

    def create_realtime_controls(self):
        """Create realtime collection controls."""
        frame = ttk.LabelFrame(self.main_frame, text="Realtime Collection", padding="5")
        frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Start button
        self.start_button = ttk.Button(
            frame,
            text="Start Collection",
            command=self.start_realtime_collection
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=2)
        
        # Stop button
        self.stop_button = ttk.Button(
            frame,
            text="Stop Collection",
            command=self.stop_realtime_collection,
            state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=2)
        
        # Status label
        self.realtime_status = ttk.Label(frame, textvariable=self.realtime_status_var)
        self.realtime_status.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

    def create_download_frame(self):
        """Create the download section of the GUI."""
        download_frame = ttk.LabelFrame(self.main_frame, text="Download Data", padding="5")
        download_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Market selection checkboxes
        self.market_vars = {
            'SP500': tk.BooleanVar(value=True),
            'NASDAQ': tk.BooleanVar(value=True),
            'FOREX': tk.BooleanVar(value=True),
            'CRYPTO': tk.BooleanVar(value=True)
        }

        for i, (market, var) in enumerate(self.market_vars.items()):
            ttk.Checkbutton(
                download_frame, 
                text=market, 
                variable=var
            ).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)

        # Date range selection
        ttk.Label(download_frame, text="Start Date:").grid(row=0, column=1, padx=5, pady=2)
        self.start_date = DateEntry(
            download_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='yyyy-mm-dd'
        )
        self.start_date.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(download_frame, text="End Date:").grid(row=1, column=1, padx=5, pady=2)
        self.end_date = DateEntry(
            download_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='yyyy-mm-dd'
        )
        self.end_date.grid(row=1, column=2, padx=5, pady=2)

        # Download buttons frame
        buttons_frame = ttk.Frame(download_frame)
        buttons_frame.grid(row=len(self.market_vars), column=0, columnspan=3, pady=10)

        # Single ticker download
        ttk.Label(buttons_frame, text="Ticker:").pack(side=tk.LEFT, padx=5)
        self.ticker_entry = ttk.Entry(buttons_frame, width=10)
        self.ticker_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(
            buttons_frame,
            text="Download Ticker",
            command=self.download_single_ticker
        ).pack(side=tk.LEFT, padx=5)

        # Download all button
        ttk.Button(
            buttons_frame,
            text="Download All Selected",
            command=self.download_all_tickers
        ).pack(side=tk.LEFT, padx=5)

    def create_progress_frame(self):
        """Create the progress section of the GUI."""
        progress_frame = ttk.LabelFrame(self.main_frame, text="Progress", padding="5")
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Progress text
        self.progress_text = tk.StringVar(value="Ready")
        ttk.Label(
            progress_frame,
            textvariable=self.progress_text
        ).grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5)

    def update_selected_tickers(self):
        """Update the set of selected tickers."""
        self.selected_tickers = {
            ticker for ticker, var in self.ticker_vars.items() 
            if var.get()
        }
        self.logger.debug(f"Selected tickers: {self.selected_tickers}")

    def add_ticker(self):
        """Add a new ticker to the selection."""
        ticker = self.new_ticker_entry.get().strip().upper()
        if ticker and ticker not in self.ticker_vars:
            var = tk.BooleanVar(value=True)
            self.ticker_vars[ticker] = var
            row = len(self.ticker_vars) // 2
            col = len(self.ticker_vars) % 2
            ttk.Checkbutton(
                self.ticker_frame,
                text=ticker,
                variable=var,
                command=self.update_selected_tickers
            ).grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
            self.update_selected_tickers()
            self.new_ticker_entry.delete(0, tk.END)

    def download_single_ticker(self):
        """Download data for a single ticker."""
        ticker = self.ticker_entry.get().strip().upper()
        if not ticker:
            messagebox.showwarning("Input Error", "Please enter a ticker symbol")
            return

        try:
            self.status_var.set(f"Downloading {ticker}...")
            self.progress_text.set(f"Downloading {ticker}...")
            self.progress_var.set(0)
            self.window.update_idletasks()

            start_date = self.start_date.get_date().strftime('%Y-%m-%d')
            end_date = self.end_date.get_date().strftime('%Y-%m-%d')

            success = self.collector._download_single_ticker_with_retry(
                ticker, start_date, end_date
            )

            if success:
                self.progress_var.set(100)
                messagebox.showinfo("Success", f"Successfully downloaded data for {ticker}")
            else:
                messagebox.showerror("Error", f"Failed to download data for {ticker}")

        except Exception as e:
            self.logger.error(f"Error downloading {ticker}: {e}")
            messagebox.showerror("Error", f"Failed to download {ticker}: {str(e)}")
        finally:
            self.status_var.set("Ready")
            self.progress_text.set("Ready")
            self.progress_var.set(0)

    def download_all_tickers(self):
        """Download data for all selected markets."""
        try:
            # Collect selected tickers
            tickers = []
            if self.market_vars['SP500'].get():
                tickers.extend(self.collector.get_sp500_tickers())
            if self.market_vars['NASDAQ'].get():
                tickers.extend(self.collector.get_nasdaq100_tickers())
            if self.market_vars['FOREX'].get():
                tickers.extend(self.collector.get_forex_pairs())
            if self.market_vars['CRYPTO'].get():
                tickers.extend(self.collector.get_crypto_tickers())

            # Remove duplicates
            tickers = list(set(tickers))

            if not tickers:
                messagebox.showwarning("No Selection", "Please select at least one market")
                return

            # Confirm download
            if not messagebox.askyesno(
                "Confirm Download",
                f"Download data for {len(tickers)} tickers?\nThis may take a while."
            ):
                return

            # Get date range
            start_date = self.start_date.get_date().strftime('%Y-%m-%d')
            end_date = self.end_date.get_date().strftime('%Y-%m-%d')

            # Update status
            self.status_var.set("Downloading market data...")
            self.progress_text.set("Initializing download...")
            self.progress_var.set(0)
            self.window.update_idletasks()

            # Download data
            total_tickers = len(tickers)
            success_count = 0
            failed_tickers = []

            for i, ticker in enumerate(tickers, 1):
                try:
                    self.progress_text.set(f"Downloading {ticker} ({i}/{total_tickers})")
                    self.progress_var.set((i - 1) / total_tickers * 100)
                    self.window.update_idletasks()

                    success = self.collector._download_single_ticker_with_retry(
                        ticker, start_date, end_date
                    )

                    if success:
                        success_count += 1
                    else:
                        failed_tickers.append(ticker)

                except Exception as e:
                    self.logger.error(f"Error downloading {ticker}: {e}")
                    failed_tickers.append(ticker)

                self.progress_var.set(i / total_tickers * 100)
                self.window.update_idletasks()

            # Show results
            self.progress_var.set(100)
            self.progress_text.set("Download completed")
            
            result_message = f"Download completed:\n"
            result_message += f"Successfully downloaded: {success_count}\n"
            result_message += f"Failed downloads: {len(failed_tickers)}"
            
            if failed_tickers:
                result_message += "\n\nFailed tickers:\n"
                result_message += "\n".join(failed_tickers)

            messagebox.showinfo("Download Complete", result_message)

        except Exception as e:
            self.logger.error(f"Error in batch download: {e}")
            messagebox.showerror("Error", f"Download failed: {str(e)}")
        finally:
            self.status_var.set("Ready")
            self.progress_text.set("Ready")
            self.progress_var.set(0)

    def start_realtime_collection(self):
        """Start realtime data collection."""
        try:
            if not self._realtime_collection_active:
                self._realtime_collection_active = True
                self._realtime_collection_worker = threading.Thread(
                    target=self._realtime_collection_loop,
                    daemon=True
                )
                self._realtime_collection_worker.start()
                self.realtime_status_var.set("Status: Running")
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_var.set("Realtime collection started")
        except Exception as e:
            self.logger.error(f"Error starting collection: {e}")
            messagebox.showerror("Error", str(e))

    def stop_realtime_collection(self):
        """Stop realtime data collection."""
        try:
            if self._realtime_collection_active:
                self._realtime_collection_active = False
                if self._realtime_collection_worker:
                    self._realtime_collection_worker.join(timeout=1)
                    self._realtime_collection_worker = None
                self.realtime_status_var.set("Status: Stopped")
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.status_var.set("Realtime collection stopped")
        except Exception as e:
            self.logger.error(f"Error stopping collection: {e}")
            messagebox.showerror("Error", str(e))

    def _realtime_collection_loop(self):
        """Background worker for realtime data collection."""
        while self._realtime_collection_active:
            try:
                # Get data for default tickers
                tickers = ['AAPL', 'MSFT', 'GOOG', 'TSLA']
                for ticker in tickers:
                    data = self.collector.get_realtime_data(ticker)
                    if data is not None:
                        self.collector.save_ticker_data(ticker, data)
                        self.logger.info(f"Updated {ticker}")
                time.sleep(60)  # Wait 1 minute
            except Exception as e:
                self.logger.error(f"Collection error: {e}")
                self.window.after(0, self.stop_realtime_collection)
                break

    def on_closing(self):
        """Handle application closing."""
        try:
            if self._realtime_collection_active:
                self.stop_realtime_collection()
            if hasattr(self, 'collector'):
                self.collector.close()
            self.window.destroy()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def run(self):
        """Start the GUI application."""
        self.window.mainloop()

if __name__ == "__main__":
    config = load_config()  # Your config loading function
    app = DataCollectorGUI(config)
    app.run() 