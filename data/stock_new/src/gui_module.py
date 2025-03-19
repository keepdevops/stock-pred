import tkinter as tk
from tkinter import ttk, messagebox
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
import threading
from queue import Queue
import datetime
from src.market_normalizations import MarketNormalizer
from src.market_analysis import analyze_market_metrics
from src.help_definitions import MARKET_DEFINITIONS
from src.help_window import HelpWindow, HelpButton
from src.data_collector import DataCollector
from src.database_connector import DatabaseConnector

class DataCollectorGUI:
    def __init__(self, root: tk.Tk):
        self.logger = logging.getLogger(__name__)
        self.root = root
        self.root.title("Market Data Collector")
        
        # Initialize components
        self.initialize_components()
        
        # Create GUI elements
        self.create_gui()
        
        # Initialize queue for thread communication
        self.queue = Queue()
        
        # Set up proper window closing handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Flag to track if application is closing
        self.is_closing = False
        
        # Apply theme
        self._apply_theme()

    def initialize_components(self):
        """Initialize all required components"""
        try:
            self.data_collector = DataCollector()
            self.config = self._load_config()
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise

    def create_gui(self):
        """Create all GUI elements"""
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self._create_ticker_section()
        self._create_date_section()
        self._create_source_section()
        self._create_status_section()
        self._create_buttons()

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            with open("config.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "tickers": [],
                "realtime": False,
                "source": "yahoo",
                "combinations": []
            }

    def _save_config(self) -> None:
        """Save current configuration to file"""
        try:
            with open("config.json", 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            messagebox.showerror("Error", "Failed to save configuration")

    def _create_ticker_section(self) -> None:
        """Create ticker selection section"""
        ticker_frame = ttk.LabelFrame(self.main_frame, text="Ticker Selection", padding="5")
        ticker_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Ticker entry
        self.ticker_var = tk.StringVar(value=",".join(self.config.get("tickers", [])))
        ttk.Label(ticker_frame, text="Tickers (comma-separated):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(ticker_frame, textvariable=self.ticker_var, width=40).grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Preset buttons
        ttk.Button(ticker_frame, text="S&P 500", command=self._load_sp500).grid(row=2, column=0, sticky=tk.W)
        ttk.Button(ticker_frame, text="NASDAQ-100", command=self._load_nasdaq).grid(row=2, column=1, sticky=tk.W)

    def _create_date_section(self) -> None:
        """Create date selection section"""
        date_frame = ttk.LabelFrame(self.main_frame, text="Date Range", padding="5")
        date_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Date entries
        self.start_date = tk.StringVar(value=datetime.datetime.now().strftime("%Y-%m-%d"))
        self.end_date = tk.StringVar(value=datetime.datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Label(date_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(date_frame, textvariable=self.start_date).grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        ttk.Label(date_frame, text="End Date (YYYY-MM-DD):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(date_frame, textvariable=self.end_date).grid(row=1, column=1, sticky=(tk.W, tk.E))

    def _create_source_section(self) -> None:
        """Create data source selection section"""
        source_frame = ttk.LabelFrame(self.main_frame, text="Data Source", padding="5")
        source_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Source selection
        self.source_var = tk.StringVar(value=self.config.get("source", "yahoo"))
        ttk.Radiobutton(source_frame, text="Yahoo Finance", variable=self.source_var, value="yahoo").grid(row=0, column=0)
        ttk.Radiobutton(source_frame, text="Alpaca (WebSocket)", variable=self.source_var, value="alpaca").grid(row=0, column=1)

    def _create_status_section(self) -> None:
        """Create status display section"""
        status_frame = ttk.LabelFrame(self.main_frame, text="Status", padding="5")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.status_text = tk.Text(status_frame, height=5, width=50)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text['yscrollcommand'] = scrollbar.set

    def _create_buttons(self) -> None:
        """Create action buttons"""
        button_frame = ttk.Frame(self.main_frame, padding="5")
        button_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        ttk.Button(button_frame, text="Start Collection", command=self._start_collection).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Stop Collection", command=self._stop_collection).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self._save_config).grid(row=0, column=2, padx=5)

    def _apply_theme(self) -> None:
        """Apply custom theme to the GUI"""
        style = ttk.Style()
        style.configure("TLabel", padding=3)
        style.configure("TButton", padding=5)
        style.configure("TFrame", padding=5)
        style.configure("TLabelframe", padding=5)

    def collect_historical_data(self, ticker: str):
        """Collect historical data for a single ticker"""
        try:
            self.logger.info(f"Collecting historical data for {ticker}...")
            
            # Download data using data collector
            data = self.data_collector.download_ticker_data(
                ticker=ticker,
                start_date=self.start_date.get(),
                end_date=self.end_date.get()
            )
            
            if data is not None:
                # Save to database using the data collector
                success = self.data_collector.save_ticker_data(ticker, data)
                if success:
                    self.logger.info(f"Successfully collected and saved data for {ticker}")
                    self.update_status(f"Successfully collected data for {ticker}")
                else:
                    self.logger.error(f"Failed to save data for {ticker}")
                    self.update_status(f"Failed to save data for {ticker}")
            else:
                self.logger.error(f"No data received for {ticker}")
                self.update_status(f"Failed to collect data for {ticker}")
                
        except Exception as e:
            self.logger.error(f"Error in historical collection: {e}")
            self.update_status(f"Error collecting data for {ticker}: {e}")

    def on_closing(self):
        """Handle window closing properly"""
        try:
            if not self.is_closing:
                self.is_closing = True
                self.logger.info("Closing application...")
                
                # Stop any running threads
                self.stop_all_threads()
                
                # Clean up resources
                self.cleanup()
                
                # Destroy the window
                self.root.quit()
                self.root.destroy()
                
        except Exception as e:
            self.logger.error(f"Error during application shutdown: {e}")
            # Ensure window is destroyed even if there's an error
            try:
                self.root.destroy()
            except:
                pass

    def stop_all_threads(self):
        """Stop all running threads"""
        try:
            # Set flag to stop collection threads
            self.stop_collection = True
            
            # Wait for threads to finish (with timeout)
            if hasattr(self, 'collection_thread') and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=2.0)
                
        except Exception as e:
            self.logger.error(f"Error stopping threads: {e}")

    def cleanup(self):
        """Clean up resources before closing"""
        try:
            # Close database connections
            if hasattr(self, 'data_collector'):
                if hasattr(self.data_collector, 'db_connector'):
                    self.data_collector.db_connector.close()
            
            # Clear queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Queue.Empty:
                    break
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _start_collection(self) -> None:
        """Start data collection process"""
        try:
            if hasattr(self, 'stop_collection'):
                self.stop_collection = False
            
            # Get tickers
            tickers = [t.strip() for t in self.ticker_var.get().split(",") if t.strip()]
            if not tickers:
                messagebox.showerror("Error", "Please enter at least one ticker")
                return
            
            # Start collection in separate thread
            self.collection_thread = threading.Thread(
                target=self._collection_worker,
                args=(tickers,)
            )
            self.collection_thread.daemon = True
            self.collection_thread.start()
            
            # Start queue checker
            self.root.after(100, self._check_queue)
            
        except Exception as e:
            self.logger.error(f"Error starting collection: {e}")
            messagebox.showerror("Error", f"Failed to start collection: {e}")

    def _collection_worker(self, tickers: List[str]) -> None:
        """Worker function for data collection"""
        try:
            self.queue.put(("status", f"Starting collection for {len(tickers)} tickers..."))
            
            for ticker in tickers:
                # Check if collection should stop
                if hasattr(self, 'stop_collection') and self.stop_collection:
                    self.queue.put(("status", "Collection stopped by user"))
                    break
                    
                try:
                    self.queue.put(("status", f"Processing {ticker}..."))
                    self.collect_historical_data(ticker)
                except Exception as e:
                    self.queue.put(("error", f"Error processing {ticker}: {e}"))
            
            self.queue.put(("status", "Collection completed"))
            
        except Exception as e:
            self.queue.put(("error", f"Collection error: {e}"))

    def update_status(self, message: str):
        """Update status in GUI"""
        self.queue.put(("status", message))

    def _check_queue(self) -> None:
        """Check queue for messages from worker thread"""
        try:
            while True:
                msg_type, message = self.queue.get_nowait()
                if msg_type == "status":
                    self.status_text.insert(tk.END, f"{message}\n")
                    self.status_text.see(tk.END)
                elif msg_type == "error":
                    messagebox.showerror("Error", message)
                self.queue.task_done()
        except Queue.Empty:
            pass
        finally:
            self.root.after(100, self._check_queue)

    def _stop_collection(self) -> None:
        """Stop data collection process"""
        # Add stop logic here
        self.status_text.insert(tk.END, "Stopping collection...\n")
        self.status_text.see(tk.END)

    def _load_sp500(self) -> None:
        """Load S&P 500 tickers"""
        # Add S&P 500 ticker loading logic here
        pass

    def _load_nasdaq(self) -> None:
        """Load NASDAQ-100 tickers"""
        # Add NASDAQ-100 ticker loading logic here
        pass

class MarketAnalysisTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.normalizer = MarketNormalizer()
        self._create_widgets()
        
    def _create_widgets(self):
        # Analysis type selection
        analysis_frame = ttk.LabelFrame(self, text="Analysis Configuration")
        analysis_frame.pack(fill="x", padx=5, pady=5)
        
        # Returns analysis row
        returns_frame = ttk.Frame(analysis_frame)
        returns_frame.pack(fill="x", padx=5, pady=2)
        
        self.returns_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            returns_frame,
            text="Returns Analysis",
            variable=self.returns_var
        ).pack(side=tk.LEFT)
        
        # Add help buttons for each metric
        HelpButton(
            returns_frame,
            "Simple Returns",
            MARKET_DEFINITIONS["returns"]["simple_return"]
        ).pack(side=tk.LEFT, padx=2)
        
        HelpButton(
            returns_frame,
            "Log Returns",
            MARKET_DEFINITIONS["returns"]["log_return"]
        ).pack(side=tk.LEFT, padx=2)
        
        HelpButton(
            returns_frame,
            "Excess Returns",
            MARKET_DEFINITIONS["returns"]["excess_return"]
        ).pack(side=tk.LEFT, padx=2)
        
        # Volume analysis row
        volume_frame = ttk.Frame(analysis_frame)
        volume_frame.pack(fill="x", padx=5, pady=2)
        
        self.volume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            volume_frame,
            text="Volume Analysis",
            variable=self.volume_var
        ).pack(side=tk.LEFT)
        
        HelpButton(
            volume_frame,
            "VWAP",
            MARKET_DEFINITIONS["volume"]["vwap"]
        ).pack(side=tk.LEFT, padx=2)
        
        HelpButton(
            volume_frame,
            "Relative Volume",
            MARKET_DEFINITIONS["volume"]["relative_volume"]
        ).pack(side=tk.LEFT, padx=2)
        
        # Add similar rows for other metrics...
        
        # Results section with help
        results_frame = ttk.LabelFrame(self, text="Analysis Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add help button for interpreting results
        results_help = HelpButton(
            results_frame,
            "Interpreting Results",
            {
                "title": "Analysis Results Guide",
                "formula": "N/A",
                "description": "This section shows the calculated metrics and their relationships.",
                "interpretation": "• Green/Red highlighting indicates positive/negative values\n"
                                "• Bold values indicate statistical significance\n"
                                "• Correlations > 0.7 are considered strong"
            }
        )
        results_help.pack(anchor="ne", padx=5, pady=5)
        
        self.results_text = tk.Text(results_frame, height=20, width=50)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(
            button_frame,
            text="Run Analysis",
            command=self._run_analysis
        ).pack(side=tk.LEFT, padx=5)
        
        # Add help button for analysis process
        HelpButton(
            button_frame,
            "Analysis Process",
            {
                "title": "Market Analysis Process",
                "formula": "Multiple metrics and normalizations",
                "description": "Performs comprehensive market analysis using selected metrics.",
                "interpretation": "• Calculates all selected metrics\n"
                                "• Normalizes data for comparison\n"
                                "• Generates correlation matrix\n"
                                "• Highlights significant patterns"
            }
        ).pack(side=tk.LEFT)
        
    def _run_analysis(self):
        try:
            ticker = self.master.ticker_var.get()
            start_date = self.master.start_date.get()
            end_date = self.master.end_date.get()
            
            results = analyze_market_metrics(ticker, start_date, end_date)
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Analysis Results:\n\n")
            
            # Format and display summary statistics
            self.results_text.insert(tk.END, "Summary Statistics:\n")
            for category, stats in results["summary_stats"].items():
                self.results_text.insert(tk.END, f"\n{category.title()}:\n")
                for metric, value in stats.items():
                    self.results_text.insert(tk.END, f"  {metric}: {value:.4f}\n")
            
            # Display correlation highlights
            self.results_text.insert(tk.END, "\nKey Correlations:\n")
            corr_matrix = results["correlation_matrix"]
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    if col1 < col2:
                        corr = corr_matrix[col1][col2]
                        if abs(corr) > 0.7:  # Show strong correlations
                            self.results_text.insert(
                                tk.END, 
                                f"  {col1} vs {col2}: {corr:.2f}\n"
                            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")

def main():
    root = tk.Tk()
    app = DataCollectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 