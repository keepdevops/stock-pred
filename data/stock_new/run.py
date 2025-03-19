import sys
import os
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import json
import threading
import pandas as pd
from datetime import datetime, timedelta

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create necessary directories
log_dir = project_root / "logs"
data_dir = project_root / "data"
config_dir = project_root / "config"
log_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)
config_dir.mkdir(exist_ok=True)

from config.config_manager import ConfigurationManager
from modules.database import DatabaseConnector
from modules.data_loader import DataLoader
from src.utils.data_transformer import YFinanceTransformer

class DataCollectorGUI:
    def __init__(self, root: tk.Tk, config_manager):
        self.root = root
        self.config = config_manager
        self.logger = logging.getLogger("DataCollector")
        
        # Initialize variables
        self.current_db_path = None
        self.db_path_var = tk.StringVar(value="No database loaded")
        self.selected_ticker = tk.StringVar()
        self.realtime_enabled = tk.BooleanVar(value=False)
        self._stop_realtime = False
        
        # Initialize database with cleanup
        try:
            self.db = DatabaseConnector(
                db_path=self.config.data_processing.database.path,
                logger=self.logger
            )
            
            # Set up cleanup on window close
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Verify table structure
            with self.db.conn.cursor() as cursor:
                cursor.execute("DESCRIBE stock_data")
                columns = cursor.fetchall()
                self.logger.info(f"Table structure: {columns}")
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize database: {str(e)}")
            raise
        
        # Initialize transformer
        self.transformer = YFinanceTransformer(self.logger)
        
        self.setup_gui()
        self.setup_periodic_check()
    
    def setup_periodic_check(self):
        """Setup periodic check for message queue."""
        self.root.after(100, self.check_message_queue)
    
    def check_message_queue(self):
        """Check for messages from worker threads."""
        # Implement message queue checking if needed
        self.root.after(100, self.check_message_queue)
    
    def setup_gui(self):
        """Initialize the GUI components."""
        self.root.title("Stock Market Data Collector")
        
        # Configure root window
        window_size = "800x600"
        self.root.geometry(window_size)
        
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Setup different sections
        self.setup_database_frame(self.main_frame)
        self.setup_status_frame(self.main_frame)
        self.setup_ticker_frame(self.main_frame)
        self.setup_control_frame(self.main_frame)
        self.setup_progress_frame(self.main_frame)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def setup_database_frame(self, parent):
        """Setup database management frame."""
        frame = ttk.LabelFrame(parent, text="Database Management", padding="5")
        frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Current database display
        ttk.Label(frame, text="Current Database:").grid(row=0, column=0, padx=5)
        ttk.Label(
            frame,
            textvariable=self.db_path_var,
            wraplength=300
        ).grid(row=0, column=1, columnspan=2, padx=5)
        
        # Database control buttons
        ttk.Button(
            frame,
            text="Open Database",
            command=self.open_database
        ).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(
            frame,
            text="Create New Database",
            command=self.create_database
        ).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(
            frame,
            text="Set as Default",
            command=self.set_default_database
        ).grid(row=1, column=2, padx=5, pady=5)
    
    def setup_status_frame(self, parent):
        """Setup the status display frame."""
        frame = ttk.LabelFrame(parent, text="Collection Status", padding="5")
        frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
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
        frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Ticker list
        self.ticker_list = tk.Listbox(frame, height=5, selectmode=tk.MULTIPLE)
        self.ticker_list.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        # Load current tickers
        for ticker in self.config.data_collection.tickers:
            self.ticker_list.insert(tk.END, ticker)
        
        # Ticker controls
        ttk.Button(
            frame,
            text="Add Ticker",
            command=self.add_ticker
        ).grid(row=1, column=0, padx=5)
        
        ttk.Button(
            frame,
            text="Remove Selected",
            command=self.remove_tickers
        ).grid(row=1, column=1, padx=5)
    
    def setup_control_frame(self, parent):
        """Setup the collection control frame."""
        frame = ttk.LabelFrame(parent, text="Collection Controls", padding="5")
        frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Historical controls
        ttk.Button(
            frame,
            text="Start Collection",
            command=self.start_historical_collection
        ).grid(row=0, column=0, padx=5)
        
        # Realtime controls
        ttk.Checkbutton(
            frame,
            text="Enable Realtime",
            variable=self.realtime_enabled,
            command=self.toggle_realtime_collection
        ).grid(row=0, column=1, padx=5)
        
        # Add CSV import controls
        csv_frame = ttk.Frame(frame)
        csv_frame.grid(row=0, column=2, columnspan=2, sticky="ew", padx=5, pady=5)
        
        ttk.Button(csv_frame, text="Import CSV", command=self.import_csv_data).grid(row=0, column=0, padx=2)
        ttk.Button(csv_frame, text="Download YFinance", command=self.download_yfinance_data).grid(row=0, column=1, padx=2)
    
    def setup_progress_frame(self, parent):
        """Setup the progress and log frame."""
        frame = ttk.LabelFrame(parent, text="Progress and Logs", padding="5")
        frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Log display
        self.log_text = tk.Text(frame, height=10, width=50)
        self.log_text.grid(row=1, column=0, sticky="ew", padx=5)
        
        # Scrollbar for logs
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)
    
    def open_database(self):
        """Open an existing database file."""
        try:
            file_types = [
                ("DuckDB files", "*.duckdb"),
                ("SQLite files", "*.db *.sqlite *.sqlite3"),
                ("All files", "*.*")
            ]
            
            db_path = filedialog.askopenfilename(
                title="Open Database",
                filetypes=file_types,
                initialdir=os.path.expanduser("~")
            )
            
            if not db_path:
                return
            
            # Update database path
            self.current_db_path = db_path
            self.db_path_var.set(f"Current: {Path(db_path).name}")
            
            # Reinitialize database connection
            self.initialize_database_connection()
            
            self.logger.info(f"Opened database: {db_path}")
            
        except Exception as e:
            self.logger.error(f"Error opening database: {str(e)}")
            messagebox.showerror("Error", f"Failed to open database: {str(e)}")
    
    def create_database(self):
        """Create a new database file."""
        try:
            file_types = [
                ("DuckDB files", "*.duckdb"),
                ("SQLite files", "*.db")
            ]
            
            db_path = filedialog.asksaveasfilename(
                title="Create Database",
                filetypes=file_types,
                initialdir=os.path.expanduser("~"),
                defaultextension=".duckdb"
            )
            
            if not db_path:
                return
            
            # Update database path
            self.current_db_path = db_path
            self.db_path_var.set(f"Current: {Path(db_path).name}")
            
            # Initialize new database
            self.initialize_database_connection(new_database=True)
            
            self.logger.info(f"Created database: {db_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating database: {str(e)}")
            messagebox.showerror("Error", f"Failed to create database: {str(e)}")
    
    def initialize_database_connection(self, new_database: bool = False):
        """Initialize or reinitialize database connection."""
        try:
            # Close existing connection if any
            if hasattr(self, 'db') and self.db:
                self.db.close()
            
            # Use config path if no current path
            if not self.current_db_path:
                self.current_db_path = self.config.data_processing.database.path
                self.db_path_var.set(f"Current: {Path(self.current_db_path).name}")
            
            # Create new database connection
            self.db = DatabaseConnector(
                db_path=self.current_db_path,
                logger=self.logger
            )
            
            # Initialize tables if new database
            if new_database:
                with self.db.conn.cursor() as cursor:
                    cursor.execute("DROP TABLE IF EXISTS stock_data")
                    cursor.execute("""
                        CREATE TABLE stock_data (
                            date TIMESTAMP NOT NULL,
                            ticker VARCHAR NOT NULL,
                            open DOUBLE,
                            high DOUBLE,
                            low DOUBLE,
                            close DOUBLE,
                            adj_close DOUBLE,
                            volume BIGINT,
                            PRIMARY KEY (date, ticker)
                        )
                    """)
                    self.db.conn.commit()
                    
                    # Verify the table structure
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = 'stock_data'
                        ORDER BY ordinal_position
                    """)
                    columns = cursor.fetchall()
                    self.logger.info("Created table structure:")
                    for col in columns:
                        self.logger.info(f"  {col}")
            
            # Initialize data loader
            self.data_loader = DataLoader(
                self.db,
                self.config.data_collection,
                self.logger
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def add_ticker(self):
        """Add a new ticker to the collection list."""
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
        self.config.data_collection.tickers = tickers
        self.config.save_configuration()
    
    def start_historical_collection(self):
        """Start historical data collection."""
        try:
            selected_indices = self.ticker_list.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "Please select at least one ticker")
                return
            
            selected_tickers = [self.ticker_list.get(idx) for idx in selected_indices]
            
            # Update status
            self.historical_status.config(text="Historical: Active")
            self.progress_var.set(0)
            
            # Start collection in a separate thread
            thread = threading.Thread(
                target=self._historical_collection_worker,
                args=(selected_tickers,)
            )
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting historical collection: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def _historical_collection_worker(self, tickers):
        """Worker function for historical data collection."""
        try:
            total = len(tickers)
            for i, ticker in enumerate(tickers, 1):
                self.log_message(f"Collecting historical data for {ticker}...")
                
                self.data_loader.collect_historical_data(ticker)
                
                progress = (i / total) * 100
                self.progress_var.set(progress)
            
            self.log_message("Historical data collection completed")
            self.historical_status.config(text="Historical: Inactive")
            
        except Exception as e:
            self.log_message(f"Error in historical collection: {str(e)}")
            self.historical_status.config(text="Historical: Error")
    
    def toggle_realtime_collection(self):
        """Toggle realtime data collection."""
        if self.realtime_enabled.get():
            self.start_realtime_collection()
        else:
            self.stop_realtime_collection()
    
    def start_realtime_collection(self):
        """Start realtime data collection."""
        try:
            selected_indices = self.ticker_list.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "Please select at least one ticker")
                self.realtime_enabled.set(False)
                return
            
            selected_tickers = [self.ticker_list.get(idx) for idx in selected_indices]
            
            # Update status
            self.realtime_status.config(text="Realtime: Active")
            
            # Start collection in a separate thread
            self._stop_realtime = False
            thread = threading.Thread(
                target=self._realtime_collection_worker,
                args=(selected_tickers,)
            )
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting realtime collection: {str(e)}")
            messagebox.showerror("Error", str(e))
            self.realtime_enabled.set(False)
    
    def stop_realtime_collection(self):
        """Stop realtime data collection."""
        self._stop_realtime = True
        self.realtime_status.config(text="Realtime: Inactive")
    
    def log_message(self, message: str):
        """Add message to log display."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def on_closing(self):
        """Handle cleanup when closing the application."""
        try:
            if hasattr(self, 'db'):
                self.db.close()
            self.root.destroy()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.root.destroy()
    
    def set_default_database(self):
        """Set the default database path from configuration."""
        try:
            default_path = self.config.data_processing.database.path
            if not default_path:
                default_path = os.path.join("data", "market_data.duckdb")
            
            # Ensure the data directory exists
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            
            self.current_db_path = default_path
            self.db_path_var.set(f"Current: {Path(default_path).name}")
            
            # Initialize database connection
            self.initialize_database_connection(new_database=not os.path.exists(default_path))
            
            self.logger.info(f"Set default database: {default_path}")
            
        except Exception as e:
            self.logger.error(f"Error setting default database: {str(e)}")
            messagebox.showerror("Error", f"Failed to set default database: {str(e)}")
    
    def import_csv_data(self):
        """Import and transform CSV data files."""
        try:
            file_paths = filedialog.askopenfilenames(
                title="Select CSV Files",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_paths:
                return
            
            total_files = len(file_paths)
            successful_imports = 0
            failed_imports = 0
            
            self.log_message(f"\nStarting import of {total_files} CSV files...")
            
            for file_path in file_paths:
                try:
                    file_name = Path(file_path).name
                    self.log_message(f"\nProcessing {file_name}...")
                    
                    # Transform CSV data
                    data = self.transformer.transform_csv_data(file_path)
                    
                    if data.empty:
                        self.log_message(f"No valid data found in {file_name}")
                        failed_imports += 1
                        continue
                    
                    # Log data info
                    self.log_message(f"Found {len(data)} rows of data")
                    self.log_message(f"Date range: {data['date'].min()} to {data['date'].max()}")
                    
                    # Save to database
                    self.db.save_ticker_data(data['ticker'].iloc[0], data)
                    
                    self.log_message(f"Successfully imported data for {data['ticker'].iloc[0]}")
                    successful_imports += 1
                    
                except Exception as e:
                    self.log_message(f"Error importing {Path(file_path).name}: {str(e)}")
                    failed_imports += 1
                    continue
            
            # Summary message
            self.log_message(f"\nImport Summary:")
            self.log_message(f"Total files processed: {total_files}")
            self.log_message(f"Successfully imported: {successful_imports}")
            self.log_message(f"Failed imports: {failed_imports}")
            
        except Exception as e:
            self.logger.error(f"Error during CSV import: {str(e)}")
            messagebox.showerror("Error", f"Failed to import CSV files: {str(e)}")
    
    def download_yfinance_data(self):
        """Download data from YFinance for selected tickers."""
        try:
            selected_indices = self.ticker_list.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "Please select at least one ticker")
                return
            
            # Get date range from user
            date_window = self.show_date_range_dialog()
            if not date_window:
                return
            
            start_date, end_date = date_window
            selected_tickers = [self.ticker_list.get(idx) for idx in selected_indices]
            
            self.log_message(f"\nDownloading data for {len(selected_tickers)} tickers...")
            
            for ticker in selected_tickers:
                try:
                    self.log_message(f"\nProcessing {ticker}...")
                    
                    # Download and transform data
                    data = self.transformer.download_yfinance_data(ticker, start_date, end_date)
                    
                    if data is None or data.empty:
                        self.log_message(f"No data found for {ticker}")
                        continue
                    
                    # Log data info
                    self.log_message(f"Found {len(data)} rows of data")
                    self.log_message(f"Date range: {data['date'].min()} to {data['date'].max()}")
                    
                    # Save to database
                    self.db.save_ticker_data(ticker, data)
                    
                    self.log_message(f"Successfully downloaded and saved data for {ticker}")
                    
                except Exception as e:
                    self.log_message(f"Error downloading {ticker}: {str(e)}")
                    self.logger.exception(f"Detailed error for {ticker}")  # This will log the full stack trace
                    continue
            
            self.log_message("\nYFinance download completed")
            
        except Exception as e:
            self.logger.error(f"Error during YFinance download: {str(e)}")
            self.logger.exception("Detailed error")  # This will log the full stack trace
            messagebox.showerror("Error", f"Failed to download data: {str(e)}")
    
    def show_date_range_dialog(self):
        """Show dialog for selecting date range."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Date Range")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Default dates
        default_end = datetime.now()
        default_start = default_end - timedelta(days=365)
        
        # Date variables
        start_var = tk.StringVar(value=default_start.strftime('%Y-%m-%d'))
        end_var = tk.StringVar(value=default_end.strftime('%Y-%m-%d'))
        
        # Create and pack widgets
        ttk.Label(dialog, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(dialog, textvariable=start_var).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="End Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(dialog, textvariable=end_var).grid(row=1, column=1, padx=5, pady=5)
        
        result = [None]
        
        def on_ok():
            try:
                start = start_var.get()
                end = end_var.get()
                # Validate dates
                datetime.strptime(start, '%Y-%m-%d')
                datetime.strptime(end, '%Y-%m-%d')
                result[0] = (start, end)
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(dialog, text="OK", command=on_ok).grid(row=2, column=0, pady=10)
        ttk.Button(dialog, text="Cancel", command=on_cancel).grid(row=2, column=1, pady=10)
        
        dialog.wait_window()
        return result[0]

def main():
    """Application entry point."""
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'data_collector.log')
            ]
        )
        
        # Initialize configuration
        config_manager = ConfigurationManager("config/data_collection.json")
        
        # Create and run GUI
        root = tk.Tk()
        app = DataCollectorGUI(root, config_manager)
        
        # Force recreate table with correct schema
        with app.db.conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS stock_data")
            cursor.execute("""
                CREATE TABLE stock_data (
                    date TIMESTAMP NOT NULL,
                    ticker VARCHAR NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (date, ticker)
                )
            """)
            app.db.conn.commit()
            
            # Verify the table structure
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'stock_data'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            app.logger.info("Recreated table structure:")
            for col in columns:
                app.logger.info(f"  {col}")
        
        root.protocol("WM_DELETE_WINDOW", lambda: (app.on_closing(), root.destroy()))
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        messagebox.showerror("Fatal Error", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main() 