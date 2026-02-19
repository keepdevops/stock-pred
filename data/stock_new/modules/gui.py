import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path
import os
import pandas as pd
import duckdb
import sqlite3

from config.config_manager import ConfigurationManager
from modules.database import DatabaseConnector
from modules.data_loader import DataLoader
from modules.stock_ai_agent import StockAIAgent
from modules.trading.real_trading_agent import RealTradingAgent
from utils.visualization import StockVisualizer

class StockGUI:
    """Main GUI interface for the Stock Market Analyzer."""
    
    def __init__(
        self,
        root: tk.Tk,
        db: DatabaseConnector,
        data_loader: DataLoader,
        ai_agent: StockAIAgent,
        trading_agent: RealTradingAgent,
        config_manager: ConfigurationManager
    ):
        self.root = root
        self.db = db
        self.data_loader = data_loader
        self.ai_agent = ai_agent
        self.trading_agent = trading_agent
        self.config_manager = config_manager
        self.logger = logging.getLogger("GUI")
        
        self.visualizer = StockVisualizer()
        
        # Initialize variables
        self.selected_ticker = tk.StringVar()
        self.model_type = tk.StringVar(value="LSTM")
        self.trading_enabled = tk.BooleanVar(value=False)
        self.realtime_enabled = tk.BooleanVar(value=False)
        
        # Add database path tracking
        self.current_db_path = None
        
        # Store available tickers
        self.available_tickers = []
        self.nasdaq_data = None
        self.db_tickers = set()  # Store tickers from database
        
        # Add database info
        self.data_dir = Path("data")
        self.databases = {}  # Store database connections
        
        # Initialize GUI components first
        self._setup_gui()
        
        # Then load the data
        self._scan_databases()
        self._load_nasdaq_data()
        self._populate_ticker_list()
    
    def _setup_gui(self):
        """Setup GUI components."""
        # Main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for database and ticker selection
        self.left_frame = ttk.Frame(self.main_container)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # Database Section
        self.db_frame = ttk.LabelFrame(self.left_frame, text="Available Databases")
        self.db_frame.pack(fill=tk.X, padx=5, pady=5)

        # Database Treeview
        self.db_tree = ttk.Treeview(self.db_frame, height=4)
        self.db_tree["columns"] = ("Type", "Tables")
        self.db_tree.column("#0", width=150)
        self.db_tree.column("Type", width=70)
        self.db_tree.column("Tables", width=100)
        
        self.db_tree.heading("#0", text="Database")
        self.db_tree.heading("Type", text="Type")
        self.db_tree.heading("Tables", text="Tables")
        
        self.db_tree.pack(fill=tk.X, padx=5, pady=5)
        self.db_tree.bind('<<TreeviewSelect>>', self.on_database_select)

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

        # Right frame for analysis
        self.right_frame = ttk.Frame(self.main_container)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Analysis section
        self.analysis_frame = ttk.LabelFrame(self.right_frame, text="Analysis")
        self.analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Button frame
        self.button_frame = ttk.Frame(self.right_frame)
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.analyze_button = ttk.Button(
            self.button_frame,
            text="Analyze Stock",
            command=self.analyze_stock
        )
        self.analyze_button.pack(side=tk.LEFT, padx=5)
    
    def _scan_databases(self):
        """Scan for databases in the data directory."""
        try:
            self.status_var.set("Scanning databases...")
            self.root.update()

            # Clear existing database info
            self.databases.clear()
            
            # Scan for DuckDB and SQLite databases
            for file in self.data_dir.glob("*.duckdb"):
                try:
                    conn = duckdb.connect(str(file))
                    # Get table names
                    tables = conn.execute("SHOW TABLES").fetchdf()
                    self.databases[file.name] = {
                        'type': 'duckdb',
                        'path': file,
                        'tables': tables['name'].tolist() if not tables.empty else [],
                        'connection': conn
                    }
                except Exception as e:
                    self.logger.error(f"Error connecting to DuckDB {file}: {e}")

            for file in self.data_dir.glob("*.db"):
                try:
                    conn = sqlite3.connect(str(file))
                    cursor = conn.cursor()
                    # Get table names
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    self.databases[file.name] = {
                        'type': 'sqlite',
                        'path': file,
                        'tables': tables,
                        'connection': conn
                    }
                except Exception as e:
                    self.logger.error(f"Error connecting to SQLite {file}: {e}")

            self._update_database_display()
            self.logger.info(f"Found {len(self.databases)} databases")
            
        except Exception as e:
            error_msg = f"Error scanning databases: {e}"
            self.logger.error(error_msg)
            self.status_var.set("Error scanning databases")
            messagebox.showerror("Error", error_msg)

    def _update_database_display(self):
        """Update the database treeview."""
        # Clear existing items
        for item in self.db_tree.get_children():
            self.db_tree.delete(item)
            
        # Add databases to treeview
        for db_name, db_info in self.databases.items():
            table_count = len(db_info['tables'])
            self.db_tree.insert(
                "",
                "end",
                text=db_name,
                values=(
                    db_info['type'],
                    f"{table_count} tables"
                )
            )

    def on_database_select(self, event):
        """Handle database selection."""
        selection = self.db_tree.selection()
        if selection:
            db_name = self.db_tree.item(selection[0])['text']
            db_info = self.databases.get(db_name)
            
            if db_info:
                details = (
                    f"Database: {db_name}\n"
                    f"Type: {db_info['type']}\n"
                    f"Tables: {', '.join(db_info['tables'])}\n"
                    f"Path: {db_info['path']}"
                )
                
                self.details_text.config(state=tk.NORMAL)
                self.details_text.delete(1.0, tk.END)
                self.details_text.insert(tk.END, details)
                self.details_text.config(state=tk.DISABLED)

    def _load_nasdaq_data(self):
        """Load data from NASDAQ screener CSV file. Uses defaults if file is missing."""
        try:
            self.status_var.set("Loading NASDAQ data...")
            self.root.update()

            # Look in current dir and common data dirs
            nasdaq_files = (
                list(Path(".").glob("nasdaq_screener_*.csv"))
                or list(Path("data").glob("nasdaq_screener_*.csv"))
            )
            if not nasdaq_files:
                self.logger.warning(
                    "NASDAQ screener CSV not found. Use a few default tickers; "
                    "run 'python src/data/download_nasdaq.py' from stock_new to download."
                )
                self.status_var.set(
                    "No NASDAQ screener file—using default tickers. "
                    "Run download_nasdaq.py to get full list."
                )
                # Default tickers so the app is usable without the CSV
                self.available_tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
                self.nasdaq_data = None
                return

            # Read the CSV file (use most recent if multiple)
            latest = max(nasdaq_files, key=lambda p: p.stat().st_mtime)
            self.nasdaq_data = pd.read_csv(latest)

            # Ensure Symbol column is string type
            if "Symbol" in self.nasdaq_data.columns:
                self.nasdaq_data["Symbol"] = self.nasdaq_data["Symbol"].astype(str)
                self.available_tickers = sorted(
                    self.nasdaq_data["Symbol"].unique().tolist(), key=str
                )
            else:
                self.available_tickers = []
                self.logger.warning("NASDAQ CSV has no 'Symbol' column")

            self.logger.info(f"Loaded {len(self.available_tickers)} NASDAQ tickers from {latest.name}")
            self.status_var.set(f"Loaded {len(self.available_tickers)} NASDAQ tickers")

        except Exception as e:
            self.logger.error(f"Error loading NASDAQ data: {e}")
            self.status_var.set("Error loading NASDAQ data—using default tickers")
            self.available_tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
            self.nasdaq_data = None

    def _populate_ticker_list(self):
        """Populate the ticker listbox with loaded symbols."""
        try:
            if self.available_tickers:
                self.ticker_listbox.delete(0, tk.END)
                for ticker in self.available_tickers:
                    # Add an indicator if the ticker is in the database
                    if ticker in self.db_tickers:
                        self.ticker_listbox.insert(tk.END, f"* {ticker}")
                    else:
                        self.ticker_listbox.insert(tk.END, ticker)
                
                total_tickers = len(self.available_tickers)
                db_ticker_count = len(self.db_tickers)
                self.status_var.set(
                    f"Loaded {total_tickers} NASDAQ tickers "
                    f"({db_ticker_count} in database)"
                )
        except Exception as e:
            self.logger.error(f"Error populating ticker list: {e}")

    def filter_tickers(self, *args):
        """Filter tickers based on search text."""
        search_text = self.search_var.get().upper()
        self.ticker_listbox.delete(0, tk.END)
        
        filtered_tickers = [ticker for ticker in self.available_tickers if search_text in ticker.upper()]
        for ticker in filtered_tickers:
            # Maintain the database indicator when filtering
            if ticker in self.db_tickers:
                self.ticker_listbox.insert(tk.END, f"* {ticker}")
            else:
                self.ticker_listbox.insert(tk.END, ticker)

    def on_ticker_select(self, event):
        """Handle ticker selection event."""
        try:
            selected = self.get_selected_tickers()
            self.selection_label.config(text=f"Selected: {len(selected)}")
            
            if selected:
                last_selected = selected[-1]
                
                # Get ticker details from NASDAQ data
                if self.nasdaq_data is not None:
                    info = self.nasdaq_data[self.nasdaq_data['Symbol'] == last_selected]
                    
                    if not info.empty:
                        info = info.iloc[0]
                        
                        # Check if ticker exists in database
                        in_database = last_selected in self.db_tickers
                        
                        # Get additional info from database if available
                        db_info = ""
                        if in_database:
                            try:
                                query = """
                                SELECT 
                                    MIN(date) as first_date,
                                    MAX(date) as last_date,
                                    COUNT(*) as records
                                FROM stock_data
                                WHERE ticker = ?
                                """
                                result = self.db.execute_query(query, [last_selected])
                                if not result.empty:
                                    db_info = (
                                        f"\nData Available: Yes\n"
                                        f"Period: {result.iloc[0]['first_date']} to {result.iloc[0]['last_date']}\n"
                                        f"Records: {result.iloc[0]['records']}"
                                    )
                            except Exception as e:
                                self.logger.error(f"Error getting database info: {e}")
                                db_info = "\nData Available: Yes (details unavailable)"
                        else:
                            db_info = "\nData Available: No"
                        
                        details = (
                            f"Symbol: {info.get('Symbol', 'N/A')}\n"
                            f"Name: {info.get('Name', 'N/A')}\n"
                            f"Sector: {info.get('Sector', 'N/A')}\n"
                            f"Industry: {info.get('Industry', 'N/A')}"
                            f"{db_info}"
                        )
                        
                        self.details_text.config(state=tk.NORMAL)
                        self.details_text.delete(1.0, tk.END)
                        self.details_text.insert(tk.END, details)
                        self.details_text.config(state=tk.DISABLED)
                    
        except Exception as e:
            self.logger.error(f"Error updating selection: {e}")

    def get_selected_tickers(self):
        """Get list of selected tickers."""
        selected = []
        for idx in self.ticker_listbox.curselection():
            ticker = self.ticker_listbox.get(idx)
            # Remove the database indicator if present
            if ticker.startswith('* '):
                ticker = ticker[2:]
            selected.append(ticker)
        return selected
    
    def setup_data_section(self):
        """Setup data management section with database loader."""
        frame = ttk.LabelFrame(self.main_frame, text="Data Management", padding="5")
        frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Database controls
        db_frame = ttk.Frame(frame)
        db_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(db_frame, text="Database:").grid(row=0, column=0, padx=5)
        self.db_path_var = tk.StringVar(value="No database loaded")
        ttk.Label(
            db_frame,
            textvariable=self.db_path_var,
            wraplength=300
        ).grid(row=0, column=1, padx=5)
        
        # Database buttons
        btn_frame = ttk.Frame(db_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(
            btn_frame,
            text="Open Database",
            command=self.open_database
        ).grid(row=0, column=0, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Create New Database",
            command=self.create_database
        ).grid(row=0, column=1, padx=5)
        
        # Existing data controls
        control_frame = ttk.Frame(frame)
        control_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Button(
            control_frame,
            text="Load Historical",
            command=self.load_historical_data
        ).grid(row=0, column=0, padx=5)
        
        ttk.Checkbutton(
            control_frame,
            text="Enable Realtime",
            variable=self.realtime_enabled,
            command=self.toggle_realtime
        ).grid(row=0, column=1, padx=5)
    
    def setup_analysis_section(self):
        """Setup analysis section."""
        frame = ttk.LabelFrame(self.main_frame, text="Analysis", padding="5")
        frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Model selection
        ttk.Label(frame, text="Model:").grid(row=0, column=0, padx=5)
        model_combo = ttk.Combobox(
            frame,
            textvariable=self.model_type,
            values=["LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Transformer"]
        )
        model_combo.grid(row=0, column=1, padx=5)
        
        # Analysis buttons
        ttk.Button(
            frame,
            text="Train Model",
            command=self.train_model
        ).grid(row=0, column=2, padx=5)
        
        ttk.Button(
            frame,
            text="Make Prediction",
            command=self.make_prediction
        ).grid(row=0, column=3, padx=5)
    
    def setup_trading_section(self):
        """Setup trading section."""
        frame = ttk.LabelFrame(self.main_frame, text="Trading", padding="5")
        frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Trading controls
        ttk.Checkbutton(
            frame,
            text="Enable Trading",
            variable=self.trading_enabled,
            command=self.toggle_trading
        ).grid(row=0, column=0, padx=5)
        
        ttk.Button(
            frame,
            text="View Positions",
            command=self.view_positions
        ).grid(row=0, column=1, padx=5)
        
        ttk.Button(
            frame,
            text="Trading History",
            command=self.view_trading_history
        ).grid(row=0, column=2, padx=5)
    
    def setup_status_bar(self):
        """Setup status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        status_bar.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
    
    # Event handlers
    def refresh_data(self):
        """Refresh data for selected ticker."""
        ticker = self.selected_ticker.get()
        if not ticker:
            messagebox.showwarning("Warning", "Please select a ticker")
            return
        
        try:
            self.status_var.set(f"Refreshing data for {ticker}...")
            self.data_loader.refresh_ticker_data(ticker)
            self.status_var.set("Data refresh complete")
        except Exception as e:
            self.logger.error(f"Error refreshing data: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh data: {str(e)}")
            self.status_var.set("Ready")
    
    def load_historical_data(self):
        """Load historical data for selected ticker and save to database."""
        ticker = (self.selected_ticker.get() or "").strip()
        if not ticker:
            messagebox.showwarning("Warning", "Please select a ticker")
            return
        ticker = ticker.lstrip("* ")  # listbox may show "* AAPL" for tickers in DB

        try:
            self.status_var.set(f"Loading historical data for {ticker}...")
            self.root.update()
            df = self.data_loader.collect_historical_data(ticker)
            if df is not None and not df.empty:
                self.db.save_ticker_data(ticker, df)
                self.status_var.set(f"Loaded {len(df)} rows for {ticker}")
                messagebox.showinfo("Done", f"Historical data saved for {ticker} ({len(df)} rows).")
            else:
                self.status_var.set("Ready")
                messagebox.showwarning("No data", f"No historical data returned for {ticker}.")
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load historical data: {str(e)}")
            self.status_var.set("Ready")
    
    def toggle_realtime(self):
        """Toggle realtime data collection."""
        if self.realtime_enabled.get():
            self.start_realtime_collection()
        else:
            self.stop_realtime_collection()
    
    def train_model(self):
        """Train model for selected ticker."""
        ticker = self.selected_ticker.get()
        if not ticker:
            messagebox.showwarning("Warning", "Please select a ticker")
            return
        
        try:
            self.status_var.set(f"Training model for {ticker}...")
            metrics = self.ai_agent.train_model(
                ticker,
                model_type=self.model_type.get()
            )
            self.status_var.set("Model training complete")
            messagebox.showinfo("Training Complete", f"Training metrics: {metrics}")
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.status_var.set("Ready")
    
    def make_prediction(self):
        """Make prediction for selected ticker."""
        ticker = self.selected_ticker.get()
        if not ticker:
            messagebox.showwarning("Warning", "Please select a ticker")
            return
        
        try:
            self.status_var.set(f"Making prediction for {ticker}...")
            predictions = self.ai_agent.make_prediction(ticker)
            self.status_var.set("Prediction complete")
            messagebox.showinfo("Prediction", f"Predictions: {predictions}")
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            messagebox.showerror("Error", f"Failed to make prediction: {str(e)}")
            self.status_var.set("Ready")
    
    def toggle_trading(self):
        """Toggle trading system."""
        if self.trading_enabled.get():
            self.trading_agent.start_trading()
            self.status_var.set("Trading system started")
        else:
            self.trading_agent.stop_trading()
            self.status_var.set("Trading system stopped")
    
    def view_positions(self):
        """View current trading positions."""
        try:
            positions = self.trading_agent.get_position_summary()
            if positions.empty:
                messagebox.showinfo("Positions", "No open positions")
            else:
                # Create a new window to display positions
                self.show_dataframe(positions, "Current Positions")
        except Exception as e:
            self.logger.error(f"Error viewing positions: {str(e)}")
            messagebox.showerror("Error", f"Failed to view positions: {str(e)}")
    
    def view_trading_history(self):
        """View trading history."""
        try:
            history = self.trading_agent.get_trade_history()
            if history.empty:
                messagebox.showinfo("History", "No trading history")
            else:
                # Create a new window to display history
                self.show_dataframe(history, "Trading History")
        except Exception as e:
            self.logger.error(f"Error viewing history: {str(e)}")
            messagebox.showerror("Error", f"Failed to view history: {str(e)}")
    
    def show_dataframe(self, df, title):
        """Display a DataFrame in a new window."""
        window = tk.Toplevel(self.root)
        window.title(title)
        
        # Create treeview
        tree = ttk.Treeview(window)
        tree["columns"] = list(df.columns)
        tree["show"] = "headings"
        
        # Set column headings
        for column in df.columns:
            tree.heading(column, text=column)
            tree.column(column, width=100)
        
        # Add data
        for i, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def open_database(self):
        """Open an existing database file."""
        try:
            # Ask for database file
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
            
            # Validate database
            if self.validate_database(db_path):
                self.load_database(db_path)
            else:
                messagebox.showerror(
                    "Error",
                    "Invalid database format or missing required tables"
                )
            
        except Exception as e:
            self.logger.error(f"Error opening database: {str(e)}")
            messagebox.showerror("Error", f"Failed to open database: {str(e)}")
    
    def create_database(self):
        """Create a new database file."""
        try:
            # Ask for save location
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
            
            # Create and initialize database
            self.load_database(db_path, create_new=True)
            
        except Exception as e:
            self.logger.error(f"Error creating database: {str(e)}")
            messagebox.showerror("Error", f"Failed to create database: {str(e)}")
    
    def validate_database(self, db_path: str) -> bool:
        """Validate database structure."""
        try:
            # Create temporary connection to validate
            temp_db = DatabaseConnector(
                db_path=db_path,
                logger=self.logger
            )
            
            # Check for required tables
            required_tables = {"stock_data", "predictions"}
            existing_tables = set(temp_db.get_tables())
            
            temp_db.close()
            
            return required_tables.issubset(existing_tables)
            
        except Exception as e:
            self.logger.error(f"Error validating database: {str(e)}")
            return False
    
    def load_database(self, db_path: str, create_new: bool = False) -> None:
        """Load or create a database."""
        try:
            # Close existing connection if any
            if hasattr(self, 'db') and self.db:
                self.db.close()
            
            # Create new database connection
            self.db = DatabaseConnector(
                db_path=db_path,
                logger=logging.getLogger("Database")
            )
            
            # Initialize if new
            if create_new:
                self.db.initialize_tables()
            
            # Update components with new database
            self.data_loader.db = self.db
            self.ai_agent.db = self.db
            self.trading_agent.db = self.db
            
            # Update display
            self.current_db_path = db_path
            self.db_path_var.set(f"Current: {Path(db_path).name}")
            
            # Update status
            self.status_var.set(f"{'Created' if create_new else 'Loaded'} database: {Path(db_path).name}")
            
            # Refresh data display if any
            self.refresh_data()
            
        except Exception as e:
            self.logger.error(f"Error loading database: {str(e)}")
            messagebox.showerror("Error", f"Failed to load database: {str(e)}")

    def cleanup(self):
        """Clean up database connections."""
        for db_info in self.databases.values():
            try:
                db_info['connection'].close()
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")

    def analyze_stock(self):
        """Analyze selected stock using ML models."""
        try:
            selected = self.ticker_listbox.curselection()
            if not selected:
                messagebox.showwarning("Warning", "Please select a ticker first")
                return
            
            ticker = self.ticker_listbox.get(selected[0])
            
            # Show progress
            self.status_var.set(f"Analyzing {ticker}...")
            self.root.update()
            
            # Run analysis
            predictions = self.ai_agent.analyze_stock(ticker)

            if predictions is not None:
                # Show results
                plot_path = f"data/plots/{ticker}_prediction.png"
                if os.path.exists(plot_path):
                    self._show_plot(plot_path)
                messagebox.showinfo(
                    "Analysis Complete",
                    f"Analysis completed for {ticker}\n"
                    f"Predicted price movement: "
                    f"{'Up' if predictions[-1] > predictions[0] else 'Down'}"
                )
            else:
                messagebox.showinfo(
                    "No data",
                    f"No price data for {ticker}. Select the ticker and click "
                    "'Load Historical' to download data, then try Analyze again."
                )
            self.status_var.set("Ready")
            
        except Exception as e:
            self.logger.error(f"Error in analyze_stock: {e}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Ready")

    def _show_plot(self, plot_path):
        """Show a plot in a new window."""
        try:
            # Implement the logic to display the plot
            # This is a placeholder and should be replaced with the actual implementation
            print(f"Plot path: {plot_path}")
        except Exception as e:
            self.logger.error(f"Error showing plot: {e}")
            messagebox.showerror("Error", f"Failed to show plot: {str(e)}")
            self.status_var.set("Ready") 