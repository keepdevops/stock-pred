import yfinance as yf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sqlite3
import duckdb  # Add DuckDB import
import os  # For file operations
import polars as pl

class StockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Viewer")
        
        # Initialize database paths
        self.sqlite_path = 'stock.db'
        self.duckdb_dir = 'duckdb_data'  # Directory for DuckDB files
        self.current_db = 'sqlite'  # Default to SQLite
        
        # Initialize current_duck_conn
        self.current_duck_conn = None
        
        # Ensure DuckDB directory exists
        os.makedirs(self.duckdb_dir, exist_ok=True)
        
        # Initialize databases
        self.init_sqlite_database()
        self.init_duckdb_connection()
        
        # Configure grid weights to allow resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Create left control panel frame
        control_frame = ttk.Frame(root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create database selection frame at the top of control frame
        db_frame = ttk.LabelFrame(control_frame, text="Database Selection", padding="5")
        db_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Add DuckDB database dropdown
        ttk.Label(db_frame, text="DuckDB Database:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.duckdb_var = tk.StringVar()
        self.duckdb_dropdown = ttk.Combobox(db_frame, textvariable=self.duckdb_var, width=30)
        self.duckdb_dropdown.grid(row=0, column=1, padx=5, pady=5)
        
        # Add database control buttons
        button_frame = ttk.Frame(db_frame)
        button_frame.grid(row=0, column=2, padx=5)
        
        ttk.Button(button_frame, text="Open DB", command=self.open_duckdb).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Refresh", command=self.refresh_duckdb_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="New DB", command=self.create_new_db).pack(side=tk.LEFT, padx=2)
        
        # Add Query Database button to database frame
        ttk.Button(button_frame, text="Query DB", 
                  command=self.query_local_database).pack(side=tk.LEFT, padx=2)
        
        # Initialize the dropdown list
        self.refresh_duckdb_list()
        
        # Add dropdown change handler
        self.duckdb_dropdown.bind('<<ComboboxSelected>>', self.on_database_change)
        
        # Database selector
        self.db_var = tk.StringVar(value='sqlite')
        self.db_dropdown = ttk.Combobox(db_frame, textvariable=self.db_var, width=20)
        self.db_dropdown.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Add database buttons
        ttk.Button(db_frame, text="New DB", command=self.create_new_db).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(db_frame, text="Refresh", command=self.update_db_list).grid(row=2, column=1, padx=5, pady=5)
        
        # Update database list
        self.update_db_list()
        
        # Create input frame (now after db_frame)
        input_frame = ttk.Frame(control_frame, padding="5")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create menu bar
        self.menubar = tk.Menu(root)
        self.root.config(menu=self.menubar)
        
        # Create File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.open_file)
        self.file_menu.add_command(label="Save", command=self.save_file)
        self.file_menu.add_command(label="Save As", command=self.save_as_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.quit)
        
        # Create Help menu
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Technical Indicators", command=self.show_indicator_help)
        self.help_menu.add_command(label="Trading Signals", command=self.show_signal_help)
        self.help_menu.add_command(label="Band Settings", command=self.show_band_help)
        self.help_menu.add_command(label="About", command=self.show_about)
        self.help_menu.add_command(label="Day Trading", command=self.show_daytrading_help)
        self.help_menu.add_command(label="Margin Trading", command=self.show_margin_help)
        
        # Create Trading Strategies menu
        self.strategy_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Trading Strategies", menu=self.strategy_menu)
        
        # Add day trading strategy submenus
        self.day_menu = tk.Menu(self.strategy_menu, tearoff=0)
        self.strategy_menu.add_cascade(label="Day Trading", menu=self.day_menu)
        
        # Add specific timeframe strategies
        self.day_menu.add_command(label="1-Day Scalping", command=lambda: self.apply_strategy("1d_scalp"))
        self.day_menu.add_command(label="2-Day Momentum", command=lambda: self.apply_strategy("2d_momentum"))
        self.day_menu.add_command(label="3-Day Swing", command=lambda: self.apply_strategy("3d_swing"))
        self.day_menu.add_command(label="5-Day Trend", command=lambda: self.apply_strategy("5d_trend"))
        
        # Add strategy settings
        self.strategy_settings = {
            "1d_scalp": {
                "atr_period": "5",
                "sr_period": "10",
                "momentum_period": "5",
                "rsi_period": "7",
                "macd_fast": "6",
                "macd_slow": "13",
                "macd_signal": "4",
                "bb_std": "1.5",
                "buy_threshold": "0.3",
                "sell_threshold": "-0.3"
            },
            "2d_momentum": {
                "atr_period": "8",
                "sr_period": "15",
                "momentum_period": "8",
                "rsi_period": "10",
                "macd_fast": "8",
                "macd_slow": "17",
                "macd_signal": "6",
                "bb_std": "2.0",
                "buy_threshold": "0.4",
                "sell_threshold": "-0.4"
            },
            "3d_swing": {
                "atr_period": "10",
                "sr_period": "20",
                "momentum_period": "10",
                "rsi_period": "12",
                "macd_fast": "10",
                "macd_slow": "21",
                "macd_signal": "7",
                "bb_std": "2.2",
                "buy_threshold": "0.5",
                "sell_threshold": "-0.5"
            },
            "5d_trend": {
                "atr_period": "14",
                "sr_period": "30",
                "momentum_period": "14",
                "rsi_period": "14",
                "macd_fast": "12",
                "macd_slow": "26",
                "macd_signal": "9",
                "bb_std": "2.5",
                "buy_threshold": "0.6",
                "sell_threshold": "-0.6"
            }
        }
        
        # Store current filename
        self.current_file = None
        
        # Ticker input
        ttk.Label(input_frame, text="Ticker Symbol:").grid(row=0, column=0, sticky=tk.W)
        self.ticker_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.ticker_var).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Plot title input
        ttk.Label(input_frame, text="Plot Title:").grid(row=1, column=0, sticky=tk.W)
        self.title_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.title_var).grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Period selection
        ttk.Label(input_frame, text="Time Period:").grid(row=2, column=0, sticky=tk.W)
        self.period_var = tk.StringVar(value="1y")
        period_choices = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
        ttk.Combobox(input_frame, textvariable=self.period_var, values=period_choices).grid(row=2, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Band settings
        ttk.Label(input_frame, text="Band Settings:").grid(row=3, column=0, sticky=tk.W)
        band_frame = ttk.Frame(input_frame)
        band_frame.grid(row=3, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Bollinger Band settings
        ttk.Label(band_frame, text="BB STD:").pack(side=tk.LEFT)
        self.bb_std_var = tk.StringVar(value="2")
        ttk.Entry(band_frame, textvariable=self.bb_std_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # RSI settings
        ttk.Label(band_frame, text="RSI Period:").pack(side=tk.LEFT, padx=(10,0))
        self.rsi_period_var = tk.StringVar(value="14")
        ttk.Entry(band_frame, textvariable=self.rsi_period_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # MACD settings
        macd_frame = ttk.Frame(input_frame)
        macd_frame.grid(row=4, column=1, padx=5)
        ttk.Label(macd_frame, text="MACD (Fast/Slow/Signal):").pack(side=tk.LEFT)
        self.macd_fast_var = tk.StringVar(value="12")
        self.macd_slow_var = tk.StringVar(value="26")
        self.macd_signal_var = tk.StringVar(value="9")
        ttk.Entry(macd_frame, textvariable=self.macd_fast_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(macd_frame, textvariable=self.macd_slow_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(macd_frame, textvariable=self.macd_signal_var, width=4).pack(side=tk.LEFT, padx=2)
        
        # Signal threshold settings
        signal_frame = ttk.Frame(input_frame)
        signal_frame.grid(row=5, column=1, padx=5)
        ttk.Label(signal_frame, text="Signal Thresholds (Buy/Sell):").pack(side=tk.LEFT)
        self.buy_threshold_var = tk.StringVar(value="0.5")
        self.sell_threshold_var = tk.StringVar(value="-0.5")
        ttk.Entry(signal_frame, textvariable=self.buy_threshold_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(signal_frame, textvariable=self.sell_threshold_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # Add Margin Trading frame
        margin_frame = ttk.LabelFrame(input_frame, text="Margin Trading Settings", padding="5")
        margin_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Margin ratio setting
        ttk.Label(margin_frame, text="Margin Ratio:").grid(row=0, column=0, sticky=tk.W)
        self.margin_ratio_var = tk.StringVar(value="2.0")  # 2:1 margin ratio
        ttk.Entry(margin_frame, textvariable=self.margin_ratio_var, width=5).grid(row=0, column=1, padx=5)
        
        # Stop loss percentage
        ttk.Label(margin_frame, text="Stop Loss %:").grid(row=0, column=2, sticky=tk.W)
        self.stop_loss_var = tk.StringVar(value="2.0")
        ttk.Entry(margin_frame, textvariable=self.stop_loss_var, width=5).grid(row=0, column=3, padx=5)
        
        # Add Short Selling menu
        self.short_menu = tk.Menu(self.strategy_menu, tearoff=0)
        self.strategy_menu.add_cascade(label="Short Selling", menu=self.short_menu)
        
        # Add short selling strategies
        self.short_menu.add_command(label="Bearish Trend", command=lambda: self.apply_short_strategy("bearish_trend"))
        self.short_menu.add_command(label="Volatility Short", command=lambda: self.apply_short_strategy("volatility_short"))
        self.short_menu.add_command(label="Technical Short", command=lambda: self.apply_short_strategy("technical_short"))
        self.short_menu.add_command(label="Margin Settings", command=self.show_margin_settings)
        
        # Plot button
        ttk.Button(input_frame, text="Plot", command=self.plot_stock).grid(row=6, column=0, columnspan=2, pady=10)
        
        # Create plot frame (now in column 1)
        self.plot_frame = ttk.Frame(root, padding="10")
        self.plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        
        # Add separator between control panel and plot
        ttk.Separator(root, orient='vertical').grid(row=0, column=0, sticky=(tk.N, tk.S), padx=(2, 0))

    def init_sqlite_database(self):
        """Initialize SQLite database connection and create tables if they don't exist"""
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                
                # Create ticker table (changed from stocks)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ticker (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL UNIQUE,
                        company_name TEXT,
                        sector TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Update foreign key reference in historical_data
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS historical_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker_id INTEGER,
                        date DATE NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        FOREIGN KEY (ticker_id) REFERENCES ticker(id),
                        UNIQUE(ticker_id, date)
                    )
                ''')
                
                # Update foreign key reference in trading_signals
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker_id INTEGER,
                        date DATE NOT NULL,
                        signal_type TEXT,
                        signal_strength REAL,
                        strategy TEXT,
                        FOREIGN KEY (ticker_id) REFERENCES ticker(id)
                    )
                ''')
                
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def init_duckdb_connection(self):
        """Initialize DuckDB database and create tables if they don't exist"""
        try:
            # Create a default database if none exists
            default_db_path = os.path.join(self.duckdb_dir, 'default.db')
            self.current_duck_conn = duckdb.connect(default_db_path)
            
            # Create tables if they don't exist with updated table name
            self.current_duck_conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker (
                    id INTEGER PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    company_name VARCHAR,
                    sector VARCHAR,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY,
                    ticker_id INTEGER,
                    date DATE NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume INTEGER,
                    FOREIGN KEY (ticker_id) REFERENCES ticker(id)
                );

                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY,
                    ticker_id INTEGER,
                    date DATE NOT NULL,
                    signal_type VARCHAR,
                    signal_strength DOUBLE,
                    strategy VARCHAR,
                    FOREIGN KEY (ticker_id) REFERENCES ticker(id)
                );
            """)
            
            # Update indexes for better query performance
            self.current_duck_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_symbol ON ticker(symbol);
                CREATE INDEX IF NOT EXISTS idx_historical_date ON historical_data(date);
                CREATE INDEX IF NOT EXISTS idx_historical_ticker_date ON historical_data(ticker_id, date);
            """)
            
            print("Database tables initialized successfully")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            if self.current_duck_conn:
                self.current_duck_conn.close()
                self.current_duck_conn = None

    def create_duckdb_tables(self, conn):
        """Create tables in DuckDB database"""
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker (
                    id INTEGER PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    company_name VARCHAR,
                    sector VARCHAR,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY,
                    ticker_id INTEGER,
                    date DATE NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume INTEGER,
                    FOREIGN KEY (ticker_id) REFERENCES ticker(id),
                    UNIQUE(ticker_id, date)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY,
                    ticker_id INTEGER,
                    date DATE NOT NULL,
                    signal_type VARCHAR,
                    signal_strength DOUBLE,
                    strategy VARCHAR,
                    FOREIGN KEY (ticker_id) REFERENCES ticker(id)
                )
            """)
            
            conn.commit()
        except Exception as e:
            print(f"Error creating DuckDB tables: {e}")

    def update_db_list(self):
        """Update the database dropdown list"""
        db_list = ['sqlite']  # Always include SQLite
        
        # Add DuckDB databases
        try:
            duckdb_files = [f[:-3] for f in os.listdir(self.duckdb_dir) 
                           if f.endswith('.db')]
            db_list.extend([f'duckdb_{f}' for f in duckdb_files])
        except Exception as e:
            print(f"Error listing DuckDB databases: {e}")
        
        self.db_dropdown['values'] = db_list
        
        # Set default if not already set
        if not self.db_var.get() in db_list:
            self.db_var.set(db_list[0])

    def create_new_db(self):
        """Create a new DuckDB database"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create New Database")
        dialog.geometry("300x100")
        
        ttk.Label(dialog, text="Database Name:").pack(pady=5)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var).pack(pady=5)
        
        def create():
            name = name_var.get().strip()
            if name:
                db_path = os.path.join(self.duckdb_dir, f'{name}.db')
                try:
                    conn = duckdb.connect(db_path)
                    self.create_duckdb_tables(conn)
                    conn.close()
                    self.update_db_list()
                    dialog.destroy()
                except Exception as e:
                    tk.messagebox.showerror("Error", f"Failed to create database: {e}")
            else:
                tk.messagebox.showwarning("Warning", "Please enter a database name")
        
        ttk.Button(dialog, text="Create", command=create).pack(pady=5)

    def get_current_connection(self):
        """Get connection to currently selected database"""
        db_type = self.db_var.get()
        
        if db_type == 'sqlite':
            return sqlite3.connect(self.sqlite_path)
        elif db_type.startswith('duckdb_'):
            db_name = db_type[7:]  # Remove 'duckdb_' prefix
            db_path = os.path.join(self.duckdb_dir, f'{db_name}.db')
            return duckdb.connect(db_path)
        else:
            raise ValueError(f"Unknown database type: {db_type}")

    def save_stock_data(self, ticker, hist_data):
        """Save stock data to DuckDB database"""
        try:
            if not self.current_duck_conn:
                print("No active database connection")
                return

            # Insert or update stock record
            self.current_duck_conn.execute("""
                INSERT INTO ticker (symbol, last_updated)
                VALUES (?, CURRENT_TIMESTAMP)
                ON CONFLICT (symbol) DO UPDATE 
                SET last_updated = CURRENT_TIMESTAMP
                RETURNING id
            """, [ticker])
            
            result = self.current_duck_conn.fetchone()
            if not result:
                print(f"Failed to get ticker_id for {ticker}")
                return
                
            ticker_id = result[0]
            
            # Prepare historical data
            for date, row in hist_data.iterrows():
                self.current_duck_conn.execute("""
                    INSERT INTO historical_data (ticker_id, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker_id, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, [
                    ticker_id, 
                    date.strftime('%Y-%m-%d'),
                    float(row['Open']), 
                    float(row['High']), 
                    float(row['Low']), 
                    float(row['Close']), 
                    int(row['Volume'])
                ])
            
            self.current_duck_conn.commit()
            
        except Exception as e:
            print(f"Error saving stock data: {e}")
            if self.current_duck_conn:
                self.current_duck_conn.rollback()

    def fetch_stock_data(self, ticker, period):
        """Fetch stock data from local DuckDB database first, fallback to yfinance if needed"""
        if not hasattr(self, 'current_duck_conn') or not self.current_duck_conn:
            print("No active database connection")
            return None
            
        try:
            # Try to get data from local DuckDB first
            query = """
                SELECT h.date, h.open as Open, h.high as High, h.low as Low, 
                       h.close as Close, h.volume as Volume
                FROM ticker s
                JOIN historical_data h ON s.id = h.ticker_id
                WHERE s.symbol = ?
                ORDER BY h.date
            """
            result = self.current_duck_conn.execute(query, [ticker]).fetchdf()
            
            if not result.empty:
                # Set date as index for compatibility with yfinance format
                result.set_index('date', inplace=True)
                return result
                
            # If no local data and internet is available, try yfinance as fallback
            print(f"No local data found for {ticker}, checking yfinance...")
            import time
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=period)
                    if not hist.empty:
                        # Save to local database for future use
                        self.save_stock_data(ticker, hist)
                        return hist
                    time.sleep(retry_delay)
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        print("All attempts to fetch online data failed")
            
            return None
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None

    def plot_stock(self):
        """Modified plot_stock method to handle local data"""
        ticker = self.ticker_var.get().upper()
        if not ticker:
            tk.messagebox.showwarning("Warning", "Please enter a ticker symbol")
            return
            
        try:
            # Get data from local database
            hist = self.fetch_stock_data(ticker, self.period_var.get())
            if hist is None or hist.empty:
                tk.messagebox.showwarning("Warning", f"No data available for {ticker}")
                return

            # Continue with plotting...
            # Create a new figure
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Plot the closing prices
            ax.plot(hist.index, hist['Close'], label='Close Price')
            
            # Customize the plot
            ax.set_title(f"{ticker} Stock Price")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.grid(True)
            ax.legend()
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Clear existing plot frame
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Create canvas and add it to the plot frame
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"Error in plot_stock: {e}")
            tk.messagebox.showerror("Error", f"Failed to plot stock data: {str(e)}")

    def open_file(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as file:
                    data = file.read().split('\n')
                    if len(data) >= 3:
                        self.ticker_var.set(data[0])
                        self.title_var.set(data[1])
                        self.period_var.set(data[2])
                        self.current_file = filename
                        self.plot_stock()
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to open file: {str(e)}")

    def save_file(self):
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_as_file()

    def save_as_file(self):
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            self._save_to_file(filename)
            self.current_file = filename

    def _save_to_file(self, filename):
        try:
            with open(filename, 'w') as file:
                file.write(f"{self.ticker_var.get()}\n")
                file.write(f"{self.title_var.get()}\n")
                file.write(f"{self.period_var.get()}\n")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save file: {str(e)}")

    def show_indicator_help(self):
        help_text = """
Technical Indicators Explained:

1. Moving Averages (MA)
   - 20-day MA: Short-term trend indicator
   - 50-day MA: Medium-term trend indicator
   - Crossovers signal potential trend changes

2. Bollinger Bands (BB)
   - Middle Band: 20-day moving average
   - Upper/Lower Bands: Middle ± (Standard Deviation × Multiplier)
   - Measures volatility and potential overbought/oversold levels
   - Default multiplier is 2, adjustable in settings

3. Relative Strength Index (RSI)
   - Momentum oscillator measuring speed/change of price movements
   - Range: 0-100
   - Traditional levels: 
     * Above 70: Overbought
     * Below 30: Oversold
   - Period is adjustable (default: 14 days)

4. Moving Average Convergence Divergence (MACD)
   - Trend-following momentum indicator
   - Components:
     * MACD Line: Difference between fast and slow EMAs
     * Signal Line: EMA of MACD line
     * Histogram: MACD Line - Signal Line
   - Parameters adjustable:
     * Fast period (default: 12)
     * Slow period (default: 26)
     * Signal period (default: 9)

5. Volume Analysis
   - Volume MA: 20-day average trading volume
   - High volume: Confirms price movements
   - Volume bands: Shows unusual trading activity
    """
        
        self.show_help_window("Technical Indicators Help", help_text)

    def show_signal_help(self):
        help_text = """
Trading Signals Explained:

1. Combined Signal Algorithm
   Weighted combination of multiple factors:
   - Trend (30%): Based on MA crossovers
   - Bollinger Bands (20%): Mean reversion
   - RSI (20%): Momentum
   - MACD (20%): Trend confirmation
   - Volume (10%): Trading activity confirmation

2. Signal Generation
   - Buy Signal: Combined signal ≥ Buy threshold
   - Sell Signal: Combined signal ≤ Sell threshold
   - Hold: Between thresholds
   
3. Signal Components:
   a) Trend Analysis
      - Positive: MA20 > MA50
      - Negative: MA20 < MA50
   
   b) Bollinger Band Signals
      - Buy: Price near lower band
      - Sell: Price near upper band
   
   c) RSI Signals
      - Buy: RSI < 30 (oversold)
      - Sell: RSI > 70 (overbought)
   
   d) MACD Signals
      - Buy: MACD crosses above Signal Line
      - Sell: MACD crosses below Signal Line
   
   e) Volume Confirmation
      - Significant: Volume > 1.5 × Volume MA
      - Normal: Volume ≤ 1.5 × Volume MA

4. Performance Metrics
   - Strategy Returns: Cumulative returns from signals
   - Buy & Hold Returns: Market benchmark
   - Risk-adjusted metrics included in summary
    """
        
        self.show_help_window("Trading Signals Help", help_text)

    def show_band_help(self):
        help_text = """
Band Settings Explained:

1. Bollinger Bands (BB)
   - STD Multiplier: Controls band width
   - Higher value = Wider bands = Fewer signals
   - Lower value = Narrower bands = More signals
   - Traditional value: 2

2. RSI Settings
   - Period: Number of days for calculation
   - Shorter period = More sensitive
   - Longer period = More stable
   - Traditional period: 14 days

3. MACD Parameters
   - Fast Period: Short-term EMA
   - Slow Period: Long-term EMA
   - Signal Period: MACD smoothing
   - Traditional values: 12/26/9

4. Signal Thresholds
   - Buy: Trigger level for buy signals
   - Sell: Trigger level for sell signals
   - Higher thresholds = More conservative
   - Lower thresholds = More aggressive

5. Band Applications
   - Price Bands: Volatility measurement
   - Volume Bands: Trading activity range
   - Return Bands: Expected price movement
   - RSI Bands: Momentum range
   - MACD Bands: Trend strength
    """
        
        self.show_help_window("Band Settings Help", help_text)

    def show_about(self):
        about_text = """
Stock Price Viewer

A technical analysis tool for stock market data.

Features:
- Real-time stock data retrieval
- Multiple technical indicators
- Custom trading signals
- Adjustable parameters
- Performance analysis

Data provided by Yahoo Finance
    """
        
        self.show_help_window("About", about_text)

    def show_daytrading_help(self):
        help_text = """
Day Trading Indicators and Algorithms:

1. Average True Range (ATR)
   - Measures volatility
   - Higher ATR = Higher volatility
   - Used for stop-loss placement
   - Adjustable period (default: 14)

2. Support/Resistance Levels
   - Dynamic price levels
   - Based on rolling high/low
   - Breakout signals on level breach
   - Volume confirmation required

3. Price Momentum
   - Measures price velocity
   - Includes confidence bands
   - Signals:
     * Above upper band: Strong upward momentum
     * Below lower band: Strong downward momentum
   - Mean reversion potential at extremes

4. Volatility Bands
   - Based on price standard deviation
   - Breakout signals on band breach
   - Volume confirmation required
   - Adjusts to market conditions

5. Intraday Momentum Index (IMI)
   - Range: 0-100
   - Above 70: Overbought
   - Below 30: Oversold
   - Confirms trend strength

Day Trading Signal Generation:
1. Volatility Breakout (40% weight)
   - Price breaks volatility bands
   - Volume confirmation required
   - Most significant for day trading

2. Support/Resistance Break (30% weight)
   - Price breaks S/R levels
   - Volume confirmation needed
   - Medium-term significance

3. Momentum Signal (30% weight)
   - Combines momentum and IMI
   - Confirms trend strength
   - Short-term focus

Risk Management:
- Use ATR for stop-loss placement
- Consider volatility for position sizing
- Monitor volume for confirmation
- Watch multiple timeframes
- Use confidence bands for entry/exit
    """
        
        self.show_help_window("Day Trading Help", help_text)

    def show_margin_help(self):
        help_text = """
Short Selling and Margin Trading Strategies:

1. Bearish Trend Strategy
   - For strong downtrends
   - Conservative margin (2:1)
   - Standard stop loss (2%)
   - Focuses on trend following
   Best for: Clear downtrend markets

2. Volatility Short Strategy
   - For high volatility stocks
   - Lower margin (1.5:1)
   - Tighter stops (1.5%)
   - Quick entry/exit
   Best for: Volatile, overextended stocks

3. Technical Short Strategy
   - Based on technical signals
   - Standard margin (2:1)
   - Wider stops (2.5%)
   - Multiple indicator confirmation
   Best for: Technical breakdowns

Risk Management:
1. Margin Requirements
   - Maintenance margin minimum
   - Available buying power
   - Margin call thresholds

2. Short Squeeze Risk
   - Volume analysis
   - Short interest monitoring
   - Squeeze risk indicator

3. Stop Loss Management
   - Percentage-based stops
   - ATR-based stops
   - Margin call prevention

4. Position Sizing
   - Account for margin requirements
   - Risk per trade calculation
   - Portfolio exposure limits

Key Metrics:
- Trend Strength: Volatility/Price ratio
- Squeeze Risk: Volume and price pressure
- Margin Risk: Position risk vs margin
- Technical Signals: Entry/exit timing

Best Practices:
1. Monitor borrowing costs
2. Watch for dividend dates
3. Keep margin cushion
4. Use strict stop losses
5. Monitor short interest
6. Plan exit strategies
    """
        
        self.show_help_window("Margin Trading Help", help_text)

    def show_help_window(self, title, text):
        help_window = tk.Toplevel(self.root)
        help_window.title(title)
        help_window.geometry("600x800")
        
        # Create text widget with scrollbar
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(help_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Insert help text
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)  # Make text read-only

    def apply_strategy(self, strategy_name):
        """Apply selected trading strategy settings"""
        if strategy_name in self.strategy_settings:
            settings = self.strategy_settings[strategy_name]
            
            # Update all settings variables
            self.bb_std_var.set(settings["bb_std"])
            self.rsi_period_var.set(settings["rsi_period"])
            self.macd_fast_var.set(settings["macd_fast"])
            self.macd_slow_var.set(settings["macd_slow"])
            self.macd_signal_var.set(settings["macd_signal"])
            self.buy_threshold_var.set(settings["buy_threshold"])
            self.sell_threshold_var.set(settings["sell_threshold"])
            
            # Update period for shorter timeframes
            if strategy_name == "1d_scalp":
                self.period_var.set("1d")
            elif strategy_name == "2d_momentum":
                self.period_var.set("2d")
            elif strategy_name == "3d_swing":
                self.period_var.set("3d")
            elif strategy_name == "5d_trend":
                self.period_var.set("5d")
            
            # Automatically update plot
            self.plot_stock()

    def show_strategy_help(self):
        help_text = """
Day Trading Strategy Presets:

1. 1-Day Scalping Strategy
   - Very short-term trades (minutes to hours)
   - Tight Bollinger Bands (1.5 STD)
   - Quick RSI (7 periods)
   - Fast MACD (6/13/4)
   - Aggressive thresholds
   Best for: High-volume, liquid stocks
   
2. 2-Day Momentum Strategy
   - Short-term momentum trades
   - Moderate bands (2.0 STD)
   - RSI (10 periods)
   - Modified MACD (8/17/6)
   - Moderate thresholds
   Best for: Trending stocks with good volume
   
3. 3-Day Swing Strategy
   - Multi-day position trades
   - Wider bands (2.2 STD)
   - Standard RSI (12 periods)
   - Balanced MACD (10/21/7)
   - Standard thresholds
   Best for: Stocks with clear support/resistance
   
4. 5-Day Trend Strategy
   - Week-long trend following
   - Conservative bands (2.5 STD)
   - Traditional RSI (14 periods)
   - Standard MACD (12/26/9)
   - Conservative thresholds
   Best for: Strong trending stocks

Strategy Components:
- ATR Period: Volatility measurement
- S/R Period: Support/Resistance levels
- Momentum Period: Price momentum
- RSI Period: Oversold/Overbought
- MACD Settings: Trend confirmation
- BB STD: Band width
- Thresholds: Signal sensitivity

Usage Tips:
1. Match strategy to market conditions
2. Consider stock volatility
3. Monitor volume for confirmation
4. Use appropriate position sizing
5. Set stops based on ATR
6. Confirm signals across timeframes
    """
        
        self.show_help_window("Trading Strategies Help", help_text)

    def apply_short_strategy(self, strategy_name):
        """Apply short selling strategy settings"""
        short_settings = {
            "bearish_trend": {
                "margin_ratio": "2.0",
                "stop_loss": "2.0",
                "atr_period": "14",
                "momentum_period": "10",
                "rsi_period": "14",
                "macd_fast": "12",
                "macd_slow": "26",
                "macd_signal": "9",
                "bb_std": "2.0",
                "sell_threshold": "-0.4",
                "buy_threshold": "0.4"  # For covering shorts
            },
            "volatility_short": {
                "margin_ratio": "1.5",
                "stop_loss": "1.5",
                "atr_period": "10",
                "momentum_period": "5",
                "rsi_period": "10",
                "macd_fast": "8",
                "macd_slow": "17",
                "macd_signal": "6",
                "bb_std": "2.5",
                "sell_threshold": "-0.5",
                "buy_threshold": "0.5"
            },
            "technical_short": {
                "margin_ratio": "2.0",
                "stop_loss": "2.5",
                "atr_period": "14",
                "momentum_period": "14",
                "rsi_period": "14",
                "macd_fast": "12",
                "macd_slow": "26",
                "macd_signal": "9",
                "bb_std": "2.2",
                "sell_threshold": "-0.6",
                "buy_threshold": "0.6"
            }
        }
        
        if strategy_name in short_settings:
            settings = short_settings[strategy_name]
            
            # Update margin settings
            self.margin_ratio_var.set(settings["margin_ratio"])
            self.stop_loss_var.set(settings["stop_loss"])
            
            # Update other settings
            self.bb_std_var.set(settings["bb_std"])
            self.rsi_period_var.set(settings["rsi_period"])
            self.macd_fast_var.set(settings["macd_fast"])
            self.macd_slow_var.set(settings["macd_slow"])
            self.macd_signal_var.set(settings["macd_signal"])
            self.buy_threshold_var.set(settings["buy_threshold"])
            self.sell_threshold_var.set(settings["sell_threshold"])
            
            self.plot_stock()

    def show_margin_settings(self):
        """Display margin settings configuration window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Margin Trading Settings")
        settings_window.geometry("400x500")
        
        # Create main frame
        main_frame = ttk.Frame(settings_window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Margin Requirements
        margin_frame = ttk.LabelFrame(main_frame, text="Margin Requirements", padding="5")
        margin_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(margin_frame, text="Initial Margin Ratio:").grid(row=0, column=0, sticky=tk.W)
        initial_margin_var = tk.StringVar(value=self.margin_ratio_var.get())
        ttk.Entry(margin_frame, textvariable=initial_margin_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(margin_frame, text="Maintenance Margin (%):").grid(row=1, column=0, sticky=tk.W)
        maintenance_margin_var = tk.StringVar(value="25.0")
        ttk.Entry(margin_frame, textvariable=maintenance_margin_var, width=8).grid(row=1, column=1, padx=5)
        
        # Risk Management
        risk_frame = ttk.LabelFrame(main_frame, text="Risk Management", padding="5")
        risk_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(risk_frame, text="Stop Loss (%):").grid(row=0, column=0, sticky=tk.W)
        stop_loss_var = tk.StringVar(value=self.stop_loss_var.get())
        ttk.Entry(risk_frame, textvariable=stop_loss_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(risk_frame, text="Max Position Size (%):").grid(row=1, column=0, sticky=tk.W)
        position_size_var = tk.StringVar(value="20.0")
        ttk.Entry(risk_frame, textvariable=position_size_var, width=8).grid(row=1, column=1, padx=5)
        
        ttk.Label(risk_frame, text="Max Portfolio Short (%):").grid(row=2, column=0, sticky=tk.W)
        portfolio_short_var = tk.StringVar(value="50.0")
        ttk.Entry(risk_frame, textvariable=portfolio_short_var, width=8).grid(row=2, column=1, padx=5)
        
        # Short Squeeze Protection
        squeeze_frame = ttk.LabelFrame(main_frame, text="Short Squeeze Protection", padding="5")
        squeeze_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(squeeze_frame, text="Max Short Interest Ratio:").grid(row=0, column=0, sticky=tk.W)
        short_interest_var = tk.StringVar(value="15.0")
        ttk.Entry(squeeze_frame, textvariable=short_interest_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(squeeze_frame, text="Volume Threshold:").grid(row=1, column=0, sticky=tk.W)
        volume_threshold_var = tk.StringVar(value="1.5")
        ttk.Entry(squeeze_frame, textvariable=volume_threshold_var, width=8).grid(row=1, column=1, padx=5)
        
        # Alert Settings
        alert_frame = ttk.LabelFrame(main_frame, text="Alert Settings", padding="5")
        alert_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(alert_frame, text="Margin Call Warning (%):").grid(row=0, column=0, sticky=tk.W)
        margin_warning_var = tk.StringVar(value="30.0")
        ttk.Entry(alert_frame, textvariable=margin_warning_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(alert_frame, text="Squeeze Alert Level:").grid(row=1, column=0, sticky=tk.W)
        squeeze_alert_var = tk.StringVar(value="2.0")
        ttk.Entry(alert_frame, textvariable=squeeze_alert_var, width=8).grid(row=1, column=1, padx=5)
        
        # Information text
        info_text = """
Risk Management Guidelines:
• Initial Margin: Minimum required to open position
• Maintenance Margin: Minimum to maintain position
• Stop Loss: Automatic exit point
• Position Size: Maximum single position
• Portfolio Short: Total short exposure limit
• Short Interest: Maximum market short ratio
• Volume Threshold: Unusual volume multiplier
• Margin Warning: Early warning threshold
• Squeeze Alert: Short squeeze risk level
    """
        
        info_label = ttk.Label(main_frame, text=info_text, wraplength=380, justify=tk.LEFT)
        info_label.grid(row=4, column=0, pady=10)
        
        def apply_settings():
            """Apply the margin settings"""
            self.margin_ratio_var.set(initial_margin_var.get())
            self.stop_loss_var.set(stop_loss_var.get())
            
            # Store other settings as instance variables
            self.maintenance_margin = float(maintenance_margin_var.get())
            self.max_position_size = float(position_size_var.get())
            self.max_portfolio_short = float(portfolio_short_var.get())
            self.max_short_interest = float(short_interest_var.get())
            self.volume_threshold = float(volume_threshold_var.get())
            self.margin_warning = float(margin_warning_var.get())
            self.squeeze_alert = float(squeeze_alert_var.get())
            
            # Update plot with new settings
            self.plot_stock()
            settings_window.destroy()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Make window modal
        settings_window.transient(self.root)
        settings_window.grab_set()
        self.root.wait_window(settings_window)

    def save_trading_signal(self, stock_symbol, date, signal_type, signal_strength, strategy):
        """Save trading signal to database"""
        try:
            conn = self.get_current_connection()
            
            if isinstance(conn, sqlite3.Connection):
                # Existing SQLite save logic
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_signals 
                    (ticker_id, date, signal_type, signal_strength, strategy)
                    VALUES (?, ?, ?, ?, ?)
                ''', (stock_id, date.strftime('%Y-%m-%d'), 
                     signal_type, signal_strength, strategy))
            
            else:  # DuckDB connection
                # Convert DataFrame to DuckDB format
                conn.execute("""
                    INSERT INTO trading_signals 
                    (ticker_id, date, signal_type, signal_strength, strategy)
                    VALUES (?, ?, ?, ?, ?)
                """, [stock_id, date.strftime('%Y-%m-%d'), 
                     signal_type, signal_strength, strategy])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving trading signal: {e}")
            if conn:
                conn.close()

    def open_duckdb(self):
        """Open an existing DuckDB database from anywhere on the system"""
        from tkinter import filedialog
        
        # Open file dialog for selecting .db files
        file_path = filedialog.askopenfilename(
            title="Select DuckDB Database",
            filetypes=[
                ("DuckDB files", "*.db"),
                ("All files", "*.*")
            ],
            initialdir="/"  # Start from root directory
        )
        
        if file_path:
            try:
                # Close existing connection if any
                if hasattr(self, 'current_duck_conn') and self.current_duck_conn:
                    self.current_duck_conn.close()
                
                # Connect to selected database
                self.current_duck_conn = duckdb.connect(file_path)
                
                # Get database name from path
                db_name = os.path.basename(file_path)
                
                # Create symlink in duckdb_dir if it doesn't exist there
                local_path = os.path.join(self.duckdb_dir, db_name)
                if not os.path.exists(local_path):
                    try:
                        os.symlink(file_path, local_path)
                    except Exception as e:
                        print(f"Warning: Could not create symlink: {e}")
                
                # Update dropdown to show the newly opened database
                self.refresh_duckdb_list()
                self.duckdb_var.set(os.path.splitext(db_name)[0])
                
                # Try to create tables if they don't exist
                try:
                    self.create_duckdb_tables(self.current_duck_conn)
                except Exception as e:
                    print(f"Warning: Could not create tables: {e}")
                
                tk.messagebox.showinfo("Success", f"Opened database: {db_name}")
                
                # Refresh the plot if there's data
                self.plot_stock()
                
            except Exception as e:
                error_msg = f"Failed to open database: {str(e)}"
                print(error_msg)
                tk.messagebox.showerror("Error", error_msg)
                
                # Reset connection
                self.current_duck_conn = None

    def refresh_duckdb_list(self):
        """Refresh the list of available DuckDB databases"""
        try:
            # Get list of .db files and symlinks in the DuckDB directory
            db_files = []
            for f in os.listdir(self.duckdb_dir):
                if f.endswith('.db'):
                    # Remove .db extension for display
                    db_files.append(f[:-3])
            
            # Update dropdown values
            self.duckdb_dropdown['values'] = sorted(db_files)
            
            # Set default selection if list is not empty and no current selection
            if db_files and not self.duckdb_var.get():
                self.duckdb_var.set(db_files[0])
                
        except Exception as e:
            error_msg = f"Error refreshing database list: {str(e)}"
            print(error_msg)
            tk.messagebox.showerror("Error", error_msg)
            self.duckdb_dropdown['values'] = []

    def on_database_change(self, event):
        """Handle database selection change"""
        selected_db = self.duckdb_var.get()
        if selected_db:
            try:
                # Close existing connection if any
                if hasattr(self, 'current_duck_conn') and self.current_duck_conn:
                    self.current_duck_conn.close()
                
                # Connect to the selected database
                db_path = os.path.join(self.duckdb_dir, f'{selected_db}.db')
                self.current_duck_conn = duckdb.connect(db_path)
                
                # Refresh the plot if there's data
                self.plot_stock()
                
            except Exception as e:
                error_msg = f"Failed to switch to database: {str(e)}"
                print(error_msg)
                tk.messagebox.showerror("Error", error_msg)
                self.current_duck_conn = None

    def __del__(self):
        """Cleanup when the application closes"""
        if hasattr(self, 'current_duck_conn') and self.current_duck_conn:
            self.current_duck_conn.close()

    def query_local_database(self):
        """Query data from local DuckDB database"""
        if not self.current_duck_conn:
            tk.messagebox.showwarning("Warning", "Please connect to a database first")
            return
            
        # Create query dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Query Database")
        dialog.geometry("600x400")
        
        # Create frames
        query_frame = ttk.LabelFrame(dialog, text="SQL Query", padding="5")
        query_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        result_frame = ttk.LabelFrame(dialog, text="Results", padding="5")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add query text area with scrollbar
        query_scroll = ttk.Scrollbar(query_frame)
        query_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        query_text = tk.Text(query_frame, height=6, yscrollcommand=query_scroll.set)
        query_text.pack(fill=tk.BOTH, expand=True)
        query_scroll.config(command=query_text.yview)
        
        # Add default query example
        default_query = """SELECT t.symbol, h.date, h.open, h.high, h.low, h.close, h.volume
FROM ticker t
JOIN historical_data h ON t.id = h.ticker_id
WHERE t.symbol = 'AAPL'
ORDER BY h.date DESC
LIMIT 10;"""
        query_text.insert('1.0', default_query)
        
        # Add result text area with scrollbar
        result_scroll = ttk.Scrollbar(result_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        result_text = tk.Text(result_frame, height=10, yscrollcommand=result_scroll.set)
        result_text.pack(fill=tk.BOTH, expand=True)
        result_scroll.config(command=result_text.yview)
        
        def execute_query():
            """Execute the SQL query and display results"""
            try:
                # Get query from text area
                query = query_text.get('1.0', tk.END).strip()
                
                # Execute query
                result = self.current_duck_conn.execute(query).fetchdf()
                
                # Clear previous results
                result_text.delete('1.0', tk.END)
                
                # Display results
                if result.empty:
                    result_text.insert(tk.END, "No results found")
                else:
                    # Convert DataFrame to string with proper formatting
                    result_str = result.to_string()
                    result_text.insert(tk.END, result_str)
                
            except Exception as e:
                error_msg = f"Query error: {str(e)}"
                print(error_msg)
                result_text.delete('1.0', tk.END)
                result_text.insert(tk.END, error_msg)
        
        def save_results():
            """Save query results to a CSV file"""
            try:
                # Get query from text area
                query = query_text.get('1.0', tk.END).strip()
                
                # Execute query
                result = self.current_duck_conn.execute(query).fetchdf()
                
                if not result.empty:
                    # Open file dialog for saving
                    file_path = filedialog.asksaveasfilename(
                        defaultextension=".csv",
                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                    )
                    
                    if file_path:
                        result.to_csv(file_path, index=False)
                        tk.messagebox.showinfo("Success", "Results saved to CSV file")
                else:
                    tk.messagebox.showwarning("Warning", "No results to save")
                    
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to save results: {str(e)}")
        
        # Add buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Execute Query", command=execute_query).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Results", command=save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Add help text
        help_text = """Available Tables:
- ticker (id, symbol, company_name, sector, last_updated)
- historical_data (id, ticker_id, date, open, high, low, close, volume)
- trading_signals (id, ticker_id, date, signal_type, signal_strength, strategy)"""
        
        ttk.Label(dialog, text=help_text, justify=tk.LEFT).pack(padx=5, pady=5)

def main():
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()