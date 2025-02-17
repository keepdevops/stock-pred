import tkinter as tk
from tkinter import ttk, messagebox
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib
import traceback
import os
import sys
import threading
import contextlib
from concurrent.futures import ThreadPoolExecutor
import functools
import glob

matplotlib.use('TkAgg')

def find_databases():
    """Find all DuckDB database files in the current directory"""
    return glob.glob('*.db')

def create_connection(db_name):
    """Create a database connection"""
    try:
        return duckdb.connect(db_name)
    except Exception as e:
        print(f"Error connecting to database {db_name}: {e}")
        return None

class ThreadSafeManager:
    def __init__(self):
        self._lock = threading.Lock()
    
    def __enter__(self):
        self._lock.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

def process_data_safely(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with ThreadSafeManager():
            return func(*args, **kwargs)
    return wrapper

class StockAnalyzerGUI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("Stock Market Analyzer")
        self.root.geometry("1600x1000")
        
        # Initialize descriptions dictionary first
        self.ticker_descriptions = {
            # Futures
            'ES=F': 'E-mini S&P 500 Futures - Tracks the S&P 500 index with 1/5th the size',
            'NQ=F': 'E-mini NASDAQ-100 Futures - Tracks the NASDAQ-100 technology index',
            'YM=F': 'E-mini Dow Futures - Tracks the Dow Jones Industrial Average',
            'RTY=F': 'E-mini Russell 2000 Futures - Tracks small-cap U.S. stocks',
            'ZB=F': 'U.S. Treasury Bond Futures - Long-term 30-year Treasury bonds',
            'ZN=F': '10-Year T-Note Futures - Medium-term Treasury notes',
            'CL=F': 'Crude Oil Futures - West Texas Intermediate (WTI) crude oil',
            'NG=F': 'Natural Gas Futures - Henry Hub natural gas benchmark',
            'GC=F': 'Gold Futures - Physical gold bullion contracts',
            'SI=F': 'Silver Futures - Physical silver bullion contracts',
            'ZC=F': 'Corn Futures - U.S. corn agricultural commodity',
            'ZS=F': 'Soybean Futures - U.S. soybean agricultural commodity',
            'ZW=F': 'Wheat Futures - U.S. wheat agricultural commodity',
            
            # ETFs
            'SPY': 'SPDR S&P 500 ETF - Tracks S&P 500 index, most liquid ETF',
            'QQQ': 'Invesco QQQ - Tracks NASDAQ-100, focus on tech companies',
            'IWM': 'iShares Russell 2000 ETF - Small-cap U.S. companies',
            'GLD': 'SPDR Gold Trust - Physical gold-backed ETF',
            'USO': 'United States Oil Fund - Tracks crude oil prices',
            'TLT': 'iShares 20+ Year Treasury Bond ETF - Long-term Treasury bonds',
            
            # Add descriptions for stocks
            'MSFT': 'Microsoft Corporation - Technology, software, and cloud computing',
            'GOOGL': 'Alphabet Inc. - Technology, search engine, and digital advertising',
            'AAPL': 'Apple Inc. - Technology, consumer electronics, and services',
            'AMZN': 'Amazon.com Inc. - E-commerce, cloud computing, and digital services',
            'NVDA': 'NVIDIA Corporation - Technology, graphics processors, and AI computing'
        }
        
        try:
            # Find all DuckDB databases in current directory
            self.available_dbs = find_databases()
            print(f"Found databases: {self.available_dbs}")
            
            # Start with first available database
            self.current_db = self.available_dbs[0] if self.available_dbs else 'stocks.db'
            self.db_conn = duckdb.connect(self.current_db)
            print(f"Connected to database: {self.current_db}")
            
            # Get available tables
            self.tables = self.get_tables()
            print(f"Found tables: {self.tables}")
            
            self.setup_gui()
            
        except Exception as e:
            print(f"Error in setup: {str(e)}")
            messagebox.showerror("Setup Error", str(e))
            raise e

    def setup_gui(self):
        self.control_panel = ttk.Frame(self.root, padding="5")
        self.control_panel.grid(row=0, column=0, sticky="nsew")
        
        self.graph_panel = ttk.Frame(self.root, padding="5")
        self.graph_panel.grid(row=0, column=1, sticky="nsew")
        
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.ticker_var = tk.StringVar()
        self.duration_var = tk.StringVar(value="1mo")
        
        self.create_controls()
        
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_panel)
        self.toolbar.update()
        
        self.loading_label = ttk.Label(self.root, text="")
        self.loading_label.grid(row=1, column=0, columnspan=2, sticky="ew")

    def create_controls(self):
        """Create control panel widgets"""
        # Database selection
        db_frame = ttk.LabelFrame(self.control_panel, text="Database Selection", padding="5")
        db_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(db_frame, text="Database:").pack(side="left", padx=5)
        self.db_var = tk.StringVar(value=self.current_db)
        self.db_combo = ttk.Combobox(db_frame, textvariable=self.db_var)
        self.db_combo.pack(side="left", fill="x", expand=True, padx=5)
        
        if self.available_dbs:
            self.db_combo['values'] = self.available_dbs
            self.db_combo.set(self.current_db)
            self.db_combo.bind('<<ComboboxSelected>>', self.on_database_change)
        else:
            self.db_combo['values'] = ['No databases found']
            self.db_combo.set('No databases found')
        
        # Refresh button for databases
        ttk.Button(db_frame, text="🔄", width=3,
                   command=self.refresh_databases).pack(side="left", padx=5)
        
        # Table selection
        table_frame = ttk.LabelFrame(self.control_panel, text="Table Selection", padding="5")
        table_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(table_frame, text="Table:").pack(side="left", padx=5)
        self.table_var = tk.StringVar()
        self.table_combo = ttk.Combobox(table_frame, textvariable=self.table_var)
        self.table_combo.pack(side="left", fill="x", expand=True, padx=5)
        
        if self.tables:
            self.table_combo['values'] = self.tables
            self.table_combo.set(self.tables[0])
            self.table_combo.bind('<<ComboboxSelected>>', self.on_table_change)
        else:
            self.table_combo['values'] = ['No tables available']
            self.table_combo.set('No tables available')
        
        # Ticker selection with help
        ticker_frame = ttk.LabelFrame(self.control_panel, text="Stock Selection", padding="5")
        ticker_frame.pack(fill="x", padx=5, pady=5)
        
        # Add help button
        help_frame = ttk.Frame(ticker_frame)
        help_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(help_frame, text="Ticker:").pack(side="left", padx=5)
        help_button = ttk.Button(help_frame, text="?", width=2)
        help_button.pack(side="right", padx=5)
        
        # Create tooltip text based on table
        def show_ticker_help():
            help_text = """
            Futures Symbols:
            ES=F : E-mini S&P 500 Futures
            NQ=F : E-mini NASDAQ-100 Futures
            YM=F : E-mini Dow Futures
            RTY=F: E-mini Russell 2000 Futures
            ZB=F : U.S. Treasury Bond Futures
            ZN=F : 10-Year T-Note Futures
            CL=F : Crude Oil Futures
            NG=F : Natural Gas Futures
            GC=F : Gold Futures
            SI=F : Silver Futures
            ZC=F : Corn Futures
            ZS=F : Soybean Futures
            ZW=F : Wheat Futures
            
            Options Symbols:
            SPY : SPDR S&P 500 ETF
            QQQ : Invesco QQQ (NASDAQ-100)
            IWM : iShares Russell 2000 ETF
            GLD : SPDR Gold Trust
            USO : United States Oil Fund
            TLT : iShares 20+ Year Treasury Bond ETF
            """
            messagebox.showinfo("Ticker Symbol Guide", help_text)
        
        help_button.config(command=show_ticker_help)
        
        # Ticker combobox with description
        self.ticker_var = tk.StringVar()
        self.ticker_combo = ttk.Combobox(ticker_frame, textvariable=self.ticker_var)
        self.ticker_combo.pack(fill="x", padx=5, pady=2)
        
        # Description label
        self.ticker_desc = ttk.Label(ticker_frame, text="", wraplength=250)
        self.ticker_desc.pack(fill="x", padx=5, pady=2)
        
        # Update description when ticker changes
        def on_ticker_change(event=None):
            ticker = self.ticker_var.get()
            description = self.ticker_descriptions.get(ticker, 'No description available')
            self.ticker_desc.config(text=description)
        
        self.ticker_combo.bind('<<ComboboxSelected>>', on_ticker_change)
        
        # Initialize tickers for selected table
        self.update_tickers()
        
        # Duration selection
        duration_frame = ttk.LabelFrame(self.control_panel, text="Duration", padding="5")
        duration_frame.pack(fill="x", padx=5, pady=5)
        
        durations = [("1 Day", "1d"), ("1 Month", "1mo"), ("3 Months", "3mo"),
                    ("6 Months", "6mo"), ("1 Year", "1y")]
        
        self.duration_var = tk.StringVar(value="1mo")
        for text, value in durations:
            ttk.Radiobutton(duration_frame, text=text, value=value,
                           variable=self.duration_var).pack(side="left", padx=5)
        
        ttk.Button(self.control_panel, text="Analyze",
                   command=self.start_analysis).pack(pady=10)

    def on_database_change(self, event=None):
        """Handle database selection change"""
        new_db = self.db_var.get()
        try:
            # Close existing connection if any
            if hasattr(self, 'db_conn') and self.db_conn:
                self.db_conn.close()
            
            # Connect to new database
            self.db_conn = duckdb.connect(new_db)
            print(f"Connected to database: {new_db}")
            
            # Get available tables
            self.tables = self.get_tables()
            print(f"Found tables: {self.tables}")
            
            # Update table combobox
            self.table_combo['values'] = self.tables if self.tables else ['No tables available']
            if self.tables:
                self.table_combo.set(self.tables[0])
                # Trigger table change to update tickers
                self.on_table_change()
            else:
                self.table_combo.set('No tables available')
                self.clear_ticker_selection()
            
        except Exception as e:
            print(f"Error switching database: {e}")
            messagebox.showerror("Database Error", str(e))

    def on_table_change(self, event=None):
        """Handle table selection change"""
        if not hasattr(self, 'db_conn') or not self.db_conn:
            return
        
        table = self.table_var.get()
        if not table or table == 'No tables available':
            self.clear_ticker_selection()
            return
        
        try:
            # Get column information
            columns = self.db_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            column_names = [col[0] for col in columns]
            print(f"Available columns in {table}: {column_names}")
            
            # Check if table has ticker column
            if 'ticker' in column_names:
                # Get unique tickers from the table
                tickers = self.db_conn.execute(f"SELECT DISTINCT ticker FROM {table}").fetchall()
                tickers = [t[0] for t in tickers]
                print(f"Found tickers using column 'ticker': {tickers[:5]}...")
                
                # Update ticker combobox
                self.ticker_combo['values'] = tickers
                if tickers:
                    self.ticker_combo.set(tickers[0])
                    # Update description
                    self.update_ticker_description()
            else:
                self.clear_ticker_selection()
            
        except Exception as e:
            print(f"Error updating table selection: {e}")
            messagebox.showerror("Table Error", str(e))

    def clear_ticker_selection(self):
        """Clear ticker selection when no valid table/database is selected"""
        self.ticker_combo['values'] = ['No tickers available']
        self.ticker_combo.set('No tickers available')
        self.ticker_desc.config(text="")

    def update_ticker_description(self):
        """Update the description for the currently selected ticker"""
        ticker = self.ticker_var.get()
        description = self.ticker_descriptions.get(ticker, "No description available")
        self.ticker_desc.config(text=description)

    def get_tables(self):
        """Get list of non-empty tables from current database"""
        try:
            tables = self.db_conn.execute(
                """SELECT name FROM sqlite_master 
                   WHERE type='table' AND name != 'sqlite_sequence'"""
            ).fetchall()
            tables = [t[0] for t in tables]
            
            # Filter out empty tables
            non_empty_tables = []
            for table in tables:
                count = self.db_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count > 0:
                    non_empty_tables.append(table)
            
            print(f"Found non-empty tables: {non_empty_tables}")
            return non_empty_tables
        except Exception as e:
            print(f"Error getting tables: {e}")
            return []

    def get_tickers(self, table_name):
        """Get list of unique tickers/symbols from the specified table"""
        try:
            # First get all column names for the table
            columns_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = ?
            """
            columns = [row[0].lower() for row in self.db_conn.execute(columns_query, [table_name]).fetchall()]
            print(f"Available columns in {table_name}: {columns}")
            
            # Try different common column names for tickers
            ticker_columns = ['ticker', 'symbol', 'pair']
            
            for col in ticker_columns:
                if col in columns:
                    query = f"""
                        SELECT DISTINCT {col}
                        FROM {table_name}
                        WHERE {col} IS NOT NULL
                        ORDER BY {col}
                    """
                    result = self.db_conn.execute(query).fetchall()
                    if result:
                        tickers = [row[0] for row in result]
                        print(f"Found tickers using column '{col}': {tickers[:5]}...")
                        return tickers
            
            # Special handling for market_data table
            if 'type' in columns and table_name == 'market_data':
                query = """
                    SELECT DISTINCT ticker 
                    FROM market_data 
                    WHERE type = 'forex'
                    ORDER BY ticker
                """
                result = self.db_conn.execute(query).fetchall()
                if result:
                    tickers = [row[0] for row in result]
                    print(f"Found forex pairs: {tickers[:5]}...")
                    return tickers
            
            print(f"No suitable ticker column found in {table_name}")
            return []
            
        except Exception as e:
            print(f"Error getting tickers: {str(e)}")
            traceback.print_exc()
            return []

    def get_historical_data(self, ticker, table_name, timeframe):
        """Get historical price data for a ticker"""
        try:
            interval_map = {
                '1d': '1 day',
                '1mo': '1 month',
                '3mo': '3 months',
                '6mo': '6 months',
                '1y': '1 year'
            }
            interval = interval_map.get(timeframe, '1 month')
            
            # Get column information
            columns_query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = ?
            """
            columns = {row[0].lower(): row[1] for row in self.db_conn.execute(columns_query, [table_name]).fetchall()}
            
            # Determine ticker and date columns
            ticker_col = next((col for col in ['ticker', 'symbol', 'pair'] if col in columns), None)
            date_col = next((col for col in ['date', 'timestamp', 'created_at'] if col in columns), None)
            
            if not ticker_col or not date_col:
                raise Exception(f"Required columns not found in {table_name}")
            
            # Build query with available columns
            select_cols = [
                f"{date_col} as date",
                "open as Open",
                "high as High",
                "low as Low",
                "close as Close",
                "volume as Volume",
                "COALESCE(adj_close, close) as Adj_Close"
            ]
            
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM {table_name}
                WHERE {ticker_col} = ?
                AND {date_col} >= CURRENT_DATE - INTERVAL '{interval}'
                ORDER BY {date_col}
            """
            
            print(f"Executing query: {query}")
            print(f"Parameters: {[ticker]}")
            
            df = self.db_conn.execute(query, [ticker]).df()
            if df is not None and not df.empty:
                print(f"Retrieved {len(df)} rows of data")
            return df
            
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_rsi(self, prices, periods=14):
        """Calculate RSI for a price series"""
        try:
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=periods).mean()
            avg_losses = losses.rolling(window=periods).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return pd.Series([50] * len(prices))  # Return neutral RSI on error

    def start_analysis(self):
        try:
            ticker = self.ticker_var.get()
            duration = self.duration_var.get()
            
            self.loading_label.config(text=f"Loading data for {ticker}...")
            
            df = self.get_historical_data(ticker, self.table_var.get(), duration)
            
            if df is not None and not df.empty:
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['RSI'] = self.calculate_rsi(df['Close'])
                
                self.update_plots(df, ticker)
                self.loading_label.config(text=f"Analysis complete for {ticker}")
            else:
                self.loading_label.config(text=f"No data available for {ticker}")
                
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            self.loading_label.config(text=f"Error analyzing {ticker}: {str(e)}")

    @process_data_safely
    def update_plots(self, df, ticker):
        """Update all plots with current data"""
        try:
            if ticker == 'No tickers available' or self.table_var.get() == 'No tables available':
                return
            
            print(f"Updating plots for {ticker} from {self.table_var.get()}")
            
            # Clear previous plots
            self.figure.clear()
            
            # Create subplots
            ax_price = self.figure.add_subplot(211)  # Price plot
            ax_rsi = self.figure.add_subplot(212)    # RSI plot
            
            # Get data for different timeframes
            timeframes = ['1mo', '3mo', '6mo', '1y']
            colors = ['blue', 'green', 'red', 'purple']
            
            for timeframe, color in zip(timeframes, colors):
                df_timeframe = self.get_historical_data(ticker, self.table_var.get(), timeframe)
                if df_timeframe is not None and not df_timeframe.empty:
                    # Convert date column to datetime if it's not already
                    df_timeframe['date'] = pd.to_datetime(df_timeframe['date'])
                    
                    # Plot price data
                    ax_price.plot(df_timeframe['date'], df_timeframe['Close'], label=f'{timeframe} Close', color=color, alpha=0.7)
                    
                    # Calculate and plot RSI
                    rsi = self.calculate_rsi(df_timeframe['Close'])
                    ax_rsi.plot(df_timeframe['date'], rsi, label=f'{timeframe} RSI', color=color, alpha=0.7)
            
            # Customize price plot
            ax_price.set_title(f'{ticker} Price History')
            ax_price.set_xlabel('Date')
            ax_price.set_ylabel('Price')
            ax_price.grid(True)
            ax_price.legend()
            
            # Format x-axis dates
            ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Customize RSI plot
            ax_rsi.set_title('RSI Indicator')
            ax_rsi.set_xlabel('Date')
            ax_rsi.set_ylabel('RSI')
            ax_rsi.grid(True)
            ax_rsi.set_ylim([0, 100])
            
            # Add RSI reference lines
            ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax_rsi.legend()
            
            # Format x-axis dates for RSI
            ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_rsi.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Adjust layout and display
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
            traceback.print_exc()

    def find_duckdb_databases(self):
        """Find all DuckDB database files in current directory"""
        try:
            dbs = []
            for file in os.listdir('.'):
                if file.endswith('.db'):
                    try:
                        # Try to connect to verify it's a valid DuckDB database
                        test_conn = duckdb.connect(file)
                        test_conn.close()
                        dbs.append(file)
                    except:
                        continue
            return dbs
        except Exception as e:
            print(f"Error finding databases: {str(e)}")
            return ['stocks.db']

    def refresh_databases(self):
        """Refresh the list of available databases"""
        try:
            self.available_dbs = find_databases()
            print(f"Refreshed database list: {self.available_dbs}")
            
            self.db_combo['values'] = self.available_dbs
            if self.available_dbs:
                if self.db_var.get() not in self.available_dbs:
                    self.db_var.set(self.available_dbs[0])
                    self.on_database_change()
        except Exception as e:
            print(f"Error refreshing databases: {e}")
            messagebox.showerror("Refresh Error", str(e))

    def run(self):
        self.root.mainloop()

    def update_tickers(self):
        """Update available tickers based on current table selection"""
        if not hasattr(self, 'db_conn') or not self.db_conn:
            self.clear_ticker_selection()
            return
            
        table = self.table_var.get()
        if not table or table == 'No tables available':
            self.clear_ticker_selection()
            return
            
        try:
            # Get column information
            columns = self.db_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            column_names = [col[0] for col in columns]
            
            # Check if table has ticker column
            if 'ticker' in column_names:
                # Get unique tickers from the table
                tickers = self.db_conn.execute(f"SELECT DISTINCT ticker FROM {table}").fetchall()
                tickers = [t[0] for t in tickers]
                
                # Update ticker combobox
                self.ticker_combo['values'] = tickers
                if tickers:
                    self.ticker_combo.set(tickers[0])
                    self.update_ticker_description()
                else:
                    self.clear_ticker_selection()
            else:
                self.clear_ticker_selection()
                
        except Exception as e:
            print(f"Error updating tickers: {e}")
            self.clear_ticker_selection()

def process_database(db_name):
    with contextlib.closing(create_connection(db_name)) as conn:
        if conn:
            print(f"Processing database: {db_name}")
            try:
                # Get list of tables
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                tables = [t[0] for t in tables]
                
                if not tables:
                    print(f"No tables found in {db_name}")
                    return
                
                print(f"Found tables: {tables}")
                
                for table in tables:
                    # Get column information
                    columns = conn.execute(f"SELECT * FROM {table} LIMIT 0").description
                    column_names = [col[0] for col in columns]
                    print(f"Columns in {table}: {column_names}")
                    
                    # Get sample data
                    sample = conn.execute(f"""
                        SELECT COUNT(*) as count, 
                               MIN(date) as earliest_date,
                               MAX(date) as latest_date 
                        FROM {table}
                    """).fetchone()
                    
                    print(f"Table {table} statistics:")
                    print(f"  Total records: {sample[0]}")
                    print(f"  Date range: {sample[1]} to {sample[2]}")
                    
                    # If table has ticker column, show unique tickers
                    if 'ticker' in column_names:
                        tickers = conn.execute(f"SELECT DISTINCT ticker FROM {table}").fetchall()
                        tickers = [t[0] for t in tickers]
                        print(f"  Available tickers: {tickers[:5]}...")
                    
                    print()
                    
            except Exception as e:
                print(f"Error processing {db_name}: {e}")

@process_data_safely
def initialize_gui(databases):
    app = StockAnalyzerGUI(databases)
    return app

def main():
    try:
        databases = find_databases()
        print(f"Found databases: {databases}")
        
        # Process databases in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(process_database, databases)
        
        # Initialize GUI
        app = initialize_gui(databases)
        app.root.mainloop()
        
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
