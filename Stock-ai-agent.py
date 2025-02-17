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
matplotlib.use('TkAgg')

class StockAnalyzerGUI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("Stock Market Analyzer")
        self.root.geometry("1600x1000")
        
        try:
            # Find all DuckDB databases in current directory
            self.available_dbs = self.find_duckdb_databases()
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
        
        # Ticker selection
        ticker_frame = ttk.LabelFrame(self.control_panel, text="Stock Selection", padding="5")
        ticker_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(ticker_frame, text="Ticker:").pack(side="left", padx=5)
        self.ticker_var = tk.StringVar()
        self.ticker_combo = ttk.Combobox(ticker_frame, textvariable=self.ticker_var)
        self.ticker_combo.pack(side="left", fill="x", expand=True, padx=5)
        
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
        try:
            new_db = self.db_var.get()
            if new_db != self.current_db:
                # Close current connection
                self.db_conn.close()
                
                # Connect to new database
                self.current_db = new_db
                self.db_conn = duckdb.connect(self.current_db)
                print(f"Connected to database: {self.current_db}")
                
                # Update tables
                self.tables = self.get_tables()
                self.table_combo['values'] = self.tables
                if self.tables:
                    self.table_combo.set(self.tables[0])
                    self.update_tickers()
                    # Clear previous plots
                    self.figure.clear()
                    self.canvas.draw()
                else:
                    self.table_combo.set('No tables available')
                    self.ticker_combo['values'] = ['No tickers available']
                    self.ticker_combo.set('No tickers available')
                
        except Exception as e:
            print(f"Error changing database: {str(e)}")
            messagebox.showerror("Database Error", f"Error connecting to {new_db}: {str(e)}")

    def refresh_databases(self):
        """Refresh the list of available databases"""
        try:
            self.available_dbs = self.find_duckdb_databases()
            self.db_combo['values'] = self.available_dbs
            print(f"Refreshed database list: {self.available_dbs}")
        except Exception as e:
            print(f"Error refreshing databases: {str(e)}")

    def on_table_change(self, event=None):
        """Handle table selection change"""
        self.update_tickers()

    def update_tickers(self):
        """Update ticker list based on selected table"""
        table = self.table_var.get()
        tickers = self.get_tickers(table)
        
        if tickers:
            self.ticker_combo['values'] = tickers
            self.ticker_combo.set(tickers[0])
        else:
            self.ticker_combo['values'] = ['No tickers available']
            self.ticker_combo.set('No tickers available')

    def get_tables(self):
        """Get list of tables in the current database"""
        try:
            query = """
                SELECT DISTINCT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                AND (
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_name = tables.table_name
                ) > 0
                ORDER BY table_name
            """
            result = self.db_conn.execute(query).fetchall()
            tables = [row[0] for row in result]
            
            # Filter out empty tables
            non_empty_tables = []
            for table in tables:
                count_query = f"SELECT COUNT(*) FROM {table}"
                count = self.db_conn.execute(count_query).fetchone()[0]
                if count > 0:
                    non_empty_tables.append(table)
            
            print(f"Found non-empty tables: {non_empty_tables}")
            return non_empty_tables
        except Exception as e:
            print(f"Error getting tables: {str(e)}")
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

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    analyzer = None
    gui = StockAnalyzerGUI(analyzer)
    gui.run()
