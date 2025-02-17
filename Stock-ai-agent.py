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
            self.tables = self.get_available_tables()
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
                self.tables = self.get_available_tables()
                self.table_combo['values'] = self.tables
                if self.tables:
                    self.table_combo.set(self.tables[0])
                    self.update_tickers()
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

    def get_tickers(self, table=None):
        """Get list of available tickers from specified table"""
        try:
            if not table:
                table = self.table_var.get()
            
            print(f"Attempting to get tickers from {table}...")
            
            # First check if ticker column exists
            columns = self.db_conn.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """).fetchall()
            columns = [col[0].lower() for col in columns]
            
            # Use 'ticker' or 'symbol' column if available
            ticker_col = 'ticker' if 'ticker' in columns else 'symbol'
            
            if ticker_col not in columns:
                print(f"No ticker/symbol column found in {table}")
                return []
            
            query = f"SELECT DISTINCT {ticker_col} FROM {table} ORDER BY {ticker_col}"
            result = self.db_conn.execute(query).fetchall()
            tickers = [ticker[0] for ticker in result] if result else []
            
            print(f"Found {len(tickers)} tickers: {tickers[:5]}...")
            return tickers
            
        except Exception as e:
            print(f"Error getting tickers: {str(e)}")
            return []

    def get_historical_data(self, ticker, duration):
        """Get historical data from selected table"""
        try:
            table = self.table_var.get()
            print(f"Retrieving data for {ticker} from {table} over {duration}")
            
            # Check table columns
            columns = self.db_conn.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """).fetchall()
            columns = [col[0].lower() for col in columns]
            
            # Map column names
            ticker_col = 'ticker' if 'ticker' in columns else 'symbol'
            date_col = 'date' if 'date' in columns else 'timestamp'
            
            duration_map = {
                '1d': 'INTERVAL 1 day',
                '1mo': 'INTERVAL 1 month',
                '3mo': 'INTERVAL 3 months',
                '6mo': 'INTERVAL 6 months',
                '1y': 'INTERVAL 1 year'
            }
            
            interval = duration_map.get(duration, 'INTERVAL 1 month')
            
            query = f"""
            SELECT 
                {date_col} as date,
                open as Open,
                high as High,
                low as Low,
                close as Close,
                volume as Volume,
                adj_close as Adj_Close
            FROM {table}
            WHERE {ticker_col} = ?
            AND {date_col} >= CURRENT_DATE - {interval}
            ORDER BY {date_col}
            """
            
            print(f"Executing query: {query}")
            df = self.db_conn.execute(query, [ticker]).df()
            
            if df.empty:
                print(f"No data found for {ticker}")
                return None
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            print(f"Retrieved {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def start_analysis(self):
        try:
            ticker = self.ticker_var.get()
            duration = self.duration_var.get()
            
            self.loading_label.config(text=f"Loading data for {ticker}...")
            
            df = self.get_historical_data(ticker, duration)
            
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
        try:
            self.figure.clear()
            
            # Create subplots with gridspec
            gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
            ax_candle = self.figure.add_subplot(gs[0])
            ax_volume = self.figure.add_subplot(gs[1])
            ax_ma = self.figure.add_subplot(gs[2])
            ax_rsi = self.figure.add_subplot(gs[3])
            
            # Prepare data for candlestick
            ohlc = []
            for date, row in df.iterrows():
                date_num = mdates.date2num(date)
                ohlc.append([date_num, row['Open'], row['High'], row['Low'], row['Close']])
            
            # Plot candlestick
            candlestick_ohlc(ax_candle, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)
            
            # Plot volume
            ax_volume.bar(df.index, df['Volume'], color='gray', alpha=0.5)
            
            # Plot moving averages
            ax_ma.plot(df.index, df['SMA_20'], label='SMA 20', color='blue')
            ax_ma.plot(df.index, df['SMA_50'], label='SMA 50', color='red')
            
            # Plot RSI
            ax_rsi.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax_rsi.axhline(y=70, color='red', linestyle='--')
            ax_rsi.axhline(y=30, color='green', linestyle='--')
            
            # Set titles and labels
            ax_candle.set_title(f'{ticker} Technical Analysis')
            ax_volume.set_ylabel('Volume')
            ax_ma.set_ylabel('Price')
            ax_rsi.set_ylabel('RSI')
            
            # Format dates for all subplots
            for ax in [ax_candle, ax_volume, ax_ma, ax_rsi]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(rotation=45)
                plt.setp(ax.get_xticklabels(), ha='right')
            
            # Add legends
            ax_ma.legend(['SMA 20', 'SMA 50'])
            ax_rsi.legend(['RSI'])
            
            # Set RSI range
            ax_rsi.set_ylim([0, 100])
            
            # Adjust layout
            self.figure.tight_layout()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            raise e

    def get_available_tables(self):
        """Get list of available tables in the current database"""
        try:
            # Get all tables
            tables = self.db_conn.execute("""
                SELECT table_name, column_count
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchall()
            
            # Filter for tables with required columns
            valid_tables = []
            for table, col_count in tables:
                # Check if table has necessary columns
                columns = self.db_conn.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                """).fetchall()
                columns = [col[0].lower() for col in columns]
                
                # Check for required columns (ticker/symbol and price data)
                has_ticker = 'ticker' in columns or 'symbol' in columns
                has_date = 'date' in columns or 'timestamp' in columns
                has_price = all(col in columns for col in ['open', 'high', 'low', 'close', 'adj_close'])
                has_volume = 'volume' in columns
                
                if has_ticker and has_date and has_price and has_volume:
                    valid_tables.append(table)
            
            print(f"Found {len(valid_tables)} valid tables: {valid_tables}")
            return valid_tables
        except Exception as e:
            print(f"Error getting tables: {str(e)}")
            return []

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
