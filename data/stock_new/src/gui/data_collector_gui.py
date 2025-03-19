import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import yfinance as yf
import polars as pl
import duckdb
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import threading
import time
from typing import List
import tkinter.messagebox as messagebox

class DataCollectorGUI:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_collector = DataCollector()  # Initialize data collector
        self.data_loader = None  # Initialize data_loader as None
        self.root = tk.Tk()
        self.root.title("Market Data Collector")
        
        # Initialize components
        self.setup_gui_components()
        
    def setup_gui_components(self):
        # Ticker input frame
        ticker_frame = ttk.LabelFrame(self.root, text="Ticker Input")
        ticker_frame.pack(padx=5, pady=5, fill="x")
        
        ttk.Label(ticker_frame, text="Enter Tickers (comma-separated):").pack(side="left", padx=5)
        self.ticker_entry = ttk.Entry(ticker_frame, width=50)
        self.ticker_entry.pack(side="left", padx=5)
        
        # Date range frame
        date_frame = ttk.LabelFrame(self.root, text="Date Range")
        date_frame.pack(padx=5, pady=5, fill="x")
        
        # Start date
        ttk.Label(date_frame, text="Start Date:").pack(side="left", padx=5)
        self.start_date = DateEntry(date_frame, width=12)
        self.start_date.pack(side="left", padx=5)
        
        # End date
        ttk.Label(date_frame, text="End Date:").pack(side="left", padx=5)
        self.end_date = DateEntry(date_frame, width=12)
        self.end_date.pack(side="left", padx=5)
        
        # Format selection frame
        format_frame = ttk.LabelFrame(self.root, text="Output Format")
        format_frame.pack(padx=5, pady=5, fill="x")
        
        # Format selection
        ttk.Label(format_frame, text="Save as:").pack(side="left", padx=5)
        self.format_var = tk.StringVar(value="csv")
        self.format_combo = ttk.Combobox(
            format_frame, 
            textvariable=self.format_var,
            values=["csv", "json", "duckdb", "csv+duckdb", "json+duckdb"],
            width=15,
            state="readonly"
        )
        self.format_combo.pack(side="left", padx=5)
        
        # Download button
        self.download_button = ttk.Button(
            format_frame,
            text="Download Data",
            command=self.download_data
        )
        self.download_button.pack(side="left", padx=5, pady=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Status")
        status_frame.pack(padx=5, pady=5, fill="x")
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate')
        self.progress_bar.pack(fill="x", padx=5, pady=5)

        # Add historical data collection button
        self.historical_button = ttk.Button(
            self.root,
            text="Collect Historical",
            command=self.collect_historical_data
        )
        self.historical_button.pack(pady=5)

        # Setup real-time collection components
        self.setup_realtime_collection()

    def setup_realtime_collection(self):
        """Setup real-time collection components."""
        # Create frame for real-time collection
        realtime_frame = ttk.LabelFrame(self.root, text="Real-time Collection", padding="5 5 5 5")
        realtime_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Add start/stop button
        self.realtime_status = tk.StringVar(value="Start Real-time")
        self.realtime_button = ttk.Button(
            realtime_frame, 
            textvariable=self.realtime_status,
            command=self.toggle_realtime_collection
        )
        self.realtime_button.grid(row=0, column=0, padx=5, pady=5)

        # Add interval selection
        ttk.Label(realtime_frame, text="Interval:").grid(row=0, column=1, padx=5, pady=5)
        self.interval_var = tk.StringVar(value="1m")
        interval_combo = ttk.Combobox(
            realtime_frame, 
            textvariable=self.interval_var,
            values=["1m", "2m", "5m", "15m", "30m", "60m", "90m"]
        )
        interval_combo.grid(row=0, column=2, padx=5, pady=5)

        # Initialize real-time collection attributes
        self.realtime_running = False
        self.realtime_thread = None

    def toggle_realtime_collection(self):
        """Toggle real-time data collection."""
        if not self.realtime_running:
            # Get selected tickers
            selected_indices = self.ticker_list.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "Please select at least one ticker")
                return

            selected_tickers = [self.ticker_list.get(idx) for idx in selected_indices]
            
            # Start collection
            self.realtime_running = True
            self.realtime_status.set("Stop Real-time")
            self.realtime_thread = threading.Thread(
                target=self.realtime_collection_worker,
                args=(selected_tickers, self.interval_var.get()),
                daemon=True
            )
            self.realtime_thread.start()
            self.log_message("Started real-time collection")
        else:
            # Stop collection
            self.realtime_running = False
            self.realtime_status.set("Start Real-time")
            if self.realtime_thread:
                self.realtime_thread.join(timeout=1.0)
            self.log_message("Stopped real-time collection")

    def realtime_collection_worker(self, tickers: List[str], interval: str):
        """Worker function for real-time data collection using YFinance."""
        try:
            while self.realtime_running:
                start_time = time.time()
                
                try:
                    # Download latest data
                    data = yf.download(
                        tickers=tickers,
                        period="1d",  # Get today's data
                        interval=interval,
                        group_by='ticker',
                        auto_adjust=False,
                        prepost=False
                    )

                    # Process each ticker's data
                    for ticker in tickers:
                        try:
                            # Extract single ticker data
                            if len(tickers) > 1:
                                ticker_data = data[ticker].copy()
                            else:
                                ticker_data = data.copy()
                            
                            # Reset index to make Date a column
                            ticker_data = ticker_data.reset_index()
                            
                            # Rename columns
                            column_mapping = {
                                'Datetime': 'date',
                                'Date': 'date',
                                'Open': 'open',
                                'High': 'high',
                                'Low': 'low',
                                'Close': 'close',
                                'Adj Close': 'adj_close',
                                'Volume': 'volume'
                            }
                            ticker_data = ticker_data.rename(columns=column_mapping)
                            
                            # Add ticker column
                            ticker_data['ticker'] = ticker
                            
                            # Ensure correct column order
                            columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                            ticker_data = ticker_data[columns]
                            
                            # Save to database
                            self.data_collector.save_ticker_data(ticker, ticker_data, realtime=True)
                            
                            self.log_message(f"Updated {ticker} - {len(ticker_data)} new records")
                            
                        except Exception as e:
                            self.log_message(f"Error processing {ticker}: {str(e)}")
                    
                except Exception as e:
                    self.log_message(f"Error downloading real-time data: {str(e)}")
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - start_time
                interval_seconds = int(interval[:-1]) * 60  # Convert interval to seconds
                sleep_time = max(0, interval_seconds - elapsed)
                
                # Sleep until next update
                time.sleep(sleep_time)
                
        except Exception as e:
            self.log_message(f"Real-time collection error: {str(e)}")
            self.realtime_running = False
            self.realtime_status.set("Start Real-time")

    def download_data(self):
        """Download data in the selected format"""
        try:
            tickers = [t.strip() for t in self.ticker_entry.get().split(',')]
            start_date = self.start_date.get_date().strftime('%Y-%m-%d')
            end_date = self.end_date.get_date().strftime('%Y-%m-%d')
            format_choice = self.format_var.get()
            
            # Create necessary directories
            Path("data/csv").mkdir(parents=True, exist_ok=True)
            Path("data/json").mkdir(parents=True, exist_ok=True)
            
            downloaded_data = {}
            
            # Download data for all tickers
            for ticker in tickers:
                self.status_label.config(text=f"Downloading {ticker}...")
                self.root.update()
                
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if data.empty:
                    self.logger.warning(f"No data returned for {ticker}")
                    continue
                    
                downloaded_data[ticker] = data
                
                # Save based on format choice
                if format_choice in ['csv', 'csv+duckdb']:
                    csv_path = f"data/csv/{ticker}_data.csv"
                    data.to_csv(csv_path)
                    self.logger.info(f"Saved {ticker} data to {csv_path}")
                    
                if format_choice in ['json', 'json+duckdb']:
                    json_path = f"data/json/{ticker}_data.json"
                    data.to_json(json_path, date_format='iso')
                    self.logger.info(f"Saved {ticker} data to {json_path}")
            
            # Save to DuckDB if selected
            if format_choice in ['duckdb', 'csv+duckdb', 'json+duckdb']:
                self.status_label.config(text="Converting to DuckDB...")
                self.root.update()
                
                conn = duckdb.connect("data/market_data.duckdb")
                
                # Create table with correct schema
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS stock_data (
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
                """
                conn.execute(create_table_sql)
                
                # Insert data for each ticker
                for ticker, data in downloaded_data.items():
                    # Prepare DataFrame
                    df = data.reset_index()
                    df['ticker'] = ticker
                    
                    # Rename columns
                    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                    
                    # Ensure correct column order
                    columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                    df = df[columns]
                    
                    # Insert into DuckDB
                    conn.execute("DELETE FROM stock_data WHERE ticker = ?", [ticker])
                    conn.execute("INSERT INTO stock_data SELECT * FROM df")
                
                conn.close()
                self.logger.info("Data saved to DuckDB")
            
            self.status_label.config(text=f"Download completed! Data saved in {format_choice} format(s)")
            
        except Exception as e:
            self.logger.error(f"Error downloading data: {e}")
            self.status_label.config(text=f"Error: {str(e)}")

    def import_csv_files(self):
        """Import CSV files from the data/csv directory"""
        try:
            csv_dir = Path("data/csv")
            csv_files = list(csv_dir.glob("*.csv"))
            
            if not csv_files:
                self.logger.info("No CSV files found in data/csv directory")
                return
                
            self.logger.info(f"\nStarting import of {len(csv_files)} CSV files...\n")
            
            success_count = 0
            fail_count = 0
            
            for csv_file in csv_files:
                try:
                    self.logger.info(f"\nProcessing {csv_file.name}...")
                    
                    # Read CSV file, skipping the first 2 rows
                    df = pd.read_csv(csv_file, skiprows=[1, 2])
                    self.logger.info(f"Found {len(df)} rows of data")
                    
                    # Convert date column
                    df['Date'] = pd.to_datetime(df['Date'])
                    self.logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
                    
                    # Create new dataframe with correct structure
                    processed_df = pd.DataFrame({
                        'date': df['Date'],
                        'ticker': df['ticker'],  # Use ticker from data
                        'open': df['Open'],
                        'high': df['High'],
                        'low': df['Low'],
                        'close': df['Close'],
                        'adj_close': df['Price'],  # Use Price as adj_close
                        'volume': df['Volume'].astype('int64')  # Ensure volume is integer
                    })
                    
                    # Ensure columns are in correct order
                    columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                    processed_df = processed_df[columns]
                    
                    # Log the column structure
                    self.logger.info(f"Processed columns: {processed_df.columns.tolist()}")
                    
                    # Save to database
                    self.data_collector.cursor.execute("""
                        INSERT INTO stock_data 
                        SELECT * FROM read_pandas(?)
                    """, [processed_df])
                    
                    self.data_collector.connection.commit()
                    success_count += 1
                    self.logger.info(f"Successfully imported {csv_file.name}")
                    
                except Exception as e:
                    self.logger.error(f"Error importing {csv_file.name}: {e}")
                    fail_count += 1
                    
            self.logger.info(f"\nImport Summary:")
            self.logger.info(f"Total files processed: {len(csv_files)}")
            self.logger.info(f"Successfully imported: {success_count}")
            self.logger.info(f"Failed imports: {fail_count}\n")
            
        except Exception as e:
            self.logger.error(f"Error during CSV import: {e}")

    def download_yfinance_data(self):
        """Download data from YFinance"""
        try:
            tickers = self.ticker_entry.get().split(',')
            tickers = [t.strip().upper() for t in tickers]
            
            if not tickers:
                self.logger.warning("No tickers specified")
                return
                
            start_date = self.start_date.get_date()
            end_date = self.end_date.get_date()
            
            self.logger.info(f"Downloading data for {len(tickers)} tickers...")
            
            for ticker in tickers:
                try:
                    # Download data
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if data.empty:
                        self.logger.warning(f"No data found for {ticker}")
                        continue
                        
                    # Save to database
                    data.index.name = 'date'
                    data = data.reset_index()
                    self.data_collector.save_ticker_data(ticker, data)
                    
                except Exception as e:
                    self.logger.error(f"Error downloading {ticker}: {e}")
                    
            self.logger.info("YFinance download completed")
            
        except Exception as e:
            self.logger.error(f"Error during YFinance download: {e}")

    def collect_historical_data(self):
        """Collect historical data for specified tickers"""
        try:
            tickers = self.ticker_entry.get().split(',')
            tickers = [t.strip().upper() for t in tickers]
            
            if not tickers:
                self.logger.warning("No tickers specified")
                return
                
            start_date = self.start_date.get_date()
            end_date = self.end_date.get_date()
            
            for ticker in tickers:
                self.logger.info(f"Collecting historical data for {ticker}...")
                try:
                    # Download data using yfinance
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if data.empty:
                        self.logger.warning(f"No historical data found for {ticker}")
                        continue
                        
                    # Save to database
                    data.index.name = 'date'
                    data = data.reset_index()
                    self.data_collector.save_ticker_data(ticker, data)
                    self.logger.info(f"Successfully saved historical data for {ticker}")
                    
                except Exception as e:
                    self.logger.error(f"Error collecting historical data for {ticker}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in historical collection: {e}")

    def run(self):
        self.root.mainloop() 