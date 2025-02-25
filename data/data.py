import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import polars as pl
import duckdb
from datetime import datetime, timedelta
import pandas as pd
import requests
import time
import numpy as np
import random
import sys
import os
import tkinter as tk
from tkinter import messagebox
import matplotlib.dates as mdates
from tqdm.asyncio import tqdm
from tqdm import tqdm as tqdm_sync
import logging
from io import StringIO

class DataManager:
    def __init__(self, db_path='stocks.db'):
        self.db_path = db_path
        self.failed_downloads = []
        self.session = None
        self.error_buffer = StringIO()
        
        # Configure logging
        self.logger = logging.getLogger('DataManager')
        self.logger.setLevel(logging.ERROR)
        handler = logging.StreamHandler(self.error_buffer)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # Delete existing database file if it exists
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Deleted existing database: {db_path}")

        self.rate_limit = asyncio.Semaphore(5)  # Limit concurrent requests
        self.delay = 0.5  # Delay between requests in seconds

    async def initialize_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit_per_host=10)  # Limit concurrent connections
            self.session = aiohttp.ClientSession(connector=connector)

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_nyse_tickers_async(self):
        """Async version of get_nyse_tickers"""
        try:
            url = "https://www.nyse.com/api/quotes/filter"
            payload = {
                "instrumentType": "EQUITY",
                "pageNumber": 1,
                "sortColumn": "SYMBOL",
                "sortOrder": "ASC",
                "maxResultsPerPage": 10000,
                "filterToken": ""
            }
            headers = {"Content-Type": "application/json"}
            
            await self.initialize_session()
            async with self.session.post(url, json=payload, headers=headers) as response:
                data = await response.json()
                tickers = [item['symbolTicker'] for item in data]
                print(f"Found {len(tickers)} NYSE tickers")
                return sorted(tickers)
        except Exception as e:
            print(f"Error fetching NYSE tickers: {str(e)}")
            return []

    async def _download_stock_data_raw(self, ticker, retries=3):
        """Downloads raw stock data from yfinance"""
        for attempt in range(retries):
            try:
                # Add delay between attempts
                if attempt > 0:
                    await asyncio.sleep(self.delay * (2 ** attempt))
                
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor() as pool:
                    df = await loop.run_in_executor(
                        pool,
                        lambda: yf.download(ticker, progress=False, period="1y")
                    )
                
                if len(df) == 0:
                    self.logger.error(f"Empty dataframe for {ticker}")
                    return None
                
                return df
                
            except Exception as e:
                if attempt < retries - 1:
                    self.logger.warning(f"Retry {attempt + 1} for {ticker}: {str(e)}")
                    await asyncio.sleep(self.delay * (2 ** attempt))
                else:
                    self.logger.error(f"Failed to download {ticker}: {str(e)}")
                    return None

    def _process_stock_data(self, df, ticker):
        """Processes raw stock data into standardized format"""
        try:
            # Handle multi-index columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure column names are unique and valid
            df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
            
            # Convert to polars and handle column types
            df = pl.from_pandas(df)
            df = df.with_columns([
                pl.col('Open').cast(pl.Float64),
                pl.col('High').cast(pl.Float64),
                pl.col('Low').cast(pl.Float64),
                pl.col('Close').cast(pl.Float64),
                pl.col('Adj_Close').cast(pl.Float64),
                pl.col('Volume').cast(pl.Float64)
            ])
            
            # Add ticker column
            df = df.with_columns(pl.lit(ticker).alias('Symbol'))
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing {ticker} data: {str(e)}")
            return None

    async def download_stock_data_async(self, ticker, retries=3):
        """Async version of download_stock_data"""
        async with self.rate_limit:  # Rate limit downloads
            # Download raw data
            raw_df = await self._download_stock_data_raw(ticker, retries)
            if raw_df is None:
                self.failed_downloads.append(ticker)
                return None
            
            # Process the data
            processed_df = self._process_stock_data(raw_df, ticker)
            if processed_df is None:
                self.failed_downloads.append(ticker)
                return None
            
            return processed_df

    async def process_letter_group_async(self, tickers_for_letter, letter):
        """Process a group of tickers for a single letter asynchronously"""
        # Reduce batch size to avoid rate limits
        batch_size = 5  # Reduced from 10
        
        tasks = [self.download_stock_data_async(ticker) for ticker in tickers_for_letter]
        letter_data = []
        
        # Create progress bars
        overall_pbar = tqdm(
            total=len(tasks),
            desc=f"Letter {letter}",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks)
            successful_results = [df for df in batch_results if df is not None]
            letter_data.extend(successful_results)
            
            # Update progress
            overall_pbar.update(len(batch_tasks))
            
            # Display accumulated errors below progress bar
            if self.error_buffer.getvalue():
                print("\nErrors:", flush=True)
                print(self.error_buffer.getvalue(), flush=True)
                self.error_buffer.truncate(0)
                self.error_buffer.seek(0)
            
            # Add delay between batches
            await asyncio.sleep(self.delay)
        
        overall_pbar.close()
        
        if letter_data:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                await loop.run_in_executor(pool, self.store_in_duckdb, letter_data)
        
        return [t for t in tickers_for_letter if t in self.failed_downloads]

    async def process_all_data_async(self, ticker_groups):
        """Process all letter groups asynchronously"""
        all_failures = []
        
        try:
            # Calculate total tickers for overall progress
            total_tickers = sum(len(tickers) for tickers in ticker_groups.values())
            
            print("\nStarting download process...")
            print("=" * 80)
            
            for letter in sorted(ticker_groups.keys()):
                tickers = ticker_groups[letter]
                failures = await self.process_letter_group_async(tickers, letter)
                all_failures.extend(failures)
                
                if failures:
                    print(f"\nFailed downloads for letter {letter}: {len(failures)}")
                
                await asyncio.sleep(2)
            
            print("\n" + "=" * 80)
            print("Download process completed!")
            
        except KeyboardInterrupt:
            print("\nProcess interrupted. Saving progress...")
        finally:
            await self.close_session()
            
        return all_failures

    def get_nyse_tickers(self):
        """Get list of NYSE tickers from NYSE website"""
        try:
            # Get NYSE listed stocks from NYSE website
            url = "https://www.nyse.com/api/quotes/filter"
            payload = {
                "instrumentType": "EQUITY",
                "pageNumber": 1,
                "sortColumn": "SYMBOL",
                "sortOrder": "ASC",
                "maxResultsPerPage": 10000,
                "filterToken": ""
            }
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            data = response.json()
            
            # Extract symbols
            tickers = [item['symbolTicker'] for item in data]
            print(f"Found {len(tickers)} NYSE tickers")
            return sorted(tickers)
        except Exception as e:
            print(f"Error fetching NYSE tickers: {str(e)}")
            return []

    def download_stock_data(self, ticker, start_date, end_date):
        try:
            print(f"\nProcessing {ticker}...", end='')
            # Download data
            df = yf.download(
                ticker,
                progress=False,
                period="1y"
            )
            
            if len(df) == 0:
                print(" Empty dataframe")
                self.failed_downloads.append(ticker)
                return None
                
            # Convert pandas DataFrame to polars and ensure all columns are float
            df = pl.from_pandas(df).with_columns([
                pl.col('Open').cast(pl.Float64),
                pl.col('High').cast(pl.Float64),
                pl.col('Low').cast(pl.Float64),
                pl.col('Close').cast(pl.Float64),
                pl.col('Adj Close').cast(pl.Float64),
                pl.col('Volume').cast(pl.Float64)
            ])
            
            # Calculate trading days coverage
            trading_days = len(df)
            if trading_days == 0:
                print(" No data found")
                self.failed_downloads.append(ticker)
                return None
                
            print(f" Done! ({trading_days} days)")
            
            # Add ticker column
            df = df.with_columns(pl.lit(ticker).alias('Symbol'))
            
            return df
            
        except Exception as e:
            print(f"\nError processing {ticker}: {str(e)}")
            self.failed_downloads.append(ticker)
            return None

    def process_tickers(self, tickers, start_date, end_date):
        """Process a group of tickers"""
        all_data = []
        download_stats = []

        for ticker in tickers:
            print(f"Downloading {ticker}...", end=" ")
            
            ticker_data = self.download_stock_data(ticker, start_date, end_date)
            
            if ticker_data is not None:
                all_data.append(ticker_data)
                
                # Calculate statistics
                trading_days = len(ticker_data)
                calendar_days = (end_date - start_date).days
                coverage_pct = round((trading_days / calendar_days) * 100, 2)
                
                download_stats.append({
                    'Symbol': ticker,
                    'Trading_Days': trading_days,
                    'Calendar_Days': calendar_days,
                    'Coverage_Pct': coverage_pct,
                    'First_Date': ticker_data.index[0],
                    'Last_Date': ticker_data.index[-1]
                })
                print("Done")
            else:
                print("Failed!")
                
            time.sleep(1)  # Rate limiting
        
        if not all_data:
            return None, self.failed_downloads, download_stats
            
        # Combine all data using join instead of concat
        combined_data = all_data[0]
        for df in all_data[1:]:
            combined_data = combined_data.join(df, on='Date', how='outer')
        
        return combined_data, self.failed_downloads, download_stats

    def process_ticker(self, ticker, max_retries=3, retry_delay=5, timeout=30):
        """Process a single ticker with retries"""
        for attempt in range(max_retries):
            try:
                # Download data with increased timeout
                df = yf.download(ticker, period="1y", progress=False, timeout=timeout)
                
                if len(df) == 0:
                    print(f" Empty dataframe")
                    return None
                
                # Debug print
                print(f" Downloaded columns: {df.columns.tolist()}")
                
                # Handle multi-index columns if they exist
                if isinstance(df.columns, pd.MultiIndex):
                    # Take first level of multi-index columns
                    df.columns = df.columns.get_level_values(0)
                
                # Reset index to make date a column
                df = df.reset_index()
                
                # Add ticker column
                df['ticker'] = ticker
                
                # Print DataFrame info for debugging
                print(f" DataFrame columns after processing: {df.columns.tolist()}")
                
                # Handle column names (case-insensitive)
                column_mapping = {
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Adj Close': 'adj_close',
                    'Volume': 'volume'
                }
                
                # Rename columns
                df = df.rename(columns=column_mapping)
                
                # If adj_close is missing, use close
                if 'adj_close' not in df.columns and 'close' in df.columns:
                    df['adj_close'] = df['close']
                
                # Ensure all required columns exist
                required_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                
                # Check for missing columns
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Select columns in specific order
                df = df[required_columns]
                
                # Convert to polars
                df = pl.from_pandas(df)
                
                # Cast columns to correct types
                df = df.with_columns([
                    pl.col('date').cast(pl.Date),
                    pl.col('ticker').cast(pl.Utf8),
                    pl.col('open').cast(pl.Float64),
                    pl.col('high').cast(pl.Float64),
                    pl.col('low').cast(pl.Float64),
                    pl.col('close').cast(pl.Float64),
                    pl.col('adj_close').cast(pl.Float64),
                    pl.col('volume').cast(pl.Int64)
                ])
                
                print(f" Done! ({len(df)} days)")
                return df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f" Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    print(f" Failed! ({str(e)})")
                    return None

    def store_in_duckdb(self, dfs, batch_size=1000):
        """Store processed data in DuckDB with batching and progress bar"""
        if not dfs:
            print("No data to store")
            return
        
        try:
            # Combine all dataframes
            combined = pl.concat(dfs)
            
            # Connect to DuckDB
            con = duckdb.connect(self.db_path)
            
            # Drop existing table if it exists
            #con.execute("DROP TABLE IF EXISTS nyse_table")
            
            # Create new table
            con.execute("""
                CREATE TABLE IF NOT EXISTS nyse_table (
                    date DATE,
                    ticker VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume BIGINT
                )
            """)
            
            # Insert data with progress bar
            total_rows = len(combined)
            with tqdm_sync(total=total_rows, desc="Storing in DB", leave=False) as pbar:
                for i in range(0, total_rows, batch_size):
                    batch = combined.slice(i, batch_size)
                    con.execute("INSERT INTO nyse_table SELECT * FROM batch")
                    pbar.update(len(batch))
            
            # Create indices if they don't exist
            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_date_ticker ON nyse_table(date, ticker);
                CREATE INDEX IF NOT EXISTS idx_ticker_date ON nyse_table(ticker, date);
            """)
            
        except Exception as e:
            print(f"Error storing data in DuckDB: {str(e)}")
        finally:
            if 'con' in locals():
                con.close()

    def group_tickers_by_letter(self, tickers):
        """Group tickers by their first letter"""
        ticker_groups = {}
        for ticker in tickers:
            first_letter = ticker[0].upper()
            if first_letter.isalpha():
                if first_letter not in ticker_groups:
                    ticker_groups[first_letter] = []
                ticker_groups[first_letter].append(ticker)
        return ticker_groups

    def sample_tickers_by_letter(self, tickers, per_letter=50):
        """Randomly sample tickers for each letter"""
        ticker_groups = self.group_tickers_by_letter(tickers)
        sampled_tickers = []
        
        for letter in sorted(ticker_groups.keys()):
            tickers_for_letter = ticker_groups[letter]
            if len(tickers_for_letter) <= per_letter:
                sampled = tickers_for_letter
            else:
                # Randomly sample without replacement
                sampled = random.sample(tickers_for_letter, per_letter)
            
            print(f"Letter {letter}: sampled {len(sampled)} from {len(tickers_for_letter)} available")
            sampled_tickers.extend(sampled)
        
        print(f"\nTotal sampled tickers: {len(sampled_tickers)}")
        return sampled_tickers

    def process_letter_group(self, tickers_for_letter, letter, max_retries=3):
        """Process a group of tickers for a single letter with appropriate pauses"""
        letter_data = []
        letter_failures = []
        
        print(f"\nStarting letter group {letter} with {len(tickers_for_letter)} tickers")
        
        try:
            for i, ticker in enumerate(tickers_for_letter):
                print(f"\nProcessing {ticker}... ({i+1}/100 for letter {letter})")
                
                # Process ticker with retries
                df = self.process_ticker(ticker, max_retries=max_retries)
                
                if df is not None:
                    letter_data.append(df)
                else:
                    letter_failures.append(ticker)
                
                # Pause between tickers
                if df is None:
                    time.sleep(5)  # Longer pause after a failure
                else:
                    time.sleep(2)  # Normal pause between successful requests
            
            # Store the letter group's data
            if letter_data:
                print(f"\nStoring data for letter {letter}...")
                self.store_in_duckdb(letter_data)
            
            # Pause between letter groups
            print(f"\nCompleted letter {letter}. Pausing before next letter...")
            time.sleep(10)  # 10 second pause between letter groups
            
        except KeyboardInterrupt:
            # Store any data collected before the interrupt
            if letter_data:
                print(f"\n\nKeyboard interrupt detected. Storing collected data for letter {letter}...")
                self.store_in_duckdb(letter_data)
                
            # Show progress
            print("\nProgress before interrupt:")
            self.show_duckdb_status()
            raise  # Re-raise the interrupt
            
        return letter_failures

    def show_duckdb_status(self):
        """Show current status of data in DuckDB"""
        try:
            con = duckdb.connect(self.db_path)
            
            # Get total count and basic stats
            result = con.execute("""
                SELECT 
                    COUNT(DISTINCT ticker) as ticker_count,
                    COUNT(*) as total_records,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    COUNT(DISTINCT date) as trading_days
                FROM nyse_table
            """).fetchone()
            
            # Get count by letter
            letter_counts = con.execute("""
                SELECT 
                    LEFT(ticker, 1) as letter,
                    COUNT(DISTINCT ticker) as ticker_count,
                    COUNT(*) as record_count
                FROM nyse_table
                GROUP BY letter
                ORDER BY letter
            """).fetchall()
            
            # Get approximate database size
            db_size = os.path.getsize(self.db_path) / (1024*1024)  # Convert to MB
            
            print("\nDuckDB Status:")
            print(f"Database size: {db_size:.2f} MB")
            print(f"Total unique tickers: {result[0]}")
            print(f"Total records: {result[1]}")
            print(f"Date range: {result[2]} to {result[3]}")
            print(f"Trading days: {result[4]}")
            
            print("\nBreakdown by letter:")
            print("Letter | Tickers | Records")
            print("-" * 30)
            for letter, ticker_count, record_count in letter_counts:
                print(f"{letter:6} | {ticker_count:7} | {record_count:7}")
                
        except Exception as e:
            print(f"Error checking DuckDB status: {str(e)}")
        finally:
            if 'con' in locals():
                con.close()

    def get_current_progress(self):
        """Check which letters have been processed in DuckDB"""
        try:
            con = duckdb.connect(self.db_path)
            
            # Get letters already in database
            result = con.execute("""
                SELECT DISTINCT LEFT(ticker, 1) as letter
                FROM nyse_table
                ORDER BY letter
            """).fetchall()
            
            if result:
                processed_letters = [row[0] for row in result]
                print("\nProcessed letters:", ", ".join(processed_letters))
                
                # Find next letter to process
                all_letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
                for letter in all_letters:
                    if letter not in processed_letters:
                        return letter
                return None  # All letters processed
            else:
                print("\nNo data in database yet")
                return 'A'  # Start from beginning
                
        except Exception as e:
            if "does not exist" in str(e):
                print("\nNo database exists yet")
                return 'A'  # Start from beginning
            else:
                print(f"\nError checking progress: {str(e)}")
                return 'A'
        finally:
            if 'con' in locals():
                con.close()

    def query_stock_data(self, query):
        """Execute a query and return results as a Pandas DataFrame"""
        try:
            con = duckdb.connect(self.db_path)
            result = con.execute(query).df()
            return result
        finally:
            con.close()

    def refresh_database_stats(self):
        """Refresh database statistics and check for new data"""
        try:
            con = duckdb.connect(self.db_path)
            
            # Get current stats
            current_stats = con.execute("""
                SELECT 
                    LEFT(ticker, 1) as letter,
                    COUNT(DISTINCT ticker) as tickers,
                    COUNT(*) as records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(DISTINCT date) as trading_days
                FROM nyse_table
                GROUP BY letter
                ORDER BY letter
            """).fetchall()
            
            print("\nDatabase Statistics by Letter:")
            print("Letter | Tickers | Records | Date Range | Trading Days")
            print("-" * 65)
            
            for (letter, tickers, records, start_date, end_date, days) in current_stats:
                print(f"{letter:6} | {tickers:7} | {records:7} | {start_date} to {end_date} | {days:5}")
            
            # Get overall statistics
            totals = con.execute("""
                SELECT 
                    COUNT(DISTINCT ticker) as total_tickers,
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(DISTINCT date) as total_trading_days
                FROM nyse_table
            """).fetchone()
            
            print("\nOverall Statistics:")
            print(f"Total Tickers: {totals[0]:,}")
            print(f"Total Records: {totals[1]:,}")
            print(f"Date Range: {totals[2]} to {totals[3]}")
            print(f"Trading Days: {totals[4]}")
            
            # Check for potentially stale data
            today = datetime.now().date()
            latest_date = totals[3]
            days_since_update = (today - latest_date).days
            
            if days_since_update > 5:  # Assuming 5 business days is stale
                print(f"\nWARNING: Data may be stale. Last update was {days_since_update} days ago")
                
        except Exception as e:
            print(f"Error refreshing database stats: {str(e)}")
        finally:
            if 'con' in locals():
                con.close()

    def __del__(self):
        try:
            if hasattr(self, 'db_conn'):
                self.db_conn.close()
        except:
            pass

    def on_database_change(self, event=None):
        try:
            # ... existing code ...
            pass
        except duckdb.Error as e:
            print(f"DuckDB error: {str(e)}")
            messagebox.showerror("Database Error", f"DuckDB error: {str(e)}")
        except Exception as e:
            print(f"General error: {str(e)}")
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")

    def get_historical_data(self, ticker, duration):
        try:
            # ... existing code ...
            stmt = self.db_conn.prepare(query)
            df = stmt.execute([ticker]).df()
            return df
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            return None

# Main execution
if __name__ == "__main__":
    try:
        data_manager = DataManager()
        
        # Create event loop
        loop = asyncio.get_event_loop()
        
        # Get NYSE tickers asynchronously
        tickers = loop.run_until_complete(data_manager.get_nyse_tickers_async())
        
        # Sample and group tickers
        sampled_tickers = data_manager.sample_tickers_by_letter(tickers, per_letter=100)
        ticker_groups = {}
        for ticker in sampled_tickers:
            letter = ticker[0].upper()
            if letter not in ticker_groups:
                ticker_groups[letter] = []
            ticker_groups[letter].append(ticker)
        
        # Process all data asynchronously
        all_failures = loop.run_until_complete(data_manager.process_all_data_async(ticker_groups))
        
        # Print final summary
        print("\nDownload Summary:")
        print(f"Total tickers attempted: {len(sampled_tickers)}")
        print(f"Successfully downloaded: {len(sampled_tickers) - len(all_failures)}")
        print(f"Failed downloads: {len(all_failures)}")
        
        if all_failures:
            print("\nFailed tickers:")
            print(", ".join(all_failures))
        
        # Show final database status
        data_manager.show_duckdb_status()
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Showing final status:")
        data_manager.show_duckdb_status()
        sys.exit(1)
    finally:
        loop.close()
