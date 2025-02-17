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

class DataManager:
    def __init__(self, db_path='stocks.db'):
        self.db_path = db_path
        self.failed_downloads = []

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
        """Store processed data in DuckDB with batching"""
        if not dfs:
            print("No data to store")
            return
        
        try:
            # Combine all dataframes
            combined = pl.concat(dfs)
            
            # Connect to DuckDB
            con = duckdb.connect(self.db_path)
            
            # Create table if it doesn't exist
            con.execute("""
                CREATE TABLE IF NOT EXISTS stock_prices (
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
            
            # Debug print
            print(f"\nStoring {len(combined)} rows with schema:")
            print(combined.schema)
            
            # Insert data
            con.execute("INSERT INTO stock_prices SELECT * FROM combined")
            
            # Create index if it doesn't exist
            con.execute("CREATE INDEX IF NOT EXISTS idx_date_ticker ON stock_prices(date, ticker)")
            
            # Add this after table creation
            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_prices(ticker, date);
            """)
            
            print(f"Successfully stored {len(combined)} records in DuckDB")
            
        except Exception as e:
            print(f"Error storing data in DuckDB: {str(e)}")
            # Print detailed error info
            print("\nDebug information:")
            if len(dfs) > 0:
                print("First dataframe schema:")
                print(dfs[0].schema)
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
                FROM stock_prices
            """).fetchone()
            
            # Get count by letter
            letter_counts = con.execute("""
                SELECT 
                    LEFT(ticker, 1) as letter,
                    COUNT(DISTINCT ticker) as ticker_count,
                    COUNT(*) as record_count
                FROM stock_prices
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
                FROM stock_prices
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
                FROM stock_prices
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
                FROM stock_prices
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
        
        # Get NYSE tickers
        tickers = data_manager.get_nyse_tickers()
        print(f"Found {len(tickers)} NYSE tickers")
        
        # Check current progress
        start_letter = data_manager.get_current_progress()
        if start_letter is None:
            print("\nAll letters have been processed!")
            sys.exit(0)
            
        print(f"\nResuming from letter {start_letter}")
        
        # Sample tickers by letter
        sampled_tickers = data_manager.sample_tickers_by_letter(tickers, per_letter=100)
        print(f"\nProcessing {len(sampled_tickers)} tickers (50 per letter):")
        
        # Group sampled tickers by letter
        ticker_groups = {}
        for ticker in sampled_tickers:
            letter = ticker[0].upper()
            if letter not in ticker_groups:
                ticker_groups[letter] = []
            ticker_groups[letter].append(ticker)
        
        all_failures = []
        
        # Process each letter group starting from the resume point
        for letter in sorted(ticker_groups.keys()):
            if start_letter and letter < start_letter:
                print(f"\nSkipping letter {letter} (already processed)")
                continue
                
            tickers_for_letter = ticker_groups[letter]
            print(f"\nProcessing letter {letter} ({len(tickers_for_letter)} tickers)")
            letter_failures = data_manager.process_letter_group(tickers_for_letter, letter)
            all_failures.extend(letter_failures)
        
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
