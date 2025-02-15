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
from bs4 import BeautifulSoup
from db_config import FOREX_DB, get_connection, create_tables

def get_nyse_tickers(limit=1500):
    """Get list of top NASDAQ tickers by market cap/volume"""
    try:
        # Use yfinance's built-in tickers from NASDAQ
        nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"
        df = pd.read_csv(nasdaq_url, sep='|')
        
        # Filter for active NASDAQ stocks (exclude ETFs, etc)
        df = df[
            (df['ETF'] == 'N') & 
            (df['Symbol'].str.contains('^[A-Z]+$', na=False))  # Only pure letter symbols
        ]
        
        # Get top tickers by market cap/volume
        top_tickers = [
            # Major tech and well-known NASDAQ stocks first
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
            "NFLX", "INTC", "AMD", "ADBE", "CSCO", "CMCSA", "COST", "PEP",
            "AVGO", "PYPL", "TMUS", "QCOM", "INTU", "TXN", "AMAT", "ISRG",
            "BKNG", "ADP", "MRNA", "ABNB", "SBUX", "LRCX", "REGN", "MELI"
        ]
        
        # Add remaining tickers from the NASDAQ list
        remaining_tickers = [t for t in df['Symbol'].tolist() if t not in top_tickers]
        
        # Combine lists and limit to specified number
        all_tickers = top_tickers + remaining_tickers
        final_tickers = all_tickers[:limit]
        
        print(f"Selected {len(final_tickers)} NASDAQ tickers")
        return sorted(final_tickers)
    except Exception as e:
        print(f"Error fetching NASDAQ tickers: {str(e)}")
        # Fallback to major NASDAQ stocks if the FTP fails
        fallback_tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", 
            "TSLA", "NFLX", "INTC", "AMD", "PYPL", "CSCO", 
            "CMCSA", "PEP", "AVGO", "COST", "ADBE", "TMUS"
        ]
        print(f"Using fallback list of {len(fallback_tickers)} major NASDAQ tickers")
        return sorted(fallback_tickers)

def download_stock_data(ticker, start_date, end_date):
    try:
        print(f"\nProcessing {ticker}...", end='')
        # Download data with explicit date range
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if len(df) == 0:
            print(" Empty dataframe")
            failed_downloads.append(ticker)
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
            failed_downloads.append(ticker)
            return None
            
        print(f" Done! ({trading_days} days)")
        
        # Add ticker column
        df = df.with_columns(pl.lit(ticker).alias('Symbol'))
        
        return df
        
    except Exception as e:
        print(f"\nError processing {ticker}: {str(e)}")
        failed_downloads.append(ticker)
        return None

def process_tickers(tickers, start_date, end_date):
    """Process a group of tickers"""
    all_data = []
    failed_downloads = []
    download_stats = []

    for ticker in tickers:
        print(f"Downloading {ticker}...", end=" ")
        
        ticker_data = download_stock_data(ticker, start_date, end_date)
        
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
            failed_downloads.append(ticker)
            print("Failed!")
            
        time.sleep(1)  # Rate limiting
    
    if not all_data:
        return None, failed_downloads, download_stats
        
    # Combine all data using join instead of concat
    combined_data = all_data[0]
    for df in all_data[1:]:
        combined_data = combined_data.join(df, on='Date', how='outer')
    
    return combined_data, failed_downloads, download_stats

def get_trading_dates():
    """Get trading date ranges"""
    end_date = datetime.now()
    calendar_year = end_date - timedelta(days=365)  # 1 calendar year
    
    # For 252 trading days, add extra calendar days to account for weekends and holidays
    # Typically need ~355 calendar days to get 252 trading days
    trading_year = end_date - timedelta(days=355)
    
    return {
        'end_date': end_date,
        'calendar_year': calendar_year,
        'trading_year': trading_year
    }

def process_ticker(ticker, max_retries=3, retry_delay=5, timeout=30):
    """Process a single ticker with retries"""
    dates = get_trading_dates()
    
    for attempt in range(max_retries):
        try:
            # Download data with trading year date range
            df = yf.download(
                ticker, 
                start=dates['trading_year'],
                end=dates['end_date'],
                progress=False, 
                timeout=timeout
            )
            
            if len(df) == 0:
                print(f" Empty dataframe for {ticker}")
                return None
            
            # Verify we have enough trading days
            if len(df) < 240:  # Allow some flexibility for holidays
                print(f" Insufficient trading days for {ticker}: {len(df)}")
                return None
                
            # Trim to exactly 252 trading days if we have more
            if len(df) > 252:
                df = df.tail(252)
            
            # Handle multi-index columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Add ticker column and Adj Close if missing
            df['ticker'] = ticker
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']  # Use regular close if adj close not available
            
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
            
            # Convert to polars with explicit schema
            schema = {
                'date': pl.Date,
                'ticker': pl.Utf8,
                'open': pl.Float64,
                'high': pl.Float64,
                'low': pl.Float64,
                'close': pl.Float64,
                'adj_close': pl.Float64,
                'volume': pl.Int64
            }
            
            df = pl.from_pandas(
                df[list(schema.keys())],  # Only keep needed columns
                schema_overrides=schema
            )
            
            print(f" Successfully processed {ticker} with {len(df)} trading days")
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f" Retry {attempt + 1}/{max_retries} for {ticker} after error: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f" Failed to process {ticker}: {str(e)}")
                return None

def store_in_duckdb(dfs, batch_size=1000):
    """Store processed data in DuckDB with batching"""
    if not dfs:
        print("No data to store")
        return
    
    try:
        # Get connection with standard configuration
        conn = get_connection(FOREX_DB)
        
        # Ensure tables exist
        create_tables(conn)
        
        # Combine all dataframes
        combined = pl.concat(dfs)
        
        # Debug print
        print(f"\nStoring {len(combined)} rows with schema:")
        print(combined.schema)
        
        # Insert data
        conn.execute("INSERT INTO forex_prices SELECT * FROM combined")
        
        # Create index if it doesn't exist
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date_pair ON forex_prices(date, pair)")
        
        print(f"Successfully stored {len(combined)} records in DuckDB")
        
    except Exception as e:
        print(f"Error storing data in DuckDB: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def group_tickers_by_letter(tickers):
    """Group tickers by their first letter"""
    ticker_groups = {}
    for ticker in tickers:
        first_letter = ticker[0].upper()
        if first_letter.isalpha():
            if first_letter not in ticker_groups:
                ticker_groups[first_letter] = []
            ticker_groups[first_letter].append(ticker)
    return ticker_groups

def sample_tickers(tickers, max_per_letter=50):
    """Sample tickers by first letter, with a maximum per letter"""
    # Group tickers by first letter
    letter_groups = {}
    for ticker in tickers:
        first_letter = ticker[0]
        if first_letter not in letter_groups:
            letter_groups[first_letter] = []
        letter_groups[first_letter].append(ticker)
    
    # Sample from each letter group
    sampled_tickers = []
    for letter in sorted(letter_groups.keys()):
        group = letter_groups[letter]
        # If group is smaller than max, take all tickers
        if len(group) <= max_per_letter:
            sample = group
        else:
            # Prioritize well-known tickers
            major_tickers = [t for t in group if t in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]]
            remaining_slots = max_per_letter - len(major_tickers)
            remaining_tickers = [t for t in group if t not in major_tickers]
            sample = major_tickers + random.sample(remaining_tickers, remaining_slots)
        
        sampled_tickers.extend(sample)
        print(f"Letter {letter}: sampled {len(sample)} from {len(group)} available")
    
    return sampled_tickers

def process_letter_group(tickers_for_letter, letter, max_retries=3):
    """Process a group of tickers for a single letter with appropriate pauses"""
    letter_data = []
    letter_failures = []
    
    print(f"\nStarting letter group {letter} with {len(tickers_for_letter)} tickers")
    
    try:
        for i, ticker in enumerate(tickers_for_letter):
            print(f"\nProcessing {ticker}... ({i+1}/50 for letter {letter})")
            
            # Process ticker with retries
            df = process_ticker(ticker, max_retries=max_retries)
            
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
            store_in_duckdb(letter_data)
        
        # Pause between letter groups
        print(f"\nCompleted letter {letter}. Pausing before next letter...")
        time.sleep(10)  # 10 second pause between letter groups
        
    except KeyboardInterrupt:
        # Store any data collected before the interrupt
        if letter_data:
            print(f"\n\nKeyboard interrupt detected. Storing collected data for letter {letter}...")
            store_in_duckdb(letter_data)
            
        # Show progress
        print("\nProgress before interrupt:")
        show_duckdb_status()
        raise  # Re-raise the interrupt
        
    return letter_failures

def show_duckdb_status():
    """Show current status of data in DuckDB"""
    try:
        conn = get_connection(FOREX_DB)
        conn.execute("SET scalar_subquery_error_on_multiple_rows=false")
        
        # Get total count and basic stats
        result = conn.execute("""
            WITH stats AS (
                SELECT 
                    COUNT(DISTINCT ticker) as ticker_count,
                    COUNT(*) as total_records,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    COUNT(DISTINCT date) as trading_days
                FROM forex_prices
            ),
            complete_tickers AS (
                SELECT COUNT(*) as complete_count
                FROM (
                    SELECT ticker
                    FROM forex_prices 
                    GROUP BY ticker 
                    HAVING COUNT(*) = 252
                ) t
            )
            SELECT * FROM stats CROSS JOIN complete_tickers
        """).fetchone()
        
        # Get count by letter
        letter_counts = conn.execute("""
            SELECT 
                LEFT(ticker, 1) as letter,
                COUNT(DISTINCT ticker) as ticker_count,
                COUNT(*) as record_count,
                CAST(COUNT(*) AS FLOAT) / COUNT(DISTINCT ticker) as avg_days_per_ticker
            FROM forex_prices
            GROUP BY letter
            ORDER BY letter
        """).fetchall()
        
        # Get approximate database size
        db_size = os.path.getsize(FOREX_DB) / (1024*1024)  # Convert to MB
        
        print("\nDuckDB Status:")
        print(f"Database size: {db_size:.2f} MB")
        print(f"Total unique tickers: {result[0]}")
        print(f"Complete tickers (252 days): {result[5]}")
        print(f"Total records: {result[1]}")
        print(f"Date range: {result[2]} to {result[3]}")
        print(f"Trading days: {result[4]}")
        
        print("\nBreakdown by letter:")
        print("Letter | Tickers | Records | Avg Days/Ticker")
        print("-" * 45)
        for letter, ticker_count, record_count, avg_days in letter_counts:
            print(f"{letter:6} | {ticker_count:7} | {record_count:7} | {avg_days:14.1f}")
            
    except Exception as e:
        print(f"Error checking DuckDB status: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def get_current_progress(db_path="forex-duckdb.db"):
    """Check which letters have been processed in DuckDB"""
    try:
        conn = get_connection(FOREX_DB)
        
        # Get letters already in database
        result = conn.execute("""
            SELECT DISTINCT LEFT(ticker, 1) as letter
            FROM forex_prices
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
        if 'conn' in locals():
            conn.close()

def get_forex_pairs():
    """Get list of major and minor forex pairs"""
    major_pairs = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", 
        "AUDUSD=X", "USDCAD=X", "NZDUSD=X"
    ]
    
    minor_pairs = [
        "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURAUD=X", "EURCAD=X",
        "GBPJPY=X", "GBPCHF=X", "GBPAUD=X", "GBPCAD=X",
        "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X"
    ]
    
    return major_pairs + minor_pairs

def process_forex_pair(pair, max_retries=3, retry_delay=5, timeout=30):
    """Process a single forex pair with retries"""
    dates = get_trading_dates()
    
    for attempt in range(max_retries):
        try:
            df = yf.download(
                pair, 
                start=dates['trading_year'],
                end=dates['end_date'],
                progress=False, 
                timeout=timeout
            )
            
            if len(df) == 0:
                print(f" Empty dataframe for {pair}")
                return None
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Add pair column and remove =X suffix
            df['pair'] = pair.replace('=X', '')
            
            # For forex, we need to ensure consistent column names
            # First, standardize the column names from yfinance
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Handle column names
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
            
            # For forex pairs, Close and Adj Close are the same
            if 'adj_close' not in df.columns:
                df['adj_close'] = df['close']
            
            # Ensure volume exists (some forex pairs don't have volume)
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            # Convert to polars with explicit schema
            schema = {
                'date': pl.Date,
                'pair': pl.Utf8,
                'open': pl.Float64,
                'high': pl.Float64,
                'low': pl.Float64,
                'close': pl.Float64,
                'adj_close': pl.Float64,
                'volume': pl.Float64
            }
            
            df = pl.from_pandas(
                df[list(schema.keys())],  # Only keep needed columns in correct order
                schema_overrides=schema
            )
            
            print(f" Successfully processed {pair} with {len(df)} trading days")
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f" Retry {attempt + 1}/{max_retries} for {pair} after error: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f" Failed to process {pair}: {str(e)}")
                return None

# Main execution
if __name__ == "__main__":
    try:
        # Set explicit date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        # Get forex pairs
        pairs = get_forex_pairs()
        print(f"Processing {len(pairs)} forex pairs")
        
        all_data = []
        failures = []
        
        # Process each forex pair
        for pair in pairs:
            print(f"\nProcessing {pair}...")
            df = process_forex_pair(pair)
            
            if df is not None:
                all_data.append(df)
            else:
                failures.append(pair)
            
            time.sleep(2)  # Rate limiting
        
        # Store the data
        if all_data:
            store_in_duckdb(all_data)
        
        # Print summary
        print("\nDownload Summary:")
        print(f"Total pairs attempted: {len(pairs)}")
        print(f"Successfully downloaded: {len(pairs) - len(failures)}")
        print(f"Failed downloads: {len(failures)}")
        
        if failures:
            print("\nFailed pairs:")
            print(", ".join(failures))
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)