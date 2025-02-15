import yfinance as yf
import polars as pl
import duckdb
from datetime import datetime
import pandas as pd
import signal
import sys

def signal_handler(signum, frame):
    print("\n\nCtrl+C detected. Gracefully stopping the download process...")
    print("Current downloads will be saved before exiting.")
    sys.exit(0)

def get_nyse_symbols():
    # Download NYSE symbols - you might want to replace this with a more reliable source
    url = "https://www.nasdaq.com/market-activity/stocks/screener?exchange=nyse&render=download"
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

def get_stock_data_bulk(tickers, period="1y", interval="1d"):
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    all_data = []
    total_tickers = len(tickers)
    completed_tickers = 0
    failed_tickers = 0
    
    print(f"\nStarting download for {total_tickers} tickers...")
    print(f"Press Ctrl+C to stop the download process at any time")
    print("-" * 50)
    
    # Create status line template
    status_template = "\rProgress: [{}/{}] tickers | Success: {} | Failed: {} | Current: {:<5}"
    
    for index, ticker in enumerate(tickers, 1):
        try:
            # Update status line (overwrite previous line with \r)
            print(status_template.format(
                index, total_tickers, completed_tickers, failed_tickers, ticker
            ), end='', flush=True)
            
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            # Reset index to make date a column
            data = data.reset_index()
            data['ticker'] = ticker
            
            # Print completion details on new line
            print(f"\n✓ {ticker}: {len(data)} days | Range: {data['Date'].iloc[0].strftime('%Y-%m-%d')} to {data['Date'].iloc[-1].strftime('%Y-%m-%d')}")
            
            # Convert to polars
            pl_df = pl.from_pandas(data)
            all_data.append(pl_df)
            
            completed_tickers += 1
            
        except Exception as e:
            print(f"\n✗ Error downloading {ticker}: {e}")
            failed_tickers += 1
            continue
    
    # Final status update
    print("\n" + "=" * 50)
    print(f"Download completed:")
    print(f"- Successfully downloaded: {completed_tickers} tickers")
    print(f"- Failed downloads: {failed_tickers} tickers")
    print("=" * 50)
    
    # Combine all data
    if all_data:
        print("\nCombining all downloaded data...")
        combined_data = pl.concat(all_data)
        total_days = len(combined_data)
        print(f"Successfully combined data for {completed_tickers} tickers (total of {total_days} daily records)")
        return combined_data
    return None

def save_to_duckdb(data, db_path="stocks.duckdb"):
    # Initialize DuckDB connection
    con = duckdb.connect(db_path)
    
    # Create table if it doesn't exist
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            Date TIMESTAMP,
            Open DOUBLE,
            High DOUBLE,
            Low DOUBLE,
            Close DOUBLE,
            Volume BIGINT,
            Dividends DOUBLE,
            "Stock Splits" DOUBLE,
            ticker VARCHAR
        )
    """)
    
    # Insert data
    con.execute("INSERT INTO stock_prices SELECT * FROM data")
    
    # Create index on date and ticker
    con.execute("CREATE INDEX IF NOT EXISTS idx_date_ticker ON stock_prices(Date, ticker)")
    
    con.close()

def main():
    print("Starting stock data collection process...")
    
    # Get NYSE symbols
    print("Fetching NYSE symbols...")
    tickers = get_nyse_symbols()
    print(f"Found {len(tickers)} NYSE symbols")
    
    # Get data for all tickers
    data = get_stock_data_bulk(tickers)
    
    if data is not None:
        # Save to DuckDB
        print("\nSaving data to DuckDB...")
        save_to_duckdb(data)
        print("Data successfully saved to DuckDB")
    else:
        print("No data was collected - process terminated")

if __name__ == "__main__":
    main()
