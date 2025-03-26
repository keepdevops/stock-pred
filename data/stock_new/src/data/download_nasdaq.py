import requests
import time
from pathlib import Path
import logging
import polars as pl
import duckdb
import pandas as pd
import json
import os

def download_nasdaq_screener():
    """Download the NASDAQ screener data."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create data directory if it doesn't exist
        data_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'data'
        data_dir.mkdir(exist_ok=True)
        
        # Use NASDAQ screener API
        url = "https://api.nasdaq.com/api/screener/stocks"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        params = {
            'download': 'true',
            'exchange': 'NASDAQ',
            'marketcap': 'all',
            'render': 'download'
        }
        
        logger.info("Downloading NASDAQ screener data...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        if 'data' in data and 'rows' in data['data']:
            df = pd.DataFrame(data['data']['rows'])
        else:
            raise ValueError("Invalid response format from NASDAQ API")
        
        # Create filename with timestamp
        timestamp = int(time.time())
        filename = data_dir / f'nasdaq_screener_{timestamp}.csv'
        
        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"NASDAQ screener data saved to {filename}")
        
        return filename
        
    except Exception as e:
        logger.error(f"Error downloading NASDAQ screener: {e}")
        
        # Use backup method - hardcoded list of major tickers
        try:
            logger.info("Using backup list of major tickers...")
            major_tickers = {
                'Symbol': [
                    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
                    'NFLX', 'INTC', 'AMD', 'ADBE', 'CSCO', 'CMCSA', 'COST', 'PEP',
                    'AVGO', 'PYPL', 'TMUS', 'QCOM', 'INTU', 'TXN', 'AMAT', 'ISRG',
                    'BKNG', 'ADP', 'MRNA', 'ABNB', 'SBUX', 'LRCX', 'REGN', 'MELI'
                ]
            }
            df = pd.DataFrame(major_tickers)
            
            # Create filename with timestamp
            timestamp = int(time.time())
            filename = data_dir / f'nasdaq_screener_{timestamp}.csv'
            
            # Save to CSV
            df.to_csv(filename, index=False)
            logger.info(f"Backup NASDAQ data saved to {filename}")
            
            return filename
            
        except Exception as backup_error:
            logger.error(f"Backup method failed: {backup_error}")
            return None

def save_polars_to_duckdb(df: pl.DataFrame, table_name: str, db_path: str = "stocks.db"):
    """
    Save a Polars DataFrame to DuckDB
    
    Args:
        df (pl.DataFrame): Polars DataFrame to save
        table_name (str): Name of the table in DuckDB
        db_path (str): Path to the DuckDB database file
    """
    # Connect to DuckDB
    con = duckdb.connect(db_path)
    
    try:
        # Convert Polars DataFrame to Arrow table
        arrow_table = df.to_arrow()
        
        # Create or replace table in DuckDB using Arrow
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM arrow_table")
        
        # Commit the changes
        con.commit()
    finally:
        # Always close the connection
        con.close()

def append_polars_to_duckdb(df: pl.DataFrame, table_name: str, db_path: str = "stocks.db"):
    con = duckdb.connect(db_path)
    try:
        arrow_table = df.to_arrow()
        con.execute(f"INSERT INTO {table_name} SELECT * FROM arrow_table")
        con.commit()
    finally:
        con.close()

def read_from_duckdb(table_name: str, db_path: str = "stocks.db") -> pl.DataFrame:
    con = duckdb.connect(db_path)
    try:
        # Query the table and convert to Polars
        result = con.execute(f"SELECT * FROM {table_name}").arrow()
        return pl.from_arrow(result)
    finally:
        con.close()

def create_indexes(db_path: str, table_name: str):
    con = duckdb.connect(db_path)
    try:
        # Create indexes for common query patterns
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ticker ON {table_name}(ticker)")
        con.commit()
    finally:
        con.close()

if __name__ == "__main__":
    download_nasdaq_screener() 