import requests
import time
from pathlib import Path
import logging
import polars as pl
import duckdb

def download_nasdaq_screener():
    """Download the NASDAQ screener data."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create data directory if it doesn't exist
        Path('data').mkdir(exist_ok=True)
        
        # NASDAQ Screener URL (you might need to update this URL)
        url = "https://www.nasdaq.com/market-activity/stocks/screener/nasdaq-stocks.csv"
        
        # Create filename with timestamp
        timestamp = int(time.time())
        filename = Path('data') / f'nasdaq_screener_{timestamp}.csv'
        
        logger.info(f"Downloading NASDAQ screener data to {filename}")
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the file
        with open(filename, 'wb') as f:
            f.write(response.content)
            
        logger.info("NASDAQ screener data downloaded successfully")
        return filename
        
    except Exception as e:
        logger.error(f"Error downloading NASDAQ screener: {e}")
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