import yfinance as yf
import polars as pl
import duckdb
from datetime import datetime, timedelta
import pandas as pd
import time
import traceback

def get_metals_tickers():
    """Get list of major metals tickers"""
    metals_tickers = [
        "GC=F",  # Gold Futures
        "SI=F",  # Silver Futures 
        "PL=F",  # Platinum Futures
        "PA=F",  # Palladium Futures
        "HG=F",  # Copper Futures
        "ALI=F", # Aluminum Futures
    ]
    return metals_tickers

def download_metals_data(tickers, start_date=None, end_date=None):
    """Download historical metals data from Yahoo Finance"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
    if end_date is None:
        end_date = datetime.now()
        
    all_data = []
    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}")
            # Download data using yfinance
            metal = yf.download(ticker, start=start_date, end=end_date)
            
            # Reset index and handle MultiIndex columns
            metal = metal.reset_index()
            
            # Print column names for debugging
            print(f"Original columns: {metal.columns}")
            
            # Flatten MultiIndex columns if they exist
            if isinstance(metal.columns, pd.MultiIndex):
                metal.columns = [col[0].lower() for col in metal.columns]
            else:
                metal.columns = [col.lower() for col in metal.columns]
                
            # Add ticker column
            metal['ticker'] = ticker
            
            # Print standardized column names for debugging
            print(f"Standardized columns: {metal.columns}")
            
            # Convert to Polars DataFrame with clean column names
            metal_pl = pl.from_pandas(metal).select([
                pl.col('date'),
                pl.col('ticker'),
                pl.col('open'),
                pl.col('high'),
                pl.col('low'),
                pl.col('close'),
                pl.col('adj close').alias('adj_close') if 'adj close' in metal.columns else pl.col('close').alias('adj_close'),
                pl.col('volume')
            ])
            
            all_data.append(metal_pl)
            time.sleep(1)  # Avoid rate limiting
            
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
            traceback.print_exc()  # Add traceback for more detailed error info
            continue
            
    if all_data:
        # Combine all data using Polars
        combined_data = pl.concat(all_data)
        return combined_data
    return None

def create_metals_database():
    """Create and populate metals database"""
    try:
        # Connect to database
        con = duckdb.connect("metals_market.db")
        
        # Create tables
        con.execute("""
            CREATE TABLE IF NOT EXISTS metals_prices (
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
        
        # Get metals data
        tickers = get_metals_tickers()
        data = download_metals_data(tickers)
        
        if data is not None:
            # Print column names for debugging
            print("DataFrame columns:", data.columns)
            
            # Insert data
            con.execute("DELETE FROM metals_prices")  # Clear existing data
            
            # Convert Polars DataFrame to pandas for DuckDB compatibility
            pandas_df = data.to_pandas()
            
            # Register the DataFrame as a table
            con.register('data_table', pandas_df)
            
            # Insert data directly from the registered table
            con.execute("""
                INSERT INTO metals_prices 
                SELECT 
                    date,
                    ticker,
                    open,
                    high,
                    low,
                    close,
                    adj_close,
                    volume
                FROM data_table
            """)
            
            print(f"Successfully loaded {len(data)} records into metals database")
        
        con.close()
        return True
        
    except Exception as e:
        print(f"Error creating metals database: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_metals_database()
