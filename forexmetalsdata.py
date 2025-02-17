import yfinance as yf
import polars as pl
import duckdb
from datetime import datetime, timedelta
import pandas as pd
import time
import traceback
import os

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
        start_date = datetime(2024, 1, 1)  # Start from January 1, 2024
    if end_date is None:
        end_date = datetime(2025, 12, 31)  # End at December 31, 2025
        
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
            traceback.print_exc()
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

def get_database_size(db_path):
    """Get the size of the database file in MB"""
    size_bytes = os.path.getsize(db_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)

def get_table_info(db_path):
    """Get information about tables in the database"""
    try:
        conn = duckdb.connect(db_path)
        
        # Get list of tables
        tables = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()
        
        table_info = []
        for table in tables:
            table_name = table[0]
            
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            # Get column info
            columns = conn.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
            """).fetchall()
            
            table_info.append({
                'table_name': table_name,
                'rows': row_count,
                'columns': len(columns),
                'column_details': columns
            })
            
        conn.close()
        return table_info
        
    except Exception as e:
        print(f"Error analyzing database: {str(e)}")
        return []

def main():
    # List all .db files in current directory
    db_files = [f for f in os.listdir('.') if f.endswith('.db')]
    
    for db_file in db_files:
        print(f"\nAnalyzing {db_file}:")
        print(f"File size: {get_database_size(db_file)} MB")
        
        table_info = get_table_info(db_file)
        for table in table_info:
            print(f"\nTable: {table['table_name']}")
            print(f"Rows: {table['rows']:,}")
            print(f"Columns: {table['columns']}")
            print("\nColumn Details:")
            for col in table['column_details']:
                print(f"- {col[0]}: {col[1]}")

if __name__ == "__main__":
    create_metals_database()
    main()
