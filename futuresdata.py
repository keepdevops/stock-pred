import yfinance as yf
import polars as pl
import duckdb
from datetime import datetime
import pandas as pd
import time
import traceback
import os
from dateutil.relativedelta import relativedelta

def get_derivatives_tickers():
    """Get list of futures and options tickers"""
    futures_tickers = [
        "ES=F",  # S&P 500 E-mini
        "NQ=F",  # NASDAQ 100 E-mini
        "YM=F",  # Dow Jones E-mini
        "RTY=F", # Russell 2000 E-mini
        "ZB=F",  # U.S. Treasury Bond
        "ZN=F",  # 10-Year T-Note
        "CL=F",  # Crude Oil
        "NG=F",  # Natural Gas
        "GC=F",  # Gold
        "SI=F",  # Silver
        "ZC=F",  # Corn
        "ZS=F",  # Soybeans
        "ZW=F"   # Wheat
    ]
    
    options_tickers = [
        "SPY",   # S&P 500 ETF
        "QQQ",   # NASDAQ 100 ETF
        "IWM",   # Russell 2000 ETF
        "GLD",   # Gold ETF
        "USO",   # Oil ETF
        "TLT"    # 20+ Year Treasury ETF
    ]
    
    return futures_tickers, options_tickers

def download_futures_data(ticker):
    """Download futures data for a single ticker"""
    try:
        # Download data
        data = yf.download(ticker, period="1y")
        
        if data.empty:
            print(f"No data found for {ticker}")
            return None
            
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Print column names and types before conversion
        print(f"\nBefore conversion for {ticker}:")
        print(data.dtypes)
        
        # Fix column names before converting to polars
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        # Convert to polars DataFrame directly from pandas
        data = pl.from_pandas(data)
        
        # Print schema after initial conversion
        print(f"\nAfter initial polars conversion for {ticker}:")
        print(data.schema)
        
        # Rename columns
        data = data.rename({
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add ticker and adj_close columns and convert volume to Float64
        data = data.with_columns([
            pl.lit(ticker).alias('ticker'),
            pl.col('close').alias('adj_close'),
            pl.col('volume').cast(pl.Float64).alias('volume')
        ])
        
        # Ensure all numeric columns are float64
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in numeric_cols:
            data = data.with_columns(pl.col(col).cast(pl.Float64))
        
        # Print final schema
        print(f"\nFinal schema for {ticker}:")
        print(data.schema)
            
        return data
        
    except Exception as e:
        print(f"Error downloading {ticker}: {str(e)}")
        traceback.print_exc()
        return None

def download_options_data(ticker):
    """Download options data for a single ticker"""
    try:
        # Get stock info
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        expirations = stock.options
        if not expirations:
            print(f"No options data available for {ticker}")
            return None
            
        # Get the closest expiration date that's at least 2 weeks away
        today = datetime.now()
        valid_dates = [datetime.strptime(d, '%Y-%m-%d') for d in expirations]
        future_dates = [d for d in valid_dates if (d - today).days >= 14]
        if not future_dates:
            print(f"No valid expiration dates found for {ticker}")
            return None
            
        expiration = min(future_dates).strftime('%Y-%m-%d')
        
        # Get options chain
        opt = stock.option_chain(expiration)
        
        if opt.calls.empty and opt.puts.empty:
            print(f"No options data found for {ticker} at {expiration}")
            return None
            
        # Combine calls and puts
        calls = opt.calls.copy()
        calls['type'] = 'call'
        puts = opt.puts.copy()
        puts['type'] = 'put'
        data = pd.concat([calls, puts])
        
        # Add required columns and ensure numeric columns are float
        data['ticker'] = ticker
        data['date'] = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))  # Convert to datetime
        numeric_cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest']
        for col in numeric_cols:
            data[col] = data[col].astype('float64')
        
        # Convert to polars
        data = pl.from_pandas(data)
        
        # Select and rename columns
        data = data.select([
            'date',
            'ticker',
            'strike',
            'lastPrice',
            'bid',
            'ask',
            'volume',
            'openInterest',
            'type'
        ]).rename({
            'lastPrice': 'last_price',
            'openInterest': 'open_interest'
        })
        
        # Ensure all numeric columns are float64
        numeric_cols = ['strike', 'last_price', 'bid', 'ask', 'volume', 'open_interest']
        for col in numeric_cols:
            data = data.with_columns(pl.col(col).cast(pl.Float64))
        
        # Print final schema
        print(f"\nFinal schema for {ticker} options:")
        print(data.schema)
        
        return data
        
    except Exception as e:
        print(f"Error downloading options for {ticker}: {str(e)}")
        traceback.print_exc()
        return None

def download_derivatives_data(futures_tickers, options_tickers):
    """Download both futures and options data"""
    all_data = []
    
    # Download futures data
    for ticker in futures_tickers:
        print(f"Downloading futures data for {ticker}")
        data = download_futures_data(ticker)
        if data is not None:
            all_data.append(data)
    
    # Download options data
    for ticker in options_tickers:
        print(f"Downloading options data for {ticker}")
        data = download_options_data(ticker)
        if data is not None:
            all_data.append(data)
    
    if not all_data:
        return None
        
    # Combine all data, ensuring consistent schema
    return pl.concat(all_data, how="diagonal")

def get_database_size(db_path):
    """Get the size of the database file in MB"""
    try:
        size_bytes = os.path.getsize(db_path)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    except Exception as e:
        print(f"Error getting database size: {str(e)}")
        return None

def create_derivatives_database():
    """Create SQLite database for derivatives data"""
    try:
        db_path = 'derivatives.db'
        
        # Create database connection
        con = duckdb.connect(db_path)
        
        # Create tables if they don't exist
        con.execute("""
            CREATE TABLE IF NOT EXISTS futures_prices (
                date TIMESTAMP,
                ticker VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)
        
        con.execute("""
            CREATE TABLE IF NOT EXISTS options_prices (
                date TIMESTAMP,
                ticker VARCHAR,
                strike DOUBLE,
                last_price DOUBLE,
                bid DOUBLE,
                ask DOUBLE,
                volume DOUBLE,
                open_interest DOUBLE,
                type VARCHAR,
                PRIMARY KEY (date, ticker, strike, type)
            )
        """)
        
        # Download and process futures data
        futures_tickers, options_tickers = get_derivatives_tickers()
        futures_data = []
        options_data = []
        
        print("\nProcessing futures data...")
        for ticker in futures_tickers:
            print(f"\nProcessing futures data for {ticker}")
            data = download_futures_data(ticker)
            if data is not None:
                futures_data.append(data)
                
        print("\nProcessing options data...")
        for ticker in options_tickers:
            print(f"\nProcessing options data for {ticker}")
            data = download_options_data(ticker)
            if data is not None:
                options_data.append(data)
                
        # Insert data if we have any
        if futures_data:
            futures_df = pl.concat(futures_data)
            print("\nFinal futures data schema:")
            print(futures_df.schema)
            print(f"Futures data rows: {len(futures_df)}")
            
            # Delete existing data before inserting new data
            con.execute("DELETE FROM futures_prices")
            con.register("futures_df_view", futures_df)
            con.execute("""
                INSERT INTO futures_prices 
                SELECT date, ticker, open, high, low, close, adj_close, volume 
                FROM futures_df_view
            """)
            
            # Get row count after insertion
            result = con.execute("SELECT COUNT(*) FROM futures_prices").fetchone()
            print(f"\nFutures data inserted successfully: {result[0]} rows")
            
        if options_data:
            options_df = pl.concat(options_data)
            print("\nFinal options data schema:")
            print(options_df.schema)
            print(f"Options data rows: {len(options_df)}")
            
            # Delete existing data before inserting new data
            con.execute("DELETE FROM options_prices")
            con.register("options_df_view", options_df)
            con.execute("""
                INSERT INTO options_prices 
                SELECT date, ticker, strike, last_price, bid, ask, volume, open_interest, type 
                FROM options_df_view
            """)
            
            # Get row count after insertion
            result = con.execute("SELECT COUNT(*) FROM options_prices").fetchone()
            print(f"\nOptions data inserted successfully: {result[0]} rows")
        
        # Get final database size
        db_size = os.path.getsize(db_path) / (1024 * 1024)  # Convert to MB
        print(f"\nFinal database size: {db_size:.2f} MB")
        
        con.close()
        return True
        
    except Exception as e:
        print(f"Error creating derivatives database: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_derivatives_database()

