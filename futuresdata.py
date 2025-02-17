import yfinance as yf
import polars as pl
import duckdb
from datetime import datetime
import pandas as pd
import time
import traceback
import os

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

def download_derivatives_data(futures_tickers, options_tickers):
    """Download futures and options data"""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    all_data = []
    
    # Download futures data
    for ticker in futures_tickers:
        try:
            print(f"Downloading futures data for {ticker}")
            data = yf.download(ticker, start=start_date, end=end_date)
            data = data.reset_index()
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]
            
            data['ticker'] = ticker
            data['type'] = 'future'
            
            data_pl = pl.from_pandas(data).select([
                pl.col('date'),
                pl.col('ticker'),
                pl.col('type'),
                pl.col('open'),
                pl.col('high'), 
                pl.col('low'),
                pl.col('close'),
                pl.col('adj close').alias('adj_close'),
                pl.col('volume')
            ])
            
            all_data.append(data_pl)
            time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
            continue
    
    # Download options data
    for ticker in options_tickers:
        try:
            print(f"Downloading options data for {ticker}")
            stock = yf.Ticker(ticker)
            
            # Get options expiration dates
            expirations = stock.options
            
            for expiry in expirations:
                # Get calls and puts
                opt = stock.option_chain(expiry)
                
                # Process calls
                calls = opt.calls
                calls['type'] = 'call'
                calls['expiry'] = expiry
                calls['ticker'] = ticker
                
                # Process puts
                puts = opt.puts
                puts['type'] = 'put'
                puts['expiry'] = expiry
                puts['ticker'] = ticker
                
                for options_data in [calls, puts]:
                    options_pl = pl.from_pandas(options_data).select([
                        pl.col('ticker'),
                        pl.col('type'),
                        pl.col('expiry'),
                        pl.col('strike'),
                        pl.col('lastPrice').alias('last_price'),
                        pl.col('bid'),
                        pl.col('ask'),
                        pl.col('volume'),
                        pl.col('openInterest').alias('open_interest'),
                        pl.col('impliedVolatility').alias('implied_volatility')
                    ])
                    
                    all_data.append(options_pl)
                
            time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading options for {ticker}: {str(e)}")
            continue
            
    if all_data:
        return pl.concat(all_data)
    return None

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
    """Create and populate derivatives database"""
    db_path = "derivatives_market.db"
    try:
        # Connect to database
        con = duckdb.connect(db_path)
        
        # Create futures table
        con.execute("""
            CREATE TABLE IF NOT EXISTS futures_prices (
                date DATE,
                ticker VARCHAR,
                type VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT
            )
        """)
        
        # Create options table
        con.execute("""
            CREATE TABLE IF NOT EXISTS options_data (
                ticker VARCHAR,
                type VARCHAR,
                expiry DATE,
                strike DOUBLE,
                last_price DOUBLE,
                bid DOUBLE,
                ask DOUBLE,
                volume BIGINT,
                open_interest BIGINT,
                implied_volatility DOUBLE
            )
        """)
        
        # Get derivatives data
        futures_tickers, options_tickers = get_derivatives_tickers()
        data = download_derivatives_data(futures_tickers, options_tickers)
        
        if data is not None:
            # Convert to pandas for DuckDB compatibility
            pandas_df = data.to_pandas()
            
            # Register DataFrame
            con.register('temp_data', pandas_df)
            
            # Insert futures data
            con.execute("""
                INSERT INTO futures_prices
                SELECT date, ticker, type, open, high, low, close, adj_close, volume
                FROM temp_data
                WHERE type = 'future'
            """)
            
            # Insert options data
            con.execute("""
                INSERT INTO options_data
                SELECT ticker, type, expiry, strike, last_price, bid, ask, volume, 
                       open_interest, implied_volatility
                FROM temp_data
                WHERE type IN ('call', 'put')
            """)
            
            print(f"Successfully loaded derivatives data into database")
            
            # Get and display database size
            db_size = get_database_size(db_path)
            if db_size:
                print(f"Database size: {db_size} MB")
        
        con.close()
        return True
        
    except Exception as e:
        print(f"Error creating derivatives database: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_derivatives_database()

