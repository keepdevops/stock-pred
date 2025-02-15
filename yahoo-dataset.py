import polars as pl
import yfinance as yf
import duckdb
import pandas as pd
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=f'stock_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_sp500_symbols():
    """Get S&P 500 symbols as a starting point"""
    try:
        # Download S&P 500 components from Wikipedia
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return table['Symbol'].tolist()
    except Exception as e:
        logging.error(f"Error fetching S&P 500 symbols: {str(e)}")
        # Fallback to a few major stocks if the download fails
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

def download_stock_data(symbol, start_date, end_date):
    """Download stock data for a given symbol"""
    try:
        # Add a small delay before each request to avoid rate limiting
        time.sleep(1)
        
        # Download data with error handling
        prices = yf.download(symbol, 
                           start=start_date, 
                           end=end_date, 
                           progress=False,
                           timeout=5)
        
        if prices.empty:
            print(f"✗ No data available for {symbol}")
            return None
            
        # Get the last row of data
        last_row = prices.iloc[-1]
        if len(last_row) > 0:
            print(f"✓ Downloaded data for {symbol}")
            return {
                'symbol': symbol,
                'date': prices.index[-1].strftime('%Y-%m-%d'),
                'open': float(last_row.iloc[0]),
                'high': float(last_row.iloc[1]),
                'low': float(last_row.iloc[2]),
                'close': float(last_row.iloc[3]),
                'volume': int(last_row.iloc[4]),
                'adj_close': float(last_row.iloc[5]) if len(last_row) > 5 else float(last_row.iloc[3])
            }
        else:
            print(f"✗ No data found for {symbol}")
            return None
            
    except Exception as e:
        print(f"✗ Failed to download {symbol}: {str(e)}")
        return None

def main():
    """Main function to download and store stock data"""
    print("Starting stock data collection...")
    
    # Read S&P 500 symbols
    symbols = get_sp500_symbols()
    print(f"Found {len(symbols)} symbols to process")
    
    # Make sure any existing connections are closed
    try:
        duckdb.sql("SELECT 1")
        duckdb.sql("CHECKPOINT")
        duckdb.sql("CLOSE ALL")
    except:
        pass
        
    # Connect to database
    try:
        con = duckdb.connect('stocks.db', read_only=False)
        
        # Drop existing table if it exists and create new one
        con.execute("DROP TABLE IF EXISTS stocks")
        
        # Create new table with additional columns
        con.execute("""
            CREATE TABLE stocks (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                adj_close DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        successful_downloads = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"Processing {symbol} ({i}/{len(symbols)})...")
            
            data = download_stock_data(symbol, start_date='2024-01-01', end_date='2025-02-10')
            if data is not None:
                # Insert the data
                con.execute("""
                    INSERT INTO stocks (
                        symbol, date, open, high, low, 
                        close, volume, adj_close
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    data['symbol'], data['date'], data['open'], 
                    data['high'], data['low'], data['close'], 
                    data['volume'], data['adj_close']
                ])
                
                successful_downloads += 1
            
            # Pause every 50 requests to respect API limits
            if i > 0 and i % 50 == 0:
                print(f"\nPausing for 30 seconds to respect API limits...")
                time.sleep(30)
        
        if successful_downloads == 0:
            raise Exception("No data was downloaded")
            
        # Create index on symbol for faster queries
        con.execute("CREATE INDEX idx_stocks_symbol ON stocks(symbol)")
        
        # Commit changes and close connection
        con.commit()
        con.close()
        
        print(f"\nSuccessfully downloaded data for {successful_downloads} stocks")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'con' in locals():
            con.close()

if __name__ == "__main__":
    main()
