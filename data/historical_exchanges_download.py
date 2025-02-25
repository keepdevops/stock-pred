import yfinance as yf
import duckdb
import pandas as pd
from datetime import datetime
import time

class ExchangeDataDownloader:
    def __init__(self):
        # Create connection for writing exchange data
        self.conn = duckdb.connect('historical_market_data.db')
        self.setup_database()
        
        # Define major sectors and their key stocks for both exchanges
        self.exchange_sectors = {
            'NASDAQ': {
                'Technology': [
                    'AAPL',  # Apple
                    'MSFT',  # Microsoft
                    'GOOGL', # Alphabet
                    'NVDA',  # NVIDIA
                    'INTC'   # Intel
                ],
                'Healthcare': [
                    'AMGN',  # Amgen
                    'GILD',  # Gilead
                    'BIIB',  # Biogen
                    'REGN',  # Regeneron
                    'VRTX'   # Vertex
                ],
                'Consumer': [
                    'AMZN',  # Amazon
                    'META',  # Meta
                    'NFLX',  # Netflix
                    'TSLA',  # Tesla
                    'COST'   # Costco
                ],
                'Communications': [
                    'CMCSA', # Comcast
                    'TMUS',  # T-Mobile
                    'ATVI',  # Activision
                    'EA',    # Electronic Arts
                    'NTES'   # NetEase
                ]
            },
            'NYSE': {
                'Financial': [
                    'JPM',   # JPMorgan
                    'BAC',   # Bank of America
                    'WFC',   # Wells Fargo
                    'GS',    # Goldman Sachs
                    'MS'     # Morgan Stanley
                ],
                'Energy': [
                    'XOM',   # ExxonMobil
                    'CVX',   # Chevron
                    'COP',   # ConocoPhillips
                    'EOG',   # EOG Resources
                    'SLB'    # Schlumberger
                ],
                'Industrial': [
                    'GE',    # General Electric
                    'HON',   # Honeywell
                    'UPS',   # United Parcel Service
                    'CAT',   # Caterpillar
                    'BA'     # Boeing
                ],
                'Materials': [
                    'LIN',   # Linde
                    'APD',   # Air Products
                    'ECL',   # Ecolab
                    'NEM',   # Newmont
                    'FCX'    # Freeport-McMoRan
                ]
            }
        }

    def setup_database(self):
        """Create necessary table for exchange data"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_exchanges (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                exchange VARCHAR,
                sector VARCHAR,
                updated_at TIMESTAMP
            )
        """)

    def download_stock_data(self, symbol, exchange, sector):
        """Download historical data for a specific stock"""
        try:
            print(f"Downloading data for {symbol} ({exchange} - {sector})")
            stock = yf.Ticker(symbol)
            
            # Download data from 2023 through 2025
            df = stock.history(start="2023-01-01", end="2025-12-31")
            
            if df.empty:
                print(f"No data available for {symbol}")
                return
            
            df.reset_index(inplace=True)
            df['symbol'] = symbol
            df['exchange'] = exchange
            df['sector'] = sector
            df['updated_at'] = datetime.now()
            
            # Handle volume being None or NaN
            df['Volume'] = df['Volume'].fillna(0)
            
            # Store in DuckDB
            self.conn.execute("""
                INSERT INTO historical_exchanges 
                SELECT 
                    symbol,
                    Date as date,
                    Open as open,
                    High as high,
                    Low as low,
                    Close as close,
                    CAST(Volume as BIGINT) as volume,
                    exchange,
                    sector,
                    updated_at
                FROM df
            """)
            
            print(f"Successfully downloaded and stored data for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

    def run(self):
        """Main execution method"""
        for exchange, sectors in self.exchange_sectors.items():
            total_stocks = sum(len(stocks) for stocks in sectors.values())
            print(f"\nStarting download for {exchange}: {total_stocks} stocks across {len(sectors)} sectors")
            
            for sector, symbols in sectors.items():
                print(f"\nProcessing {exchange} - {sector} sector...")
                for symbol in symbols:
                    self.download_stock_data(symbol, exchange, sector)
                    time.sleep(1)  # Rate limiting
            
        # Show summary after completion
        self.show_summary()

    def show_summary(self):
        """Display summary of downloaded exchange data"""
        print("\nExchange Download Summary:")
        result = self.conn.execute("""
            SELECT 
                exchange,
                sector,
                COUNT(DISTINCT symbol) as stocks,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(*) as total_records
            FROM historical_exchanges
            GROUP BY exchange, sector
            ORDER BY exchange, sector
        """).fetchall()
        
        current_exchange = None
        for row in result:
            if current_exchange != row[0]:
                current_exchange = row[0]
                print(f"\n{current_exchange} Exchange:")
                print("=" * 50)
            
            print(f"\nSector: {row[1]}")
            print(f"Number of Stocks: {row[2]}")
            print(f"Date Range: {row[3]} to {row[4]}")
            print(f"Total Records: {row[5]}")
            print("-" * 40)

    def cleanup(self):
        """Clean up database connection"""
        try:
            self.conn.close()
            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    downloader = None
    try:
        downloader = ExchangeDataDownloader()
        downloader.run()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if downloader:
            downloader.cleanup()

if __name__ == "__main__":
    main() 