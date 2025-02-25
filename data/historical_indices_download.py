import yfinance as yf
import duckdb
import pandas as pd
from datetime import datetime
import time

class IndicesDataDownloader:
    def __init__(self):
        # Create connection for writing indices data
        self.conn = duckdb.connect('historical_market_data.db')
        self.setup_database()
        
        # Define major indices and their components
        self.indices = {
            'Major US': {
                '^GSPC': 'S&P 500',
                '^DJI':  'Dow Jones Industrial Average',
                '^IXIC': 'NASDAQ Composite',
                '^RUT':  'Russell 2000',
                '^VIX':  'Volatility Index'
            },
            'NASDAQ': {
                '^NDX':   'NASDAQ-100',
                '^IXCO':  'NASDAQ Computer',
                '^IXHC':  'NASDAQ Healthcare',
                '^IXFN':  'NASDAQ Financial',
                '^IXBK':  'NASDAQ Bank'
            },
            'NYSE': {
                '^NYA':   'NYSE Composite',
                '^XMI':   'NYSE ARCA Major Market',
                '^XAX':   'NYSE AMEX Composite',
                '^NYE':   'NYSE Energy',
                '^NYF':   'NYSE Financial'
            },
            'Sector': {
                '^SP500-45': 'S&P 500 Information Technology',
                '^SP500-40': 'S&P 500 Financials',
                '^SP500-35': 'S&P 500 Healthcare',
                '^SP500-30': 'S&P 500 Consumer Staples',
                '^SP500-25': 'S&P 500 Consumer Discretionary'
            }
        }

    def setup_database(self):
        """Create necessary table for indices data"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_indices (
                symbol VARCHAR,
                name VARCHAR,
                category VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                updated_at TIMESTAMP
            )
        """)

    def download_index_data(self, symbol, name, category):
        """Download historical data for a specific index"""
        try:
            print(f"Downloading data for {name} ({symbol})")
            index = yf.Ticker(symbol)
            
            # Download data from 2023 through 2025
            df = index.history(start="2023-01-01", end="2025-12-31")
            
            if df.empty:
                print(f"No data available for {symbol}")
                return
            
            df.reset_index(inplace=True)
            df['symbol'] = symbol
            df['name'] = name
            df['category'] = category
            df['updated_at'] = datetime.now()
            
            # Handle volume being None or NaN
            df['Volume'] = df['Volume'].fillna(0)
            
            # Store in DuckDB
            self.conn.execute("""
                INSERT INTO historical_indices 
                SELECT 
                    symbol,
                    name,
                    category,
                    Date as date,
                    Open as open,
                    High as high,
                    Low as low,
                    Close as close,
                    CAST(Volume as BIGINT) as volume,
                    updated_at
                FROM df
            """)
            
            print(f"Successfully downloaded and stored data for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

    def run(self):
        """Main execution method"""
        total_indices = sum(len(indices) for indices in self.indices.values())
        print(f"Starting download for {total_indices} indices across {len(self.indices)} categories")
        
        for category, indices in self.indices.items():
            print(f"\nProcessing {category} indices...")
            for symbol, name in indices.items():
                self.download_index_data(symbol, name, category)
                time.sleep(1)  # Rate limiting
            
        # Show summary after completion
        self.show_summary()

    def show_summary(self):
        """Display summary of downloaded indices data"""
        print("\nIndices Download Summary:")
        result = self.conn.execute("""
            SELECT 
                category,
                COUNT(DISTINCT symbol) as indices,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(*) as total_records
            FROM historical_indices
            GROUP BY category
            ORDER BY category
        """).fetchall()
        
        for row in result:
            print(f"\nCategory: {row[0]}")
            print(f"Number of Indices: {row[1]}")
            print(f"Date Range: {row[2]} to {row[3]}")
            print(f"Total Records: {row[4]}")
            print("-" * 50)
            
        # Show detailed index list
        print("\nDetailed Index List:")
        result = self.conn.execute("""
            SELECT 
                category,
                name,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM historical_indices
            GROUP BY category, name
            ORDER BY category, name
        """).fetchall()
        
        current_category = None
        for row in result:
            if current_category != row[0]:
                current_category = row[0]
                print(f"\n{current_category}:")
                print("=" * 40)
            print(f"- {row[1]}")
            print(f"  Date Range: {row[2]} to {row[3]}")

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
        downloader = IndicesDataDownloader()
        downloader.run()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if downloader:
            downloader.cleanup()

if __name__ == "__main__":
    main() 