import yfinance as yf
import duckdb
import pandas as pd
from datetime import datetime
import time

class HistoricalStockDownloader:
    def __init__(self):
        # Connect to existing database in read mode to get sectors
        self.read_conn = duckdb.connect('industry_market_data.db', read_only=True)
        # Create new connection for writing
        self.write_conn = duckdb.connect('historical_market_data.db')
        self.setup_database()

    def setup_database(self):
        """Create necessary tables for historical data"""
        self.write_conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_prices (
                ticker VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                sector VARCHAR,
                industry VARCHAR,
                updated_at TIMESTAMP
            )
        """)

    def get_existing_tickers(self):
        """Get unique tickers and their sectors from existing database"""
        try:
            result = self.read_conn.execute("""
                SELECT DISTINCT 
                    ticker,
                    sector,
                    industry
                FROM companies
                WHERE ticker IS NOT NULL
                  AND sector IS NOT NULL
            """).fetchall()
            return [(row[0], row[1], row[2]) for row in result]
        except Exception as e:
            print(f"Error getting existing tickers: {e}")
            return []

    def download_historical_data(self, ticker, sector, industry):
        """Download historical data for a specific ticker"""
        try:
            print(f"Downloading data for {ticker} ({sector})")
            stock = yf.Ticker(ticker)
            
            # Download data from 2023 through 2025
            df = stock.history(start="2023-01-01", end="2025-12-31")
            
            if df.empty:
                print(f"No data available for {ticker}")
                return
            
            df.reset_index(inplace=True)
            df['ticker'] = ticker
            df['sector'] = sector
            df['industry'] = industry
            df['updated_at'] = datetime.now()
            
            # Store in DuckDB
            self.write_conn.execute("""
                INSERT INTO historical_prices 
                SELECT 
                    ticker,
                    Date as date,
                    Open as open,
                    High as high,
                    Low as low,
                    Close as close,
                    Volume as volume,
                    sector,
                    industry,
                    updated_at
                FROM df
            """)
            
            print(f"Successfully downloaded and stored data for {ticker}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    def run(self):
        """Main execution method"""
        tickers = self.get_existing_tickers()
        print(f"Found {len(tickers)} tickers to process")
        
        for ticker, sector, industry in tickers:
            self.download_historical_data(ticker, sector, industry)
            time.sleep(1)  # Rate limiting
            
        # Show summary after completion
        self.show_summary()

    def show_summary(self):
        """Display summary of downloaded data"""
        print("\nDownload Summary:")
        result = self.write_conn.execute("""
            SELECT 
                sector,
                COUNT(DISTINCT ticker) as tickers,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(*) as total_records
            FROM historical_prices
            GROUP BY sector
            ORDER BY sector
        """).fetchall()
        
        for row in result:
            print(f"\nSector: {row[0]}")
            print(f"Number of Tickers: {row[1]}")
            print(f"Date Range: {row[2]} to {row[3]}")
            print(f"Total Records: {row[4]}")
            print("-" * 50)

    def cleanup(self):
        """Clean up database connections"""
        try:
            self.read_conn.close()
            self.write_conn.close()
            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    downloader = None
    try:
        downloader = HistoricalStockDownloader()
        downloader.run()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if downloader:
            downloader.cleanup()

if __name__ == "__main__":
    main() 