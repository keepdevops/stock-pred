import yfinance as yf
import duckdb
import pandas as pd
from datetime import datetime
import time

DATABASE_PATH = 'historical_market_data.db'

class CommoditiesDataDownloader:
    def __init__(self):
        # Create connection for writing commodities data
        self.conn = duckdb.connect(DATABASE_PATH)
        self.setup_database()
        
        # Define commodities by sector
        self.commodities = {
            'Energy': [
                'CL=F',    # Crude Oil
                'BZ=F',    # Brent Crude
                'NG=F',    # Natural Gas
                'RB=F',    # Gasoline
                'HO=F'     # Heating Oil
            ],
            'Metals': [
                'GC=F',    # Gold
                'SI=F',    # Silver
                'PL=F',    # Platinum
                'PA=F',    # Palladium
                'HG=F',    # Copper
                'ALI=F'    # Aluminum
            ],
            'Agriculture': [
                'ZC=F',    # Corn
                'ZW=F',    # Wheat
                'ZS=F',    # Soybeans
                'KC=F',    # Coffee
                'CT=F',    # Cotton
                'SB=F'     # Sugar
            ],
            'Livestock': [
                'LE=F',    # Live Cattle
                'HE=F',    # Lean Hogs
                'GF=F'     # Feeder Cattle
            ],
            'Softs': [
                'CC=F',    # Cocoa
                'OJ=F',    # Orange Juice
                'LBS=F'    # Lumber
            ]
        }

    def setup_database(self):
        """Create necessary table for commodities data"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_commodities (
                ticker VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                sector VARCHAR,
                updated_at TIMESTAMP
            )
        """)

    def download_commodity_data(self, symbol, sector):
        """Download historical data for a specific commodity"""
        try:
            print(f"Downloading data for {symbol} ({sector})")
            commodity = yf.Ticker(symbol)
            
            # Download data from 2023 through 2025
            df = commodity.history(start="2023-01-01", end="2025-12-31")
            
            if df.empty:
                print(f"No data available for {symbol}")
                return
            
            df.reset_index(inplace=True)
            df['ticker'] = symbol
            df['sector'] = sector
            df['updated_at'] = datetime.now()
            
            # Handle volume being None or NaN
            df['Volume'] = df['Volume'].fillna(0)
            
            # Store in DuckDB
            self.conn.execute("""
                INSERT INTO historical_commodities 
                SELECT 
                    ticker,
                    date as date,
                    open as open,
                    high as high,
                    low as low,
                    close as close,
                    CAST(Volume as BIGINT) as volume,
                    sector,
                    updated_at
                FROM df
            """)
            
            print(f"Successfully downloaded and stored data for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

    def run(self):
        """Main execution method"""
        total_commodities = sum(len(commodities) for commodities in self.commodities.values())
        print(f"Starting download for {total_commodities} commodities across {len(self.commodities)} sectors")
        
        for sector, symbols in self.commodities.items():
            print(f"\nProcessing {sector} sector...")
            for symbol in symbols:
                self.download_commodity_data(symbol, sector)
                time.sleep(1)  # Rate limiting
            
        # Show summary after completion
        self.show_summary()

    def show_summary(self):
        """Display summary of downloaded commodities data"""
        print("\nCommodities Download Summary:")
        result = self.conn.execute("""
            SELECT 
                sector,
                COUNT(DISTINCT ticker) as commodities,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(*) as total_records
            FROM historical_commodities
            GROUP BY sector
            ORDER BY sector
        """).fetchall()
        
        for row in result:
            print(f"\nSector: {row[0]}")
            print(f"Number of Commodities: {row[1]}")
            print(f"Date Range: {row[2]} to {row[3]}")
            print(f"Total Records: {row[4]}")
            print("-" * 50)

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
        downloader = CommoditiesDataDownloader()
        downloader.run()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if downloader:
            downloader.cleanup()

if __name__ == "__main__":
    main() 
