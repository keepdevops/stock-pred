import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import duckdb
import csv
from concurrent.futures import ThreadPoolExecutor

class DataLoader:
    """Fetches, cleans, and organizes raw stock data into DuckDB."""
    
    def __init__(self, output_db: Union[str, Path]):
        self.output_db = Path(output_db)
        self.engine = create_engine(f"duckdb:///{self.output_db}")
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_and_sort_csv(
        self,
        csv_path: Path,
        sector: str,
        ticker_column: str = 'ticker'
    ) -> bool:
        """Load CSV data and insert into appropriate sector table."""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', ticker_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Rename ticker column if different
            if ticker_column != 'ticker':
                df = df.rename(columns={ticker_column: 'ticker'})
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add updated_at timestamp
            df['updated_at'] = datetime.now()
            
            # Create sector table if it doesn't exist
            table_name = f"{sector.lower()}_stocks"
            with self.engine.connect() as conn:
                conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    date DATETIME,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    adj_close FLOAT,
                    volume FLOAT,
                    ticker VARCHAR,
                    date DATETIME
                )
                """)
            
            # Insert data
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            self.logger.info(f"Loaded {len(df)} rows into {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return False
    
    def fetch_yfinance_data(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch stock data from YFinance."""
        try:
            # Set default dates if not provided
            end_date = end_date or datetime.now()
            start_date = start_date or (end_date - timedelta(days=365))
            
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                self.logger.warning(f"No data found for {ticker}")
                return None
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Rename columns to match schema
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            # Add ticker column
            df['ticker'] = ticker
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def bulk_download_tickers(
        self,
        sector_mappings: Dict[str, List[str]],
        start_date: Optional[datetime] = None,
        max_workers: int = 5
    ) -> bool:
        """Download data for multiple tickers and organize by sector."""
        try:
            for sector, tickers in sector_mappings.items():
                self.logger.info(f"Processing {sector} sector...")
                
                # Create sector table
                table_name = f"{sector.lower()}_stocks"
                with self.engine.connect() as conn:
                    conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        date DATETIME,
                        open FLOAT,
                        high FLOAT,
                        low FLOAT,
                        close FLOAT,
                        adj_close FLOAT,
                        volume FLOAT,
                        ticker VARCHAR,
                        da DATETIME
                    )
                    """)
                
                # Download data in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ticker = {
                        executor.submit(self.fetch_yfinance_data, ticker, start_date): ticker
                        for ticker in tickers
                    }
                    
                    # Process completed downloads
                    all_data = []
                    for future in future_to_ticker:
                        ticker = future_to_ticker[future]
                        try:
                            df = future.result()
                            if df is not None:
                                all_data.append(df)
                        except Exception as e:
                            self.logger.error(f"Error processing {ticker}: {e}")
                    
                    if all_data:
                        # Combine all data for this sector
                        sector_df = pd.concat(all_data, ignore_index=True)
                        sector_df['updated_at'] = datetime.now()
                        
                        # Save to database
                        sector_df.to_sql(
                            table_name,
                            self.engine,
                            if_exists='append',
                            index=False,
                            method='multi',
                            chunksize=1000
                        )
                        
                        self.logger.info(f"Saved {len(sector_df)} rows to {table_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in bulk download: {e}")
            return False
    
    def create_sector_tables(self, sector_mappings: Dict[str, List[str]]) -> bool:
        """Create tables for different market sectors."""
        try:
            with self.engine.connect() as conn:
                for sector in sector_mappings:
                    table_name = f"{sector.lower()}_stocks"
                    
                    # Create table
                    conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        date DATETIME,
                        open FLOAT,
                        high FLOAT,
                        low FLOAT,
                        close FLOAT,
                        adj_close FLOAT,
                        volume FLOAT,
                        ticker VARCHAR,
                        date DATETIME
                    )
                    """)
                    
                    # Create indexes
                    conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_date_ticker 
                    ON {table_name}(date, ticker)
                    """)
                    
                    self.logger.info(f"Created sector table: {table_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating sector tables: {e}")
            return False 