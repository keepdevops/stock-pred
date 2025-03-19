import yfinance as yf
import pandas as pd
import logging
from src.modules.database import DatabaseConnector
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import duckdb

class DataCollector:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.db = None
        self._initialize_database()
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.download_sp500_var = tk.BooleanVar()
        self.download_nasdaq_var = tk.BooleanVar()
        self.download_forex_var = tk.BooleanVar()
        self.download_crypto_var = tk.BooleanVar()
        self.connect_to_database()
        self.initialize_database()

    def _initialize_database(self):
        """Initialize database connection with specified table name."""
        try:
            self.db = DatabaseConnector(
                db_path=self.config.database.path,
                table_name="stock_data",  # You can make this configurable
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    def connect_to_database(self):
        """Connect to the DuckDB database"""
        try:
            db_path = "data/market_data.duckdb"
            self.connection = duckdb.connect(db_path)
            self.cursor = self.connection.cursor()
            self.logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise

    def initialize_database(self):
        """Initialize database with correct schema"""
        try:
            # Drop existing table to ensure clean slate
            self.cursor.execute("DROP TABLE IF EXISTS stock_data")
            
            # Create table with all required columns including adj_close
            create_table_sql = """
            CREATE TABLE stock_data (
                date TIMESTAMP NOT NULL,
                ticker VARCHAR NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (date, ticker)
            )
            """
            
            self.cursor.execute(create_table_sql)
            self.connection.commit()
            
            # Verify the table structure
            self.cursor.execute("""
                SELECT column_name, data_type, is_nullable, is_primary_key
                FROM information_schema.columns 
                WHERE table_name = 'stock_data'
                ORDER BY ordinal_position
            """)
            columns = self.cursor.fetchall()
            self.logger.info("Table structure:")
            for col in columns:
                self.logger.info(f"  {col}")
            
            self.logger.info("Database tables initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    def close(self):
        """Properly close all resources."""
        if self.db is not None:
            self.db.close()
            self.db = None

    def __del__(self):
        """Ensure resources are cleaned up."""
        self.close()

    def download_ticker_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download data for a single ticker and transform it to the required format."""
        try:
            self.logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            
            # Download data from YFinance
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False  # Ensure we get both adjusted and unadjusted prices
            )
            
            if data.empty:
                self.logger.warning(f"No data found for {ticker}")
                return None

            # Ensure all required columns are present
            required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns for {ticker}: {missing_columns}")
                if 'Adj Close' in missing_columns:
                    data['Adj Close'] = data['Close']
            
            # Log the data structure
            self.logger.debug(f"Downloaded data columns: {data.columns.tolist()}")
            self.logger.debug(f"First row of downloaded data:\n{data.iloc[0]}")
            
            return data

        except Exception as e:
            self.logger.error(f"Error downloading {ticker}: {str(e)}")
            return None 

    def download_multiple_tickers(self, tickers, start_date=None, end_date=None, batch_size=5):
        """
        Download data for multiple tickers with batching and error handling.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            batch_size (int): Number of tickers to process simultaneously
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        self.logger.info(f"Starting download for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Track success and failures
        results = {
            'success': [],
            'failed': [],
            'errors': {}
        }

        # Process tickers in batches
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}: {batch_tickers}")

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_ticker = {
                    executor.submit(
                        self._download_single_ticker_with_retry, 
                        ticker, 
                        start_date, 
                        end_date
                    ): ticker for ticker in batch_tickers
                }

                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        success = future.result()
                        if success:
                            results['success'].append(ticker)
                            self.logger.info(f"Successfully downloaded and saved {ticker}")
                        else:
                            results['failed'].append(ticker)
                            self.logger.error(f"Failed to download {ticker}")
                    except Exception as e:
                        results['failed'].append(ticker)
                        results['errors'][ticker] = str(e)
                        self.logger.error(f"Error processing {ticker}: {str(e)}")

            # Add a small delay between batches to avoid rate limiting
            time.sleep(1)

        # Log summary
        self.logger.info("\nDownload Summary:")
        self.logger.info(f"Total tickers processed: {len(tickers)}")
        self.logger.info(f"Successfully downloaded: {len(results['success'])}")
        self.logger.info(f"Failed downloads: {len(results['failed'])}")
        
        if results['failed']:
            self.logger.info("\nFailed tickers:")
            for ticker in results['failed']:
                error_msg = results['errors'].get(ticker, 'Unknown error')
                self.logger.info(f"{ticker}: {error_msg}")

        return results

    def _download_single_ticker_with_retry(self, ticker, start_date, end_date):
        """Download data for a single ticker with retries."""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Downloading {ticker} (Attempt {attempt + 1}/{self.max_retries})")
                
                # Download data
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False  # Get both adjusted and unadjusted prices
                )

                if data.empty:
                    self.logger.warning(f"No data found for {ticker}")
                    return False

                # Save to database
                self.save_ticker_data(ticker, data)
                return True

            except Exception as e:
                self.logger.error(f"Error downloading {ticker} (Attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue

        return False

    def get_sp500_tickers(self):
        """Get S&P 500 tickers using yfinance."""
        try:
            sp500 = yf.Ticker('^GSPC')
            return sp500.index_components
        except:
            # Fallback to pandas_datareader if yfinance fails
            import pandas_datareader.data as web
            sp500 = web.get_iex_symbols()
            return [symbol for symbol in sp500['symbol'] if symbol.isalpha()]

    def get_nasdaq100_tickers(self):
        """Get NASDAQ-100 tickers using yfinance."""
        try:
            nasdaq = yf.Ticker('^NDX')
            return nasdaq.index_components
        except:
            # Return some major NASDAQ stocks as fallback
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

    def get_forex_pairs(self):
        """Get major forex pairs."""
        return [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 
            'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X'
        ]

    def get_crypto_tickers(self):
        """Get major cryptocurrency tickers."""
        return [
            'BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 
            'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOT-USD'
        ]

    def save_ticker_data(self, ticker: str, data: pd.DataFrame) -> None:
        """Save ticker data to database"""
        try:
            self.logger.info(f"Saving data for {ticker}")
            
            # Log incoming data structure
            self.logger.info(f"Incoming data columns: {data.columns.tolist()}")
            
            # Prepare data for insertion
            insert_data = data.copy()
            insert_data['ticker'] = ticker
            
            # Rename columns to match database schema
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
            insert_data = insert_data.rename(columns=column_mapping)
            
            # Ensure columns are in correct order
            columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            insert_data = insert_data[columns]
            
            # Log prepared data structure
            self.logger.info(f"Prepared data columns: {insert_data.columns.tolist()}")
            
            # Delete existing data for this ticker
            self.cursor.execute("DELETE FROM stock_data WHERE ticker = ?", [ticker])
            
            # Insert new data
            self.cursor.execute("""
                INSERT INTO stock_data 
                SELECT * FROM read_pandas(?)
            """, [insert_data])
            
            self.connection.commit()
            self.logger.info(f"Successfully saved {len(insert_data)} records for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error saving data for {ticker}: {e}")
            raise

if __name__ == "__main__":
    config = load_config()  # Your config loading function
    app = DataCollectorGUI(config)
    app.run() 