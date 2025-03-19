import logging
import yfinance as yf
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self):
        """Initialize the data collector"""
        self.logger = logging.getLogger(__name__)
        self.connect_to_database()
        self.recreate_table()  # Force recreate table with correct schema

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

    def recreate_table(self):
        """Force recreate the table with correct schema"""
        try:
            # Drop existing table
            self.cursor.execute("DROP TABLE IF EXISTS stock_data")
            
            # Create table with correct schema including adj_close
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
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'stock_data'
                ORDER BY ordinal_position
            """)
            columns = self.cursor.fetchall()
            self.logger.info("Created table with columns:")
            for col in columns:
                self.logger.info(f"  {col[0]}: {col[1]} (Nullable: {col[2]})")
            
        except Exception as e:
            self.logger.error(f"Error creating table: {e}")
            raise

    def scan_and_create_schema(self, ticker: str) -> list:
        """Scan YFinance data and create matching schema"""
        try:
            # Download a small sample of data to inspect structure
            self.logger.info(f"Downloading sample data for {ticker} to determine schema")
            sample_data = yf.download(
                ticker,
                start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                end=datetime.now().strftime('%Y-%m-%d'),
                progress=False
            )
            
            if sample_data.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Get column information
            columns = []
            for col in sample_data.columns:
                # Determine data type
                dtype = sample_data[col].dtype
                if dtype == 'datetime64[ns]':
                    sql_type = 'TIMESTAMP'
                elif dtype == 'float64':
                    sql_type = 'DOUBLE'
                elif dtype == 'int64':
                    sql_type = 'BIGINT'
                else:
                    sql_type = 'VARCHAR'
                
                # Add to columns list
                columns.append({
                    'name': col.lower().replace(' ', '_'),
                    'type': sql_type,
                    'original_name': col
                })
            
            self.logger.info(f"Detected columns: {columns}")
            return columns
        
        except Exception as e:
            self.logger.error(f"Error scanning schema: {e}")
            raise

    def create_table_from_schema(self, columns: list):
        """Create DuckDB table based on detected schema"""
        try:
            # Drop existing table
            self.cursor.execute("DROP TABLE IF EXISTS stock_data")
            self.connection.commit()
            
            # Build CREATE TABLE statement
            column_defs = [
                "date TIMESTAMP NOT NULL",
                "ticker VARCHAR NOT NULL"
            ]
            
            for col in columns:
                if col['name'] not in ['date', 'ticker']:
                    column_defs.append(f"{col['name']} {col['type']}")
            
            create_table_sql = f"""
            CREATE TABLE stock_data (
                {','.join(column_defs)},
                PRIMARY KEY (date, ticker)
            )
            """
            
            self.logger.info(f"Creating table with SQL: {create_table_sql}")
            self.cursor.execute(create_table_sql)
            self.connection.commit()
            
            # Verify the table structure
            self.cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'stock_data'
                ORDER BY ordinal_position
            """)
            created_columns = self.cursor.fetchall()
            self.logger.info(f"Created table with columns: {created_columns}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error creating table: {e}")
            raise

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

    def download_ticker_data(self, ticker: str, start_date: str, end_date: str) -> None:
        """Download and save ticker data"""
        try:
            self.logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty:
                self.logger.warning(f"No data returned for {ticker}")
                return
                
            self.save_ticker_data(ticker, data)
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {ticker}: {e}")
            raise

    def get_realtime_data(self, ticker):
        """Get realtime data for a ticker."""
        try:
            # Use yfinance to get current data
            data = yf.download(ticker, period='1d', interval='1m')
            if not data.empty:
                # Get the latest record
                latest_data = data.iloc[-1:].copy()
                # Ensure the data has all required columns
                if 'Adj Close' in latest_data.columns:
                    latest_data.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
                else:
                    latest_data['adj_close'] = latest_data['Close']
                return latest_data
            return None
        except Exception as e:
            self.logger.error(f"Error getting realtime data for {ticker}: {e}")
            return None

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