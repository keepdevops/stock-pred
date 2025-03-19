import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from pathlib import Path
import os
import glob

class YFinanceTransformer:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def download_yfinance_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download data directly from YFinance."""
        try:
            self.logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            
            # Download data with auto_adjust=False to get both Close and Adj Close
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            
            # Log raw data structure
            self.logger.info(f"Raw data columns: {data.columns.tolist()}")
            self.logger.info(f"Raw data index: {type(data.index)}")
            
            # Reset index to make Date a column
            data = data.reset_index()
            
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
            
            # If we have multi-level columns, get the first level only
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Rename columns
            data = data.rename(columns=column_mapping)
            
            # Add ticker column
            data['ticker'] = ticker
            
            # Convert numeric columns to appropriate types
            numeric_columns = ['open', 'high', 'low', 'close', 'adj_close']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Convert volume to integer
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0).astype('int64')
            
            # Remove any rows with invalid data
            data = data.dropna(subset=['date'] + numeric_columns)
            
            # Ensure correct column order
            columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            data = data[columns]
            
            # Log processed data info
            self.logger.info(f"Processed data shape: {data.shape}")
            self.logger.info(f"Processed columns: {data.columns.tolist()}")
            self.logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {ticker}: {str(e)}")
            self.logger.exception(f"Detailed error for {ticker}")
            raise

    def verify_data_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Verify and fix data structure if needed."""
        required_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        
        # Verify all columns exist
        for col in required_columns:
            if col not in data.columns:
                if col == 'adj_close':
                    data['adj_close'] = data['close']
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Verify data types
        try:
            data['date'] = pd.to_datetime(data['date'])
            numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Error converting data types: {str(e)}")
        
        # Reorder columns
        return data[required_columns]

    def transform_csv_data(self, file_path: str) -> pd.DataFrame:
        """Transform CSV data to match the database schema."""
        try:
            # Read CSV file, skipping the first two rows (Ticker and Date headers)
            df = pd.read_csv(file_path, skiprows=[1, 2])
            
            self.logger.info(f"Original columns: {df.columns.tolist()}")
            
            # Rename columns to match database schema
            column_mapping = {
                'Price': 'adj_close',  # Using Price as adj_close
                'Close': 'close',
                'High': 'high',
                'Low': 'low',
                'Open': 'open',
                'Volume': 'volume',
                'Date': 'date',
                'ticker': 'ticker'
            }
            
            # Rename columns that exist in the mapping
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df.index if df.index.name == 'Date' else df['date'])
            
            # Select and order columns to match database schema
            required_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            
            # Log data info before processing
            self.logger.info(f"Data columns after renaming: {df.columns.tolist()}")
            self.logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    raise ValueError(f"Missing required column: {col}")
            
            # Select only the required columns in the correct order
            df = df[required_columns]
            
            # Log final data structure
            self.logger.info(f"Final columns: {df.columns.tolist()}")
            self.logger.info(f"Number of rows: {len(df)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error transforming CSV data: {str(e)}")
            raise

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the data format and content.
        
        Args:
            data (pd.DataFrame): DataFrame to validate
        """
        try:
            # Check required columns
            required_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                raise ValueError("Date column must be datetime type")
            
            # Check for numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(f"Column {col} must be numeric type")
            
            # Check for missing values
            if data.isnull().any().any():
                raise ValueError("Dataset contains missing values")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise

    def import_csv_file(self, filepath: str) -> None:
        """Import a single CSV file."""
        try:
            # Extract ticker from filename (first 3-4 letters before _historical.csv)
            filename = os.path.basename(filepath)
            ticker = filename.split('_')[0].upper()  # Get characters before '_' and convert to uppercase
            
            self.logger.info(f"Processing {filename} for ticker {ticker}...")
            
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Rename columns to match schema
            column_mapping = {
                'Date': 'date',
                'Price': 'adj_close',  # Using Price as adj_close
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure date is in correct format
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'adj_close']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert volume to integer
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')
            
            # Remove any rows with invalid data
            df = df.dropna(subset=['date'] + numeric_columns)
            
            # Ensure correct column order
            columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            df = df[columns]
            
            # Log data info
            self.logger.info(f"Found {len(df)} rows of data for {ticker}")
            self.logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Save to database
            self.db.save_ticker_data(ticker, df)
            self.logger.info(f"Successfully imported {ticker} data")
            
        except Exception as e:
            self.logger.error(f"Error importing {filepath}: {str(e)}")
            raise

    def import_csv_files(self, directory: str) -> None:
        """Import all CSV files from a directory."""
        try:
            # Get list of CSV files
            csv_files = glob.glob(os.path.join(directory, '*_historical.csv'))
            self.logger.info(f"Found {len(csv_files)} CSV files to process")
            
            success_count = 0
            fail_count = 0
            
            for filepath in csv_files:
                try:
                    self.import_csv_file(filepath)
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to import {filepath}: {str(e)}")
                    fail_count += 1
            
            self.logger.info("\nImport Summary:")
            self.logger.info(f"Total files processed: {len(csv_files)}")
            self.logger.info(f"Successfully imported: {success_count}")
            self.logger.info(f"Failed imports: {fail_count}")
            
        except Exception as e:
            self.logger.error(f"Error during CSV import: {str(e)}")
            raise 