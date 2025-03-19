import polars as pl
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import json

class DatabaseConverter:
    def __init__(self, config_path: str = "config.json"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.batch_size = self.config.get('batch_validation', {}).get('batch_size', 10)
        
    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"{config_path} not found, using defaults")
            return {'batch_validation': {'batch_size': 10}}

    def convert_to_database(self, input_dir: str = "data/clean", 
                          output_path: str = "data/database/market_data.parquet") -> None:
        """Convert cleaned CSV files to Polars database"""
        try:
            input_path = Path(input_dir)
            output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get list of all clean CSV files
            csv_files = list(input_path.glob("clean_*.csv"))
            
            # Process files in batches
            for i in range(0, len(csv_files), self.batch_size):
                batch_files = csv_files[i:i + self.batch_size]
                self._process_batch(batch_files, output_path)
                
        except Exception as e:
            self.logger.error(f"Error in convert_to_database: {e}")
            raise

    def _process_batch(self, files: List[Path], output_path: Path) -> None:
        """Process a batch of CSV files"""
        try:
            # Validate and load each file in the batch
            dataframes = []
            for file in files:
                if self._validate_csv(file):
                    df = pl.read_csv(file)
                    dataframes.append(df)
            
            if not dataframes:
                return
                
            # Combine all dataframes in the batch
            combined_df = pl.concat(dataframes)
            
            # Create or append to the database
            if output_path.exists():
                existing_df = pl.read_parquet(output_path)
                combined_df = pl.concat([existing_df, combined_df])
            
            # Sort and deduplicate
            combined_df = (combined_df
                         .sort(['ticker', 'date'])
                         .unique(subset=['ticker', 'date'], keep='last'))
            
            # Save to parquet with compression
            combined_df.write_parquet(
                output_path,
                compression='snappy',
                statistics=True
            )
            
            self.logger.info(f"Processed batch of {len(files)} files")
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise

    def _validate_csv(self, file_path: Path) -> bool:
        """Validate a CSV file before conversion"""
        try:
            # Read the first few rows to check structure
            df = pd.read_csv(file_path, nrows=5)
            
            # Check required columns
            required_columns = {'date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume'}
            if not required_columns.issubset(set(map(str.lower, df.columns))):
                self.logger.error(f"Missing required columns in {file_path}")
                return False
            
            # Check date format
            try:
                pd.to_datetime(df['date'])
            except Exception:
                self.logger.error(f"Invalid date format in {file_path}")
                return False
            
            # Check numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in numeric_columns:
                if not pd.to_numeric(df[col], errors='coerce').notna().all():
                    self.logger.error(f"Invalid numeric values in {col} column of {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating {file_path}: {e}")
            return False 