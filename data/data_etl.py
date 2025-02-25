import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import polars as pl
import duckdb
from datetime import datetime, timedelta
import pandas as pd
import requests
import time
import numpy as np
import random
import sys
import os
import logging
from io import StringIO
from typing import List, Dict, Optional, Tuple, Any, Set
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from dataclasses import dataclass
import json

# Configure logging for external libraries
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3.connectionpool').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').propagate = False
logging.getLogger('urllib3.connectionpool').propagate = False

# Configuration management
@dataclass
class ETLConfig:
    """Configuration for the ETL pipeline"""
    # Database settings
    db_path: str = 'stocks.db'
    create_new_db: bool = True
    
    # API settings
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    # Retry settings
    max_retries: int = 3
    base_delay: float = 1.0
    backoff_factor: float = 2.0
    
    # Batch settings
    batch_size: int = 50
    db_batch_size: int = 1000
    
    # Logging settings
    log_level: int = logging.INFO
    
    # Data settings
    download_period: str = "1y"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "db_path": self.db_path,
            "create_new_db": self.create_new_db,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "backoff_factor": self.backoff_factor,
            "batch_size": self.batch_size,
            "db_batch_size": self.db_batch_size,
            "log_level": self.log_level,
            "download_period": self.download_period
        }


# 1. Extraction Layer
class StockExtractor:
    """Handles extraction of stock data from various sources"""
    
    def __init__(self, config: ETLConfig):
        self.config = config
        self.session = None
        self.rate_limit = asyncio.Semaphore(config.max_concurrent_requests)
        self.logger = logging.getLogger('StockExtractor')
        self.failed_downloads: Set[str] = set()
        
    async def initialize_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit_per_host=self.config.max_concurrent_requests)
            self.session = aiohttp.ClientSession(
                connector=connector, 
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            )

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_tickers(self) -> List[str]:
        """Fetch list of NYSE tickers"""
        try:
            url = "https://www.nyse.com/api/quotes/filter"
            payload = {
                "instrumentType": "EQUITY",
                "pageNumber": 1,
                "sortColumn": "SYMBOL",
                "sortOrder": "ASC",
                "maxResultsPerPage": 10000,
                "filterToken": ""
            }
            headers = {"Content-Type": "application/json"}
            
            await self.initialize_session()
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to fetch tickers: HTTP {response.status}")
                    return []
                    
                data = await response.json()
                tickers = [item['symbolTicker'] for item in data]
                self.logger.info(f"Found {len(tickers)} NYSE tickers")
                return sorted(tickers)
        except Exception as e:
            self.logger.error(f"Error fetching NYSE tickers: {str(e)}")
            return []
    
    async def fetch_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch stock data for a single ticker"""
        async with self.rate_limit:
            for attempt in range(self.config.max_retries):
                try:
                    # Add delay between attempts
                    if attempt > 0:
                        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
                        await asyncio.sleep(delay)
                    
                    # Use ThreadPoolExecutor for the blocking yfinance call
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor() as pool:
                        df = await loop.run_in_executor(
                            pool,
                            lambda: yf.download(
                                ticker, 
                                progress=False, 
                                period=self.config.download_period
                            )
                        )
                    
                    if df.empty:
                        self.logger.warning(f"Empty dataframe for {ticker}")
                        continue
                    
                    return df
                    
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        self.logger.warning(f"Retry {attempt + 1} for {ticker}: {str(e)}")
                    else:
                        self.logger.error(f"Failed to download {ticker}: {str(e)}")
                        self.failed_downloads.add(ticker)
            
            return None
    
    async def batch_fetch_stock_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch stock data for multiple tickers in parallel"""
        await self.initialize_session()
        
        # Don't log individual failures
        result = {}
        failed_count = 0
        empty_count = 0
        
        # Create tasks for all tickers
        tasks = [self.fetch_stock_data(ticker) for ticker in tickers]
        
        # Process tasks with progress bar
        for i, (ticker, future) in enumerate(zip(tickers, asyncio.as_completed(tasks))):
            try:
                df = await future
                if df is not None and not df.empty:
                    result[ticker] = df
                else:
                    empty_count += 1
            except Exception as e:
                failed_count += 1
            
            # Don't log anything here - let the progress bars show progress
        
        # At the end, log summaries
        if failed_count > 0:
            self.logger.warning(f"Failed to download {failed_count} tickers")
        
        if empty_count > 0:
            self.logger.warning(f"Empty dataframes for {empty_count} tickers")
        
        self.logger.info(f"Successfully downloaded {len(result)}/{len(tickers)} tickers")
        return result


# 2. Transformation Layer
class StockTransformer:
    """Handles transformation of raw stock data into standardized format"""
    
    def __init__(self, config: ETLConfig):
        self.config = config
        self.logger = logging.getLogger('StockTransformer')
    
    def standardize_format(self, raw_df: pd.DataFrame, ticker: str) -> Optional[pl.DataFrame]:
        """Standardize the format of a raw dataframe"""
        try:
            # Handle multi-index columns if they exist
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)
            
            # Ensure column names are unique and valid
            raw_df.columns = [str(col).strip().replace(' ', '_') for col in raw_df.columns]
            
            # Reset index to make date a column
            raw_df = raw_df.reset_index()
            
            # Add ticker column
            raw_df['ticker'] = ticker
            
            # Convert to polars
            df = pl.from_pandas(raw_df)
            
            # Standardize column names (case-insensitive mapping)
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj_Close': 'adj_close',
                'Volume': 'volume'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename({old_name: new_name})
            
            # If adj_close is missing, use close
            if 'adj_close' not in df.columns and 'close' in df.columns:
                df = df.with_columns(pl.col('close').alias('adj_close'))
            
            # Ensure all required columns exist
            required_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            
            # Check for missing columns
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                self.logger.error(f"Missing required columns for {ticker}: {missing_columns}")
                return None
            
            # Select columns in specific order
            df = df.select(required_columns)
            
            # Cast columns to correct types
            df = df.with_columns([
                pl.col('date').cast(pl.Date),
                pl.col('ticker').cast(pl.Utf8),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('adj_close').cast(pl.Float64),
                pl.col('volume').cast(pl.Int64)
            ])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error transforming data for {ticker}: {str(e)}")
            return None
    
    def validate_data(self, df: pl.DataFrame, ticker: str) -> bool:
        """Validate transformed data"""
        try:
            # Check if dataframe is empty
            if df.is_empty():
                self.logger.warning(f"Empty dataframe for {ticker}")
                return False
            
            # Check for missing values
            for col in df.columns:
                null_count = df.select(pl.col(col).is_null().sum()).item()
                if null_count > 0:
                    self.logger.warning(f"{ticker} has {null_count} null values in column {col}")
            
            # Check date range
            date_range = df.select([
                pl.min('date').alias('min_date'),
                pl.max('date').alias('max_date')
            ])
            min_date = date_range.item(0, 0)
            max_date = date_range.item(0, 1)
            
            # Calculate expected trading days (rough estimate)
            days_diff = (max_date - min_date).days
            expected_trading_days = int(days_diff * 5/7)  # Approximation for weekdays
            actual_trading_days = df.height
            
            # Check if we have at least 70% of expected trading days
            coverage = actual_trading_days / max(expected_trading_days, 1)
            if coverage < 0.7:
                self.logger.warning(
                    f"{ticker} has low coverage: {actual_trading_days}/{expected_trading_days} "
                    f"trading days ({coverage:.2%})"
                )
            
            # Check for price anomalies
            price_cols = ['open', 'high', 'low', 'close', 'adj_close']
            for col in price_cols:
                stats = df.select([
                    pl.min(col).alias('min'),
                    pl.max(col).alias('max'),
                    pl.mean(col).alias('mean'),
                    pl.std(col).alias('std')
                ]).to_dicts()[0]
                
                # Check for zero or negative prices
                if stats['min'] <= 0:
                    self.logger.warning(f"{ticker} has zero or negative values in {col}")
                
                # Check for extreme price changes
                if stats['std'] > stats['mean'] * 2:
                    self.logger.warning(f"{ticker} has high volatility in {col}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data for {ticker}: {str(e)}")
            return False
    
    def calculate_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate additional metrics for the data"""
        try:
            # Calculate daily returns
            df = df.with_columns(
                ((pl.col('close') - pl.col('close').shift(1)) / pl.col('close').shift(1)).alias('daily_return')
            )
            
            # Calculate moving averages
            df = df.with_columns([
                pl.col('close').rolling_mean(window_size=5).alias('ma_5'),
                pl.col('close').rolling_mean(window_size=20).alias('ma_20')
            ])
            
            # Calculate trading volume moving average
            df = df.with_columns(
                pl.col('volume').rolling_mean(window_size=5).alias('volume_ma_5')
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return df  # Return original dataframe if calculation fails
    
    def batch_transform(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pl.DataFrame]:
        """Transform multiple dataframes"""
        # Don't log individual failures
        result = {}
        error_count = 0
        
        for ticker, raw_df in raw_data.items():
            try:
                # Standardize format
                df = self.standardize_format(raw_df, ticker)
                if df is None:
                    continue
                
                # Validate data
                if not self.validate_data(df, ticker):
                    continue
                
                # Calculate additional metrics
                df = self.calculate_metrics(df)
                
                result[ticker] = df
                
            except Exception as e:
                error_count += 1
                continue
        
        # At the end, log summaries
        if error_count > 0:
            self.logger.warning(f"Failed to transform {error_count} tickers")
        
        self.logger.info(f"Successfully transformed {len(result)}/{len(raw_data)} dataframes")
        return result


# 3. Loading Layer
class DuckDBLoader:
    """Handles loading of transformed data into DuckDB"""
    
    def __init__(self, config: ETLConfig):
        self.config = config
        self.logger = logging.getLogger('DuckDBLoader')
        self.db_path = config.db_path
        
        # Delete existing database if requested
        if config.create_new_db and os.path.exists(self.db_path):
            os.remove(self.db_path)
            self.logger.info(f"Deleted existing database: {self.db_path}")
    
    def init_db_schema(self):
        """Initialize database schema"""
        try:
            con = duckdb.connect(self.db_path)
            
            # Create stock data table
            con.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    date DATE,
                    ticker VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume BIGINT,
                    daily_return DOUBLE,
                    ma_5 DOUBLE,
                    ma_20 DOUBLE,
                    volume_ma_5 DOUBLE,
                    PRIMARY KEY (date, ticker)
                )
            """)
            
            # Create metadata table
            con.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR
                )
            """)
            
            # Create indices
            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker ON stock_data(ticker);
                CREATE INDEX IF NOT EXISTS idx_date ON stock_data(date);
            """)
            
            # Store creation timestamp
            con.execute("""
                INSERT OR REPLACE INTO metadata VALUES ('created_at', ?)
            """, [datetime.now().isoformat()])
            
            self.logger.info("Database schema initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing database schema: {str(e)}")
            raise
        finally:
            if 'con' in locals():
                con.close()
    
    def load_data(self, df: pl.DataFrame, ticker: str):
        """Load a single dataframe into the database"""
        try:
            con = duckdb.connect(self.db_path)
            
            # Convert to pandas for DuckDB compatibility
            pdf = df.to_pandas()
            
            # Insert data
            con.execute("""
                INSERT OR REPLACE INTO stock_data 
                SELECT * FROM pdf
            """)
            
            self.logger.info(f"Loaded {len(df)} rows for {ticker}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {str(e)}")
            return False
        finally:
            if 'con' in locals():
                con.close()
    
    def batch_load_data(self, data: Dict[str, pl.DataFrame]):
        """Load multiple dataframes into the database"""
        if not data:
            self.logger.warning("No data to load")
            return
        
        try:
            # Combine all dataframes
            all_dfs = list(data.values())
            if not all_dfs:
                return
                
            combined = pl.concat(all_dfs)
            
            # Connect to DuckDB
            con = duckdb.connect(self.db_path)
            
            # Insert data in batches with progress bar
            total_rows = len(combined)
            batch_size = self.config.db_batch_size
            
            with tqdm(total=total_rows, desc="Loading data", unit="rows") as pbar:
                for i in range(0, total_rows, batch_size):
                    batch = combined.slice(i, min(batch_size, total_rows - i))
                    batch_pd = batch.to_pandas()
                    
                    con.execute("""
                        INSERT OR REPLACE INTO stock_data 
                        SELECT * FROM batch_pd
                    """)
                    
                    pbar.update(len(batch))
            
            # Update metadata
            con.execute("""
                INSERT OR REPLACE INTO metadata VALUES ('last_updated', ?),
                                                      ('total_tickers', ?),
                                                      ('total_records', ?)
            """, [
                datetime.now().isoformat(),
                len(data),
                total_rows
            ])
            
            self.logger.info(f"Successfully loaded {total_rows} rows for {len(data)} tickers")
            
        except Exception as e:
            self.logger.error(f"Error in batch loading: {str(e)}")
        finally:
            if 'con' in locals():
                con.close()
    
    def update_indexes(self):
        """Update database indexes"""
        try:
            con = duckdb.connect(self.db_path)
            
            # Reindex
            con.execute("REINDEX")
            
            self.logger.info("Database indexes updated")
            
        except Exception as e:
            self.logger.error(f"Error updating indexes: {str(e)}")
        finally:
            if 'con' in locals():
                con.close()
    
    def optimize_storage(self):
        """Optimize database storage"""
        try:
            con = duckdb.connect(self.db_path)
            
            # Vacuum database
            con.execute("VACUUM")
            
            # Analyze for query optimization
            con.execute("ANALYZE")
            
            self.logger.info("Database storage optimized")
            
        except Exception as e:
            self.logger.error(f"Error optimizing storage: {str(e)}")
        finally:
            if 'con' in locals():
                con.close()
    
    def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            con = duckdb.connect(self.db_path)
            
            # Get ticker count
            ticker_count = con.execute("""
                SELECT COUNT(DISTINCT ticker) FROM stock_data
            """).fetchone()[0]
            
            # Get record count
            record_count = con.execute("""
                SELECT COUNT(*) FROM stock_data
            """).fetchone()[0]
            
            # Get date range
            date_range = con.execute("""
                SELECT MIN(date), MAX(date) FROM stock_data
            """).fetchone()
            
            # Get ticker breakdown
            ticker_breakdown = con.execute("""
                SELECT 
                    LEFT(ticker, 1) as letter,
                    COUNT(DISTINCT ticker) as ticker_count,
                    COUNT(*) as record_count
                FROM stock_data
                GROUP BY letter
                ORDER BY letter
            """).fetchall()
            
            # Get database size
            db_size = os.path.getsize(self.db_path) / (1024*1024)  # MB
            
            return {
                "ticker_count": ticker_count,
                "record_count": record_count,
                "date_range": {
                    "start": date_range[0].isoformat() if date_range[0] else None,
                    "end": date_range[1].isoformat() if date_range[1] else None
                },
                "ticker_breakdown": [
                    {"letter": t[0], "ticker_count": t[1], "record_count": t[2]}
                    for t in ticker_breakdown
                ],
                "db_size_mb": round(db_size, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}
        finally:
            if 'con' in locals():
                con.close()


# 4. Orchestration Layer
class ETLOrchestrator:
    """Orchestrates the ETL pipeline"""
    
    def __init__(self, config: ETLConfig = None):
        # Use default config if none provided
        self.config = config or ETLConfig()
        
        # Configure logging
        self.setup_logging()
        self.logger = logging.getLogger('ETLOrchestrator')
        
        # Initialize components
        self.extractor = StockExtractor(self.config)
        self.transformer = StockTransformer(self.config)
        self.loader = DuckDBLoader(self.config)
        
        # Pipeline statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_tickers": 0,
            "successful_extractions": 0,
            "successful_transformations": 0,
            "successful_loads": 0,
            "failed_tickers": [],
            "processing_time": 0
        }
    
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('etl_pipeline.log')
            ]
        )
    
    def _create_progress_bar(self, total, desc):
        """Create a consistent progress bar"""
        return tqdm(total=total, desc=desc, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

    def _log_batch_summary(self, letter, batch_results):
        """Log a summary of batch processing results"""
        total_processed = sum(count for status, count in batch_results.items())
        success_count = batch_results.get('success', 0)
        
        # Only log the summary, not individual batches
        self.logger.info(f"Letter {letter}: {success_count}/{total_processed} tickers processed successfully")
        
        # Log error categories if any failures
        if total_processed > success_count:
            error_count = total_processed - success_count
            self.logger.info(f"  Failed tickers: {error_count} (see detailed report in etl_report.json)")

    async def process_ticker_batch(self, tickers: List[str]) -> Tuple[int, List[str]]:
        """Process a batch of tickers through the ETL pipeline"""
        # Process silently without individual logs
        raw_data = await self.extractor.batch_fetch_stock_data(tickers)
        transformed_data = self.transformer.batch_transform(raw_data)
        self.loader.batch_load_data(transformed_data)
        
        # Calculate failed tickers
        successful_tickers = set(transformed_data.keys())
        failed_tickers = [t for t in tickers if t not in successful_tickers]
        
        return len(successful_tickers), failed_tickers
    
    async def run_pipeline(self, tickers: List[str] = None):
        """Run the complete ETL pipeline"""
        self.stats["start_time"] = datetime.now()
        
        try:
            # Initialize database
            self.loader.init_db_schema()
            
            # Fetch tickers if not provided
            if tickers is None:
                self.logger.info("Fetching tickers")
                tickers = await self.extractor.fetch_tickers()
            
            self.stats["total_tickers"] = len(tickers)
            
            if not tickers:
                self.logger.error("No tickers to process")
                return
            
            # Group tickers by first letter
            ticker_groups = {}
            for ticker in tickers:
                first_letter = ticker[0].upper()
                if first_letter not in ticker_groups:
                    ticker_groups[first_letter] = []
                ticker_groups[first_letter].append(ticker)
            
            # Create progress bar for letter groups
            letter_pbar = self._create_progress_bar(len(ticker_groups), "Processing letter groups")
            
            # Process each letter group
            all_failed_tickers = []
            
            for letter, letter_tickers in ticker_groups.items():
                letter_pbar.set_description(f"Processing letter {letter}")
                
                # Process in batches
                batch_count = (len(letter_tickers) + self.config.batch_size - 1) // self.config.batch_size
                
                # Create progress bar for batches within this letter
                batch_pbar = self._create_progress_bar(len(letter_tickers), f"Letter {letter}")
                
                # Track results for this letter
                letter_results = {'success': 0, 'failed': 0}
                
                for i in range(0, len(letter_tickers), self.config.batch_size):
                    batch = letter_tickers[i:i + self.config.batch_size]
                    
                    # Process batch
                    successful, failed = await self.process_ticker_batch(batch)
                    
                    # Update stats
                    self.stats["successful_loads"] += successful
                    all_failed_tickers.extend(failed)
                    
                    # Update letter results
                    letter_results['success'] += successful
                    letter_results['failed'] += len(failed)
                    
                    # Update progress bar
                    batch_pbar.update(len(batch))
                    
                    # Pause between batches
                    if i + self.config.batch_size < len(letter_tickers):
                        await asyncio.sleep(1)
                
                # Close batch progress bar
                batch_pbar.close()
                
                # Log letter summary
                self._log_batch_summary(letter, letter_results)
                
                # Update letter progress bar
                letter_pbar.update(1)
                
                # Pause between letter groups
                if letter != list(ticker_groups.keys())[-1]:
                    await asyncio.sleep(2)
            
            # Close letter progress bar
            letter_pbar.close()
            
            # Update stats
            self.stats["failed_tickers"] = all_failed_tickers
            
            # Optimize database
            self.logger.info("Optimizing database...")
            self.loader.update_indexes()
            self.loader.optimize_storage()
            
            # Final summary
            self.logger.info(f"ETL pipeline summary:")
            self.logger.info(f"  Processed: {self.stats['total_tickers']} tickers")
            self.logger.info(f"  Successful: {self.stats['successful_loads']} tickers")
            self.logger.info(f"  Failed: {len(self.stats['failed_tickers'])} tickers")
            
        except Exception as e:
            self.logger.error(f"Error in ETL pipeline: {str(e)}")
        finally:
            # Close extractor session
            await self.extractor.close_session()
            
            # Update final stats
            self.stats["end_time"] = datetime.now()
            self.stats["processing_time"] = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            
            self.logger.info(f"ETL pipeline completed in {self.stats['processing_time']:.2f} seconds")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a report of the ETL process"""
        report = {
            "pipeline_stats": {
                "start_time": self.stats["start_time"].isoformat() if self.stats["start_time"] else None,
                "end_time": self.stats["end_time"].isoformat() if self.stats["end_time"] else None,
                "processing_time_seconds": self.stats["processing_time"],
                "total_tickers": self.stats["total_tickers"],
                "successful_loads": self.stats["successful_loads"],
                "failed_tickers_count": len(self.stats["failed_tickers"]),
                "success_rate": (
                    self.stats["successful_loads"] / self.stats["total_tickers"] 
                    if self.stats["total_tickers"] > 0 else 0
                )
            },
            "database_stats": self.loader.get_db_stats(),
            "config": self.config.to_dict()
        }
        
        # Save report to file
        with open('etl_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def handle_failures(self, max_retries: int = 1):
        """Handle failed tickers with retries"""
        failed_tickers = self.stats["failed_tickers"]
        if not failed_tickers:
            self.logger.info("No failed tickers to retry")
            return
        
        self.logger.info(f"Retrying {len(failed_tickers)} failed tickers")
        
        # Create new event loop for retries
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for retry in range(max_retries):
                self.logger.info(f"Retry attempt {retry + 1}/{max_retries}")
                
                # Run pipeline for failed tickers
                loop.run_until_complete(self.run_pipeline(failed_tickers))
                
                # Update failed tickers list
                failed_tickers = self.stats["failed_tickers"]
                
                if not failed_tickers:
                    self.logger.info("All retries successful")
                    break
                
                self.logger.info(f"Still have {len(failed_tickers)} failed tickers after retry {retry + 1}")
                
                # Pause between retries
                if retry < max_retries - 1:
                    time.sleep(10)
        finally:
            loop.close()


# Main execution
if __name__ == "__main__":
    # Create configuration
    config = ETLConfig(
        db_path='stocks_etl.db',
        create_new_db=True,
        max_concurrent_requests=10,
        batch_size=50,
        log_level=logging.INFO
    )
    
    # Create orchestrator
    orchestrator = ETLOrchestrator(config)
    
    try:
        # Create event loop
        loop = asyncio.get_event_loop()
        
        # Run pipeline
        loop.run_until_complete(orchestrator.run_pipeline())
        
        # Generate report
        report = orchestrator.generate_report()
        
        # Handle failures
        orchestrator.handle_failures()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        # Close orchestrator
        orchestrator.logger.info("ETL orchestrator completed") 