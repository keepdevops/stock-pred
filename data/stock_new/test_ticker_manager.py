import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import logging
from datetime import datetime, timedelta
from src.data.ticker_manager import TickerManager

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def test_ticker_manager():
    """Test the TickerManager functionality."""
    setup_logging()
    logging.info("Starting TickerManager test")
    
    # Initialize manager
    manager = TickerManager()
    
    # Test dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Test single ticker
    ticker = "AAPL"
    
    try:
        # Test historical data
        logging.info(f"Testing historical data for {ticker}")
        data = manager.get_historical_data(ticker, start_date, end_date)
        
        if data is not None:
            logging.info(f"Successfully retrieved data with shape: {data.shape}")
            print("\nFirst few rows of data:")
            print(data.head())
            
            # Test indicators
            logging.info("Testing indicator calculation")
            indicators = manager.calculate_indicators(data)
            logging.info(f"Calculated indicators: {indicators}")
            
            # Test info
            logging.info("Testing ticker info")
            info = manager.get_ticker_info(ticker)
            logging.info(f"Retrieved info: {info}")
            
            # Test cache
            logging.info("Testing cache")
            cached_data = manager.get_historical_data(ticker, start_date, end_date)
            logging.info("Successfully retrieved cached data")
            
        else:
            logging.error("No data retrieved")
            
    except Exception as e:
        logging.error(f"Error during test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_ticker_manager() 