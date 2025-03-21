import logging
from datetime import datetime, timedelta
from src.data.ticker_manager import TickerManager

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    setup_logging()
    logging.info("Starting test")
    
    # Create manager
    manager = TickerManager()
    
    # Set test parameters
    ticker = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Test data fetching
    logging.info(f"Fetching data for {ticker}")
    data = manager.get_historical_data(ticker, start_date, end_date)
    
    if data is not None:
        logging.info(f"Data shape: {data.shape}")
        logging.info("\nFirst few rows:")
        print(data.head())
        
        # Test indicators
        indicators = manager.calculate_indicators(data)
        logging.info(f"\nIndicators: {indicators}")
        
        # Test info
        info = manager.get_ticker_info(ticker)
        logging.info(f"\nTicker info: {info}")
        
        # Test cache
        logging.info("\nTesting cache...")
        cached_data = manager.get_historical_data(ticker, start_date, end_date)
        logging.info("Cache test complete")
    else:
        logging.error("Failed to fetch data")

if __name__ == "__main__":
    main() 