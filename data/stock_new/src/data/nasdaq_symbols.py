import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def load_nasdaq_symbols():
    """Load symbols from NASDAQ screener CSV."""
    try:
        # Look for CSV file in current directory
        csv_path = 'nasdaq_screener_1742359747180.csv'
        
        logger.info(f"Looking for CSV file at: {csv_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Verify file exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at: {csv_path}")
            return []
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"CSV columns found: {df.columns.tolist()}")
        
        # Extract and clean symbols
        symbols = sorted(
            df['Symbol']
            .dropna()
            .apply(lambda x: str(x).strip().upper())
            .unique()
        )
        
        logger.info(f"Loaded {len(symbols)} symbols")
        return symbols
        
    except Exception as e:
        logger.error(f"Error loading NASDAQ symbols: {e}")
        return []

# Load symbols when module is imported
NASDAQ_SYMBOLS = load_nasdaq_symbols() 