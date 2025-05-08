import logging
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.download_nasdaq import download_nasdaq_screener

def setup():
    """Initialize the application data directory and download initial data."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create data directory
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        logger.info("Data directory created/verified")
        
        # Download NASDAQ data
        logger.info("Downloading initial NASDAQ data...")
        filename = download_nasdaq_screener()
        if filename:
            logger.info(f"NASDAQ data downloaded successfully to {filename}")
        else:
            logger.error("Failed to download NASDAQ data")
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")

if __name__ == "__main__":
    setup() 