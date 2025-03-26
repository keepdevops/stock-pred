from setuptools import setup, find_packages
import logging
from pathlib import Path
from src.data.download_nasdaq import download_nasdaq_screener

def init_setup():
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
    init_setup()

# Package setup configuration
setup(
    name="stock_market_analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'duckdb',
        'polars',
        # Add other dependencies
    ],
) 