import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_nasdaq_data():
    """Verify NASDAQ data files exist and are valid"""
    data_path = Path('/Users/porupine/Documents/GitHub/stock-pred/data/stock_new/data')
    
    files_to_check = [
        'nasdaq_screener.csv',
        'nasdaq_screener_1742967072.csv'
    ]
    
    for file in files_to_check:
        file_path = data_path / file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                logger.info(f"\nVerified {file}:")
                logger.info(f"Rows: {len(df)}")
                logger.info(f"Columns: {df.columns.tolist()}")
                logger.info("\nFirst few rows:")
                logger.info(df.head())
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
        else:
            logger.error(f"File not found: {file}")

if __name__ == '__main__':
    verify_nasdaq_data() 