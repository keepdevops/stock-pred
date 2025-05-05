import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_stock_data():
    """Initialize stock data with guaranteed tickers"""
    # Define the exact path where your application is looking
    data_path = Path('/Users/porupine/Documents/GitHub/stock-pred/data/stock_new/data')
    data_path.mkdir(parents=True, exist_ok=True)

    # Define core ticker data
    stock_data = {
        'Symbol': [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'CRM', 'ADBE',
            # Finance
            'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'WFC', 'BLK', 'SCHW', 'AXP',
            # Consumer
            'AMZN', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'PG', 'KO',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
            # Industrial
            'BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'LMT', 'RTX', 'DE', 'FDX'
        ],
        'Security Name': [
            # Technology
            'Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'NVIDIA Corporation',
            'Meta Platforms Inc.', 'Tesla Inc.', 'Advanced Micro Devices', 'Intel Corporation',
            'Salesforce Inc.', 'Adobe Inc.',
            # Finance
            'JPMorgan Chase & Co.', 'Bank of America Corp.', 'Goldman Sachs Group',
            'Morgan Stanley', 'Visa Inc.', 'Mastercard Inc.', 'Wells Fargo & Co.',
            'BlackRock Inc.', 'Charles Schwab Corp.', 'American Express Co.',
            # Consumer
            'Amazon.com Inc.', 'Walmart Inc.', 'Home Depot Inc.', 'McDonald\'s Corporation',
            'Nike Inc.', 'Starbucks Corporation', 'Target Corporation', 'Costco Wholesale',
            'Procter & Gamble', 'Coca-Cola Company',
            # Healthcare
            'Johnson & Johnson', 'Pfizer Inc.', 'UnitedHealth Group', 'AbbVie Inc.',
            'Merck & Co.', 'Eli Lilly and Company', 'Thermo Fisher Scientific',
            'Abbott Laboratories', 'Bristol-Myers Squibb', 'Amgen Inc.',
            # Industrial
            'Boeing Company', 'Caterpillar Inc.', 'General Electric', 'Honeywell International',
            '3M Company', 'United Parcel Service', 'Lockheed Martin', 'Raytheon Technologies',
            'Deere & Company', 'FedEx Corporation'
        ],
        'Sector': [
            # Technology
            *['Technology']*10,
            # Finance
            *['Finance']*10,
            # Consumer
            *['Consumer']*10,
            # Healthcare
            *['Healthcare']*10,
            # Industrial
            *['Industrial']*10
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(stock_data)
    
    # Add additional columns
    df['Exchange'] = 'NASDAQ'
    df['Market Cap'] = 0
    df['Category'] = df['Sector']  # Category matches Sector

    # Save to both required filenames
    filenames = ['nasdaq_screener.csv', 'nasdaq_screener_1742967072.csv']
    for filename in filenames:
        file_path = data_path / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} tickers to {file_path}")

        # Verify file was created and is readable
        try:
            test_df = pd.read_csv(file_path)
            logger.info(f"Verified {filename} is readable with {len(test_df)} rows")
        except Exception as e:
            logger.error(f"Error verifying {filename}: {e}")

    return df

if __name__ == '__main__':
    df = initialize_stock_data()
    print("\nStock data initialized successfully!")
    print(f"Total tickers: {len(df)}")
    print("\nDistribution by sector:")
    print(df['Sector'].value_counts()) 