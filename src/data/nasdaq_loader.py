import pandas as pd
import yfinance as yf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_nasdaq_data():
    """Create and save NASDAQ ticker data"""
    # Define the exact path where the application looks for the file
    data_path = Path('/Users/porupine/Documents/GitHub/stock-pred/data/stock_new/data')
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Core NASDAQ tickers with categories
    tickers_data = {
        'Technology': [
            ('AAPL', 'Apple Inc.'), ('MSFT', 'Microsoft Corporation'),
            ('GOOGL', 'Alphabet Inc.'), ('NVDA', 'NVIDIA Corporation'),
            ('AMD', 'Advanced Micro Devices'), ('INTC', 'Intel Corporation'),
            ('META', 'Meta Platforms Inc.'), ('CSCO', 'Cisco Systems Inc.')
        ],
        'Finance': [
            ('JPM', 'JPMorgan Chase'), ('BAC', 'Bank of America'),
            ('GS', 'Goldman Sachs'), ('MS', 'Morgan Stanley'),
            ('V', 'Visa Inc.'), ('MA', 'Mastercard Inc.')
        ],
        'Consumer': [
            ('AMZN', 'Amazon.com Inc.'), ('TSLA', 'Tesla Inc.'),
            ('WMT', 'Walmart Inc.'), ('HD', 'Home Depot Inc.'),
            ('NKE', 'Nike Inc.'), ('SBUX', 'Starbucks Corporation')
        ],
        'Healthcare': [
            ('JNJ', 'Johnson & Johnson'), ('PFE', 'Pfizer Inc.'),
            ('UNH', 'UnitedHealth Group'), ('ABBV', 'AbbVie Inc.'),
            ('MRK', 'Merck & Co.'), ('BMY', 'Bristol-Myers Squibb')
        ],
        'Industrial': [
            ('BA', 'Boeing Company'), ('CAT', 'Caterpillar Inc.'),
            ('GE', 'General Electric'), ('MMM', '3M Company'),
            ('HON', 'Honeywell'), ('UPS', 'United Parcel Service')
        ]
    }
    
    # Create DataFrame structure
    data = []
    for sector, companies in tickers_data.items():
        for symbol, name in companies:
            data.append({
                'Symbol': symbol,
                'Security Name': name,
                'Sector': sector,
                'Exchange': 'NASDAQ',
                'Market Cap': 0,  # Will be updated with real data
                'Category': sector
            })
    
    df = pd.DataFrame(data)
    
    # Update market caps using yfinance
    for symbol in df['Symbol']:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            market_cap = info.get('marketCap', 0)
            df.loc[df['Symbol'] == symbol, 'Market Cap'] = market_cap
            logger.info(f"Updated market cap for {symbol}")
        except Exception as e:
            logger.warning(f"Could not get market cap for {symbol}: {e}")
    
    # Save to both required filenames
    files_to_save = [
        'nasdaq_screener.csv',
        'nasdaq_screener_1742967072.csv'
    ]
    
    for file in files_to_save:
        file_path = data_path / file
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} tickers to {file_path}")
    
    return df

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = create_nasdaq_data()
    print(f"\nCreated NASDAQ data with {len(df)} tickers")
    print("\nSector distribution:")
    print(df['Sector'].value_counts()) 