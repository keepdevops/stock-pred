import pandas as pd
import requests
from pathlib import Path
import logging
from datetime import datetime

class NasdaqLoader:
    def __init__(self, data_dir=None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir) if data_dir else Path('data')
        self.data_dir.mkdir(exist_ok=True)

    def update_nasdaq_data(self):
        """Download latest NASDAQ screener data."""
        try:
            # Use NASDAQ screener API
            url = "https://api.nasdaq.com/api/screener/stocks"
            headers = {
                'User-Agent': 'Mozilla/5.0'
            }
            
            self.logger.info("Downloading NASDAQ data...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Save raw data
            timestamp = int(datetime.now().timestamp())
            filename = self.data_dir / f'nasdaq_screener_{timestamp}.csv'
            
            # Convert to DataFrame and save
            df = pd.DataFrame(response.json()['data']['rows'])
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Saved NASDAQ data to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error updating NASDAQ data: {e}")
            raise

    def get_latest_file(self):
        """Get the most recent NASDAQ screener file."""
        try:
            files = list(self.data_dir.glob('nasdaq_screener_*.csv'))
            if not files:
                return None
            return max(files, key=lambda x: x.stat().st_mtime)
        except Exception as e:
            self.logger.error(f"Error getting latest file: {e}")
            return None

    def get_nasdaq_symbols(self):
        """Get NASDAQ symbols from the most recent file."""
        try:
            latest_file = self.get_latest_file()
            if not latest_file:
                self.logger.warning("No NASDAQ data file found")
                return pd.DataFrame()
                
            df = pd.read_csv(latest_file)
            return self._clean_data(df)
            
        except Exception as e:
            self.logger.error(f"Error getting NASDAQ symbols: {e}")
            return pd.DataFrame()

    def _clean_data(self, df):
        """Clean the NASDAQ data."""
        try:
            # Ensure required columns exist
            required_cols = ['Symbol', 'Name', 'Market Cap', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Missing required columns in NASDAQ data")

            # Clean market cap
            df['Market Cap'] = df['Market Cap'].replace('[\$,]', '', regex=True)
            df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')

            # Clean volume
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

            # Fill missing values
            df = df.fillna({
                'Market Cap': 0,
                'Volume': 0,
                'Name': '',
                'Country': '',
                'IPO Year': 0,
                'Sector': '',
                'Industry': ''
            })

            return df

        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            raise

    def get_symbol_info(self, symbol):
        """Get information for a specific symbol."""
        try:
            df = self.get_nasdaq_symbols()
            if df.empty:
                return None
                
            symbol_info = df[df['Symbol'] == symbol].to_dict('records')
            return symbol_info[0] if symbol_info else None
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            raise 