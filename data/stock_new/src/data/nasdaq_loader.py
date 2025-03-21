import pandas as pd
import yfinance as yf
import logging
from pathlib import Path
from datetime import datetime
import requests
import time
from typing import Optional, List, Dict
import io

class StockLoader:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)

    def download_all_stocks(self) -> Optional[Path]:
        """Download complete stock listings from NASDAQ, NYSE, and AMEX."""
        try:
            # URLs for different exchanges
            urls = {
                'NASDAQ': "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
                'NYSE': "https://www.nyse.com/api/quotes/filter",
                'OTHER': "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
            }
            
            all_stocks = []
            
            # Download NASDAQ stocks
            self.logger.info("Downloading NASDAQ listings...")
            nasdaq_df = self._download_nasdaq_data(urls['NASDAQ'])
            if nasdaq_df is not None:
                all_stocks.append(nasdaq_df)
                self.logger.info(f"Downloaded {len(nasdaq_df)} NASDAQ symbols")

            # Download NYSE stocks
            self.logger.info("Downloading NYSE listings...")
            nyse_df = self._download_nyse_data(urls['NYSE'])
            if nyse_df is not None:
                all_stocks.append(nyse_df)
                self.logger.info(f"Downloaded {len(nyse_df)} NYSE symbols")

            # Download other listings (AMEX, etc.)
            self.logger.info("Downloading other listings...")
            other_df = self._download_other_data(urls['OTHER'])
            if other_df is not None:
                all_stocks.append(other_df)
                self.logger.info(f"Downloaded {len(other_df)} other symbols")

            # Combine all listings
            if not all_stocks:
                raise ValueError("No stock data downloaded")
                
            combined_df = pd.concat(all_stocks, ignore_index=True)
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['Symbol'])
            
            # Save the data
            timestamp = int(time.time())
            output_file = self.data_dir / f'all_stocks_{timestamp}.csv'
            combined_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Total unique symbols: {len(combined_df)}")
            self.logger.info(f"Data saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error downloading stock data: {e}")
            return self._use_backup_source()

    def _download_nasdaq_data(self, url: str) -> Optional[pd.DataFrame]:
        """Download NASDAQ listings."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text), delimiter='|', dtype=str)
            df = df[df['Test Issue'] == 'N']  # Exclude test issues
            
            processed_df = pd.DataFrame({
                'Symbol': df['Symbol'].astype(str).str.strip(),
                'Name': df['Security Name'].astype(str).str.strip(),
                'Exchange': 'NASDAQ',
                'Market Cap': 0.0,
                'Volume': 0.0,
                'Sector': '',
                'Industry': ''
            })
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error downloading NASDAQ data: {e}")
            return None

    def _download_nyse_data(self, url: str) -> Optional[pd.DataFrame]:
        """Download NYSE listings."""
        try:
            # Use backup source for NYSE data
            url = "https://www.nyse.com/listings_directory/stock"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            processed_df = pd.DataFrame({
                'Symbol': df['Symbol'].astype(str).str.strip(),
                'Name': df['Company Name'].astype(str).str.strip(),
                'Exchange': 'NYSE',
                'Market Cap': 0.0,
                'Volume': 0.0,
                'Sector': '',
                'Industry': ''
            })
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error downloading NYSE data: {e}")
            return self._download_nyse_backup()

    def _download_nyse_backup(self) -> Optional[pd.DataFrame]:
        """Download NYSE listings from backup source."""
        try:
            url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
            df = pd.read_csv(url, delimiter='|', dtype=str)
            df = df[df['Exchange'] == 'N']  # Filter NYSE listings
            
            processed_df = pd.DataFrame({
                'Symbol': df['ACT Symbol'].astype(str).str.strip(),
                'Name': df['Security Name'].astype(str).str.strip(),
                'Exchange': 'NYSE',
                'Market Cap': 0.0,
                'Volume': 0.0,
                'Sector': '',
                'Industry': ''
            })
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error downloading NYSE backup data: {e}")
            return None

    def _download_other_data(self, url: str) -> Optional[pd.DataFrame]:
        """Download other exchange listings."""
        try:
            df = pd.read_csv(url, delimiter='|', dtype=str)
            df = df[df['Test Issue'] == 'N']  # Exclude test issues
            
            processed_df = pd.DataFrame({
                'Symbol': df['ACT Symbol'].astype(str).str.strip(),
                'Name': df['Security Name'].astype(str).str.strip(),
                'Exchange': df['Exchange'].astype(str).str.strip(),
                'Market Cap': 0.0,
                'Volume': 0.0,
                'Sector': '',
                'Industry': ''
            })
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error downloading other exchange data: {e}")
            return None

    def update_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Update market data for all symbols using yfinance."""
        try:
            self.logger.info("Fetching market data for all symbols...")
            symbols = df['Symbol'].tolist()
            
            # Process in batches
            batch_size = 100
            total_batches = len(symbols) // batch_size + 1
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                batch_num = i // batch_size + 1
                self.logger.info(f"Processing batch {batch_num} of {total_batches}")
                
                # Get data for batch
                tickers = yf.Tickers(' '.join(batch))
                
                # Update information
                for symbol in batch:
                    try:
                        if hasattr(tickers.tickers[symbol], 'info'):
                            info = tickers.tickers[symbol].info
                            idx = df.index[df['Symbol'] == symbol][0]
                            
                            df.at[idx, 'Market Cap'] = info.get('marketCap', 0)
                            df.at[idx, 'Volume'] = info.get('volume', 0)
                            df.at[idx, 'Sector'] = info.get('sector', '')
                            df.at[idx, 'Industry'] = info.get('industry', '')
                    except Exception as e:
                        self.logger.warning(f"Could not fetch info for {symbol}: {e}")
                
                time.sleep(1)  # Rate limiting
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return df

    def get_latest_file(self) -> Optional[Path]:
        """Get the most recent stock data file."""
        try:
            files = list(self.data_dir.glob('all_stocks_*.csv'))
            if not files:
                return None
            return max(files, key=lambda x: x.stat().st_mtime)
        except Exception as e:
            self.logger.error(f"Error getting latest file: {e}")
            return None

    def load_symbols(self) -> Optional[pd.DataFrame]:
        """Load symbols from the most recent file."""
        try:
            latest_file = self.get_latest_file()
            if not latest_file:
                self.logger.warning("No stock data file found")
                return None
                
            df = pd.read_csv(latest_file)
            self.logger.info(f"Loaded {len(df)} symbols from {latest_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading symbols: {e}")
            return None 