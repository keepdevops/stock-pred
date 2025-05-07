import requests
import time
import pandas as pd
from io import StringIO
import os

class TwelveDataDownloader:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_calls_this_minute = 0
        self.api_calls_today = 0
        self.minute_window_start = time.time()
        self.day_window_start = time.time()

    def download_historical(self, symbol, interval, start_date, end_date):
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "apikey": self.api_key,
            "format": "csv"
        }
        response = requests.get(url, params=params)
        csv_data = response.text
        if csv_data:
            df = pd.read_csv(StringIO(csv_data))
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            return df
        else:
            return None

    def fetch_forex_symbols(self):
        url = f"https://api.twelvedata.com/forex_pairs?apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            if 'symbol' in df.columns:
                return df['symbol'].dropna().astype(str).str.strip().tolist()
        return []

    def fetch_stock_symbols(self):
        url = f"https://api.twelvedata.com/stocks?apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            if 'symbol' in df.columns:
                return df['symbol'].dropna().astype(str).str.strip().tolist()
        return []

    def fetch_crypto_symbols(self):
        url = f"https://api.twelvedata.com/cryptocurrencies?apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            if 'symbol' in df.columns:
                return df['symbol'].dropna().astype(str).str.strip().tolist()
        return []

    def convert_symbol(self, symbol, target_provider, instrument_type='forex'):
        symbol = symbol.strip()
        if target_provider == 'yfinance':
            # Twelve Data: 'EUR/USD' → yfinance: 'EURUSD=X'
            if instrument_type == 'forex':
                if '/' in symbol:
                    return symbol.replace('/', '') + '=X'
                return symbol + '=X'
            elif instrument_type == 'crypto':
                if '/' in symbol:
                    return symbol.replace('/', '-')
                return symbol
            else:
                return symbol
        elif target_provider == 'alphavantage':
            # Twelve Data: 'EUR/USD' → Alpha Vantage: ('EUR', 'USD')
            if instrument_type == 'forex' or instrument_type == 'crypto':
                if '/' in symbol:
                    parts = symbol.split('/')
                    if len(parts) == 2:
                        return (parts[0], parts[1])
                return (symbol, None)
            else:
                return symbol
        elif target_provider == 'tiingo':
            # Twelve Data: 'BTC/USD' → Tiingo: 'btcusd'
            if instrument_type == 'crypto':
                if '/' in symbol:
                    return symbol.replace('/', '').lower()
                return symbol.lower()
            else:
                return symbol
        else:
            return symbol

    def update_ticker_manager(self, ticker_manager):
        # Forex
        forex_symbols = self.fetch_forex_symbols()
        yfinance_forex_symbols = [self.convert_symbol(s, 'yfinance', 'forex') for s in forex_symbols]
        ticker_manager.tickers['yFinance Forex'] = yfinance_forex_symbols
        if 'yFinance Forex' not in ticker_manager.categories:
            ticker_manager.categories.append('yFinance Forex')
            if hasattr(ticker_manager, 'category_combo'):
                ticker_manager.category_combo.addItem('yFinance Forex')
        tiingo_crypto_symbols = [self.convert_symbol(s, 'tiingo', 'crypto') for s in forex_symbols]
        ticker_manager.tickers['Tiingo Crypto'] = tiingo_crypto_symbols
        if 'Tiingo Crypto' not in ticker_manager.categories:
            ticker_manager.categories.append('Tiingo Crypto')
            if hasattr(ticker_manager, 'category_combo'):
                ticker_manager.category_combo.addItem('Tiingo Crypto')
        alphavantage_forex_symbols = [self.convert_symbol(s, 'alphavantage', 'forex') for s in forex_symbols]
        ticker_manager.tickers['AlphaVantage Forex'] = alphavantage_forex_symbols
        if 'AlphaVantage Forex' not in ticker_manager.categories:
            ticker_manager.categories.append('AlphaVantage Forex')
            if hasattr(ticker_manager, 'category_combo'):
                ticker_manager.category_combo.addItem('AlphaVantage Forex')
        # Stocks
        stock_symbols = self.fetch_stock_symbols()
        yfinance_stock_symbols = [self.convert_symbol(s, 'yfinance', 'stock') for s in stock_symbols]
        ticker_manager.tickers['yFinance Stocks'] = yfinance_stock_symbols
        if 'yFinance Stocks' not in ticker_manager.categories:
            ticker_manager.categories.append('yFinance Stocks')
            if hasattr(ticker_manager, 'category_combo'):
                ticker_manager.category_combo.addItem('yFinance Stocks')
        tiingo_stock_symbols = [self.convert_symbol(s, 'tiingo', 'stock') for s in stock_symbols]
        ticker_manager.tickers['Tiingo Stocks'] = tiingo_stock_symbols
        if 'Tiingo Stocks' not in ticker_manager.categories:
            ticker_manager.categories.append('Tiingo Stocks')
            if hasattr(ticker_manager, 'category_combo'):
                ticker_manager.category_combo.addItem('Tiingo Stocks')
        alphavantage_stock_symbols = [self.convert_symbol(s, 'alphavantage', 'stock') for s in stock_symbols]
        ticker_manager.tickers['AlphaVantage Stocks'] = alphavantage_stock_symbols
        if 'AlphaVantage Stocks' not in ticker_manager.categories:
            ticker_manager.categories.append('AlphaVantage Stocks')
            if hasattr(ticker_manager, 'category_combo'):
                ticker_manager.category_combo.addItem('AlphaVantage Stocks')
        # Crypto
        crypto_symbols = self.fetch_crypto_symbols()
        yfinance_crypto_symbols = [self.convert_symbol(s, 'yfinance', 'crypto') for s in crypto_symbols]
        ticker_manager.tickers['yFinance Crypto'] = yfinance_crypto_symbols
        if 'yFinance Crypto' not in ticker_manager.categories:
            ticker_manager.categories.append('yFinance Crypto')
            if hasattr(ticker_manager, 'category_combo'):
                ticker_manager.category_combo.addItem('yFinance Crypto')
        tiingo_crypto_symbols = [self.convert_symbol(s, 'tiingo', 'crypto') for s in crypto_symbols]
        ticker_manager.tickers['Tiingo Crypto'] = tiingo_crypto_symbols
        if 'Tiingo Crypto' not in ticker_manager.categories:
            ticker_manager.categories.append('Tiingo Crypto')
            if hasattr(ticker_manager, 'category_combo'):
                ticker_manager.category_combo.addItem('Tiingo Crypto')
        alphavantage_crypto_symbols = [self.convert_symbol(s, 'alphavantage', 'crypto') for s in crypto_symbols]
        ticker_manager.tickers['AlphaVantage Crypto'] = alphavantage_crypto_symbols
        if 'AlphaVantage Crypto' not in ticker_manager.categories:
            ticker_manager.categories.append('AlphaVantage Crypto')
            if hasattr(ticker_manager, 'category_combo'):
                ticker_manager.category_combo.addItem('AlphaVantage Crypto')

    def batch_download_category_to_cache(self, category, ticker_manager, interval, start_date, end_date, cache_dir=None, verbose=True):
        """
        Batch download all symbols in a category from Twelve Data in CSV format, sequential batches of 8, save to cache.
        Warn if unsupported (e.g., 2 years of 1min data). Basic rate limiting for free tier.
        Args:
            category (str): Category name (e.g., 'Commodities')
            ticker_manager (TickerManager): Instance to get symbols
            interval (str): e.g., '1min', '1day'
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD'
            cache_dir (Path or str): Where to save CSVs (default: ticker_manager.cache_dir)
            verbose (bool): Print progress
        Returns:
            dict: {symbol: DataFrame}
        """
        import time
        from datetime import datetime
        from pathlib import Path
        import pandas as pd
        from io import StringIO

        # --- Free tier limits ---
        MAX_PER_MIN = 8
        BATCH_SIZE = 8
        # 1min interval: warn if > 7 days
        if interval in ["1min", "5min", "15min", "30min", "45min"]:
            dt1 = datetime.strptime(start_date, "%Y-%m-%d")
            dt2 = datetime.strptime(end_date, "%Y-%m-%d")
            days = (dt2 - dt1).days
            if days > 7:
                print(f"WARNING: Twelve Data free tier only allows a few days of 1-minute data. You requested {days} days. Reduce the date range or use a lower granularity.")
        # Get symbols
        symbols = ticker_manager.tickers.get(category, [])
        if not symbols:
            print(f"No symbols found for category '{category}'")
            return {}
        if cache_dir is None:
            cache_dir = getattr(ticker_manager, 'cache_dir', Path('./cache'))
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        all_data = {}
        calls_this_minute = 0
        minute_window_start = time.time()
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i+BATCH_SIZE]
            symbol_str = ",".join(batch)
            # --- Rate limiting ---
            now = time.time()
            if now - minute_window_start >= 60:
                calls_this_minute = 0
                minute_window_start = now
            if calls_this_minute >= MAX_PER_MIN:
                sleep_time = 60 - (now - minute_window_start)
                if sleep_time > 0:
                    if verbose:
                        print(f"Twelve Data per-minute limit reached. Sleeping {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                calls_this_minute = 0
                minute_window_start = time.time()
            # --- Download batch as CSV ---
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": symbol_str,
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date,
                "apikey": self.api_key,
                "format": "csv"
            }
            response = requests.get(url, params=params)
            if response.status_code == 429 or 'quota' in response.text.lower() or 'limit' in response.text.lower():
                # Quota exhausted
                if verbose:
                    print("Twelve Data API quota exhausted. No more data can be downloaded today.")
                # Raise an exception to be caught by the GUI
                raise RuntimeError("Twelve Data API quota exhausted. Please wait for your quota to reset or upgrade your plan.")
            csv_data = response.text
            # Parse CSV: may contain multiple symbols
            try:
                df = pd.read_csv(StringIO(csv_data))
            except Exception as e:
                print(f"Error parsing CSV for batch {batch}: {e}")
                continue
            # If 'symbol' column exists, split by symbol
            if 'symbol' in df.columns:
                for symbol in batch:
                    symbol_df = df[df['symbol'] == symbol].copy()
                    if not symbol_df.empty:
                        symbol_df = symbol_df.set_index('datetime')
                        cache_file = cache_dir / f"{symbol}_{start_date}_{end_date}_{interval}_twelvedata.csv"
                        symbol_df.to_csv(cache_file)
                        all_data[symbol] = symbol_df
                        if verbose:
                            print(f"Saved {symbol} to {cache_file}")
            else:
                # Single symbol
                if not df.empty:
                    df = df.set_index('datetime')
                    symbol = batch[0]
                    cache_file = cache_dir / f"{symbol}_{start_date}_{end_date}_{interval}_twelvedata.csv"
                    df.to_csv(cache_file)
                    all_data[symbol] = df
                    if verbose:
                        print(f"Saved {symbol} to {cache_file}")
            calls_this_minute += 1
            if self.api_calls_this_minute >= 7:
                self.progress.emit(i, "Warning: Approaching per-minute API limit!")
            if self.api_calls_today >= 790:
                self.progress.emit(i, "Warning: Approaching daily API limit!")
        return all_data

    def batch_download_category_to_cache_gui(self):
        if not self.ticker_manager:
            QMessageBox.warning(self, "Warning", "Please apply database settings first.")
            return
        category = self.category_combo.currentText()
        interval = self.interval_combo.currentText()
        start_date = self.start_date.date().toString('yyyy-MM-dd')
        end_date = self.end_date.date().toString('yyyy-MM-dd')
        api_key = self.twelvedata_api_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter your Twelve Data API key in the input field.")
            return

        # Confirm with user if 1min/5min/15min/30min/45min and >7 days
        from datetime import datetime
        if interval in ["1min", "5min", "15min", "30min", "45min"]:
            dt1 = datetime.strptime(start_date, "%Y-%m-%d")
            dt2 = datetime.strptime(end_date, "%Y-%m-%d")
            days = (dt2 - dt1).days
            if days > 7:
                QMessageBox.warning(self, "Warning", f"Twelve Data free tier only allows a few days of 1-minute data. You requested {days} days. Reduce the date range or use a lower granularity.")
                return

        # Create downloader
        from src.data.twelvedata_downloader import TwelveDataDownloader
        downloader = TwelveDataDownloader(api_key)

        # Run batch download (optionally in a thread for GUI responsiveness)
        self.status_label.setText("Batch downloading category to cache...")
        try:
            result = downloader.batch_download_category_to_cache(
                category=category,
                ticker_manager=self.ticker_manager,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                cache_dir=None,
                verbose=False
            )
            QMessageBox.information(self, "Batch Download Complete", f"Downloaded and cached data for {len(result)} symbols in category '{category}'.")
            self.status_label.setText(f"Batch download complete for {category}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch download failed: {e}")
            self.status_label.setText("Batch download failed.")

    def get_usage(self):
        url = f"https://api.twelvedata.com/usage?apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        if 'data' in data:
            return data['data']
        else:
            return None
