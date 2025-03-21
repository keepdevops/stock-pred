import ccxt
import requests
from forex_python.converter import CurrencyRates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import yfinance as yf
import sqlite3
from src.data.ticker_manager import TickerManager
import json
import os

class MarketDataManager:
    def __init__(self):
        self.sectors = [
            'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
            'Industrials', 'Consumer Defensive', 'Energy', 'Basic Materials',
            'Communication Services', 'Utilities', 'Real Estate'
        ]
        
        self.major_tickers = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'TSM', 'AVGO', 'ORCL', 'CRM', 'ADBE'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'DHR', 'ABBV', 'BMY', 'LLY'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'F', 'GM'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'ATVI', 'EA'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'KMI'],
            'Industrials': ['BA', 'HON', 'UPS', 'CAT', 'DE', 'LMT', 'GE', 'MMM', 'RTX', 'UNP'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'EL', 'CL', 'KMB'],
            'Basic Materials': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'SCCO'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'PCG', 'XEL', 'ED', 'ETR'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'WELL', 'AVB', 'EQR', 'DLR']
        }
        
        self.crypto_tickers = {
            'Major Cryptocurrencies': [
                'BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'XRP-USD',
                'ADA-USD', 'DOGE-USD', 'SOL-USD', 'DOT-USD', 'MATIC-USD',
                'SHIB-USD', 'TRX-USD', 'LTC-USD', 'UNI-USD', 'LINK-USD'
            ]
        }
        
        self.index_tickers = {
            'Major Indices': [
                '^GSPC',    # S&P 500
                '^DJI',     # Dow Jones Industrial Average
                '^IXIC',    # NASDAQ Composite
                '^RUT',     # Russell 2000
                '^VIX',     # Volatility Index
                '^FTSE',    # FTSE 100
                '^N225',    # Nikkei 225
                '^HSI',     # Hang Seng Index
                '^GDAXI',   # DAX
                '^FCHI'     # CAC 40
            ]
        }
        
        self.commodity_tickers = {
            'Precious Metals': ['GC=F', 'SI=F', 'PL=F', 'PA=F'],
            'Energy': ['CL=F', 'NG=F', 'HO=F', 'RB=F'],
            'Agriculture': ['ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CT=F', 'CC=F']
        }
        
        self.etf_tickers = {
            'Major ETFs': [
                'SPY',  # S&P 500 ETF
                'QQQ',  # NASDAQ 100 ETF
                'IWM',  # Russell 2000 ETF
                'DIA',  # Dow Jones ETF
                'VTI',  # Total Stock Market ETF
                'VOO',  # Vanguard S&P 500 ETF
                'VEA',  # Developed Markets ETF
                'VWO',  # Emerging Markets ETF
                'AGG',  # Total Bond Market ETF
                'BND',  # Vanguard Total Bond Market ETF
                'GLD',  # Gold ETF
                'SLV',  # Silver ETF
                'VNQ',  # Real Estate ETF
                'XLE',  # Energy ETF
                'XLF',  # Financial ETF
                'XLK',  # Technology ETF
                'XLV',  # Healthcare ETF
                'XLI',  # Industrial ETF
                'XLP',  # Consumer Staples ETF
                'XLY'   # Consumer Discretionary ETF
            ]
        }

    def get_tickers(self, include_stocks=True, include_crypto=True, 
                    include_indices=True, include_commodities=True, 
                    include_etfs=True, sectors=None):
        """
        Get list of tickers based on selected markets and sectors.
        
        Args:
            include_stocks (bool): Include stock tickers
            include_crypto (bool): Include cryptocurrency tickers
            include_indices (bool): Include market indices
            include_commodities (bool): Include commodity futures
            include_etfs (bool): Include ETF tickers
            sectors (list): List of sectors to include (None for all)
        """
        tickers = []
        
        try:
            if include_stocks:
                if sectors:
                    # Add tickers from specified sectors
                    for sector in sectors:
                        if sector in self.major_tickers:
                            tickers.extend(self.major_tickers[sector])
                else:
                    # Add tickers from all sectors
                    for sector_tickers in self.major_tickers.values():
                        tickers.extend(sector_tickers)
            
            if include_crypto:
                tickers.extend(self.crypto_tickers['Major Cryptocurrencies'])
            
            if include_indices:
                tickers.extend(self.index_tickers['Major Indices'])
            
            if include_commodities:
                for commodity_type in self.commodity_tickers.values():
                    tickers.extend(commodity_type)
            
            if include_etfs:
                tickers.extend(self.etf_tickers['Major ETFs'])
            
            # Remove duplicates while preserving order
            return list(dict.fromkeys(tickers))
            
        except Exception as e:
            logging.error(f"Error getting tickers: {str(e)}")
            # Return a minimal set of reliable tickers as fallback
            return ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD', '^GSPC', 'GLD']

    def validate_ticker(self, ticker):
        """Validate a single ticker."""
        try:
            # Get ticker info with a timeout
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            # Check if we got valid info
            is_valid = bool(info and 'regularMarketPrice' in info)
            
            if is_valid:
                logging.info(f"Validated ticker: {ticker}")
            else:
                logging.warning(f"Invalid ticker: {ticker}")
            
            return is_valid
            
        except Exception as e:
            logging.error(f"Failed to validate ticker '{ticker}': {str(e)}")
            return False

    def get_sector_tickers(self, sector):
        """Get tickers for a specific sector."""
        return self.major_tickers.get(sector, [])

    def get_available_sectors(self):
        """Get list of available sectors."""
        return list(self.major_tickers.keys())

    def _get_exchange_tickers(self):
        """Get all tickers from NASDAQ and NYSE."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            all_tickers = set()
            
            # Source 1: NASDAQ FTP
            try:
                nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"
                df = pd.read_csv(nasdaq_url, sep='|')
                if 'Symbol' in df.columns:
                    nasdaq_tickers = df['Symbol'].tolist()
                    all_tickers.update([t for t in nasdaq_tickers if isinstance(t, str)])
            except Exception as e:
                logging.error(f"Error fetching from NASDAQ FTP: {str(e)}")

            # Source 2: NYSE API
            try:
                nyse_url = "https://www.nyse.com/api/quotes/filter"
                payload = {
                    "instrumentType": "EQUITY",
                    "pageNumber": 1,
                    "sortColumn": "SYMBOL",
                    "sortOrder": "ASC",
                    "maxResultsPerPage": 10000,
                    "filterToken": ""
                }
                response = requests.post(nyse_url, json=payload, headers=headers)
                if response.status_code == 200:
                    nyse_data = response.json()
                    nyse_tickers = [item['symbolTicker'] for item in nyse_data if 'symbolTicker' in item]
                    all_tickers.update(nyse_tickers)
            except Exception as e:
                logging.error(f"Error fetching from NYSE API: {str(e)}")

            # Source 3: Alternative source (Finnhub)
            try:
                finnhub_token = 'YOUR_FINNHUB_TOKEN'  # Get from finnhub.io
                exchanges = ['US']
                for exchange in exchanges:
                    url = f"https://finnhub.io/api/v1/stock/symbol?exchange={exchange}&token={finnhub_token}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        finnhub_tickers = [item['symbol'] for item in data if 'symbol' in item]
                        all_tickers.update(finnhub_tickers)
            except Exception as e:
                logging.error(f"Error fetching from Finnhub: {str(e)}")

            # Source 4: Local backup file
            try:
                backup_file = "data/tickers_backup.json"
                if os.path.exists(backup_file):
                    with open(backup_file, 'r') as f:
                        backup_tickers = json.load(f)
                    all_tickers.update(backup_tickers)
            except Exception as e:
                logging.error(f"Error loading backup tickers: {str(e)}")

            # Filter out invalid tickers
            filtered_tickers = []
            for ticker in all_tickers:
                if isinstance(ticker, str) and ticker.strip():
                    # Basic validation
                    if (ticker.isalnum() or '-' in ticker or '.' in ticker) and \
                       len(ticker) <= 5 and ticker not in ['', 'nan', 'None']:
                        filtered_tickers.append(ticker.strip())

            # Save to backup file
            try:
                os.makedirs('data', exist_ok=True)
                with open("data/tickers_backup.json", 'w') as f:
                    json.dump(list(filtered_tickers), f)
            except Exception as e:
                logging.error(f"Error saving backup tickers: {str(e)}")

            return filtered_tickers

        except Exception as e:
            logging.error(f"Error in _get_exchange_tickers: {str(e)}")
            return []

    def get_all_tickers(self, include_exchange=True, include_categorized=True):
        """
        Get all available tickers, including both categorized and exchange-listed.
        
        Args:
            include_exchange (bool): Include all exchange-listed tickers
            include_categorized (bool): Include categorized tickers (sectors, crypto, etc.)
        """
        all_tickers = set()
        
        try:
            # Get exchange-listed tickers
            if include_exchange:
                exchange_tickers = self._get_exchange_tickers()
                all_tickers.update(exchange_tickers)
                logging.info(f"Found {len(exchange_tickers)} exchange-listed tickers")
            
            # Get categorized tickers
            if include_categorized:
                # Stocks by sector
                for sector_tickers in self.major_tickers.values():
                    all_tickers.update(sector_tickers)
                
                # Crypto
                all_tickers.update(self.crypto_tickers['Major Cryptocurrencies'])
                
                # Indices
                all_tickers.update(self.index_tickers['Major Indices'])
                
                # Commodities
                for commodity_tickers in self.commodity_tickers.values():
                    all_tickers.update(commodity_tickers)
                
                # ETFs
                all_tickers.update(self.etf_tickers['Major ETFs'])
            
            # Convert to list and sort
            all_tickers_list = sorted(list(all_tickers))
            logging.info(f"Total tickers found: {len(all_tickers_list)}")
            
            return all_tickers_list
            
        except Exception as e:
            logging.error(f"Error getting all tickers: {str(e)}")
            return self.get_tickers()  # Return basic tickers as fallback

    def validate_tickers(self, tickers):
        """Validate tickers using yfinance."""
        valid_tickers = []
        for ticker in tickers:
            try:
                if '/' in ticker:  # Crypto pair
                    if self._validate_crypto_pair(ticker):
                        valid_tickers.append(ticker)
                else:  # Stock/Index/Commodity
                    info = yf.Ticker(ticker).info
                    if info and 'regularMarketPrice' in info:
                        valid_tickers.append(ticker)
            except:
                continue
        return valid_tickers

    def _validate_crypto_pair(self, pair):
        """Validate cryptocurrency pair."""
        try:
            base, quote = pair.split('/')
            for exchange in self.crypto_exchanges.values():
                if exchange.has['fetchTicker']:
                    ticker = exchange.fetch_ticker(pair)
                    if ticker and ticker['last']:
                        return True
            return False
        except:
            return False

    def get_crypto_data(self, symbol: str, timeframe: str = '1d', 
                       limit: int = 1000) -> dict:
        """
        Get cryptocurrency data from multiple exchanges.
        
        Args:
            symbol: Cryptocurrency pair (e.g., 'BTC/USD')
            timeframe: Time interval ('1m', '5m', '1h', '1d', etc.)
            limit: Number of candles to fetch
        """
        crypto_data = {}
        
        for exchange_name, exchange in self.crypto_exchanges.items():
            try:
                if exchange.has['fetchOHLCV']:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 
                                                     'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Add exchange-specific metrics
                    if exchange.has['fetchTicker']:
                        ticker = exchange.fetch_ticker(symbol)
                        df['bid'] = ticker['bid']
                        df['ask'] = ticker['ask']
                        df['spread'] = ticker['ask'] - ticker['bid']
                    
                    crypto_data[exchange_name] = df.to_dict('records')
                    
            except Exception as e:
                logging.error(f"Error fetching {symbol} from {exchange_name}: {str(e)}")
                continue
        
        return crypto_data

    def get_forex_data(self, base_currency: str, quote_currency: str, 
                      days: int = 365) -> dict:
        """Get foreign exchange rates data."""
        try:
            forex_data = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            current_date = start_date
            while current_date <= end_date:
                try:
                    rate = self.currency_rates.get_rate(base_currency, 
                                                      quote_currency, 
                                                      current_date)
                    forex_data.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'rate': rate
                    })
                except:
                    pass
                current_date += timedelta(days=1)
            
            return {'forex_rates': forex_data}
            
        except Exception as e:
            logging.error(f"Error fetching forex data: {str(e)}")
            return {}

    def get_commodity_data(self) -> dict:
        """Get commodity prices and data."""
        commodities = {
            'GC=F': 'Gold',
            'SI=F': 'Silver',
            'CL=F': 'Crude Oil',
            'NG=F': 'Natural Gas',
            'ZC=F': 'Corn',
            'ZS=F': 'Soybeans',
            'KC=F': 'Coffee',
            'CT=F': 'Cotton'
        }
        
        commodity_data = {}
        for symbol, name in commodities.items():
            try:
                commodity = yf.Ticker(symbol)
                hist = commodity.history(period='1y')
                commodity_data[name] = hist.to_dict('records')
            except Exception as e:
                logging.error(f"Error fetching {name} data: {str(e)}")
                continue
        
        return commodity_data

    def get_economic_indicators(self) -> dict:
        """Get economic indicators and market indices."""
        try:
            # FRED API for economic data
            fred_api_key = 'YOUR_FRED_API_KEY'  # Get from FRED website
            indicators = {
                'GDP': 'GDP',
                'UNRATE': 'Unemployment Rate',
                'CPIAUCSL': 'Consumer Price Index',
                'DFF': 'Federal Funds Rate',
                'T10Y2Y': 'Treasury Yield Spread',
                'VIXCLS': 'VIX Volatility Index'
            }
            
            economic_data = {}
            for symbol, name in indicators.items():
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': symbol,
                    'api_key': fred_api_key,
                    'file_type': 'json',
                    'frequency': 'm',  # monthly
                    'observation_start': '2020-01-01'
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    economic_data[name] = data['observations']
            
            return economic_data
            
        except Exception as e:
            logging.error(f"Error fetching economic indicators: {str(e)}")
            return {}

    def get_market_sentiment(self) -> dict:
        """Get market sentiment indicators."""
        try:
            sentiment_data = {}
            
            # Fear & Greed Index
            fear_greed_url = "https://api.alternative.me/fng/"
            response = requests.get(fear_greed_url)
            if response.status_code == 200:
                sentiment_data['fear_greed'] = response.json()
            
            # Get VIX data
            vix = yf.Ticker('^VIX')
            sentiment_data['vix'] = vix.history(period='1y').to_dict('records')
            
            # Get market breadth indicators
            spx = yf.Ticker('^SPX')
            sentiment_data['advance_decline'] = spx.info.get('advancingVolume', 0) / \
                                              spx.info.get('declineVolume', 1)
            
            return sentiment_data
            
        except Exception as e:
            logging.error(f"Error fetching market sentiment: {str(e)}")
            return {}

class EnhancedTickerManager(TickerManager):
    def __init__(self):
        super().__init__()
        self.market_data = MarketDataManager()
    
    def fetch_all_market_data(self, include_crypto: bool = True, 
                            include_forex: bool = True,
                            include_commodities: bool = True) -> dict:
        """Fetch all available market data."""
        all_data = {}
        
        # Get stock data
        stock_data = self.fetch_all_data(self.get_tickers())
        all_data['stocks'] = stock_data
        
        # Get cryptocurrency data
        if include_crypto:
            crypto_pairs = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'DOGE/USD', 
                          'ADA/USD', 'DOT/USD', 'UNI/USD']
            crypto_data = {}
            for pair in crypto_pairs:
                crypto_data[pair] = self.market_data.get_crypto_data(pair)
            all_data['crypto'] = crypto_data
        
        # Get forex data
        if include_forex:
            currency_pairs = [
                ('EUR', 'USD'), ('GBP', 'USD'), ('JPY', 'USD'),
                ('AUD', 'USD'), ('CAD', 'USD'), ('CHF', 'USD')
            ]
            forex_data = {}
            for base, quote in currency_pairs:
                forex_data[f"{base}/{quote}"] = self.market_data.get_forex_data(base, quote)
            all_data['forex'] = forex_data
        
        # Get commodity data
        if include_commodities:
            all_data['commodities'] = self.market_data.get_commodity_data()
        
        # Get economic indicators
        all_data['economic'] = self.market_data.get_economic_indicators()
        
        # Get market sentiment
        all_data['sentiment'] = self.market_data.get_market_sentiment()
        
        return all_data

    def save_market_data(self, data: dict, db_path: str) -> None:
        """Save all market data to SQLite database."""
        conn = sqlite3.connect(db_path)
        
        # Save stock data
        if 'stocks' in data:
            pd.DataFrame(data['stocks']).to_sql('stocks', conn, 
                                              if_exists='replace')
        
        # Save crypto data
        if 'crypto' in data:
            for pair, exchanges in data['crypto'].items():
                for exchange, ohlcv in exchanges.items():
                    table_name = f"crypto_{pair.replace('/', '_')}_{exchange}"
                    pd.DataFrame(ohlcv).to_sql(table_name, conn, 
                                             if_exists='replace')
        
        # Save forex data
        if 'forex' in data:
            for pair, rates in data['forex'].items():
                table_name = f"forex_{pair.replace('/', '_')}"
                pd.DataFrame(rates['forex_rates']).to_sql(table_name, conn, 
                                                        if_exists='replace')
        
        # Save commodity data
        if 'commodities' in data:
            for commodity, prices in data['commodities'].items():
                table_name = f"commodity_{commodity.lower()}"
                pd.DataFrame(prices).to_sql(table_name, conn, 
                                          if_exists='replace')
        
        # Save economic indicators
        if 'economic' in data:
            for indicator, values in data['economic'].items():
                table_name = f"economic_{indicator.lower()}"
                pd.DataFrame(values).to_sql(table_name, conn, 
                                          if_exists='replace')
        
        # Save sentiment data
        if 'sentiment' in data:
            pd.DataFrame(data['sentiment']).to_sql('market_sentiment', conn, 
                                                 if_exists='replace')
        
        conn.close() 