import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class StockDataManager:
    """Manager class for handling stock market data operations."""

    def __init__(self):
        """Initialize the StockDataManager."""
        # Initialize private attributes
        self.__cache = {}
        self.__last_request = time.time()
        self.__request_delay = 0.5
        logging.info("StockDataManager initialized")

    def get_historical_data(self, ticker: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Get historical data for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical data and indicators
        """
        try:
            # Generate cache key
            cache_key = f"{ticker}_{start_date}_{end_date}"
            
            # Check cache
            if cache_key in self.__cache:
                logging.info(f"Returning cached data for {ticker}")
                return self.__cache[cache_key]

            logging.info(f"Fetching new data for {ticker}")
            
            # Apply rate limiting
            current_time = time.time()
            elapsed = current_time - self.__last_request
            if elapsed < self.__request_delay:
                time.sleep(self.__request_delay - elapsed)
            self.__last_request = current_time

            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logging.warning(f"No data found for {ticker}")
                return None
            
            # Calculate technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            # Cache the results
            self.__cache[cache_key] = df
            return df
            
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators from historical data."""
        if df is None or df.empty:
            return {}
        
        try:
            latest = df.iloc[-1]
            earliest = df.iloc[0]
            
            return {
                'last_price': float(latest['Close']),
                'volume': float(latest['Volume']),
                'sma_20': float(latest['SMA_20']) if not np.isnan(latest['SMA_20']) else None,
                'sma_50': float(latest['SMA_50']) if not np.isnan(latest['SMA_50']) else None,
                'volatility': float(latest['Volatility']) if not np.isnan(latest['Volatility']) else None,
                'daily_return': float(latest['Daily_Return']) if not np.isnan(latest['Daily_Return']) else None,
                'price_change': float(latest['Close'] - earliest['Close']),
                'price_change_pct': float(((latest['Close'] - earliest['Close']) / earliest['Close']) * 100)
            }
        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            return {}

    def get_ticker_info(self, ticker: str) -> Dict:
        """Get basic information about a ticker."""
        try:
            # Apply rate limiting
            current_time = time.time()
            elapsed = current_time - self.__last_request
            if elapsed < self.__request_delay:
                time.sleep(self.__request_delay - elapsed)
            self.__last_request = current_time

            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'symbol': ticker,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logging.error(f"Error getting info for {ticker}: {str(e)}")
            return {'symbol': ticker, 'error': str(e)}

    def clear_cache(self):
        """Clear the cached data."""
        self.__cache.clear()
        logging.info("Cache cleared") 