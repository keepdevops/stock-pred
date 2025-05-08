import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

class DataPreprocessor:
    """Class for preprocessing stock data for model training and prediction."""
    
    def __init__(self, lookback: int = 60):
        """Initialize the data preprocessor."""
        self.logger = logging.getLogger(__name__)
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        
    def load_data(self, ticker: str) -> pd.DataFrame:
        """
        Load historical data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Download data from Yahoo Finance
            data = yf.download(ticker, period="1y", interval="1d")
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
                
            # Select relevant columns
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Calculate returns and volatility
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            
            # Drop NaN values
            data = data.dropna()
            
            self.logger.info(f"Loaded data for {ticker}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {e}")
            raise
            
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            data: DataFrame with historical data
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            # Split into train and validation sets
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise
            
    def prepare_prediction_data(self, data: pd.DataFrame, days: int) -> np.ndarray:
        """
        Prepare data for making predictions.
        
        Args:
            data: DataFrame with historical data
            days: Number of days to predict
            
        Returns:
            Array of prepared data
        """
        try:
            # Scale the data
            scaled_data = self.scaler.transform(data)
            
            # Get the most recent sequence
            X = scaled_data[-self.lookback:]
            
            # Reshape for model input
            X = X.reshape(1, self.lookback, -1)
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction data: {e}")
            raise
            
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences of data for training."""
        X, y = [], []
        
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback, 3])  # Close price
            
        return np.array(X), np.array(y) 