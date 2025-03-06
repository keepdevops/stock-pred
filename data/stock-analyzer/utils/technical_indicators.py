"""
Technical indicator calculations for stock data
"""
import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    """Calculate technical indicators for the given dataframe"""
    try:
        if 'close' in df.columns:
            # Moving averages
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma50'] = df['close'].rolling(window=50).mean()
            
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss.replace(0, 0.001)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (std * 2)
            df['bb_lower'] = df['bb_middle'] - (std * 2)
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
        return df.ffill().bfill().fillna(0)
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return df
