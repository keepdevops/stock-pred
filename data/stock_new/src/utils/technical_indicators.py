import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ma_periods: List[int] = (20, 50, 200)

class TechnicalIndicators:
    """Calculate technical indicators for stock data."""
    
    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        config: Optional[IndicatorConfig] = None
    ) -> pd.DataFrame:
        """Calculate all technical indicators."""
        config = config or IndicatorConfig()
        df = df.copy()
        
        # Add RSI
        df = TechnicalIndicators.add_rsi(df, period=config.rsi_period)
        
        # Add MACD
        df = TechnicalIndicators.add_macd(
            df,
            fast=config.macd_fast,
            slow=config.macd_slow,
            signal=config.macd_signal
        )
        
        # Add Bollinger Bands
        df = TechnicalIndicators.add_bollinger_bands(
            df,
            period=config.bb_period,
            std=config.bb_std
        )
        
        # Add Moving Averages
        for period in config.ma_periods:
            df = TechnicalIndicators.add_moving_average(df, period)
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        return df
    
    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df['BB_upper'] = df['BB_middle'] + (bb_std * std)
        df['BB_lower'] = df['BB_middle'] - (bb_std * std)
        return df
    
    @staticmethod
    def add_moving_average(
        df: pd.DataFrame,
        period: int
    ) -> pd.DataFrame:
        """Calculate Simple Moving Average."""
        df[f'MA_{period}'] = df['close'].rolling(window=period).mean()
        return df 