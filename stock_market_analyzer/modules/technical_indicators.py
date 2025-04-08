"""
Technical indicators and pattern recognition for stock market analysis.
This module provides enhanced technical analysis capabilities for the Stock Market Analyzer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import ta
from scipy.signal import argrelextrema
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Enhanced technical indicators for stock analysis."""
    
    @staticmethod
    def add_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud indicators.
        
        Parameters:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added Ichimoku Cloud indicators
        """
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = df['high'].rolling(window=9).max()
        period9_low = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (period9_high + period9_low) / 2
        
        # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df['high'].rolling(window=26).max()
        period26_low = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2
        
        # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = df['high'].rolling(window=52).max()
        period52_low = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Calculate Chikou Span (Lagging Span): Close price shifted backwards by 26 periods
        df['chikou_span'] = df['close'].shift(-26)
        
        return df
    
    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Parameters:
            df: DataFrame with OHLC data and volume
            
        Returns:
            DataFrame with added VWAP indicator
        """
        df = df.copy()
        
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Calculate VWAP bands
        df['vwap_std'] = df.rolling(window=20)['close'].std()
        df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
        df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
        df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
        df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']
        
        # Clean up
        df.drop('typical_price', axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def add_advanced_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced momentum indicators using ta library.
        
        Parameters:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added momentum indicators
        """
        df = df.copy()
        
        # Money Flow Index (MFI)
        df['mfi'] = ta.volume.money_flow_index(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            volume=df['volume'], 
            window=14,
            fillna=True
        )
        
        # Chaikin Money Flow (CMF)
        df['cmf'] = ta.volume.chaikin_money_flow(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            volume=df['volume'], 
            window=20,
            fillna=True
        )
        
        # Rate of Change (ROC)
        df['roc'] = ta.momentum.roc(df['close'], window=14, fillna=True)
        
        # Commodity Channel Index (CCI)
        df['cci'] = ta.trend.cci(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            window=20,
            constant=0.015,
            fillna=True
        )
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            lbp=14,
            fillna=True
        )
        
        return df
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
        """
        Detect chart patterns in price data.
        
        Parameters:
            df: DataFrame with OHLC data
            window_size: Window size for pattern detection
            
        Returns:
            DataFrame with added pattern detection columns
        """
        df = df.copy()
        
        # Initialize pattern columns
        df['head_and_shoulders'] = 0
        df['double_top'] = 0
        df['double_bottom'] = 0
        df['triangle'] = 0
        df['wedge'] = 0
        
        # Find local maxima and minima
        order = 5  # Number of points on each side to use for comparison
        df['local_max'] = df['high'].rolling(window=order*2+1, center=True).apply(
            lambda x: 1 if np.argmax(x) == order else 0, raw=True)
        df['local_min'] = df['low'].rolling(window=order*2+1, center=True).apply(
            lambda x: 1 if np.argmin(x) == order else 0, raw=True)
        
        # Detect patterns
        for i in range(window_size, len(df)):
            window = df.iloc[i-window_size:i+1]
            
            # Detect Head and Shoulders pattern
            peaks = window[window['local_max'] == 1]['high'].values
            if len(peaks) >= 3:
                # Look for 3 peaks with middle peak higher
                for j in range(len(peaks)-2):
                    if (peaks[j+1] > peaks[j] and 
                        peaks[j+1] > peaks[j+2] and
                        abs(peaks[j] - peaks[j+2]) < 0.03 * peaks[j+1]):  # Shoulders have similar height
                        df.loc[df.index[i], 'head_and_shoulders'] = 1
                        break
            
            # Detect Double Top
            if len(peaks) >= 2:
                for j in range(len(peaks)-1):
                    if abs(peaks[j] - peaks[j+1]) < 0.02 * peaks[j]:  # Tops are within 2% of each other
                        # Check if there's a significant drop between the tops
                        window_between = window.iloc[window[window['high'] == peaks[j]].index[0]:
                                                     window[window['high'] == peaks[j+1]].index[0]]
                        if len(window_between) > 0 and min(window_between['low']) < min(peaks) * 0.97:
                            df.loc[df.index[i], 'double_top'] = 1
                            break
            
            # Detect Double Bottom
            troughs = window[window['local_min'] == 1]['low'].values
            if len(troughs) >= 2:
                for j in range(len(troughs)-1):
                    if abs(troughs[j] - troughs[j+1]) < 0.02 * troughs[j]:  # Bottoms are within 2% of each other
                        # Check if there's a significant rise between the bottoms
                        window_between = window.iloc[window[window['low'] == troughs[j]].index[0]:
                                                     window[window['low'] == troughs[j+1]].index[0]]
                        if len(window_between) > 0 and max(window_between['high']) > max(troughs) * 1.03:
                            df.loc[df.index[i], 'double_bottom'] = 1
                            break
        
        return df
    
    @staticmethod
    def calculate_sentiment_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators that can be used for sentiment analysis.
        
        Parameters:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added sentiment indicators
        """
        df = df.copy()
        
        # Calculate fear and greed indicators
        
        # Volatility - High volatility might indicate fear (using ta library's volatility indicators)
        df['volatility'] = ta.volatility.ulcer_index(df['close'], window=20, fillna=True)
        
        # Advance/Decline - Market breadth indicator
        df['price_trend'] = np.where(df['close'].diff() > 0, 1, -1)
        df['adv_decline'] = df['price_trend'].rolling(window=10).sum()
        
        # Put/Call ratio proxy using price movement
        df['down_days'] = (df['close'] < df['open']).astype(int).rolling(window=10).sum()
        df['put_call_proxy'] = df['down_days'] / 10
        
        # Momentum using ta library
        df['momentum'] = ta.momentum.roc(df['close'], window=10, fillna=True)
        
        # Calculate market strength index (combines multiple indicators)
        df['market_strength'] = (
            (df['close'] > df['close'].rolling(window=50).mean()).astype(int) +
            (df['close'] > df['close'].rolling(window=200).mean()).astype(int) +
            (df['volume'] > df['volume'].rolling(window=50).mean()).astype(int) +
            (df['adv_decline'] > 0).astype(int) +
            (df['momentum'] > 0).astype(int)
        )
        
        # Scale to 0-100
        df['market_sentiment'] = df['market_strength'] * 20  # 5 indicators * 20 = max 100
        
        return df

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.
        
        Parameters:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all technical indicators
        """
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Apply all indicators
        df = TechnicalIndicators.add_ichimoku_cloud(df)
        df = TechnicalIndicators.add_vwap(df)
        df = TechnicalIndicators.add_advanced_momentum(df)
        df = TechnicalIndicators.detect_patterns(df)
        df = TechnicalIndicators.calculate_sentiment_indicators(df)
        
        # Make sure there are no missing values
        df = df.ffill().bfill().fillna(0)
        
        return df


class PatternRecognition:
    """Pattern recognition techniques for stock market analysis."""
    
    @staticmethod
    def find_head_and_shoulders(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Find Head and Shoulders patterns in the data.
        
        Parameters:
            df: DataFrame with OHLC data
            window: Window size for pattern detection
            
        Returns:
            DataFrame with head and shoulders pattern indication
        """
        df = df.copy()
        
        # Initialize result column
        df['head_shoulders'] = 0
        
        # Need at least 60 periods for meaningful pattern detection
        if len(df) < window:
            return df
        
        # Iterate through the data using a rolling window
        for i in range(window, len(df)):
            segment = df.iloc[i-window:i]
            
            # Find local maxima for potential head and shoulders
            order = 5
            local_max = argrelextrema(segment['high'].values, np.greater, order=order)[0]
            
            # Need at least 3 peaks for head and shoulders
            if len(local_max) >= 3:
                # Check for sequence of increasing then decreasing peaks
                peaks = segment['high'].values[local_max]
                
                for j in range(len(peaks) - 2):
                    # Check if middle peak is higher (head) and the shoulders are of similar height
                    if peaks[j+1] > peaks[j] and peaks[j+1] > peaks[j+2]:
                        # Check if shoulders are at similar levels (within 5%)
                        shoulder_diff = abs(peaks[j] - peaks[j+2]) / peaks[j]
                        if shoulder_diff < 0.05:
                            # We have a head and shoulders pattern
                            df.loc[df.index[i], 'head_shoulders'] = 1
                            break
        
        return df
    
    @staticmethod
    def find_double_top(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Find Double Top patterns in the data.
        
        Parameters:
            df: DataFrame with OHLC data
            window: Window size for pattern detection
            
        Returns:
            DataFrame with double top pattern indication
        """
        df = df.copy()
        
        # Initialize result column
        df['double_top'] = 0
        
        # Need at least window periods for meaningful pattern detection
        if len(df) < window:
            return df
        
        # Iterate through the data using a rolling window
        for i in range(window, len(df)):
            segment = df.iloc[i-window:i]
            
            # Find local maxima for potential double top
            order = 5
            local_max = argrelextrema(segment['high'].values, np.greater, order=order)[0]
            
            # Need at least 2 peaks for double top
            if len(local_max) >= 2:
                peaks = segment['high'].values[local_max]
                
                for j in range(len(peaks) - 1):
                    # Check if peaks are at similar levels (within 3%)
                    peak_diff = abs(peaks[j] - peaks[j+1]) / peaks[j]
                    if peak_diff < 0.03:
                        # Check if there's a significant drop between the peaks (>3%)
                        idx1 = local_max[j]
                        idx2 = local_max[j+1]
                        if idx2 - idx1 > 10:  # Ensure peaks are not too close
                            between_low = segment['low'].values[idx1:idx2].min()
                            drop_pct = (peaks[j] - between_low) / peaks[j]
                            if drop_pct > 0.03:
                                # We have a double top pattern
                                df.loc[df.index[i], 'double_top'] = 1
                                break
        
        return df
    
    @staticmethod
    def find_double_bottom(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Find Double Bottom patterns in the data.
        
        Parameters:
            df: DataFrame with OHLC data
            window: Window size for pattern detection
            
        Returns:
            DataFrame with double bottom pattern indication
        """
        df = df.copy()
        
        # Initialize result column
        df['double_bottom'] = 0
        
        # Need at least window periods for meaningful pattern detection
        if len(df) < window:
            return df
        
        # Iterate through the data using a rolling window
        for i in range(window, len(df)):
            segment = df.iloc[i-window:i]
            
            # Find local minima for potential double bottom
            order = 5
            local_min = argrelextrema(segment['low'].values, np.less, order=order)[0]
            
            # Need at least 2 troughs for double bottom
            if len(local_min) >= 2:
                troughs = segment['low'].values[local_min]
                
                for j in range(len(troughs) - 1):
                    # Check if troughs are at similar levels (within 3%)
                    trough_diff = abs(troughs[j] - troughs[j+1]) / troughs[j]
                    if trough_diff < 0.03:
                        # Check if there's a significant rise between the troughs (>3%)
                        idx1 = local_min[j]
                        idx2 = local_min[j+1]
                        if idx2 - idx1 > 10:  # Ensure troughs are not too close
                            between_high = segment['high'].values[idx1:idx2].max()
                            rise_pct = (between_high - troughs[j]) / troughs[j]
                            if rise_pct > 0.03:
                                # We have a double bottom pattern
                                df.loc[df.index[i], 'double_bottom'] = 1
                                break
        
        return df
    
    @staticmethod
    def find_triangles(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Find triangle patterns (ascending, descending, symmetrical) in the data.
        
        Parameters:
            df: DataFrame with OHLC data
            window: Window size for pattern detection
            
        Returns:
            DataFrame with triangle pattern indications
        """
        df = df.copy()
        
        # Initialize result columns
        df['ascending_triangle'] = 0
        df['descending_triangle'] = 0
        df['symmetrical_triangle'] = 0
        
        # Need at least window periods for meaningful pattern detection
        if len(df) < window:
            return df
        
        # Iterate through the data using a rolling window
        for i in range(window, len(df)):
            segment = df.iloc[i-window:i]
            
            # Find local maxima and minima
            order = 5
            local_max = argrelextrema(segment['high'].values, np.greater, order=order)[0]
            local_min = argrelextrema(segment['low'].values, np.less, order=order)[0]
            
            # Need at least 2 peaks and 2 troughs for triangles
            if len(local_max) >= 2 and len(local_min) >= 2:
                # Get highs and lows
                highs = segment['high'].values[local_max]
                lows = segment['low'].values[local_min]
                
                # Check for triangle patterns
                
                # Check for horizontal resistance (ascending triangle)
                if len(highs) >= 2:
                    # Calculate the slope of the highs
                    high_slope = np.polyfit([local_max[0], local_max[-1]], [highs[0], highs[-1]], 1)[0]
                    
                    # Check if highs are relatively flat (slope near zero)
                    if abs(high_slope) < 0.005:
                        # Check if lows have positive slope
                        if len(lows) >= 2:
                            low_slope = np.polyfit([local_min[0], local_min[-1]], [lows[0], lows[-1]], 1)[0]
                            if low_slope > 0.005:
                                df.loc[df.index[i], 'ascending_triangle'] = 1
                
                # Check for horizontal support (descending triangle)
                if len(lows) >= 2:
                    # Calculate the slope of the lows
                    low_slope = np.polyfit([local_min[0], local_min[-1]], [lows[0], lows[-1]], 1)[0]
                    
                    # Check if lows are relatively flat (slope near zero)
                    if abs(low_slope) < 0.005:
                        # Check if highs have negative slope
                        if len(highs) >= 2:
                            high_slope = np.polyfit([local_max[0], local_max[-1]], [highs[0], highs[-1]], 1)[0]
                            if high_slope < -0.005:
                                df.loc[df.index[i], 'descending_triangle'] = 1
                
                # Check for symmetrical triangle
                if len(highs) >= 2 and len(lows) >= 2:
                    high_slope = np.polyfit([local_max[0], local_max[-1]], [highs[0], highs[-1]], 1)[0]
                    low_slope = np.polyfit([local_min[0], local_min[-1]], [lows[0], lows[-1]], 1)[0]
                    
                    # Highs have negative slope, lows have positive slope
                    if high_slope < -0.005 and low_slope > 0.005:
                        df.loc[df.index[i], 'symmetrical_triangle'] = 1
        
        return df
    
    @staticmethod
    def find_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Find all chart patterns in the data.
        
        Parameters:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all pattern indications
        """
        df = df.copy()
        
        # Apply all pattern recognition functions
        df = PatternRecognition.find_head_and_shoulders(df)
        df = PatternRecognition.find_double_top(df)
        df = PatternRecognition.find_double_bottom(df)
        df = PatternRecognition.find_triangles(df)
        
        return df


class SentimentAnalysis:
    """News and social media sentiment analysis for stock market."""
    
    @staticmethod
    def analyze_news_sentiment(symbol: str, days: int = 7) -> float:
        """
        Analyze news sentiment for a given symbol.
        
        Parameters:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            Sentiment score from -1 (negative) to 1 (positive)
        """
        try:
            # This is a placeholder for actual news API integration
            # In a real implementation, you would connect to news APIs like Alpha Vantage News
            # or sentiment analysis APIs like Sentim-API
            
            # Mock implementation:
            import random
            sentiment = random.uniform(-1, 1)
            
            logger.info(f"Analyzed news sentiment for {symbol}: {sentiment}")
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return 0.0
    
    @staticmethod
    def analyze_social_sentiment(symbol: str, days: int = 7) -> float:
        """
        Analyze social media sentiment for a given symbol.
        
        Parameters:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            Sentiment score from -1 (negative) to 1 (positive)
        """
        try:
            # This is a placeholder for actual social media API integration
            # In a real implementation, you would connect to social APIs like Twitter API
            # or Reddit API to analyze sentiment
            
            # Mock implementation:
            import random
            sentiment = random.uniform(-1, 1)
            
            logger.info(f"Analyzed social media sentiment for {symbol}: {sentiment}")
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing social media sentiment for {symbol}: {e}")
            return 0.0
    
    @staticmethod
    def get_combined_sentiment(symbol: str) -> Dict[str, float]:
        """
        Get combined sentiment analysis from multiple sources.
        
        Parameters:
            symbol: Stock symbol
            
        Returns:
            Dictionary with sentiment scores
        """
        results = {
            'news_sentiment': SentimentAnalysis.analyze_news_sentiment(symbol),
            'social_sentiment': SentimentAnalysis.analyze_social_sentiment(symbol),
            'overall_sentiment': 0.0  # Will be calculated
        }
        
        # Calculate combined sentiment (weighted average)
        news_weight = 0.6
        social_weight = 0.4
        results['overall_sentiment'] = (
            results['news_sentiment'] * news_weight + 
            results['social_sentiment'] * social_weight
        )
        
        # Add sentiment interpretation
        if results['overall_sentiment'] > 0.5:
            results['sentiment_interpretation'] = 'Very Bullish'
        elif results['overall_sentiment'] > 0.2:
            results['sentiment_interpretation'] = 'Bullish'
        elif results['overall_sentiment'] > -0.2:
            results['sentiment_interpretation'] = 'Neutral'
        elif results['overall_sentiment'] > -0.5:
            results['sentiment_interpretation'] = 'Bearish'
        else:
            results['sentiment_interpretation'] = 'Very Bearish'
            
        return results 