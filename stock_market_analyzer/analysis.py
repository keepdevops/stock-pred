import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

class StockAnalyzer:
    """
    Class for performing stock data analysis and generating technical indicators.
    """
    
    def __init__(self, logger=None):
        """Initialize the stock analyzer with optional logger."""
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
    
    def analyze_stock(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on stock data and return results.
        
        Args:
            data: DataFrame containing stock price data
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing analysis results categorized by type
        """
        self.logger.info(f"Analyzing data for {symbol} with {len(data)} rows")
        
        try:
            # Create a copy of data to avoid modifying the original
            df = data.copy()
            
            # Ensure data types are correct for calculations
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Initialize results dictionary
            results = {
                'technical': {},
                'fundamental': {},
                'statistical': {},
                'predictions': {},
                'summary': f"Analysis for {symbol} completed successfully."
            }
            
            # Calculate technical indicators if we have price data
            if 'close' in df.columns:
                self.logger.debug(f"Calculating technical indicators for {symbol}")
                
                # Add technical indicators to the dataframe
                df = self._calculate_technical_indicators(df)
                
                # Get the latest values for reporting
                latest = df.iloc[-1].copy()
                
                # Store technical indicators in results
                for indicator in ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 
                                'rsi', 'macd', 'macd_signal', 'macd_hist', 
                                'bb_upper', 'bb_middle', 'bb_lower', 
                                'atr', 'volatility']:
                    if indicator in latest and not pd.isna(latest[indicator]):
                        results['technical'][indicator] = float(latest[indicator])
                
                # Add current price
                if 'close' in latest:
                    results['technical']['current_price'] = float(latest['close'])
                
                # Add simple trend analysis
                results['technical']['trend'] = self._determine_trend(df)
                
                # Add support and resistance levels
                sr_levels = self._calculate_support_resistance(df)
                results['technical']['support_levels'] = sr_levels['support']
                results['technical']['resistance_levels'] = sr_levels['resistance']
            
            # Calculate statistical metrics
            if 'close' in df.columns:
                self.logger.debug(f"Calculating statistical metrics for {symbol}")
                
                # Returns and volatility metrics
                returns = df['close'].pct_change().dropna()
                
                results['statistical']['daily_returns'] = {
                    'mean': float(returns.mean()),
                    'median': float(returns.median()),
                    'min': float(returns.min()),
                    'max': float(returns.max()),
                    'std': float(returns.std()),
                    'skew': float(returns.skew()),
                    'kurtosis': float(returns.kurtosis())
                }
                
                # Calculate annualized metrics
                trading_days = 252
                results['statistical']['annualized'] = {
                    'return': float(returns.mean() * trading_days),
                    'volatility': float(returns.std() * np.sqrt(trading_days)),
                    'sharpe_ratio': float((returns.mean() * trading_days) / (returns.std() * np.sqrt(trading_days))) 
                    if returns.std() > 0 else 0
                }
            
            self.logger.info(f"Analysis completed for {symbol}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return minimal results with error
            return {
                'technical': {'error': str(e)},
                'summary': f"Error analyzing {symbol}: {str(e)}"
            }
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate and add technical indicators to the dataframe."""
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volatility (20-day)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(20)
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
        return df
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine the current trend based on various indicators."""
        latest = df.iloc[-1]
        
        # Initialize signals counter
        bullish_signals = 0
        bearish_signals = 0
        
        # Check price in relation to moving averages
        if 'close' in latest and 'sma_50' in latest and not pd.isna(latest['sma_50']):
            if latest['close'] > latest['sma_50']:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
        if 'close' in latest and 'sma_200' in latest and not pd.isna(latest['sma_200']):
            if latest['close'] > latest['sma_200']:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Check if shorter MA is above longer MA (golden cross or death cross)
        if 'sma_50' in latest and 'sma_200' in latest and not pd.isna(latest['sma_50']) and not pd.isna(latest['sma_200']):
            if latest['sma_50'] > latest['sma_200']:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
        # Check RSI
        if 'rsi' in latest and not pd.isna(latest['rsi']):
            if latest['rsi'] > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            # Check for overbought/oversold
            if latest['rsi'] > 70:
                bearish_signals += 1  # Overbought
            elif latest['rsi'] < 30:
                bullish_signals += 1  # Oversold
        
        # Check MACD
        if 'macd' in latest and 'macd_signal' in latest and not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
            if latest['macd'] > latest['macd_signal']:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
        # Determine overall trend
        if bullish_signals > bearish_signals:
            return "Bullish"
        elif bearish_signals > bullish_signals:
            return "Bearish"
        else:
            return "Neutral"
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate support and resistance levels using recent pivots."""
        window = 10  # Window for finding pivot points
        
        result = {
            'support': [],
            'resistance': []
        }
        
        # Need enough data for calculation
        if len(df) < window * 2:
            return result
        
        # Get recent data for pivot calculation
        recent_df = df.iloc[-100:].copy() if len(df) > 100 else df.copy()
        
        # Find pivot highs and lows
        pivot_high = (recent_df['high'] > recent_df['high'].shift(window)) & (recent_df['high'] > recent_df['high'].shift(-window))
        pivot_low = (recent_df['low'] < recent_df['low'].shift(window)) & (recent_df['low'] < recent_df['low'].shift(-window))
        
        # Get the values of the pivots
        resistance_levels = recent_df.loc[pivot_high, 'high'].sort_values().unique()
        support_levels = recent_df.loc[pivot_low, 'low'].sort_values().unique()
        
        # Convert to built-in types for serialization
        result['resistance'] = [float(level) for level in resistance_levels[-3:]]  # Get top 3 resistance levels
        result['support'] = [float(level) for level in support_levels[:3]]  # Get bottom 3 support levels
        
        return result 