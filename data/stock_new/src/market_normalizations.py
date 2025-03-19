import numpy as np
import polars as pl
from typing import Dict, List, Optional, Union
import talib
from datetime import datetime, timedelta

class MarketNormalizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_returns(self,
                        df: pl.DataFrame,
                        price_col: str = "close",
                        methods: List[str] = ["simple", "log"]) -> pl.DataFrame:
        """
        Calculate and normalize different types of returns
        """
        try:
            result = df.clone()
            prices = df[price_col].to_numpy()
            
            if "simple" in methods:
                # Simple returns (Pt - Pt-1)/Pt-1
                simple_returns = np.diff(prices) / prices[:-1]
                simple_returns = np.insert(simple_returns, 0, 0)
                result = result.with_columns([
                    pl.Series(name=f"{price_col}_simple_return", values=simple_returns)
                ])
            
            if "log" in methods:
                # Log returns ln(Pt/Pt-1)
                log_returns = np.log(prices[1:] / prices[:-1])
                log_returns = np.insert(log_returns, 0, 0)
                result = result.with_columns([
                    pl.Series(name=f"{price_col}_log_return", values=log_returns)
                ])
            
            if "excess" in methods:
                # Excess returns over risk-free rate
                risk_free_rate = 0.02 / 252  # Assuming daily data
                excess_returns = simple_returns - risk_free_rate
                result = result.with_columns([
                    pl.Series(name=f"{price_col}_excess_return", values=excess_returns)
                ])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in return normalization: {e}")
            raise

    def normalize_volume(self,
                        df: pl.DataFrame,
                        volume_col: str = "volume",
                        methods: List[str] = ["vwap", "relative"]) -> pl.DataFrame:
        """
        Normalize trading volume using various methods
        """
        try:
            result = df.clone()
            volume = df[volume_col].to_numpy()
            prices = df["close"].to_numpy()
            
            if "vwap" in methods:
                # Volume Weighted Average Price normalization
                vwap = np.cumsum(prices * volume) / np.cumsum(volume)
                result = result.with_columns([
                    pl.Series(name="vwap_normalized", values=vwap)
                ])
            
            if "relative" in methods:
                # Relative volume to moving average
                vol_ma = talib.SMA(volume, timeperiod=20)
                rel_volume = volume / vol_ma
                result = result.with_columns([
                    pl.Series(name="relative_volume", values=rel_volume)
                ])
            
            if "dollar" in methods:
                # Dollar volume normalization
                dollar_volume = volume * prices
                normalized_dollar_vol = dollar_volume / np.mean(dollar_volume)
                result = result.with_columns([
                    pl.Series(name="dollar_volume_normalized", values=normalized_dollar_vol)
                ])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in volume normalization: {e}")
            raise

    def normalize_volatility(self,
                           df: pl.DataFrame,
                           price_col: str = "close",
                           methods: List[str] = ["garch", "parkinson"]) -> pl.DataFrame:
        """
        Normalize price volatility using various methods
        """
        try:
            result = df.clone()
            prices = df[price_col].to_numpy()
            returns = np.diff(np.log(prices))
            
            if "garch" in methods:
                # GARCH(1,1) volatility normalization
                from arch import arch_model
                model = arch_model(returns, vol="Garch", p=1, q=1)
                model_fit = model.fit(disp="off")
                conditional_vol = model_fit.conditional_volatility
                result = result.with_columns([
                    pl.Series(name="garch_volatility", values=np.insert(conditional_vol, 0, np.nan))
                ])
            
            if "parkinson" in methods:
                # Parkinson volatility using high-low range
                high = df["high"].to_numpy()
                low = df["low"].to_numpy()
                park_vol = np.sqrt(1/(4*np.log(2)) * np.log(high/low)**2)
                result = result.with_columns([
                    pl.Series(name="parkinson_volatility", values=park_vol)
                ])
            
            if "realized" in methods:
                # Realized volatility
                window = 20
                realized_vol = np.array([
                    np.std(returns[max(0, i-window):i])
                    for i in range(len(returns))
                ])
                result = result.with_columns([
                    pl.Series(name="realized_volatility", values=np.insert(realized_vol, 0, np.nan))
                ])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in volatility normalization: {e}")
            raise

    def normalize_price_levels(self,
                             df: pl.DataFrame,
                             price_col: str = "close",
                             methods: List[str] = ["ma", "bb"]) -> pl.DataFrame:
        """
        Normalize price levels using technical indicators
        """
        try:
            result = df.clone()
            prices = df[price_col].to_numpy()
            
            if "ma" in methods:
                # Moving average normalization
                ma20 = talib.SMA(prices, timeperiod=20)
                ma50 = talib.SMA(prices, timeperiod=50)
                price_to_ma20 = prices / ma20 - 1
                price_to_ma50 = prices / ma50 - 1
                
                result = result.with_columns([
                    pl.Series(name="price_to_ma20", values=price_to_ma20),
                    pl.Series(name="price_to_ma50", values=price_to_ma50)
                ])
            
            if "bb" in methods:
                # Bollinger Bands normalization
                upper, middle, lower = talib.BBANDS(prices, timeperiod=20)
                bb_position = (prices - lower) / (upper - lower)
                
                result = result.with_columns([
                    pl.Series(name="bb_position", values=bb_position)
                ])
            
            if "atr" in methods:
                # ATR-based normalization
                atr = talib.ATR(df["high"].to_numpy(), 
                               df["low"].to_numpy(), 
                               prices, 
                               timeperiod=14)
                price_to_atr = prices / atr
                
                result = result.with_columns([
                    pl.Series(name="price_to_atr", values=price_to_atr)
                ])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in price level normalization: {e}")
            raise

    def normalize_momentum(self,
                         df: pl.DataFrame,
                         price_col: str = "close",
                         methods: List[str] = ["rsi", "macd"]) -> pl.DataFrame:
        """
        Normalize momentum indicators
        """
        try:
            result = df.clone()
            prices = df[price_col].to_numpy()
            
            if "rsi" in methods:
                # RSI normalization
                rsi = talib.RSI(prices, timeperiod=14)
                normalized_rsi = (rsi - 50) / 25  # Center and scale
                
                result = result.with_columns([
                    pl.Series(name="normalized_rsi", values=normalized_rsi)
                ])
            
            if "macd" in methods:
                # MACD normalization
                macd, signal, hist = talib.MACD(prices)
                normalized_hist = hist / np.std(hist[~np.isnan(hist)])
                
                result = result.with_columns([
                    pl.Series(name="normalized_macd_hist", values=normalized_hist)
                ])
            
            if "roc" in methods:
                # Rate of Change normalization
                roc = talib.ROC(prices, timeperiod=10)
                normalized_roc = roc / np.std(roc[~np.isnan(roc)])
                
                result = result.with_columns([
                    pl.Series(name="normalized_roc", values=normalized_roc)
                ])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in momentum normalization: {e}")
            raise

    def normalize_market_impact(self,
                              df: pl.DataFrame,
                              methods: List[str] = ["amihud", "kyle_lambda"]) -> pl.DataFrame:
        """
        Normalize market impact measures
        """
        try:
            result = df.clone()
            returns = np.diff(np.log(df["close"].to_numpy()))
            volume = df["volume"].to_numpy()
            
            if "amihud" in methods:
                # Amihud illiquidity ratio
                dollar_volume = volume * df["close"].to_numpy()
                illiq = np.abs(returns) / dollar_volume[1:]
                illiq = np.insert(illiq, 0, np.nan)
                normalized_illiq = illiq / np.nanmean(illiq)
                
                result = result.with_columns([
                    pl.Series(name="normalized_illiquidity", values=normalized_illiq)
                ])
            
            if "kyle_lambda" in methods:
                # Kyle's lambda (price impact)
                window = 20
                impact = np.array([
                    np.corrcoef(returns[i:i+window], volume[i+1:i+window+1])[0,1]
                    if i < len(returns)-window else np.nan
                    for i in range(len(returns))
                ])
                
                result = result.with_columns([
                    pl.Series(name="price_impact", values=impact)
                ])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in market impact normalization: {e}")
            raise 