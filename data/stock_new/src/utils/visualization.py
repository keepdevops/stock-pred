import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from src.utils.technical_indicators import TechnicalIndicators, IndicatorConfig

class StockVisualizer:
    """Handle advanced stock data visualization."""
    
    def __init__(self, style: str = 'yahoo'):
        self.style = style
        self.colors = {
            'up': '#26a69a',
            'down': '#ef5350',
            'volume': '#9b59b6',
            'prediction': '#3498db',
            'actual': '#2ecc71'
        }
    
    def plot_candlestick(
        self,
        df: pd.DataFrame,
        title: str,
        volume: bool = True,
        indicators: List[str] = None
    ) -> None:
        """Create candlestick chart with optional indicators."""
        # Prepare data
        df_plot = df.copy()
        df_plot.index = pd.DatetimeIndex(df_plot['date'])
        
        # Setup subplots
        if volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1)
        
        # Plot candlesticks
        mpf.plot(
            df_plot,
            type='candle',
            style=self.style,
            ax=ax1,
            volume=False,
            ylabel='Price ($)'
        )
        
        # Add indicators
        if indicators:
            for indicator in indicators:
                if indicator in df_plot.columns:
                    ax1.plot(
                        df_plot.index,
                        df_plot[indicator],
                        label=indicator,
                        alpha=0.7
                    )
        
        # Add volume
        if volume:
            ax2.bar(
                df_plot.index,
                df_plot['volume'],
                color=self.colors['volume'],
                alpha=0.5
            )
            ax2.set_ylabel('Volume')
        
        # Customize plot
        ax1.set_title(title)
        ax1.legend()
        plt.tight_layout()
    
    def plot_predictions(
        self,
        actual: pd.Series,
        historical_pred: np.ndarray,
        future_pred: np.ndarray,
        confidence_intervals: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """Plot historical and future predictions with confidence intervals."""
        plt.figure(figsize=(12, 6))
        
        # Plot actual data
        plt.plot(
            actual.index,
            actual.values,
            label='Actual',
            color=self.colors['actual']
        )
        
        # Plot historical predictions
        pred_dates = actual.index[-len(historical_pred):]
        plt.plot(
            pred_dates,
            historical_pred,
            label='Historical Predictions',
            color=self.colors['prediction'],
            linestyle='--'
        )
        
        # Plot future predictions
        future_dates = pd.date_range(
            start=actual.index[-1],
            periods=len(future_pred)+1
        )[1:]
        plt.plot(
            future_dates,
            future_pred,
            label='Future Predictions',
            color=self.colors['prediction']
        )
        
        # Add confidence intervals
        if confidence_intervals:
            for interval, (lower, upper) in confidence_intervals.items():
                plt.fill_between(
                    future_dates,
                    lower,
                    upper,
                    alpha=0.2,
                    label=f'{interval} Confidence'
                )
        
        plt.title('Stock Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def plot_training_metrics(
        self,
        history: Dict[str, List[float]],
        metrics: List[str] = None
    ) -> None:
        """Plot training metrics history."""
        metrics = metrics or ['loss', 'val_loss']
        
        plt.figure(figsize=(10, 6))
        for metric in metrics:
            if metric in history:
                plt.plot(
                    history[metric],
                    label=metric.replace('_', ' ').title()
                )
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def plot_technical_analysis(
        self,
        df: pd.DataFrame,
        indicators: List[str]
    ) -> None:
        """Create technical analysis dashboard."""
        n_indicators = len(indicators)
        fig, axes = plt.subplots(n_indicators + 1, 1, figsize=(12, 3*n_indicators + 6))
        
        # Plot price
        axes[0].plot(df['close'], label='Close Price')
        axes[0].set_title('Price History')
        axes[0].legend()
        
        # Plot indicators
        for i, indicator in enumerate(indicators, 1):
            if indicator in df.columns:
                axes[i].plot(df[indicator], label=indicator)
                axes[i].set_title(indicator)
                axes[i].legend()
        
        plt.tight_layout()

# Configuration
from src.utils.config_manager import ConfigManager

config_manager = ConfigManager()
model_config = config_manager.get_config('model')
config_manager.update_config('trading', {'max_position_size': 0.15})

# Technical Indicators
config = IndicatorConfig(rsi_period=14, macd_fast=12, macd_slow=26)
df = TechnicalIndicators.calculate_all(stock_data, config)

# Visualization
from src.utils.visualization import StockVisualizer

visualizer = StockVisualizer()
visualizer.plot_candlestick(df, "AAPL Analysis", indicators=['RSI', 'MACD'])
visualizer.plot_predictions(actual_data, hist_pred, future_pred) 