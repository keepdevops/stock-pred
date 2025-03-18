import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

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
    ) -> plt.Figure:
        """Create candlestick chart with optional indicators."""
        try:
            # Prepare data
            df_plot = df.copy()
            if 'date' in df_plot.columns:
                df_plot.set_index('date', inplace=True)
            
            # Create figure
            fig = plt.figure(figsize=(12, 8))
            
            # Setup subplots
            if volume:
                ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
                ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
            else:
                ax1 = plt.subplot2grid((1, 1), (0, 0))
            
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
            
            return fig
            
        except Exception as e:
            plt.close()
            raise Exception(f"Error plotting candlestick chart: {str(e)}")
    
    def plot_predictions(
        self,
        actual: pd.Series,
        predictions: np.ndarray,
        confidence_intervals: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        title: str = "Stock Price Predictions"
    ) -> plt.Figure:
        """Plot actual values and predictions with confidence intervals."""
        try:
            fig = plt.figure(figsize=(12, 6))
            
            # Plot actual data
            plt.plot(
                actual.index,
                actual.values,
                label='Actual',
                color=self.colors['actual']
            )
            
            # Plot predictions
            pred_dates = pd.date_range(
                start=actual.index[-1],
                periods=len(predictions)+1
            )[1:]
            plt.plot(
                pred_dates,
                predictions,
                label='Predictions',
                color=self.colors['prediction'],
                linestyle='--'
            )
            
            # Add confidence intervals
            if confidence_intervals:
                for interval, (lower, upper) in confidence_intervals.items():
                    plt.fill_between(
                        pred_dates,
                        lower,
                        upper,
                        alpha=0.2,
                        label=f'{interval} Confidence'
                    )
            
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return fig
            
        except Exception as e:
            plt.close()
            raise Exception(f"Error plotting predictions: {str(e)}")
    
    def plot_technical_analysis(
        self,
        df: pd.DataFrame,
        indicators: List[str],
        title: str = "Technical Analysis"
    ) -> plt.Figure:
        """Create technical analysis dashboard."""
        try:
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
            
            plt.suptitle(title)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            plt.close()
            raise Exception(f"Error plotting technical analysis: {str(e)}")
    
    def plot_performance_metrics(
        self,
        metrics: Dict[str, float],
        title: str = "Trading Performance"
    ) -> plt.Figure:
        """Plot trading performance metrics."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot win rate pie chart
            win_rate = metrics.get('win_rate', 0)
            ax1.pie(
                [win_rate, 1-win_rate],
                labels=['Wins', 'Losses'],
                colors=[self.colors['up'], self.colors['down']],
                autopct='%1.1f%%'
            )
            ax1.set_title('Win Rate')
            
            # Plot other metrics
            metrics_to_plot = {
                k: v for k, v in metrics.items()
                if k not in ['win_rate'] and isinstance(v, (int, float))
            }
            
            ax2.bar(
                metrics_to_plot.keys(),
                metrics_to_plot.values(),
                color=self.colors['volume']
            )
            ax2.set_title('Performance Metrics')
            plt.xticks(rotation=45)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            plt.close()
            raise Exception(f"Error plotting performance metrics: {str(e)}")
    
    @staticmethod
    def save_plot(fig: plt.Figure, path: str) -> None:
        """Save plot to file."""
        try:
            fig.savefig(path)
            plt.close(fig)
        except Exception as e:
            plt.close(fig)
            raise Exception(f"Error saving plot: {str(e)}") 