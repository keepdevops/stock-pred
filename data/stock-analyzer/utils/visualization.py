"""
Visualization functions for stock data and predictions
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

def create_prediction_figure(dark_mode=True):
    """Create a figure for prediction visualization"""
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    if dark_mode:
        fig.patch.set_facecolor("#2E2E2E")
        ax.set_facecolor("#3E3E3E")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
    
    return fig, ax

def plot_predictions(ax, tickers, df, historical_predictions, future_predictions, sequence_length, colors):
    """Plot historical data and predictions"""
    for i, ticker in enumerate(tickers):
        ticker_df = df[df['ticker'] == ticker]
        if ticker_df.empty:
            continue
            
        ticker_dates = ticker_df['date'].iloc[sequence_length:]
        ticker_close = ticker_df['close'].iloc[sequence_length:]
        
        # Get indices for this ticker's predictions
        indices = ticker_df.index[sequence_length:] - sequence_length
        if len(indices) > len(historical_predictions):
            indices = indices[:len(historical_predictions)]
        
        ticker_hist_pred = historical_predictions[indices] if len(indices) > 0 else []
        
        # Get future predictions
        future_pred = future_predictions.get(ticker, [])
        last_date = pd.to_datetime(ticker_df['date'].iloc[-1]) if not ticker_df.empty else None
        future_dates = [last_date + pd.Timedelta(days=j + 1) for j in range(len(future_pred))] if last_date else []
        
        # Plot with color index
        color = colors(i / max(1, len(tickers) - 1))
        ax.plot(ticker_df['date'], ticker_df['close'], label=f'{ticker} Historical', color=color)
        
        if len(ticker_hist_pred) > 0 and len(ticker_dates) > 0:
            ax.plot(ticker_dates[:len(ticker_hist_pred)], ticker_hist_pred, '--', 
                   label=f'{ticker} Hist. Pred.', color=color)
        
        if len(future_pred) > 0 and len(future_dates) > 0:
            ax.plot(future_dates, future_pred, ':', 
                   label=f'{ticker} Future Pred.', color=color)
    
    ax.set_title("Historical and Future Predictions", color="white")
    ax.set_xlabel('Date', color="white")
    ax.set_ylabel('Price', color="white")
    ax.legend(facecolor="#2E2E2E", labelcolor="white", loc='best')
    
    # Format x-axis dates
    fig = ax.figure
    fig.autofmt_xdate()
    
    return fig
