"""
Performance metrics calculations for stock predictions
"""
import pandas as pd
import numpy as np

def calculate_ticker_metrics(ticker, historical_data, future_predictions):
    """Calculate performance metrics for a ticker"""
    ticker_data = historical_data[historical_data['ticker'] == ticker]
    if len(ticker_data) < 2:
        return None
        
    # Historical metrics
    start_price = ticker_data['close'].iloc[0]
    end_price = ticker_data['close'].iloc[-1]
    hist_change_pct = ((end_price - start_price) / start_price) * 100
    
    # Calculate historical return on $1,000 investment
    hist_shares = 1000 / start_price
    hist_dollar_return = hist_shares * (end_price - start_price)
    
    # Future metrics
    future_pred = future_predictions.get(ticker, [])
    if not future_pred:
        future_price = end_price
        future_change_pct = 0
        future_dollar_return = 0
    else:
        future_price = future_pred[-1]
        future_change_pct = ((future_price - end_price) / end_price) * 100
        future_shares = 1000 / end_price
        future_dollar_return = future_shares * (future_price - end_price)
    
    # Combined metrics
    combined_change_pct = hist_change_pct + future_change_pct
    
    # Calculate combined return (compounded)
    initial_investment = 1000
    historical_value = initial_investment * (1 + hist_change_pct/100)
    final_value = historical_value * (1 + future_change_pct/100)
    combined_dollar_return = final_value - initial_investment
    
    return {
        'ticker': ticker,
        'historical': {
            'start_price': start_price,
            'end_price': end_price,
            'change_pct': hist_change_pct,
            'dollar_return': hist_dollar_return
        },
        'future': {
            'start_price': end_price,
            'end_price': future_price,
            'change_pct': future_change_pct,
            'dollar_return': future_dollar_return
        },
        'combined': {
            'change_pct': combined_change_pct,
            'dollar_return': combined_dollar_return
        }
    }

def rank_tickers(metrics_list):
    """Rank tickers based on their combined performance metrics"""
    if not metrics_list:
        return []
        
    # Sort by combined percentage change (descending)
    return sorted(metrics_list, key=lambda x: x['combined']['change_pct'], reverse=True)

def get_best_investment(metrics_list):
    """Get the best ticker for investment based on metrics"""
    ranked = rank_tickers(metrics_list)
    return ranked[0] if ranked else None
