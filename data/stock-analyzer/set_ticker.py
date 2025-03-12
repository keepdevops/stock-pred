"""
Set Ticker - Simple utility to set the current ticker before training
"""
from model_save_patch import set_current_ticker

def set_ticker_for_training(ticker):
    """
    Call this function before training to set the ticker name
    Example:
        from set_ticker import set_ticker_for_training
        set_ticker_for_training('AAPL')
        model.fit(...)  # Model will be saved as AAPL_lstm_model.keras
    """
    set_current_ticker(ticker)
    print(f"Ticker set to {ticker} for next model training") 