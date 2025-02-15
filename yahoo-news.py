

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Example for Apple stock
symbol = "AAPL"
stock = yf.Ticker(symbol)

# Get the most recent trading day's data
end_date = datetime.now()
start_date = end_date - timedelta(days=1)  # For today's data, we need to go back one day due to time zones
data = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

if not data.empty:
    # Extract the closing price of the most recent day
    latest_price = data['Close'].iloc[-1]
    print(f"Latest closing price for {symbol}: ${latest_price:.2f}")
else:
    print("No data available for the specified date.")
