import duckdb
import pandas as pd

con = duckdb.connect('stocks.db')
con.execute("""
    CREATE TABLE IF NOT EXISTS stocks AS SELECT * FROM df
""")

# 1. Basic data exploration
# Get the date range and number of tickers
con.execute("""
    SELECT 
        MIN(date) as earliest_date,
        MAX(date) as latest_date,
        COUNT(DISTINCT ticker) as num_tickers,
        COUNT(*) as total_records
    FROM stock_prices;
""").df()

# 2. Get the latest prices for all stocks
con.execute("""
    SELECT ticker, date, close, volume
    FROM stock_prices
    WHERE date = (SELECT MAX(date) FROM stock_prices)
    ORDER BY ticker;
""").df()

# 3. Calculate daily returns for a specific stock
con.execute("""
    SELECT 
        ticker,
        date,
        close,
        (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) / LAG(close) OVER (PARTITION BY ticker ORDER BY date) as daily_return
    FROM stock_prices
    WHERE ticker = 'AAPL'
    ORDER BY date;
""").df()

# 4. Find highest volume trading days
con.execute("""
    SELECT 
        date,
        ticker,
        volume,
        close
    FROM stock_prices
    ORDER BY volume DESC
    LIMIT 10;
""").df()

# 5. Calculate moving averages
con.execute("""
    SELECT 
        ticker,
        date,
        close,
        AVG(close) OVER (
            PARTITION BY ticker 
            ORDER BY date 
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) as ma_20
    FROM stock_prices
    WHERE ticker = 'AAPL'
    ORDER BY date;
""").df()
