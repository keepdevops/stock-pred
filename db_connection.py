import duckdb
import pandas as pd

# Create or load your DataFrame first
df = pd.DataFrame({
    'symbol': ['AAPL', 'GOOGL', 'MSFT'],
    'price': [150.0, 2800.0, 300.0]
})  # Replace this with your actual data

# Connect to DuckDB
con = duckdb.connect('stocks.db')

# Register the DataFrame with DuckDB
con.register('df', df)

# Create the table from the registered DataFrame
con.execute("""
    CREATE TABLE IF NOT EXISTS stocks AS SELECT * FROM df
""") 