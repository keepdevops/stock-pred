import duckdb
import pandas as pd
import plotly.express as px

# Connect to the database
con = duckdb.connect('stocks.db')

# Read the data
df = con.execute("""
    SELECT *
    FROM stocks
""").df()

# Create price comparison bar plot
fig = px.bar(df, 
             x='symbol', 
             y='close',
             title='Stock Price Comparison',
             labels={'close': 'Price ($)', 'symbol': 'Stock Symbol'})

# Update layout
fig.update_layout(
    xaxis_title="Stock Symbol",
    yaxis_title="Price ($)",
    bargap=0.2,
    bargroupgap=0.1
)

# Show the plot
fig.show()

# Close the database connection
con.close() 