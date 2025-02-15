import duckdb

con = duckdb.connect("s.db")

# First, let's see what columns are available
columns = con.execute("""
    SELECT * 
    FROM stocks_prices 
    LIMIT 0
""").df()
print("\nAvailable Columns:")
for column in columns.columns:
    print(f"- {column}")

# Now get and display the actual data
df = con.execute("""
    SELECT *
    FROM forex_prices
    ORDER BY pair, date
    LIMIT 10  -- Limiting to 10 rows for readability
""").df()

print("\nSample Data (First 10 rows):")
print(df)
print(f"\nTotal rows in dataset: {len(df)}")

con.close()
