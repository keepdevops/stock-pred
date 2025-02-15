import duckdb

con = duckdb.connect("forex-duckdb.db")

# First, let's see what columns are available
columns = con.execute("""
    SELECT * 
    FROM forex_prices 
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

# Get table schema information
schema = con.execute("""
    DESCRIBE forex_prices;
""").df()

print("\nTable Schema:")
print(schema)

# Alternative detailed way to get column info
column_info = con.execute("""
    SELECT 
        column_name,
        data_type,
        is_nullable
    FROM information_schema.columns
    WHERE table_name = 'forex_prices'
    ORDER BY ordinal_position;
""").df()

print("\nDetailed Column Information:")
print(column_info)

con.close()
