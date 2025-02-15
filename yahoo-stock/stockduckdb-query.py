import duckdb
import os

# First, let's check what database files exist in the current directory
print("\nDatabase files in current directory:")
for file in os.listdir('.'):
    if file.endswith('.db'):
        print(f"- {file}")

# Try to connect to the database
db_name = "stocks.db"  # Changed to an existing database file
print(f"\nAttempting to connect to: {db_name}")

con = duckdb.connect(db_name)

# First, let's see what tables exist in the database
print("\nAvailable Tables:")
tables = con.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'main'
""").df()
print(tables)

# Let's also add a query to show table structure if any tables exist
if not tables.empty:
    print("\nTable Structure:")
    for table_name in tables['table_name']:
        print(f"\nStructure for table: {table_name}")
        schema = con.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """).df()
        print(schema)

# Once we know the correct table name, we can look at its data
print("\nSample Data:")
sample_data = con.execute("""
    SELECT *
    FROM stock_prices
    LIMIT 5
""").df()
print(sample_data)

# Let's also get some basic statistics
print("\nAvailable Tickers:")
tickers = con.execute("""
    SELECT DISTINCT ticker
    FROM stock_prices
    ORDER BY ticker
""").df()
print(tickers)

print("\nDate Range:")
date_range = con.execute("""
    SELECT 
        MIN(date) as earliest_date,
        MAX(date) as latest_date,
        COUNT(*) as total_records
    FROM stock_prices
""").df()
print(date_range)

con.close()
