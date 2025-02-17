import duckdb

# Connect to the database
conn = duckdb.connect('stocks.db')

# List all tables
print("Available tables:")
tables = conn.execute("SELECT * FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
for table in tables:
    print(table[2])  # table name is in the third column

# Close connection
conn.close()

conn.execute("ALTER TABLE stocks RENAME TO stock_prices")

conn.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices (
        ticker VARCHAR,
        date DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT,
        adj_close DOUBLE,
        PRIMARY KEY (ticker, date)
    )
""")
