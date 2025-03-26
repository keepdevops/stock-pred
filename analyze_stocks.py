import duckdb

def analyze_stock_movements():
    # Connect to DuckDB database
    con = duckdb.connect('stocks.db')

    # Display table structure
    print("\n=== Table Structure ===")
    columns = con.execute("SELECT * FROM stocks LIMIT 1").description
    print("Columns in the database:", [col[0] for col in columns])

    # Get stocks by price range
    print("\n=== Stocks by Price Range ===")
    price_range = con.execute("""
        SELECT symbol,
            ROUND(MIN(close), 2) as min_price,
            ROUND(MAX(close), 2) as max_price,
            ROUND(AVG(close), 2) as avg_price,
            ROUND((MAX(close) - MIN(close)) / MIN(close) * 100, 2) as price_range_percent
        FROM stocks
        GROUP BY symbol
        ORDER BY price_range_percent DESC
        LIMIT 10
    """).fetchall()
    print("Symbol | Min Price | Max Price | Avg Price | Price Range %")
    print("-" * 60)
    for row in price_range:
        print(f"{row[0]:<6} | ${row[1]:<9} | ${row[2]:<9} | ${row[3]:<9} | {row[4]}%")

    # Get most expensive stocks
    print("\n=== Most Expensive Stocks ===")
    expensive = con.execute("""
        SELECT symbol, ROUND(MAX(close), 2) as max_price
        FROM stocks
        GROUP BY symbol
        ORDER BY max_price DESC
        LIMIT 5
    """).fetchall()
    print("Symbol | Max Price")
    print("-" * 20)
    for row in expensive:
        print(f"{row[0]:<6} | ${row[1]}")

    # Get least expensive stocks
    print("\n=== Least Expensive Stocks ===")
    cheap = con.execute("""
        SELECT symbol, ROUND(MIN(close), 2) as min_price
        FROM stocks
        GROUP BY symbol
        ORDER BY min_price ASC
        LIMIT 5
    """).fetchall()
    print("Symbol | Min Price")
    print("-" * 20)
    for row in cheap:
        print(f"{row[0]:<6} | ${row[1]}")

    # Get average stock prices
    print("\n=== Average Stock Prices ===")
    averages = con.execute("""
        SELECT symbol,
            COUNT(*) as records,
            ROUND(AVG(close), 2) as avg_price
        FROM stocks
        GROUP BY symbol
        HAVING COUNT(*) > 1
        ORDER BY avg_price DESC
        LIMIT 5
    """).fetchall()
    print("Symbol | Records | Avg Price")
    print("-" * 30)
    for row in averages:
        print(f"{row[0]:<6} | {row[1]:<7} | ${row[2]}")

    # Close the connection
    con.close()

if __name__ == "__main__":
    analyze_stock_movements() 