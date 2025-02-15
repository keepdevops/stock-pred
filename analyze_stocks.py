import duckdb

def analyze_stock_movements():
    con = duckdb.connect('stocks.db')
    
    # First, let's check the actual table structure
    print("\n=== Table Structure ===")
    table_info = con.execute("""
        SELECT * FROM stocks LIMIT 1;
    """).fetchdf()
    print("Columns in the database:", table_info.columns.tolist())

    print("\n=== Stocks by Price Range ===")
    price_range = con.execute("""
        SELECT 
            symbol,
            ROUND(MIN(price), 2) as min_price,
            ROUND(MAX(price), 2) as max_price,
            ROUND(AVG(price), 2) as avg_price,
            ROUND((MAX(price) - MIN(price)) / AVG(price) * 100, 2) as price_range_pct
        FROM stocks
        GROUP BY symbol
        HAVING COUNT(*) > 1
        ORDER BY price_range_pct DESC
        LIMIT 10;
    """).fetchdf()
    print(price_range)

    print("\n=== Most Expensive Stocks ===")
    expensive = con.execute("""
        SELECT 
            symbol,
            ROUND(price, 2) as price
        FROM stocks
        ORDER BY price DESC
        LIMIT 10;
    """).fetchdf()
    print(expensive)

    print("\n=== Least Expensive Stocks ===")
    cheap = con.execute("""
        SELECT 
            symbol,
            ROUND(price, 2) as price
        FROM stocks
        WHERE price > 0  -- Exclude any zero prices
        ORDER BY price ASC
        LIMIT 10;
    """).fetchdf()
    print(cheap)

    print("\n=== Average Stock Prices ===")
    averages = con.execute("""
        SELECT 
            symbol,
            ROUND(AVG(price), 2) as avg_price,
            COUNT(*) as num_records
        FROM stocks
        GROUP BY symbol
        HAVING COUNT(*) > 1
        ORDER BY avg_price DESC
        LIMIT 10;
    """).fetchdf()
    print(averages)

    con.close()

if __name__ == "__main__":
    analyze_stock_movements() 