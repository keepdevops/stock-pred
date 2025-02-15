import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import seaborn as sns

def connect_db():
    """Connect to the database and print its structure"""
    print("Connecting to database...")
    try:
        # Try to connect with read-only access to avoid lock issues
        return duckdb.connect('stocks.db', read_only=True)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("\nTrying to fix the issue...")
        
        # If there's an error, try to force close any existing connections
        try:
            duckdb.default_connection.close()
        except:
            pass
            
        # Try one more time with read-only access
        return duckdb.connect('stocks.db', read_only=True)

def plot_stock_prices(symbols, con):
    """Plot candlestick charts for symbols"""
    print(f"Plotting candlestick charts for: {symbols}")
    
    for symbol in symbols:
        # Get data for plotting
        query = """
            SELECT date, open, high, low, close, volume
            FROM stocks 
            WHERE symbol = ?
            AND date >= CURRENT_DATE - INTERVAL '6 months'
            ORDER BY date
        """
        df = con.execute(query, [symbol]).fetchdf()
        
        if not df.empty:
            # Set date as index and ensure it's datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Create the candlestick plot
            mpf.plot(df,
                    type='candle',
                    title=f'\n{symbol} Stock Price',
                    ylabel='Price ($)',
                    volume=True,
                    style='yahoo',
                    figsize=(12, 8),
                    savefig=f'candlestick_{symbol}.png')
            
            print(f"\nPlotted {len(df)} candlesticks for {symbol}")
            print(f"Plot saved as 'candlestick_{symbol}.png'")
        else:
            print(f"No data found for {symbol}")

def plot_price_distribution(symbols, con):
    """Create a box plot of price distributions"""
    print(f"Creating price distribution plot for: {symbols}")
    
    if not symbols:  # Check if symbols list is empty
        print("No symbols to plot!")
        return
    
    plt.figure(figsize=(12, 6))
    
    data = []
    labels = []
    
    for symbol in symbols:
        query = """
            SELECT close
            FROM stocks 
            WHERE symbol = ?
            AND date >= CURRENT_DATE - INTERVAL '6 months'
        """
        df = con.execute(query, [symbol]).fetchdf()
        
        if not df.empty:
            data.append(df['close'])
            labels.append(symbol)
            print(f"Added {len(df)} price points for {symbol}")
    
    if not data:  # Check if we got any data
        print("No data to plot!")
        return
    
    plt.boxplot(data, tick_labels=labels)
    plt.title('Stock Price Distributions')
    plt.ylabel('Price ($)')
    plt.grid(True)
    
    # Save the plot and show it
    plt.savefig('price_distribution.png')
    plt.show()
    plt.close()
    print("\nPlot saved as 'price_distribution.png'")

def main():
    con = connect_db()
    
    # First, let's check what data we have for the last 6 months
    print("\nChecking available data...")
    check_query = """
        SELECT 
            symbol,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(*) as num_records
        FROM stocks 
        WHERE date >= CURRENT_DATE - INTERVAL '6 months'
        GROUP BY symbol
        ORDER BY num_records DESC
        LIMIT 5;
    """
    check_df = con.execute(check_query).fetchdf()
    print("\nAvailable data summary:")
    print(check_df)
    
    # Now get symbols with sufficient data in the last 6 months
    symbols = con.execute("""
        WITH symbol_counts AS (
            SELECT 
                symbol,
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as data_points
            FROM stocks 
            WHERE date >= CURRENT_DATE - INTERVAL '6 months'
            GROUP BY symbol
            HAVING data_points >= 50  -- Lowered threshold since we're looking at less data
            ORDER BY data_points DESC
            LIMIT 5
        )
        SELECT symbol
        FROM symbol_counts
    """).fetchdf()['symbol'].tolist()
    
    print(f"\nAnalyzing these symbols: {symbols}")
    
    if not symbols:
        print("No symbols found with sufficient data!")
        return
    
    # Create plots
    plot_stock_prices(symbols, con)
    plot_price_distribution(symbols, con)
    
    con.close()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 