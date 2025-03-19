from datetime import datetime, timedelta

def download_market_data(collector):
    """Download data for different market segments."""
    
    # Get tickers for different markets
    sp500_tickers = collector.get_sp500_tickers()
    nasdaq_tickers = collector.get_nasdaq100_tickers()
    forex_pairs = collector.get_forex_pairs()
    crypto_tickers = collector.get_crypto_tickers()

    # Combine all tickers
    all_tickers = list(set(sp500_tickers + nasdaq_tickers + forex_pairs + crypto_tickers))

    # Set date range (e.g., last 2 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    # Download data
    results = collector.download_multiple_tickers(
        tickers=all_tickers,
        start_date=start_date,
        end_date=end_date,
        batch_size=5  # Process 5 tickers at a time
    )

    return results 