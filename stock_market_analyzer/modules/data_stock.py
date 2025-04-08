import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
import traceback
import argparse
import json
import sys
from pathlib import Path

class DataStock:
    """Standalone library for handling stock data operations with yfinance."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize the data stock library."""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(ch)
        
        self.logger.info("Data stock library initialized")
    
    def get_ticker_info(self, symbol: str) -> Dict:
        """Get basic information about a ticker."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            ticker_info = {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            return ticker_info
            
        except Exception as e:
            self.logger.error(f"Error getting ticker info for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_historical_data(self, symbol: str, start_date: str = None, end_date: str = None, 
                          interval: str = '1d', output_file: str = None) -> pd.DataFrame:
        """Get historical data for a ticker."""
        try:
            self.logger.info(f"Fetching historical data for {symbol}")
            
            # Set default dates if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Rename columns to match our schema
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Add additional calculated columns
            data['daily_return'] = data['close'].pct_change()
            data['cumulative_return'] = (1 + data['daily_return']).cumprod() - 1
            
            # Save to file if specified
            if output_file:
                data.to_csv(output_file, index=False)
                self.logger.info(f"Data saved to {output_file}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_multiple_tickers(self, symbols: List[str], start_date: str = None, 
                           end_date: str = None, interval: str = '1d',
                           output_dir: str = None) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple tickers."""
        try:
            self.logger.info(f"Fetching data for {len(symbols)} tickers")
            results = {}
            
            for symbol in symbols:
                try:
                    output_file = None
                    if output_dir:
                        output_file = Path(output_dir) / f"{symbol}_data.csv"
                    
                    data = self.get_historical_data(symbol, start_date, end_date, interval, output_file)
                    results[symbol] = data
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error fetching multiple tickers: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_realtime_data(self, symbol: str, output_file: str = None) -> Dict:
        """Get real-time data for a ticker."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            realtime_data = {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(realtime_data, f, indent=4)
                self.logger.info(f"Realtime data saved to {output_file}")
            
            return realtime_data
            
        except Exception as e:
            self.logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_technical_indicators(self, data: pd.DataFrame, output_file: str = None) -> Dict:
        """Calculate technical indicators for the data."""
        try:
            indicators = {}
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_hist'] = macd - signal
            
            # Calculate Bollinger Bands
            ma20 = data['close'].rolling(window=20).mean()
            std20 = data['close'].rolling(window=20).std()
            indicators['bb_upper'] = ma20 + (std20 * 2)
            indicators['bb_middle'] = ma20
            indicators['bb_lower'] = ma20 - (std20 * 2)
            
            # Save to file if specified
            if output_file:
                indicators_df = pd.DataFrame(indicators)
                indicators_df.to_csv(output_file, index=False)
                self.logger.info(f"Technical indicators saved to {output_file}")
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Stock Data Fetcher')
    
    # Add arguments
    parser.add_argument('--symbol', type=str, help='Stock symbol to fetch data for')
    parser.add_argument('--symbols', type=str, nargs='+', help='Multiple stock symbols to fetch data for')
    parser.add_argument('--start-date', type=str, help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d', 
                       choices=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
                       help='Data interval')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize DataStock
    data_stock = DataStock(log_level=args.log_level)
    
    try:
        if args.symbol:
            # Get ticker info
            info = data_stock.get_ticker_info(args.symbol)
            print(f"\nTicker Information for {args.symbol}:")
            print(json.dumps(info, indent=4))
            
            # Get historical data
            output_file = None
            if args.output_dir:
                output_file = Path(args.output_dir) / f"{args.symbol}_data.csv"
            
            data = data_stock.get_historical_data(
                args.symbol,
                args.start_date,
                args.end_date,
                args.interval,
                output_file
            )
            print(f"\nHistorical Data for {args.symbol}:")
            print(data.head())
            
            # Get real-time data
            realtime_file = None
            if args.output_dir:
                realtime_file = Path(args.output_dir) / f"{args.symbol}_realtime.json"
            
            realtime = data_stock.get_realtime_data(args.symbol, realtime_file)
            print(f"\nRealtime Data for {args.symbol}:")
            print(json.dumps(realtime, indent=4))
            
            # Calculate technical indicators
            indicators_file = None
            if args.output_dir:
                indicators_file = Path(args.output_dir) / f"{args.symbol}_indicators.csv"
            
            indicators = data_stock.get_technical_indicators(data, indicators_file)
            print(f"\nTechnical Indicators for {args.symbol}:")
            print(pd.DataFrame(indicators).tail())
            
        elif args.symbols:
            # Get data for multiple symbols
            results = data_stock.get_multiple_tickers(
                args.symbols,
                args.start_date,
                args.end_date,
                args.interval,
                args.output_dir
            )
            
            print(f"\nData for {len(results)} symbols:")
            for symbol, data in results.items():
                print(f"\n{symbol}:")
                print(data.head())
                
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 