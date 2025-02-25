import yfinance as yf
import duckdb
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import numpy as np

class MarketSentimentDownloader:
    def __init__(self):
        # Create connection for writing sentiment data
        self.conn = duckdb.connect('historical_market_data.db')
        self.setup_database()
        
        # Define sectors and their representative ETFs/symbols
        self.sector_symbols = {
            'Technology': ['XLK', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],
            'Healthcare': ['XLV', 'IYH', 'JNJ', 'UNH', 'PFE'],
            'Financial': ['XLF', 'VFH', 'JPM', 'BAC', 'GS'],
            'Consumer': ['XLY', 'XLP', 'AMZN', 'PG', 'KO'],
            'Industrial': ['XLI', 'VIS', 'HON', 'UPS', 'CAT'],
            'Energy': ['XLE', 'VDE', 'XOM', 'CVX', 'COP'],
            'Materials': ['XLB', 'VAW', 'LIN', 'APD', 'DD'],
            'Utilities': ['XLU', 'VPU', 'NEE', 'DUK', 'SO'],
            'Real Estate': ['XLRE', 'VNQ', 'PLD', 'AMT', 'CCI']
        }

    def setup_database(self):
        """Create necessary tables for sentiment data"""
        # Technical sentiment indicators
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_sentiment (
                date DATE,
                sector VARCHAR,
                symbol VARCHAR,
                rsi DOUBLE,
                macd DOUBLE,
                bollinger_position DOUBLE,
                volume_trend DOUBLE,
                price_momentum DOUBLE,
                volatility DOUBLE,
                sentiment_score DOUBLE,
                updated_at TIMESTAMP
            )
        """)
        
        # Aggregated sector sentiment
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sector_sentiment (
                date DATE,
                sector VARCHAR,
                average_sentiment DOUBLE,
                volatility_index DOUBLE,
                momentum_index DOUBLE,
                volume_index DOUBLE,
                overall_score DOUBLE,
                updated_at TIMESTAMP
            )
        """)

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for sentiment analysis"""
        try:
            # RSI (14 periods)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2

            # Bollinger Bands Position
            sma = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            df['BB_Position'] = (df['Close'] - sma) / (2 * std)

            # Volume Trend
            df['Volume_Trend'] = df['Volume'].rolling(window=20).mean() / df['Volume'].rolling(window=60).mean()

            # Price Momentum
            df['Price_Momentum'] = df['Close'].pct_change(periods=10)

            # Volatility
            df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()

            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return None

    def calculate_sentiment_score(self, row):
        """Calculate overall sentiment score from technical indicators"""
        try:
            score = 0
            # RSI contribution
            if 30 <= row['RSI'] <= 70:
                score += 0.5
            elif row['RSI'] < 30:  # Oversold
                score += 1
            else:  # Overbought
                score += 0

            # MACD contribution
            if row['MACD'] > 0:
                score += 1
            else:
                score += 0

            # Bollinger Position contribution
            if -1 <= row['BB_Position'] <= 1:
                score += 0.5
            elif row['BB_Position'] < -1:  # Oversold
                score += 1
            else:  # Overbought
                score += 0

            # Volume Trend contribution
            if row['Volume_Trend'] > 1:
                score += 1
            else:
                score += 0

            # Normalize score to 0-1 range
            return score / 4
        except Exception as e:
            print(f"Error calculating sentiment score: {e}")
            return None

    def download_sector_data(self, sector, symbols):
        """Download and process data for a sector"""
        try:
            print(f"\nProcessing {sector} sector...")
            
            for symbol in symbols:
                print(f"Downloading data for {symbol}")
                stock = yf.Ticker(symbol)
                
                # Download data from 2023 through 2025
                df = stock.history(start="2023-01-01", end="2025-12-31")
                
                if df.empty:
                    print(f"No data available for {symbol}")
                    continue
                
                df = self.calculate_technical_indicators(df)
                if df is None:
                    continue
                
                df.reset_index(inplace=True)
                df['sector'] = sector
                df['symbol'] = symbol
                df['sentiment_score'] = df.apply(self.calculate_sentiment_score, axis=1)
                df['updated_at'] = datetime.now()
                
                # Store individual symbol sentiment
                self.conn.execute("""
                    INSERT INTO market_sentiment 
                    SELECT 
                        Date as date,
                        sector,
                        symbol,
                        RSI as rsi,
                        MACD as macd,
                        BB_Position as bollinger_position,
                        Volume_Trend as volume_trend,
                        Price_Momentum as price_momentum,
                        Volatility as volatility,
                        sentiment_score,
                        updated_at
                    FROM df
                """)
                
                time.sleep(1)  # Rate limiting
                
            # Calculate and store aggregated sector sentiment
            self.calculate_sector_sentiment(sector)
            
        except Exception as e:
            print(f"Error processing sector {sector}: {e}")

    def calculate_sector_sentiment(self, sector):
        """Calculate and store aggregated sector sentiment"""
        try:
            print(f"Calculating aggregated sentiment for {sector}")
            
            self.conn.execute("""
                INSERT INTO sector_sentiment
                SELECT 
                    date,
                    sector,
                    AVG(sentiment_score) as average_sentiment,
                    AVG(volatility) as volatility_index,
                    AVG(price_momentum) as momentum_index,
                    AVG(volume_trend) as volume_index,
                    (AVG(sentiment_score) * 0.4 + 
                     AVG(price_momentum) * 0.3 +
                     AVG(volume_trend) * 0.3) as overall_score,
                    MAX(updated_at) as updated_at
                FROM market_sentiment
                WHERE sector = ?
                GROUP BY date, sector
            """, [sector])
            
        except Exception as e:
            print(f"Error calculating sector sentiment: {e}")

    def run(self):
        """Main execution method"""
        print(f"Starting sentiment analysis for {len(self.sector_symbols)} sectors")
        
        for sector, symbols in self.sector_symbols.items():
            self.download_sector_data(sector, symbols)
            
        # Show summary after completion
        self.show_summary()

    def show_summary(self):
        """Display summary of sentiment data"""
        print("\nSentiment Analysis Summary:")
        
        # Show sector-level summary
        result = self.conn.execute("""
            SELECT 
                sector,
                AVG(overall_score) as avg_sentiment,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(*) as total_records
            FROM sector_sentiment
            GROUP BY sector
            ORDER BY avg_sentiment DESC
        """).fetchall()
        
        print("\nSector Sentiment Overview:")
        for row in result:
            print(f"\nSector: {row[0]}")
            print(f"Average Sentiment: {row[1]:.2f}")
            print(f"Date Range: {row[2]} to {row[3]}")
            print(f"Total Records: {row[4]}")
            print("-" * 50)

    def cleanup(self):
        """Clean up database connection"""
        try:
            self.conn.close()
            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    downloader = None
    try:
        downloader = MarketSentimentDownloader()
        downloader.run()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if downloader:
            downloader.cleanup()

if __name__ == "__main__":
    main() 