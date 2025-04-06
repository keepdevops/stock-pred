#!/usr/bin/env python
"""
Enhanced Technical Analysis Demo

This script demonstrates the enhanced technical analysis features added to the
Stock Market Analyzer application, including:
1. Ichimoku Cloud
2. VWAP (Volume Weighted Average Price)
3. Pattern Recognition (Head & Shoulders, Double Top/Bottom)
4. Sentiment Analysis

Usage:
    python enhanced_charts_demo.py
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules
from modules.technical_indicators import TechnicalIndicators, PatternRecognition, SentimentAnalysis
from modules.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sample_data(symbol: str = 'AAPL', period: str = '6mo'):
    """Load sample data for demonstration."""
    logger.info(f"Loading sample data for {symbol} over {period}")
    
    try:
        # Create config for DataLoader
        config = {
            'source': 'yahoo',
            'start_date': (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
            'end_date': 'today',
            'data_dir': 'data'
        }
        
        # Use DataLoader to get data
        data_loader = DataLoader(config)
        
        # Since we're directly using Yahoo Finance, we'll use the built-in _load_from_yahoo method
        data = data_loader._load_from_yahoo(symbol)
        
        # Ensure we have required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data is missing required columns. Available columns: {data.columns.tolist()}")
            
        # Ensure date is datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Sort by date
        data = data.sort_values('date')
        
        logger.info(f"Successfully loaded {len(data)} data points")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def demo_ichimoku_cloud(data: pd.DataFrame):
    """Demonstrate Ichimoku Cloud indicator."""
    logger.info("Demonstrating Ichimoku Cloud")
    
    try:
        # Calculate Ichimoku Cloud
        ic_data = TechnicalIndicators.add_ichimoku_cloud(data)
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        plt.title("Ichimoku Cloud Indicator")
        
        # Plot price
        plt.plot(ic_data['date'], ic_data['close'], label='Price', color='blue')
        
        # Plot Ichimoku components
        plt.plot(ic_data['date'], ic_data['tenkan_sen'], label='Tenkan-sen (Conversion Line)', color='red')
        plt.plot(ic_data['date'], ic_data['kijun_sen'], label='Kijun-sen (Base Line)', color='green')
        
        # Plot Senkou span (Cloud)
        plt.fill_between(ic_data['date'], ic_data['senkou_span_a'], ic_data['senkou_span_b'], 
                        where=ic_data['senkou_span_a'] >= ic_data['senkou_span_b'], 
                        color='lightgreen', alpha=0.3, label='Bullish Cloud')
        plt.fill_between(ic_data['date'], ic_data['senkou_span_a'], ic_data['senkou_span_b'], 
                        where=ic_data['senkou_span_a'] < ic_data['senkou_span_b'], 
                        color='lightcoral', alpha=0.3, label='Bearish Cloud')
        
        # Plot Chikou span (Lagging Span)
        plt.plot(ic_data['date'][:-26], ic_data['chikou_span'].dropna(), label='Chikou Span (Lagging Span)', color='purple')
        
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('ichimoku_cloud_demo.png')
        logger.info("Saved Ichimoku Cloud chart to ichimoku_cloud_demo.png")
        
    except Exception as e:
        logger.error(f"Error demonstrating Ichimoku Cloud: {e}")

def demo_vwap(data: pd.DataFrame):
    """Demonstrate VWAP indicator."""
    logger.info("Demonstrating VWAP")
    
    try:
        # Calculate VWAP
        vwap_data = TechnicalIndicators.add_vwap(data)
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        plt.title("Volume Weighted Average Price (VWAP)")
        
        # Plot price
        plt.plot(vwap_data['date'], vwap_data['close'], label='Price', color='blue')
        
        # Plot VWAP
        plt.plot(vwap_data['date'], vwap_data['vwap'], label='VWAP', color='red', linewidth=2)
        
        # Plot VWAP bands
        plt.plot(vwap_data['date'], vwap_data['vwap_upper_1'], label='Upper Band (1σ)', color='green', linestyle='--')
        plt.plot(vwap_data['date'], vwap_data['vwap_lower_1'], label='Lower Band (1σ)', color='green', linestyle='--')
        
        # Plot volume on secondary y-axis
        ax2 = plt.twinx()
        ax2.bar(vwap_data['date'], vwap_data['volume'], label='Volume', color='gray', alpha=0.3)
        ax2.set_ylabel('Volume')
        
        # Combine legends
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('vwap_demo.png')
        logger.info("Saved VWAP chart to vwap_demo.png")
        
    except Exception as e:
        logger.error(f"Error demonstrating VWAP: {e}")

def demo_pattern_recognition(data: pd.DataFrame):
    """Demonstrate pattern recognition."""
    logger.info("Demonstrating Pattern Recognition")
    
    try:
        # Find patterns
        pattern_data = PatternRecognition.find_all_patterns(data)
        
        # Create a figure
        plt.figure(figsize=(12, 8))
        plt.title("Pattern Recognition")
        
        # Plot price
        plt.plot(pattern_data['date'], pattern_data['close'], label='Price', color='blue')
        
        # Find patterns and mark them
        patterns_found = False
        
        # Mark Head and Shoulders pattern
        if 'head_shoulders' in pattern_data.columns:
            hs_dates = pattern_data[pattern_data['head_shoulders'] == 1]['date']
            if not hs_dates.empty:
                patterns_found = True
                plt.scatter(hs_dates, pattern_data.loc[pattern_data['date'].isin(hs_dates), 'high'] * 1.02, 
                           marker='v', color='red', s=100, label='Head & Shoulders')
                
                # Annotate
                for date in hs_dates:
                    price = pattern_data.loc[pattern_data['date'] == date, 'high'].values[0] * 1.02
                    plt.annotate('H&S', (date, price), xytext=(0, 5), 
                               textcoords='offset points', ha='center')
        
        # Mark Double Top pattern
        if 'double_top' in pattern_data.columns:
            dt_dates = pattern_data[pattern_data['double_top'] == 1]['date']
            if not dt_dates.empty:
                patterns_found = True
                plt.scatter(dt_dates, pattern_data.loc[pattern_data['date'].isin(dt_dates), 'high'] * 1.04, 
                           marker='^', color='purple', s=100, label='Double Top')
                
                # Annotate
                for date in dt_dates:
                    price = pattern_data.loc[pattern_data['date'] == date, 'high'].values[0] * 1.04
                    plt.annotate('DT', (date, price), xytext=(0, 5), 
                               textcoords='offset points', ha='center')
        
        # Mark Double Bottom pattern
        if 'double_bottom' in pattern_data.columns:
            db_dates = pattern_data[pattern_data['double_bottom'] == 1]['date']
            if not db_dates.empty:
                patterns_found = True
                plt.scatter(db_dates, pattern_data.loc[pattern_data['date'].isin(db_dates), 'low'] * 0.98, 
                           marker='^', color='green', s=100, label='Double Bottom')
                
                # Annotate
                for date in db_dates:
                    price = pattern_data.loc[pattern_data['date'] == date, 'low'].values[0] * 0.98
                    plt.annotate('DB', (date, price), xytext=(0, -15), 
                               textcoords='offset points', ha='center')
        
        # Add note if no patterns found
        if not patterns_found:
            plt.figtext(0.5, 0.5, 'No patterns detected', ha='center', fontsize=20, 
                      bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('pattern_recognition_demo.png')
        logger.info("Saved Pattern Recognition chart to pattern_recognition_demo.png")
        
    except Exception as e:
        logger.error(f"Error demonstrating pattern recognition: {e}")

def demo_sentiment_analysis(data: pd.DataFrame, symbol: str):
    """Demonstrate sentiment analysis."""
    logger.info("Demonstrating Sentiment Analysis")
    
    try:
        # Get sentiment analysis
        sentiment = SentimentAnalysis.get_combined_sentiment(symbol)
        
        # Create a sentiment dataframe with random past values for visualization
        dates = data['date'].iloc[-30:].tolist()
        np.random.seed(42)  # For reproducibility
        
        # Create artificial sentiment history
        sentiment_history = pd.DataFrame({
            'date': dates,
            'news_sentiment': np.random.uniform(-0.8, 0.8, len(dates)),
            'social_sentiment': np.random.uniform(-0.8, 0.8, len(dates)),
        })
        
        # Calculate overall sentiment
        sentiment_history['overall_sentiment'] = (
            sentiment_history['news_sentiment'] * 0.6 + 
            sentiment_history['social_sentiment'] * 0.4
        )
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        plt.title(f"Sentiment Analysis for {symbol}")
        
        # Plot price on primary axis
        plt.plot(data['date'].iloc[-30:], data['close'].iloc[-30:], label='Price', color='blue', linewidth=2)
        
        # Create secondary axis for sentiment
        ax2 = plt.twinx()
        
        # Plot sentiment metrics (scaled to 0-100)
        ax2.plot(sentiment_history['date'], (sentiment_history['news_sentiment'] + 1) * 50, 
               label='News Sentiment', color='green', linestyle='-')
        ax2.plot(sentiment_history['date'], (sentiment_history['social_sentiment'] + 1) * 50, 
               label='Social Sentiment', color='orange', linestyle='-')
        ax2.plot(sentiment_history['date'], (sentiment_history['overall_sentiment'] + 1) * 50, 
               label='Overall Sentiment', color='red', linewidth=2)
        
        # Add current sentiment value
        current_date = data['date'].iloc[-1] + timedelta(days=1)
        current_sentiment_value = (sentiment['overall_sentiment'] + 1) * 50
        ax2.scatter([current_date], [current_sentiment_value], color='purple', s=100, 
                  label=f"Current: {sentiment['sentiment_interpretation']}")
        
        # Add sentiment zones
        ax2.axhline(y=80, color='green', linestyle='--', alpha=0.3)
        ax2.axhline(y=60, color='lightgreen', linestyle='--', alpha=0.3)
        ax2.axhline(y=40, color='lightcoral', linestyle='--', alpha=0.3)
        ax2.axhline(y=20, color='red', linestyle='--', alpha=0.3)
        
        # Add sentiment zone labels
        ax2.text(data['date'].iloc[-30], 90, 'Very Bullish', color='green', alpha=0.7)
        ax2.text(data['date'].iloc[-30], 70, 'Bullish', color='green', alpha=0.7)
        ax2.text(data['date'].iloc[-30], 50, 'Neutral', color='gray', alpha=0.7)
        ax2.text(data['date'].iloc[-30], 30, 'Bearish', color='red', alpha=0.7)
        ax2.text(data['date'].iloc[-30], 10, 'Very Bearish', color='red', alpha=0.7)
        
        # Configure axis labels and limits
        plt.xlabel('Date')
        plt.ylabel('Price', color='blue')
        ax2.set_ylabel('Sentiment (0-100)', color='red')
        ax2.set_ylim(0, 100)
        
        # Combine legends
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('sentiment_analysis_demo.png')
        logger.info("Saved Sentiment Analysis chart to sentiment_analysis_demo.png")
        
    except Exception as e:
        logger.error(f"Error demonstrating sentiment analysis: {e}")

def run_demo():
    """Run the full demonstration."""
    logger.info("Starting enhanced technical analysis demonstration")
    
    try:
        # Load sample data
        symbol = 'AAPL'
        data = load_sample_data(symbol, '6mo')
        
        # Run individual demos
        demo_ichimoku_cloud(data)
        demo_vwap(data)
        demo_pattern_recognition(data)
        demo_sentiment_analysis(data, symbol)
        
        logger.info("Demonstration completed successfully")
        print("\nDemonstration completed! The following files were created:")
        print("  - ichimoku_cloud_demo.png")
        print("  - vwap_demo.png")
        print("  - pattern_recognition_demo.png")
        print("  - sentiment_analysis_demo.png")
        
    except Exception as e:
        logger.error(f"Error running demonstration: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    run_demo() 