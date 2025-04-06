#!/usr/bin/env python
"""
Test AI Agent with Enhanced Technical Indicators

This script tests the StockAIAgent with the new enhanced technical indicators.
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules
from modules.stock_ai_agent import StockAIAgent
from modules.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run a test of the StockAIAgent with enhanced technical indicators."""
    logger.info("Testing StockAIAgent with enhanced technical indicators")
    
    try:
        # Create config
        config = {
            'source': 'yahoo',
            'start_date': (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
            'end_date': 'today',
            'data_dir': 'data',
            'model_type': 'LSTM',
            'lookback_days': 60,
            'prediction_days': 5,
            'feature_columns': ['close', 'volume', 'rsi', 'ma20'],
            'target_column': 'close',
            'models_dir': 'models',
            'n_estimators': 100,
            'max_depth': 5
        }
        
        # Create DataLoader
        data_loader = DataLoader(config)
        
        # Create StockAIAgent
        ai_agent = StockAIAgent(config)
        
        # Load data for a stock symbol
        symbol = 'AAPL'
        data = data_loader._load_from_yahoo(symbol)
        
        # Perform technical analysis
        analysis_results = ai_agent.analyze_technical(data)
        
        # Print results
        logger.info("Technical Analysis Results:")
        print("\nTechnical Analysis Results:")
        print("-" * 50)
        
        # Print traditional indicators
        print("\nTraditional Indicators:")
        traditional = ['sma_20', 'sma_50', 'sma_200', 'rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower']
        for key in traditional:
            if key in analysis_results['technical']:
                print(f"{key}: {analysis_results['technical'][key]:.2f}")
        
        # Print Ichimoku Cloud indicators
        print("\nIchimoku Cloud Indicators:")
        ichimoku = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']
        for key in ichimoku:
            if key in analysis_results['technical']:
                print(f"{key}: {analysis_results['technical'][key]:.2f}")
        
        # Print VWAP
        if 'vwap' in analysis_results['technical']:
            print(f"\nVWAP: {analysis_results['technical']['vwap']:.2f}")
        
        # Print pattern recognition
        print("\nPattern Recognition:")
        patterns = ['head_shoulders', 'double_top', 'double_bottom', 
                   'ascending_triangle', 'descending_triangle', 'symmetrical_triangle']
        for key in patterns:
            if key in analysis_results['technical']:
                print(f"{key}: {analysis_results['technical'][key]}")
        
        # Print sentiment analysis
        print("\nSentiment Analysis:")
        sentiment = ['news_sentiment', 'social_sentiment', 'overall_sentiment', 'sentiment_interpretation']
        for key in sentiment:
            if key in analysis_results['technical']:
                if isinstance(analysis_results['technical'][key], float):
                    print(f"{key}: {analysis_results['technical'][key]:.2f}")
                else:
                    print(f"{key}: {analysis_results['technical'][key]}")
                    
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing StockAIAgent: {e}")
        
if __name__ == "__main__":
    main() 