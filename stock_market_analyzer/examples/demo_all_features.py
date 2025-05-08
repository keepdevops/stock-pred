#!/usr/bin/env python
"""
Comprehensive Features Demo for Stock Market Analyzer

This script demonstrates all the enhanced technical analysis features added
to the Stock Market Analyzer application, including:
1. Ichimoku Cloud
2. VWAP (Volume Weighted Average Price)
3. Pattern Recognition
4. Sentiment Analysis
5. Combined Technical Chart

It also shows how to integrate these features into the AI agent for automated analysis.
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
from modules.stock_ai_agent import StockAIAgent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_config():
    """Create configuration dictionary for components."""
    return {
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

def load_data_for_symbols(symbols, config):
    """Load data for multiple symbols."""
    data_loader = DataLoader(config)
    data_dict = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Loading data for {symbol}")
            data = data_loader._load_from_yahoo(symbol)
            if data is not None and not data.empty:
                data_dict[symbol] = data
                logger.info(f"Successfully loaded {len(data)} data points for {symbol}")
            else:
                logger.warning(f"No data available for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
    
    return data_dict

def analyze_symbols(data_dict, ai_agent):
    """Perform technical analysis on multiple symbols."""
    analysis_results = {}
    
    for symbol, data in data_dict.items():
        try:
            logger.info(f"Analyzing {symbol}")
            # Add all technical indicators
            data = TechnicalIndicators.add_all_indicators(data)
            data = PatternRecognition.find_all_patterns(data)
            
            # Get AI agent analysis
            technical_analysis = ai_agent.analyze_technical(data)
            
            analysis_results[symbol] = {
                'data': data,
                'technical_analysis': technical_analysis
            }
            
            logger.info(f"Analysis completed for {symbol}")
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    return analysis_results

def demo_all_charts(analysis_results, output_dir="charts"):
    """Generate all chart types for each symbol."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, result in analysis_results.items():
        data = result['data']
        logger.info(f"Generating charts for {symbol}")
        
        # 1. Ichimoku Cloud Chart
        try:
            plt.figure(figsize=(12, 8))
            plt.title(f"Ichimoku Cloud - {symbol}")
            
            # Plot price
            plt.plot(data['date'], data['close'], label='Price', color='black', linewidth=1.5)
            
            # Plot Ichimoku Cloud components
            plt.plot(data['date'], data['tenkan_sen'], label='Tenkan-sen', color='#0496ff', linewidth=0.8)
            plt.plot(data['date'], data['kijun_sen'], label='Kijun-sen', color='#991515', linewidth=0.8)
            
            # Plot Cloud (Kumo)
            plt.fill_between(
                data['date'],
                data['senkou_span_a'],
                data['senkou_span_b'],
                where=data['senkou_span_a'] >= data['senkou_span_b'],
                color='#c9daf8',
                alpha=0.5,
                label='Bullish Cloud'
            )
            
            plt.fill_between(
                data['date'],
                data['senkou_span_a'],
                data['senkou_span_b'],
                where=data['senkou_span_a'] < data['senkou_span_b'],
                color='#f4cccc',
                alpha=0.5,
                label='Bearish Cloud'
            )
            
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{symbol}_ichimoku.png"))
            plt.close()
            logger.info(f"Saved Ichimoku Cloud chart for {symbol}")
        except Exception as e:
            logger.error(f"Error generating Ichimoku chart for {symbol}: {e}")
        
        # 2. VWAP Chart
        try:
            plt.figure(figsize=(12, 8))
            plt.title(f"VWAP - {symbol}")
            
            # Plot price
            plt.plot(data['date'], data['close'], label='Price', color='blue', linewidth=1)
            
            # Plot VWAP
            plt.plot(data['date'], data['vwap'], label='VWAP', color='purple', linewidth=1.5)
            
            # Plot VWAP bands
            plt.plot(data['date'], data['vwap_upper_1'], label='Upper Band (1σ)', color='red', linestyle='--')
            plt.plot(data['date'], data['vwap_lower_1'], label='Lower Band (1σ)', color='green', linestyle='--')
            
            # Plot volume on secondary y-axis
            ax2 = plt.twinx()
            ax2.bar(data['date'], data['volume'], label='Volume', color='gray', alpha=0.3)
            ax2.set_ylabel('Volume')
            
            # Combine legends
            lines, labels = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, loc='upper left')
            
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{symbol}_vwap.png"))
            plt.close()
            logger.info(f"Saved VWAP chart for {symbol}")
        except Exception as e:
            logger.error(f"Error generating VWAP chart for {symbol}: {e}")
        
        # 3. Pattern Recognition Chart
        try:
            plt.figure(figsize=(12, 8))
            plt.title(f"Pattern Recognition - {symbol}")
            
            # Plot price
            plt.plot(data['date'], data['close'], label='Price', color='blue')
            
            # Find patterns and mark them
            patterns_found = False
            
            # Dictionary of pattern columns and their markers/colors
            pattern_dict = {
                'head_shoulders': {'marker': 'v', 'color': 'red', 'label': 'Head & Shoulders'},
                'double_top': {'marker': '^', 'color': 'purple', 'label': 'Double Top'},
                'double_bottom': {'marker': '^', 'color': 'green', 'label': 'Double Bottom'},
                'ascending_triangle': {'marker': 's', 'color': 'blue', 'label': 'Ascending Triangle'},
                'descending_triangle': {'marker': 's', 'color': 'orange', 'label': 'Descending Triangle'},
                'symmetrical_triangle': {'marker': 's', 'color': 'brown', 'label': 'Symmetrical Triangle'}
            }
            
            # Y-axis ranges for positioning markers
            y_height = data['high'].max() if 'high' in data.columns else data['close'].max()
            y_low = data['low'].min() if 'low' in data.columns else data['close'].min()
            y_range = y_height - y_low
            
            # Check each pattern
            for pattern, props in pattern_dict.items():
                if pattern in data.columns:
                    pattern_indices = data[data[pattern] == 1].index
                    
                    if not pattern_indices.empty:
                        patterns_found = True
                        pattern_dates = data.loc[pattern_indices, 'date']
                        
                        # Position markers above or below based on pattern type
                        if pattern in ['double_bottom']:
                            y_pos = y_low - (y_range * 0.02)
                        else:
                            y_pos = y_height + (y_range * 0.02)
                            
                        plt.scatter(pattern_dates, [y_pos] * len(pattern_dates), 
                                   marker=props['marker'], color=props['color'], 
                                   s=100, label=props['label'])
                        
                        # Annotate patterns
                        for date in pattern_dates:
                            plt.annotate(pattern[:2].upper(), xy=(date, y_pos), 
                                       xytext=(0, 5 if pattern != 'double_bottom' else -15), 
                                       textcoords='offset points', ha='center', fontsize=8)
            
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
            plt.savefig(os.path.join(output_dir, f"{symbol}_patterns.png"))
            plt.close()
            logger.info(f"Saved Pattern Recognition chart for {symbol}")
        except Exception as e:
            logger.error(f"Error generating Pattern Recognition chart for {symbol}: {e}")
        
        # 4. Sentiment Analysis Chart
        try:
            plt.figure(figsize=(12, 8))
            plt.title(f"Sentiment Analysis - {symbol}")
            
            # Plot price
            plt.plot(data['date'], data['close'], label='Price', color='blue')
            
            # Create twin axis for sentiment
            ax2 = plt.twinx()
            
            # Plot sentiment metrics if available
            if 'market_sentiment' in data.columns:
                ax2.plot(data['date'], data['market_sentiment'], 
                       label='Market Sentiment', color='purple', linewidth=1.5)
            
            # Get current sentiment data
            sentiment_data = SentimentAnalysis.get_combined_sentiment(symbol)
            
            # Add current sentiment marker if available
            if sentiment_data:
                last_date = data['date'].iloc[-1]
                sentiment_value = (sentiment_data['overall_sentiment'] + 1) * 50  # Scale to 0-100
                
                ax2.scatter([last_date], [sentiment_value], s=100, color='red', 
                          label=f"Current: {sentiment_data['sentiment_interpretation']}")
                
                # Annotate
                ax2.annotate(sentiment_data['sentiment_interpretation'], 
                           xy=(last_date, sentiment_value),
                           xytext=(30, 0),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            # Add sentiment zones
            ax2.axhline(y=80, color='green', linestyle='--', alpha=0.3)
            ax2.axhline(y=60, color='lightgreen', linestyle='--', alpha=0.3)
            ax2.axhline(y=40, color='lightcoral', linestyle='--', alpha=0.3)
            ax2.axhline(y=20, color='red', linestyle='--', alpha=0.3)
            
            # Add zone labels
            ax2.text(data['date'].iloc[0], 90, 'Very Bullish', color='green', alpha=0.7)
            ax2.text(data['date'].iloc[0], 70, 'Bullish', color='green', alpha=0.7)
            ax2.text(data['date'].iloc[0], 50, 'Neutral', color='gray', alpha=0.7)
            ax2.text(data['date'].iloc[0], 30, 'Bearish', color='red', alpha=0.7)
            ax2.text(data['date'].iloc[0], 10, 'Very Bearish', color='red', alpha=0.7)
            
            # Configure axis
            plt.xlabel('Date')
            plt.ylabel('Price', color='blue')
            ax2.set_ylabel('Sentiment (0-100)', color='purple')
            ax2.set_ylim(0, 100)
            
            # Combine legends
            lines, labels = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, loc='upper left')
            
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{symbol}_sentiment.png"))
            plt.close()
            logger.info(f"Saved Sentiment Analysis chart for {symbol}")
        except Exception as e:
            logger.error(f"Error generating Sentiment chart for {symbol}: {e}")
        
        # 5. Combined Technical Chart (4-panel chart)
        try:
            fig = plt.figure(figsize=(12, 16))
            
            # Create 4 subplots
            ax1 = fig.add_subplot(411)  # Price and Ichimoku
            ax2 = fig.add_subplot(412)  # Volume and VWAP
            ax3 = fig.add_subplot(413)  # Momentum indicators
            ax4 = fig.add_subplot(414)  # Sentiment
            
            # 1. Price plot with Ichimoku Cloud
            ax1.set_title(f'Price and Ichimoku Cloud - {symbol}')
            ax1.plot(data['date'], data['close'], label='Price', color='black', linewidth=1.5)
            ax1.plot(data['date'], data['tenkan_sen'], label='Tenkan-sen', color='#0496ff', linewidth=0.8)
            ax1.plot(data['date'], data['kijun_sen'], label='Kijun-sen', color='#991515', linewidth=0.8)
            
            # Plot Cloud (Kumo)
            ax1.fill_between(
                data['date'],
                data['senkou_span_a'],
                data['senkou_span_b'],
                where=data['senkou_span_a'] >= data['senkou_span_b'],
                color='#c9daf8',
                alpha=0.5,
                label='Bullish Cloud'
            )
            
            ax1.fill_between(
                data['date'],
                data['senkou_span_a'],
                data['senkou_span_b'],
                where=data['senkou_span_a'] < data['senkou_span_b'],
                color='#f4cccc',
                alpha=0.5,
                label='Bearish Cloud'
            )
            
            ax1.set_ylabel('Price')
            ax1.legend(loc='upper left', fontsize='small')
            ax1.grid(True, alpha=0.3)
            
            # 2. Volume and VWAP
            ax2.set_title('Volume and VWAP')
            ax2.bar(data['date'], data['volume'], color='gray', alpha=0.3, label='Volume')
            ax2.set_ylabel('Volume')
            
            vwap_ax = ax2.twinx()
            vwap_ax.plot(data['date'], data['vwap'], color='purple', linewidth=1.5, label='VWAP')
            vwap_ax.plot(data['date'], data['vwap_upper_1'], color='red', linestyle='--', alpha=0.7,
                       linewidth=0.8, label='Upper Band (1σ)')
            vwap_ax.plot(data['date'], data['vwap_lower_1'], color='green', linestyle='--', alpha=0.7,
                       linewidth=0.8, label='Lower Band (1σ)')
            vwap_ax.set_ylabel('VWAP')
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = vwap_ax.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')
            ax2.grid(True, alpha=0.3)
            
            # 3. Momentum Indicators
            ax3.set_title('Momentum Indicators')
            momentum_ax2 = ax3.twinx()
            
            # RSI and oscillators (0-100 scale)
            ax3.plot(data['date'], data['rsi'], color='blue', linewidth=1, label='RSI')
            ax3.plot(data['date'], data['mfi'], color='purple', linewidth=1, label='MFI')
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.3)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.3)
            
            # CCI (different scale)
            momentum_ax2.plot(data['date'], data['cci'], color='green', linewidth=1, label='CCI')
            momentum_ax2.axhline(y=100, color='red', linestyle=':', alpha=0.3)
            momentum_ax2.axhline(y=-100, color='green', linestyle=':', alpha=0.3)
            
            ax3.set_ylabel('Oscillators (0-100)')
            ax3.set_ylim(0, 100)
            momentum_ax2.set_ylabel('CCI')
            
            # Combine legends
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = momentum_ax2.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')
            ax3.grid(True, alpha=0.3)
            
            # 4. Sentiment Analysis
            ax4.set_title('Sentiment Analysis')
            ax4.plot(data['date'], data['market_sentiment'], color='blue', linewidth=1.5, label='Market Sentiment')
            
            # Add sentiment zones
            ax4.axhline(y=80, color='green', linestyle='--', alpha=0.3)
            ax4.axhline(y=60, color='lightgreen', linestyle='--', alpha=0.3)
            ax4.axhline(y=40, color='lightcoral', linestyle='--', alpha=0.3)
            ax4.axhline(y=20, color='red', linestyle='--', alpha=0.3)
            
            # Add sentiment zone labels
            ax4.text(data['date'].iloc[0], 90, 'Very Bullish', color='green', alpha=0.7)
            ax4.text(data['date'].iloc[0], 70, 'Bullish', color='green', alpha=0.7)
            ax4.text(data['date'].iloc[0], 50, 'Neutral', color='gray', alpha=0.7)
            ax4.text(data['date'].iloc[0], 30, 'Bearish', color='red', alpha=0.7)
            ax4.text(data['date'].iloc[0], 10, 'Very Bearish', color='red', alpha=0.7)
            
            ax4.set_ylabel('Sentiment (0-100)')
            ax4.set_ylim(0, 100)
            ax4.legend(loc='upper left', fontsize='small')
            ax4.grid(True, alpha=0.3)
            
            # Final formatting
            ax4.set_xlabel('Date')
            for ax in [ax1, ax2, ax3, ax4]:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{symbol}_combined.png"))
            plt.close()
            logger.info(f"Saved Combined Technical chart for {symbol}")
        except Exception as e:
            logger.error(f"Error generating Combined Technical chart for {symbol}: {e}")
    
    logger.info(f"All charts saved to {output_dir} directory")

def print_technical_analysis_summary(analysis_results):
    """Print a summary of technical analysis results."""
    print("\n" + "=" * 80)
    print("TECHNICAL ANALYSIS SUMMARY".center(80))
    print("=" * 80)
    
    for symbol, result in analysis_results.items():
        tech_analysis = result['technical_analysis']['technical']
        
        print(f"\nSymbol: {symbol}")
        print("-" * 80)
        
        # Print current price and moving averages
        print(f"Current Price: ${result['data']['close'].iloc[-1]:.2f}")
        print(f"SMA (20): ${tech_analysis['sma_20']:.2f}")
        print(f"SMA (50): ${tech_analysis['sma_50']:.2f}")
        
        # Price relative to moving averages
        price = result['data']['close'].iloc[-1]
        if price > tech_analysis['sma_20'] and price > tech_analysis['sma_50']:
            trend = "BULLISH - Price above 20 & 50 SMAs"
        elif price < tech_analysis['sma_20'] and price < tech_analysis['sma_50']:
            trend = "BEARISH - Price below 20 & 50 SMAs"
        elif price > tech_analysis['sma_20'] and price < tech_analysis['sma_50']:
            trend = "NEUTRAL (Short-term bullish) - Price above 20 SMA but below 50 SMA"
        else:
            trend = "NEUTRAL (Short-term bearish) - Price below 20 SMA but above 50 SMA"
        print(f"Trend: {trend}")
        
        # RSI
        rsi = tech_analysis['rsi']
        if rsi > 70:
            rsi_signal = "OVERBOUGHT"
        elif rsi < 30:
            rsi_signal = "OVERSOLD"
        else:
            rsi_signal = "NEUTRAL"
        print(f"RSI: {rsi:.2f} - {rsi_signal}")
        
        # MACD
        macd = tech_analysis['macd']
        macd_signal = tech_analysis['macd_signal']
        if macd > macd_signal:
            macd_interp = "BULLISH - MACD above signal line"
        else:
            macd_interp = "BEARISH - MACD below signal line"
        print(f"MACD: {macd:.4f} (Signal: {macd_signal:.4f}) - {macd_interp}")
        
        # Ichimoku Cloud
        if all(x in tech_analysis for x in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']):
            tenkan = tech_analysis['tenkan_sen']
            kijun = tech_analysis['kijun_sen']
            span_a = tech_analysis['senkou_span_a']
            span_b = tech_analysis['senkou_span_b']
            
            ichimoku_signals = []
            if price > span_a and price > span_b:
                ichimoku_signals.append("Price above cloud (bullish)")
            elif price < span_a and price < span_b:
                ichimoku_signals.append("Price below cloud (bearish)")
            else:
                ichimoku_signals.append("Price in cloud (neutral)")
                
            if tenkan > kijun:
                ichimoku_signals.append("Tenkan-sen above Kijun-sen (bullish)")
            else:
                ichimoku_signals.append("Tenkan-sen below Kijun-sen (bearish)")
                
            if span_a > span_b:
                ichimoku_signals.append("Bullish cloud")
            else:
                ichimoku_signals.append("Bearish cloud")
                
            print(f"Ichimoku: {', '.join(ichimoku_signals)}")
        
        # VWAP
        if 'vwap' in tech_analysis:
            vwap = tech_analysis['vwap']
            if price > vwap:
                vwap_signal = "BULLISH - Price above VWAP"
            else:
                vwap_signal = "BEARISH - Price below VWAP"
            print(f"VWAP: ${vwap:.2f} - {vwap_signal}")
        
        # Pattern Recognition
        patterns = [pattern for pattern in [
            'head_shoulders', 'double_top', 'double_bottom', 
            'ascending_triangle', 'descending_triangle', 'symmetrical_triangle'
        ] if pattern in tech_analysis and tech_analysis[pattern]]
        
        if patterns:
            print("Patterns Detected:", ", ".join(p.replace('_', ' ').title() for p in patterns))
        else:
            print("Patterns: None detected")
        
        # Sentiment
        if 'overall_sentiment' in tech_analysis:
            sentiment = tech_analysis['overall_sentiment']
            sentiment_interp = tech_analysis.get('sentiment_interpretation', 'Unknown')
            print(f"Sentiment: {sentiment_interp} ({sentiment:.2f})")
        
        # Momentum Indicators
        if 'mfi' in tech_analysis:
            mfi = tech_analysis['mfi']
            if mfi > 80:
                mfi_signal = "OVERBOUGHT"
            elif mfi < 20:
                mfi_signal = "OVERSOLD"
            else:
                mfi_signal = "NEUTRAL"
            print(f"MFI: {mfi:.2f} - {mfi_signal}")
            
        if 'cci' in tech_analysis:
            cci = tech_analysis['cci']
            if cci > 100:
                cci_signal = "OVERBOUGHT"
            elif cci < -100:
                cci_signal = "OVERSOLD"
            else:
                cci_signal = "NEUTRAL"
            print(f"CCI: {cci:.2f} - {cci_signal}")
        
        # Overall assessment based on technical indicators
        bullish_signals = 0
        bearish_signals = 0
        
        # Count bullish/bearish signals
        if price > tech_analysis['sma_20']: bullish_signals += 1
        else: bearish_signals += 1
        
        if price > tech_analysis['sma_50']: bullish_signals += 1
        else: bearish_signals += 1
        
        if rsi > 50: bullish_signals += 1
        elif rsi < 50: bearish_signals += 1
        
        if macd > macd_signal: bullish_signals += 1
        else: bearish_signals += 1
        
        if 'vwap' in tech_analysis:
            if price > vwap: bullish_signals += 1
            else: bearish_signals += 1
        
        if 'overall_sentiment' in tech_analysis:
            if tech_analysis['overall_sentiment'] > 0: bullish_signals += 1
            elif tech_analysis['overall_sentiment'] < 0: bearish_signals += 1
        
        signal_strength = abs(bullish_signals - bearish_signals)
        if bullish_signals > bearish_signals:
            if signal_strength >= 4:
                overall = "STRONGLY BULLISH"
            else:
                overall = "BULLISH"
        elif bearish_signals > bullish_signals:
            if signal_strength >= 4:
                overall = "STRONGLY BEARISH"
            else:
                overall = "BEARISH"
        else:
            overall = "NEUTRAL"
            
        print(f"\nOVERALL: {overall} ({bullish_signals} bullish vs {bearish_signals} bearish signals)")
        print("-" * 80)
    
    print("\n" + "=" * 80)

def main():
    """Run the comprehensive demo."""
    logger.info("Starting comprehensive demo")
    
    # Create configuration
    config = create_config()
    
    # Create AI agent
    ai_agent = StockAIAgent(config)
    
    # Define symbols to analyze
    symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
    
    # Load data for symbols
    data_dict = load_data_for_symbols(symbols, config)
    
    # Analyze data
    analysis_results = analyze_symbols(data_dict, ai_agent)
    
    # Generate all chart types
    demo_all_charts(analysis_results)
    
    # Print summary of technical analysis
    print_technical_analysis_summary(analysis_results)
    
    logger.info("Comprehensive demo completed")
    print("\nDemo completed! Check the 'charts' directory for visualization results.")

if __name__ == "__main__":
    main() 