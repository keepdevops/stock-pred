import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import copy
import os
from scipy import stats
import matplotlib.gridspec as gridspec
from mplfinance.original_flavor import candlestick_ohlc
from PIL import Image, ImageTk

class StockMarketAnalyzer:
    def __init__(self, db_path='stocks.db'):  # Changed default path
        """Initialize the analyzer with database connection and account settings"""
        try:
            self.conn = duckdb.connect(db_path)
            # Add default account settings
            self.account = {
                'balance': 100000.0,  # Initial balance
                'max_position_size': 0.25,  # Maximum position size as percentage of account
                'risk_per_trade': 0.02,  # Risk per trade as percentage of account
                'max_drawdown': 0.0,  # Track maximum drawdown
                'peak_balance': 100000.0,  # Track peak balance
                'trades': []  # Track trade history
            }
        except Exception as e:
            print(f"Error initializing analyzer: {str(e)}")
            self.conn = None  # Set conn to None if connection fails
            raise
    
    def get_historical_data(self, ticker, duration):
        """Get historical data using DuckDB"""
        try:
            # Map duration to SQL interval
            duration_map = {
                '1d': 'INTERVAL 1 day',
                '5d': 'INTERVAL 5 days',
                '1mo': 'INTERVAL 1 month',
                '3mo': 'INTERVAL 3 months',
                '6mo': 'INTERVAL 6 months',
                '1y': 'INTERVAL 1 year',
                '2y': 'INTERVAL 2 years',
                '5y': 'INTERVAL 5 years'
            }
            
            interval = duration_map.get(duration)
            if not interval:
                # For 'max' duration or invalid duration, get all data
                query = """
                SELECT 
                    date,
                    open as Open,
                    high as High,
                    low as Low,
                    close as Close,
                    volume as Volume,
                    adj_close as Adj_Close
                FROM stock_prices 
                WHERE ticker = ?
                ORDER BY date
                """
                params = [ticker]
            else:
                query = """
                SELECT 
                    date,
                    open as Open,
                    high as High,
                    low as Low,
                    close as Close,
                    volume as Volume,
                    adj_close as Adj_Close
                FROM stock_prices 
                WHERE ticker = ?
                AND date >= CURRENT_DATE - {interval}
                ORDER BY date
                """.format(interval=interval)
                params = [ticker]
            
            df = self.conn.execute(query, params).df()
            
            if df.empty:
                print(f"No data found for ticker {ticker}")
                return pd.DataFrame()
            
            # Convert date column to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            print(f"Retrieved {len(df)} records for {ticker} over {duration}")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_options_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks (Delta, Gamma, Vega, Theta)"""
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (-S*sigma*np.exp(-d1**2/2)/(2*np.sqrt(2*np.pi*T)) - 
                    r*K*np.exp(-r*T)*norm.cdf(d2))
        else:  # put
            delta = -norm.cdf(-d1)
            theta = (-S*sigma*np.exp(-d1**2/2)/(2*np.sqrt(2*np.pi*T)) + 
                    r*K*np.exp(-r*T)*norm.cdf(-d2))
        
        gamma = np.exp(-d1**2/2)/(S*sigma*np.sqrt(2*np.pi*T))
        vega = S*np.sqrt(T)*np.exp(-d1**2/2)/np.sqrt(2*np.pi)
        
        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

    def calculate_position_sizing(self, df, risk_per_trade=0.02):
        """Calculate position sizing using various methods"""
        # Standard Deviation for volatility
        std_dev = df['Close'].pct_change().std()
        
        # Fixed Fractional
        account_size = 100000  # Example account size
        fixed_fractional = account_size * risk_per_trade
        
        # Fixed Ratio
        delta = 5000  # Example delta value
        contracts = np.floor(np.sqrt(2 * account_size / delta))
        
        # Kelly Criterion
        wins = len(df[df['Close'].pct_change() > 0])
        total_trades = len(df) - 1
        win_prob = wins / total_trades if total_trades > 0 else 0
        avg_win = df[df['Close'].pct_change() > 0]['Close'].pct_change().mean()
        avg_loss = df[df['Close'].pct_change() < 0]['Close'].pct_change().mean()
        kelly = win_prob - ((1 - win_prob) / (avg_win/abs(avg_loss))) if avg_loss != 0 else 0
        
        # Optimal F
        optimal_f = kelly * 0.5  # Conservative approach
        
        return {
            'std_dev': std_dev,
            'fixed_fractional': fixed_fractional,
            'fixed_ratio': contracts,
            'kelly': kelly,
            'optimal_f': optimal_f
        }

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators including ATR"""
        df = df.copy()
        
        # Calculate True Range for ATR
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        # Calculate ATR with 14 period moving average
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Calculate other basic indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df

    def generate_trading_signals(self, df):
        """Enhanced trading signals with various strategies"""
        signals = []
        current_price = df['Close'].iloc[-1]
        
        # Moving average crossover
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] and \
           df['SMA_20'].iloc[-2] <= df['SMA_50'].iloc[-2]:
            signals.append(('BUY', 'Moving Average Crossover'))
        elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] and \
             df['SMA_20'].iloc[-2] >= df['SMA_50'].iloc[-2]:
            signals.append(('SELL', 'Moving Average Crossover'))
        
        # RSI strategy
        if df['RSI'].iloc[-1] < 30:
            signals.append(('BUY', 'RSI Oversold'))
        elif df['RSI'].iloc[-1] > 70:
            signals.append(('SELL', 'RSI Overbought'))
        
        # MACD crossover
        if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and \
           df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
            signals.append(('BUY', 'MACD Crossover'))
        elif df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and \
             df['MACD'].iloc[-2] >= df['Signal_Line'].iloc[-2]:
            signals.append(('SELL', 'MACD Crossover'))
        
        # Calculate stop levels
        stop_loss = current_price * 0.95  # 5% stop loss
        trailing_stop = max(df['Close'].iloc[-5:]) * 0.97  # 3% trailing stop
        
        return {
            'signals': signals,
            'stop_loss': stop_loss,
            'trailing_stop': trailing_stop,
            'limit_buy': current_price * 0.98,  # 2% below current price
            'limit_sell': current_price * 1.02   # 2% above current price
        }

    def analyze_volume_trends(self, df):
        """Analyze volume patterns and trends"""
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volume trend analysis
        df['Volume_Trend'] = np.where(df['Volume'] > df['Volume_SMA'], 1, -1)
        df['Price_ROC'] = df['Close'].pct_change()
        
        # Volume price trends
        df['VP_Trend'] = np.where((df['Volume_Trend'] == 1) & (df['Price_ROC'] > 0), 'Bullish',
                                np.where((df['Volume_Trend'] == 1) & (df['Price_ROC'] < 0), 'Distribution',
                                np.where((df['Volume_Trend'] == -1) & (df['Price_ROC'] < 0), 'Bearish', 'Accumulation')))
        
        return df

    def identify_patterns(self, df):
        """Identify common chart patterns"""
        patterns = []
        
        # Double Top
        highs = df['High'].rolling(window=20).max()
        if abs(highs.iloc[-1] - highs.iloc[-2]) / highs.iloc[-1] < 0.02:
            patterns.append('Double Top')
        
        # Double Bottom
        lows = df['Low'].rolling(window=20).min()
        if abs(lows.iloc[-1] - lows.iloc[-2]) / lows.iloc[-1] < 0.02:
            patterns.append('Double Bottom')
        
        # Head and Shoulders (simple check)
        if len(df) > 60:
            left = df['High'].iloc[-60:-40].max()
            head = df['High'].iloc[-40:-20].max()
            right = df['High'].iloc[-20:].max()
            if head > left and head > right and abs(left - right) / left < 0.02:
                patterns.append('Head and Shoulders')
        
        return patterns

    def calculate_optimal_daily_points(self, df):
        """Calculate optimal daily entry and exit points"""
        df = df.copy()
        
        # Calculate daily volatility and average true range
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Calculate price momentum and volatility
        df['ROC'] = df['Close'].pct_change()
        df['Volatility'] = df['ROC'].rolling(window=10).std()
        
        # Identify optimal entry points (green triangles)
        df['buy_signal'] = (
            (df['Close'] < df['SMA_20']) &  # Price below short-term MA
            (df['RSI'] < 40) &              # RSI showing oversold
            (df['MACD'] > df['Signal_Line']) & # MACD crossover
            (df['Volume'] > df['Volume_SMA'])   # Above average volume
        )
        
        # Identify optimal exit points (red triangles)
        df['sell_signal'] = (
            (df['Close'] > df['SMA_20']) &   # Price above short-term MA
            (df['RSI'] > 60) &               # RSI showing overbought
            (df['MACD'] < df['Signal_Line']) & # MACD crossover
            (df['Volume'] > df['Volume_SMA'])   # Above average volume
        )
        
        # Calculate optimal entry/exit prices
        df['optimal_entry'] = np.where(df['buy_signal'],
            df['Low'] + df['ATR'] * 0.5,  # Entry above support
            np.nan
        )
        
        df['optimal_exit'] = np.where(df['sell_signal'],
            df['High'] - df['ATR'] * 0.5,  # Exit below resistance
            np.nan
        )
        
        return df

    def calculate_probability_based_orders(self, df):
        """Calculate order recommendations based on probability analysis"""
        df = df.copy()
        
        # Calculate price distribution statistics
        returns = df['Close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Calculate price levels based on probability distributions
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1] if 'ATR' in df else df['Close'].std()
        
        # Calculate probability-based price levels
        one_sigma = current_price * (1 + std_return)
        two_sigma = current_price * (1 + (2 * std_return))
        
        # Calculate support/resistance based on recent price action
        support = df['Low'].rolling(window=20).min().iloc[-1]
        resistance = df['High'].rolling(window=20).max().iloc[-1]
        
        # Probability of price movement
        prob_up = len(returns[returns > 0]) / len(returns)
        prob_down = 1 - prob_up
        
        # Expected move based on ATR
        expected_move = atr * np.sqrt(5)  # 5-day expected move
        
        # Calculate optimal entry points with probabilities
        entry_points = {
            'aggressive': support + (0.25 * atr),
            'moderate': support + (0.5 * atr),
            'conservative': support + (0.75 * atr)
        }
        
        # Calculate optimal exit points with probabilities
        exit_points = {
            'aggressive': resistance - (0.25 * atr),
            'moderate': resistance - (0.5 * atr),
            'conservative': resistance - (0.75 * atr)
        }
        
        # Calculate stop levels based on probability
        stops = {
            'tight': current_price - (1 * atr),
            'medium': current_price - (1.5 * atr),
            'wide': current_price - (2 * atr)
        }
        
        return {
            'price_levels': {
                'current': current_price,
                'one_sigma': one_sigma,
                'two_sigma': two_sigma,
                'support': support,
                'resistance': resistance
            },
            'probabilities': {
                'up': prob_up,
                'down': prob_down,
                'expected_move': expected_move
            },
            'entry_points': entry_points,
            'exit_points': exit_points,
            'stops': stops,
            'atr': atr
        }

    def identify_candlestick_patterns(self, df):
        """Identify candlestick patterns and their significance"""
        df = df.copy()
        
        # Calculate basic candlestick properties
        df['body'] = df['Close'] - df['Open']
        df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['body_size'] = abs(df['body'])
        
        # Calculate ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        patterns = []
        for i in range(len(df)):
            if i < 2:  # Skip first two rows as we need previous data
                continue
                
            # Current and previous candles
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # Use ATR for pattern detection, with fallback to a percentage of price
            atr = curr['ATR'] if not pd.isna(curr['ATR']) else curr['Close'] * 0.02
            
            # Doji pattern (small body)
            if curr['body_size'] <= 0.1 * atr:
                patterns.append({
                    'date': curr.name,
                    'pattern': 'Doji',
                    'significance': 'Indecision',
                    'action': 'Wait',
                    'strength': 0.5
                })
            
            # Hammer pattern
            if (curr['body_size'] > 0 and  # Positive close
                curr['lower_shadow'] > 2 * curr['body_size'] and  # Long lower shadow
                curr['upper_shadow'] < 0.2 * curr['body_size']):  # Small upper shadow
                patterns.append({
                    'date': curr.name,
                    'pattern': 'Hammer',
                    'significance': 'Bullish Reversal',
                    'action': 'Buy',
                    'strength': 0.8
                })
            
            # Shooting Star pattern
            if (curr['body_size'] < 0 and  # Negative close
                curr['upper_shadow'] > 2 * abs(curr['body_size']) and  # Long upper shadow
                curr['lower_shadow'] < 0.2 * abs(curr['body_size'])):  # Small lower shadow
                patterns.append({
                    'date': curr.name,
                    'pattern': 'Shooting Star',
                    'significance': 'Bearish Reversal',
                    'action': 'Sell',
                    'strength': 0.7
                })
            
            # Engulfing patterns
            if abs(curr['body']) > abs(prev['body']):
                if curr['body'] > 0 and prev['body'] < 0:  # Bullish engulfing
                    patterns.append({
                        'date': curr.name,
                        'pattern': 'Bullish Engulfing',
                        'significance': 'Strong Bullish Reversal',
                        'action': 'Buy',
                        'strength': 0.9
                    })
                elif curr['body'] < 0 and prev['body'] > 0:  # Bearish engulfing
                    patterns.append({
                        'date': curr.name,
                        'pattern': 'Bearish Engulfing',
                        'significance': 'Strong Bearish Reversal',
                        'action': 'Sell',
                        'strength': 0.9
                    })
            
            # Morning Star pattern
            if (prev2['body'] < 0 and  # First day bearish
                abs(prev['body']) <= 0.1 * prev['ATR'] and  # Second day doji
                curr['body'] > 0):  # Third day bullish
                patterns.append({
                    'date': curr.name,
                    'pattern': 'Morning Star',
                    'significance': 'Strong Bullish Reversal',
                    'action': 'Buy',
                    'strength': 0.95
                })
            
            # Evening Star pattern
            if (prev2['body'] > 0 and  # First day bullish
                abs(prev['body']) <= 0.1 * prev['ATR'] and  # Second day doji
                curr['body'] < 0):  # Third day bearish
                patterns.append({
                    'date': curr.name,
                    'pattern': 'Evening Star',
                    'significance': 'Strong Bearish Reversal',
                    'action': 'Sell',
                    'strength': 0.95
                })
        
        return patterns

    def plot_analysis(self, df, ticker):
        """Create comprehensive analysis plots with candlestick patterns"""
        # Set global font size
        plt.rcParams.update({'font.size': 9})
        
        # Prepare the data
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # Calculate optimal points and probabilities
        df = self.calculate_optimal_daily_points(df)
        prob_orders = self.calculate_probability_based_orders(df)
        
        # Create the main figure
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(f'{ticker} Market Analysis', fontsize=9, y=0.95)
        
        # Create subplots with specific spacing
        gs = plt.GridSpec(7, 1, figure=fig, height_ratios=[3, 1, 1, 1, 1, 1, 1], hspace=0.3)
        
        # Price Plot with signals and probability levels
        ax1 = fig.add_subplot(gs[0])
        mpf.plot(df, type='candle', style='charles',
                ax=ax1,
                volume=False,
                ylabel='Price',
                returnfig=False,
                tight_layout=False)
        
        # Add probability-based levels
        last_date = df.index[-1]
        ax1.axhline(y=prob_orders['price_levels']['resistance'], color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=prob_orders['price_levels']['support'], color='g', linestyle='--', alpha=0.5)
        
        # Add entry/exit points
        for level, price in prob_orders['entry_points'].items():
            ax1.plot(last_date, price, '^', color='g', markersize=8, alpha=0.7,
                    label=f'{level.capitalize()} Entry')
        
        for level, price in prob_orders['exit_points'].items():
            ax1.plot(last_date, price, 'v', color='r', markersize=8, alpha=0.7,
                    label=f'{level.capitalize()} Exit')
        
        # Add stop levels
        for level, price in prob_orders['stops'].items():
            ax1.axhline(y=price, color='orange', linestyle=':', alpha=0.5,
                       label=f'{level.capitalize()} Stop')
        
        ax1.set_title('Price Chart with Probability-Based Orders', fontsize=9)
        ax1.tick_params(labelsize=9)
        ax1.legend(fontsize=8, loc='upper left')
        
        # Volume Plot
        ax2 = fig.add_subplot(gs[1])
        ax2.bar(df.index, df['Volume'], color='blue', alpha=0.5)
        ax2.set_title('Volume', fontsize=9)
        ax2.tick_params(labelsize=9)
        
        # Moving Averages
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(df.index, df['SMA_20'], label='SMA 20')
        ax3.plot(df.index, df['SMA_50'], label='SMA 50')
        ax3.set_title('Moving Averages', fontsize=9)
        ax3.tick_params(labelsize=9)
        ax3.legend(fontsize=9)
        
        # RSI
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(df.index, df['RSI'], label='RSI')
        ax4.axhline(y=70, color='r', linestyle='--')
        ax4.axhline(y=30, color='g', linestyle='--')
        ax4.set_title('RSI', fontsize=9)
        ax4.tick_params(labelsize=9)
        ax4.legend(fontsize=9)
        
        # MACD
        ax5 = fig.add_subplot(gs[4])
        ax5.plot(df.index, df['MACD'], label='MACD')
        ax5.plot(df.index, df['Signal_Line'], label='Signal Line')
        ax5.set_title('MACD', fontsize=9)
        ax5.tick_params(labelsize=9)
        ax5.legend(fontsize=9)
        
        # Add Probability Analysis
        ax7 = fig.add_subplot(gs[6])
        prob_data = pd.DataFrame({
            'Entry': [v for v in prob_orders['entry_points'].values()],
            'Exit': [v for v in prob_orders['exit_points'].values()],
            'Stop': [v for v in prob_orders['stops'].values()]
        }, index=['Aggressive', 'Moderate', 'Conservative'])
        prob_data.plot(ax=ax7, kind='bar', width=0.8)
        ax7.set_title('Probability-Based Order Levels', fontsize=9)
        ax7.tick_params(labelsize=9)
        ax7.legend(fontsize=9)
        
        # Add candlestick pattern markers
        patterns = self.identify_candlestick_patterns(df)
        
        for pattern in patterns:
            if pattern['action'] == 'Buy':
                marker = '^'
                color = 'g'
            elif pattern['action'] == 'Sell':
                marker = 'v'
                color = 'r'
            else:
                marker = 'o'
                color = 'y'
            
            # Plot marker with size based on pattern strength
            size = pattern['strength'] * 15  # Scale marker size by pattern strength
            ax1.plot(pattern['date'], df.loc[pattern['date'], 'Low'] * 0.99, 
                    marker=marker, color=color, markersize=size, alpha=0.7,
                    label=f"{pattern['pattern']} ({pattern['significance']})")
        
        # Create custom legend
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), 
                  fontsize=8, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        return fig

    def calculate_probability_of_ruin(self, df):
        """Calculate probability of ruin with weighted outcomes"""
        try:
            # Get trading parameters
            initial_capital = float(self.capital_var.get())
            risk_per_trade = float(self.risk_var.get()) / 100
            stop_loss = float(self.stop_loss_var.get()) / 100
            
            # Calculate trade statistics
            returns = df['Close'].pct_change().dropna()
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
            
            # Calculate edge (advantage per trade)
            edge = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Calculate average trade size based on current position sizing
            avg_trade_size = initial_capital * risk_per_trade
            
            # Calculate number of trades to ruin
            if edge > 0 and avg_loss > 0:
                n = int(initial_capital / (avg_trade_size * avg_loss))
            else:
                n = 0
            
            # Calculate classic probability of ruin
            try:
                if edge > 0 and win_rate > 0:
                    q = 1 - win_rate
                    p = win_rate
                    classic_ruin = ((q/p) ** n) if p != 0 else 1.0
                    classic_ruin = min(1.0, max(0.0, classic_ruin))  # Bound between 0 and 1
                else:
                    classic_ruin = 1.0
            except:
                classic_ruin = 1.0
            
            # Calculate extended probability of ruin (with Kelly criterion)
            try:
                if edge > 0:
                    extended_ruin = np.exp(-2 * edge * (initial_capital / avg_trade_size))
                    extended_ruin = min(1.0, max(0.0, extended_ruin))  # Bound between 0 and 1
                else:
                    extended_ruin = 1.0
            except:
                extended_ruin = 1.0
            
            # Calculate optimal f with safeguards
            if avg_loss > 0 and avg_win > 0:
                try:
                    optimal_f = (win_rate - ((1 - win_rate) / (avg_win/avg_loss)))
                    optimal_f = max(0, min(1, optimal_f))  # Bound between 0 and 1
                except:
                    optimal_f = 0
            else:
                optimal_f = 0
            
            # Create probability table with different weights
            weights = {
                'conservative': {
                    'win_rate': 0.4,
                    'edge': 0.3,
                    'capital': 0.3
                },
                'moderate': {
                    'win_rate': 0.33,
                    'edge': 0.34,
                    'capital': 0.33
                },
                'aggressive': {
                    'win_rate': 0.3,
                    'edge': 0.4,
                    'capital': 0.3
                }
            }
            
            weighted_outcomes = {}
            for style, weight in weights.items():
                weighted_ruin = (
                    classic_ruin * weight['win_rate'] +
                    extended_ruin * weight['edge'] +
                    (1 - optimal_f) * weight['capital']
                ) / 3
                weighted_outcomes[style] = round(weighted_ruin * 100, 2)
            
            return {
                'classic_ruin_probability': classic_ruin,
                'extended_ruin_probability': extended_ruin,
                'win_rate': win_rate,
                'edge_per_trade': edge,
                'optimal_f': optimal_f,
                'weighted_outcomes': weighted_outcomes,
                'risk_metrics': {
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'win_loss_ratio': avg_win/avg_loss if avg_loss > 0 else float('inf'),
                    'trades_to_ruin': n
                }
            }
            
        except Exception as e:
            print(f"Error calculating probability of ruin: {str(e)}")
            return None

    def analyze_stock(self, ticker):
        """Enhanced complete analysis for a given stock ticker"""
        # Get historical data
        df = self.get_historical_data(ticker, "1mo")
        if len(df) == 0:
            return f"No data found for {ticker}"
            
        # Calculate all indicators
        df = self.calculate_technical_indicators(df)
        df = self.analyze_volume_trends(df)
        patterns = self.identify_patterns(df)
        
        # Generate signals and orders
        trading_info = self.generate_trading_signals(df)
        position_sizing = self.calculate_position_sizing(df)
        
        # Calculate probability of ruin
        ruin_analysis = self.calculate_probability_of_ruin(df)
        
        current_price = df['Close'].iloc[-1]
        volatility = df['Close'].pct_change().std() * np.sqrt(252)
        
        # Calculate Greeks
        greeks = self.calculate_options_greeks(
            S=current_price,
            K=current_price,
            T=30/365,
            r=0.05,
            sigma=volatility
        )
        
        # Create plots
        fig = self.plot_analysis(df.copy(), ticker)
        
        analysis = {
            'ticker': ticker,
            'current_price': current_price,
            'volatility': volatility,
            'patterns': patterns,
            'trading_signals': trading_info['signals'],
            'volume_analysis': {
                'current_volume': df['Volume'].iloc[-1],
                'avg_volume': df['Volume_SMA'].iloc[-1],
                'volume_trend': df['VP_Trend'].iloc[-1]
            },
            'stop_levels': {
                'stop_loss': trading_info['stop_loss'],
                'trailing_stop': trading_info['trailing_stop']
            },
            'order_levels': {
                'limit_buy': trading_info['limit_buy'],
                'limit_sell': trading_info['limit_sell']
            },
            'position_sizing': position_sizing,
            'greeks': greeks,
            'risk_analysis': ruin_analysis,  # Added probability of ruin analysis
            'last_updated': datetime.now()
        }
        
        plt.close(fig)  # Clean up the figure
        return analysis
    
    def __del__(self):
        """Close database connection"""
        self.conn.close()

    def calculate_moving_averages(self, df, duration):
        """Calculate moving averages with periods adjusted for timeframe"""
        # Define MA periods based on duration
        if duration in ['1d', '5d']:
            short_period = 5
            long_period = 10
        elif duration in ['1mo', '3mo']:
            short_period = 20
            long_period = 50
        elif duration in ['6mo', '1y']:
            short_period = 50
            long_period = 200
        else:  # 2y, 5y, max
            short_period = 100
            long_period = 200
        
        # Calculate MAs
        df[f'SMA_{short_period}'] = df['Close'].rolling(window=short_period).mean()
        df[f'SMA_{long_period}'] = df['Close'].rolling(window=long_period).mean()
        
        return df, short_period, long_period

    def calculate_strategy_performance(self, df):
        """Calculate performance metrics for each strategy"""
        try:
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            
            # Calculate drawdown
            df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
            df['Rolling_Max'] = df['Cumulative_Returns'].expanding().max()
            df['Drawdown'] = (df['Cumulative_Returns'] - df['Rolling_Max']) / df['Rolling_Max']
            
            # Calculate key metrics
            total_return = df['Returns'].sum()
            volatility = df['Returns'].std()
            max_drawdown = df['Drawdown'].min()
            sharpe_ratio = total_return / volatility if volatility != 0 else 0
            
            # Calculate win rate
            winning_trades = len(df[df['Returns'] > 0])
            total_trades = len(df) - 1  # Subtract 1 since first return is NaN
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average return
            avg_return = df['Returns'].mean()
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'avg_return': avg_return
            }
            
        except Exception as e:
            print(f"Error calculating strategy performance: {str(e)}")
            return None

    def update_account(self, trade_result):
        """Update account balance and metrics after a trade"""
        self.account['balance'] += trade_result
        
        # Update peak balance if necessary
        if self.account['balance'] > self.account['peak_balance']:
            self.account['peak_balance'] = self.account['balance']
        
        # Calculate current drawdown
        current_drawdown = (self.account['peak_balance'] - self.account['balance']) / self.account['peak_balance']
        
        # Update max drawdown if necessary
        if current_drawdown > self.account['max_drawdown']:
            self.account['max_drawdown'] = current_drawdown
        
        # Add trade to history
        self.account['trades'].append({
            'timestamp': datetime.now(),
            'result': trade_result,
            'balance': self.account['balance'],
            'drawdown': current_drawdown
        })

    def get_account_metrics(self):
        """Get current account metrics"""
        return {
            'current_balance': self.account['balance'],
            'peak_balance': self.account['peak_balance'],
            'max_drawdown': self.account['max_drawdown'],
            'total_trades': len(self.account['trades']),
            'win_rate': self.calculate_win_rate(),
            'risk_per_trade': self.account['risk_per_trade'],
            'max_position_size': self.account['max_position_size']
        }

    def calculate_win_rate(self):
        """Calculate win rate from trade history"""
        if not self.account['trades']:
            return 0.0
        
        winning_trades = sum(1 for trade in self.account['trades'] if trade['result'] > 0)
        return winning_trades / len(self.account['trades'])

    def recommend_strategy(self, df):
        """Recommend the best strategy based on market conditions and performance"""
        try:
            # Define strategy characteristics
            strategies = {
                "MA Crossover": {
                    'volatility_weight': 0.3,
                    'trend_weight': 0.4,
                    'volume_weight': 0.3,
                    'min_periods': 20,
                    'best_for': 'trending markets'
                },
                "RSI": {
                    'volatility_weight': 0.4,
                    'trend_weight': 0.2,
                    'volume_weight': 0.4,
                    'min_periods': 14,
                    'best_for': 'ranging markets'
                },
                "MACD": {
                    'volatility_weight': 0.3,
                    'trend_weight': 0.5,
                    'volume_weight': 0.2,
                    'min_periods': 26,
                    'best_for': 'trending markets with momentum'
                },
                "Bollinger Bands": {
                    'volatility_weight': 0.5,
                    'trend_weight': 0.3,
                    'volume_weight': 0.2,
                    'min_periods': 20,
                    'best_for': 'volatile markets'
                }
            }
            
            # Calculate market conditions
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            trend = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            volume_trend = (df['Volume'].iloc[-1] - df['Volume'].iloc[0]) / df['Volume'].iloc[0]
            
            # Calculate market characteristics
            market_conditions = {
                'volatility': volatility,
                'trend_strength': abs(trend),
                'trend_direction': np.sign(trend),
                'volume_trend': volume_trend,
                'avg_volume': df['Volume'].mean(),
                'price_range': (df['High'].max() - df['Low'].min()) / df['Close'].mean()
            }
            
            # Calculate strategy scores
            strategy_scores = {}
            for strategy_name, characteristics in strategies.items():
                # Skip if not enough data for strategy
                if len(df) < characteristics['min_periods']:
                    continue
                
                # Get strategy performance
                performance = self.calculate_strategy_performance(df)
                
                if performance:
                    # Market condition score
                    market_score = (
                        characteristics['volatility_weight'] * (1 - volatility if volatility < 0.5 else 0.5 - volatility/2) +
                        characteristics['trend_weight'] * (trend if trend > 0 else -trend/2) +
                        characteristics['volume_weight'] * (volume_trend if volume_trend > 0 else 0)
                    )
                    
                    # Performance score
                    performance_score = (
                        performance['win_rate'] * 0.3 +
                        performance['avg_return'] * 0.3 +
                        performance['sharpe_ratio'] * 0.2 +
                        (1 + performance['max_drawdown']) * 0.2
                    )
                    
                    # Calculate final score
                    total_score = (market_score + performance_score) / 2
                    
                    strategy_scores[strategy_name] = {
                        'total_score': total_score,
                        'market_score': market_score,
                        'performance_score': performance_score,
                        'performance_metrics': performance,
                        'best_for': characteristics['best_for']
                    }
            
            # Find best strategy
            if strategy_scores:
                best_strategy = max(strategy_scores.items(), key=lambda x: x[1]['total_score'])
                
                # Add market condition analysis
                market_analysis = {
                    'conditions': market_conditions,
                    'recommended': best_strategy[0],
                    'scores': strategy_scores,
                    'market_type': self.determine_market_type(market_conditions)
                }
                
                return market_analysis
            
            return None
            
        except Exception as e:
            print(f"Error recommending strategy: {str(e)}")
            return None

    def determine_market_type(self, conditions):
        """Determine market type based on conditions"""
        if conditions['volatility'] > 0.3:
            if abs(conditions['trend_strength']) > 0.15:
                return "Volatile Trend"
            return "Volatile Range"
        elif abs(conditions['trend_strength']) > 0.1:
            if conditions['trend_direction'] > 0:
                return "Strong Uptrend"
            return "Strong Downtrend"
        elif abs(conditions['trend_strength']) > 0.05:
            if conditions['trend_direction'] > 0:
                return "Weak Uptrend"
            return "Weak Downtrend"
        return "Ranging Market"

    def update_strategy_recommendation(self, df):
        """Update the strategy recommendation display"""
        try:
            if df is None or df.empty:
                return "Unable to generate strategy recommendation\n"
            
            # Calculate market conditions
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            trend = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            volume_trend = (df['Volume'].iloc[-1] - df['Volume'].iloc[0]) / df['Volume'].iloc[0]
            
            text = "Strategy Analysis:\n"
            text += "=" * 40 + "\n\n"
            
            # Market conditions
            text += "Market Conditions:\n"
            text += f"Volatility: {volatility:.2%}\n"
            text += f"Trend: {trend:.2%}\n"
            text += f"Volume Trend: {volume_trend:.2%}\n\n"
            
            # Strategy analysis based on conditions
            if abs(trend) > 0.1:  # Strong trend
                if trend > 0:
                    text += "Market Type: Strong Uptrend\n"
                    text += "Recommended Strategies:\n"
                    text += "1. Trend Following\n"
                    text += "2. Moving Average Crossover\n"
                    text += "3. MACD\n"
                else:
                    text += "Market Type: Strong Downtrend\n"
                    text += "Recommended Strategies:\n"
                    text += "1. Trend Following (Short)\n"
                    text += "2. Moving Average Crossover\n"
                    text += "3. RSI Oversold\n"
            elif volatility > 0.2:  # High volatility
                text += "Market Type: Volatile\n"
                text += "Recommended Strategies:\n"
                text += "1. Bollinger Bands\n"
                text += "2. RSI\n"
                text += "3. Mean Reversion\n"
            else:  # Range-bound
                text += "Market Type: Range-bound\n"
                text += "Recommended Strategies:\n"
                text += "1. Mean Reversion\n"
                text += "2. RSI\n"
                text += "3. Support/Resistance\n"
            
            return text
            
        except Exception as e:
            print(f"Error in strategy recommendation: {str(e)}")
            return "Error generating strategy recommendation\n"

class StockAnalyzerGUI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("Stock Market Analyzer")
        self.root.geometry("1600x1000")
        
        # Initialize data structures
        self.data_cache = {}
        self.strategy_metrics = {}
        
        # Create main container frame
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create left frame for controls
        self.left_frame = ttk.Frame(self.main_container, padding="5")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create right frame for plot and analysis
        self.right_frame = ttk.Frame(self.main_container, padding="5")
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize database connection
        try:
            self.db_conn = duckdb.connect('./stocks.db')
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            messagebox.showerror("Database Error", 
                               "Could not connect to database. Please ensure 'stocks.db' exists.")
            return
        
        # Create input variables
        self.ticker_var = tk.StringVar()
        self.duration_var = tk.StringVar(value='1mo')
        self.capital_var = tk.StringVar(value='100000')
        self.risk_var = tk.StringVar(value='2')
        self.stop_loss_var = tk.StringVar(value='2')
        self.strategy_var = tk.StringVar(value='MA Crossover')
        
        # Create controls in left frame
        self.create_controls()
        
        # Create loading label
        self.loading_label = ttk.Label(self.right_frame, text="")
        self.loading_label.pack(pady=5)
        
        # Initialize matplotlib figure
        self.fig_main = Figure(figsize=(12, 8))
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=self.right_frame)
        self.canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create analysis widgets below the plot
        self.create_analysis_widgets(self.right_frame)

    def create_controls(self):
        """Create control widgets"""
        # Trading Account Frame
        account_frame = ttk.LabelFrame(self.left_frame, text="Account Settings", padding="5")
        account_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Initial Balance
        balance_frame = ttk.Frame(account_frame)
        balance_frame.pack(fill=tk.X, pady=2)
        ttk.Label(balance_frame, text="Initial Balance ($):").pack(side=tk.LEFT, padx=5)
        self.balance_var = tk.StringVar(value=str(self.analyzer.account['balance']))
        ttk.Entry(balance_frame, textvariable=self.balance_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Risk per Trade
        risk_frame = ttk.Frame(account_frame)
        risk_frame.pack(fill=tk.X, pady=2)
        ttk.Label(risk_frame, text="Risk per Trade (%):").pack(side=tk.LEFT, padx=5)
        self.risk_var = tk.StringVar(value=str(self.analyzer.account['risk_per_trade'] * 100))
        ttk.Entry(risk_frame, textvariable=self.risk_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Max Position Size
        position_frame = ttk.Frame(account_frame)
        position_frame.pack(fill=tk.X, pady=2)
        ttk.Label(position_frame, text="Max Position Size (%):").pack(side=tk.LEFT, padx=5)
        self.position_var = tk.StringVar(value=str(self.analyzer.account['max_position_size'] * 100))
        ttk.Entry(position_frame, textvariable=self.position_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Stop Loss
        stop_frame = ttk.Frame(account_frame)
        stop_frame.pack(fill=tk.X, pady=2)
        ttk.Label(stop_frame, text="Stop Loss (%):").pack(side=tk.LEFT, padx=5)
        self.stop_loss_var = tk.StringVar(value="2")
        ttk.Entry(stop_frame, textvariable=self.stop_loss_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Update Account Button
        ttk.Button(account_frame, text="Update Account Settings", 
                   command=self.update_account_settings).pack(pady=5)
        
        # Trading Controls Frame
        trading_frame = ttk.LabelFrame(self.left_frame, text="Trading Controls", padding="5")
        trading_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Ticker selection
        ticker_frame = ttk.Frame(trading_frame)
        ticker_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ticker_frame, text="Ticker:").pack(side=tk.LEFT, padx=5)
        self.ticker_combo = ttk.Combobox(ticker_frame, textvariable=self.ticker_var)
        self.ticker_combo['values'] = self.get_tickers()
        self.ticker_combo.pack(side=tk.LEFT, padx=5)
        
        # Duration selection
        duration_frame = ttk.Frame(trading_frame)
        duration_frame.pack(fill=tk.X, pady=2)
        ttk.Label(duration_frame, text="Duration:").pack(side=tk.LEFT, padx=5)
        durations = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
        self.duration_combo = ttk.Combobox(duration_frame, textvariable=self.duration_var, values=durations)
        self.duration_combo.pack(side=tk.LEFT, padx=5)
        
        # Strategy selection
        strategy_frame = ttk.Frame(trading_frame)
        strategy_frame.pack(fill=tk.X, pady=2)
        ttk.Label(strategy_frame, text="Strategy:").pack(side=tk.LEFT, padx=5)
        strategies = ['MA Crossover', 'RSI', 'MACD', 'Bollinger Bands']
        self.strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.strategy_var, values=strategies)
        self.strategy_combo.pack(side=tk.LEFT, padx=5)
        
        # Analysis button
        ttk.Button(trading_frame, text="Analyze", command=self.start_analysis).pack(pady=5)

    def on_strategy_change(self, event=None):
        """Handle strategy change event"""
        if hasattr(self, 'current_df') and self.current_df is not None:
            self.update_analysis(self.current_df)

    def update_analysis(self, df):
        """Update all analysis components"""
        try:
            if df is None or df.empty:
                return
            
            # Calculate current price and basic metrics
            current_price = df['Close'].iloc[-1]
            
            # Calculate position sizing
            position_data = self.calculate_position_sizing(current_price)
            
            # Generate trading signals using the analyzer instance
            signals = self.analyzer.generate_trading_signals(df)
            
            # Calculate probability of ruin
            ruin_analysis = self.calculate_probability_of_ruin(df)
            
            # Perform pattern analysis
            pattern_analysis = self.analyze_patterns(df)
            
            # Perform probability analysis
            prob_analysis = self.analyze_probability(df)
            
            # Perform returns scenario analysis
            returns_analysis = self.analyze_returns_scenarios(df)
            
            # Update analysis text
            self.update_analysis_text(df, position_data, signals, ruin_analysis)
            
            # Update pattern analysis text
            if hasattr(self, 'pattern_text'):
                self.pattern_text.delete(1.0, tk.END)
                self.pattern_text.insert(tk.END, pattern_analysis)
            
            # Update probability analysis text
            if hasattr(self, 'prob_text'):
                self.prob_text.delete(1.0, tk.END)
                self.prob_text.insert(tk.END, prob_analysis)
            
            # Update returns analysis text
            if hasattr(self, 'returns_text'):
                self.returns_text.delete(1.0, tk.END)
                self.returns_text.insert(tk.END, returns_analysis)
            
        except Exception as e:
            print(f"Error updating analysis: {str(e)}")
            raise e

    def update_analysis_text(self, df, position_data, signals, ruin_analysis):
        """Update the analysis text display"""
        try:
            if df is None or df.empty:
                return "No data available for analysis\n"
            
            analysis = "Market Analysis:\n"
            analysis += "=" * 40 + "\n\n"
            
            # Price Analysis
            current_price = df['Close'].iloc[-1]
            price_change = df['Close'].pct_change().iloc[-1]
            analysis += f"Current Price: ${current_price:.2f}\n"
            analysis += f"Daily Change: {price_change:.2%}\n"
            
            # Volume Analysis
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].mean()
            analysis += f"Current Volume: {current_volume:,.0f}\n"
            analysis += f"Average Volume: {avg_volume:,.0f}\n"
            analysis += f"Volume Ratio: {current_volume/avg_volume:.2f}\n\n"
            
            # Position Sizing
            if position_data is not None:
                analysis += "Position Sizing:\n"
                for key, value in position_data.items():
                    if isinstance(value, float):
                        analysis += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
                    else:
                        analysis += f"{key.replace('_', ' ').title()}: {value}\n"
                analysis += "\n"
            
            # Trading Signals
            if signals is not None and 'signals' in signals and len(signals['signals']) > 0:
                analysis += "Trading Signals:\n"
                for signal, reason in signals['signals']:
                    analysis += f"{signal}: {reason}\n"
                analysis += "\n"
            
            # Add probability analysis
            if ruin_analysis is not None:
                analysis += "Risk Analysis:\n"
                if isinstance(ruin_analysis, dict):
                    for key, value in ruin_analysis.items():
                        if isinstance(value, dict):
                            analysis += f"\n{key.replace('_', ' ').title()}:\n"
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, float):
                                    analysis += f"  {sub_key.replace('_', ' ').title()}: {sub_value:.2%}\n"
                                else:
                                    analysis += f"  {sub_key.replace('_', ' ').title()}: {sub_value}\n"
                        elif isinstance(value, float):
                            analysis += f"{key.replace('_', ' ').title()}: {value:.2%}\n"
                        else:
                            analysis += f"{key.replace('_', ' ').title()}: {value}\n"
                analysis += "\n"
            
            # Add strategy recommendation
            strategy_rec = self.recommend_strategy(df)
            if strategy_rec is not None:
                analysis += "Strategy Recommendation:\n"
                analysis += "=" * 40 + "\n"
                analysis += strategy_rec
            
            # Update text widgets
            if hasattr(self, 'risk_text'):
                self.risk_text.delete(1.0, tk.END)
                self.risk_text.insert(tk.END, analysis)
            
            return analysis
        
        except Exception as e:
            print(f"Error updating analysis text: {str(e)}")
            return "Error generating analysis\n"

    def recommend_strategy(self, df):
        """Recommend trading strategy based on market conditions"""
        try:
            if df is None or df.empty:
                return "No data available for strategy recommendation"
            
            # Calculate market conditions
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            trend = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            volume_trend = (df['Volume'].iloc[-1] - df['Volume'].iloc[0]) / df['Volume'].iloc[0]
            
            recommendation = ""
            
            # Market conditions summary
            recommendation += f"Market Conditions:\n"
            recommendation += f"Volatility: {volatility:.2%}\n"
            recommendation += f"Trend: {trend:.2%}\n"
            recommendation += f"Volume Trend: {volume_trend:.2%}\n\n"
            
            # Strategy recommendation based on conditions
            if abs(trend) > 0.1:  # Strong trend
                if trend > 0:
                    recommendation += "Market Type: Strong Uptrend\n"
                    recommendation += "Recommended Strategies:\n"
                    recommendation += "1. Trend Following\n"
                    recommendation += "2. Moving Average Crossover\n"
                    recommendation += "3. MACD\n"
                else:
                    recommendation += "Market Type: Strong Downtrend\n"
                    recommendation += "Recommended Strategies:\n"
                    recommendation += "1. Trend Following (Short)\n"
                    recommendation += "2. Moving Average Crossover\n"
                    recommendation += "3. RSI Oversold\n"
            elif volatility > 0.2:  # High volatility
                recommendation += "Market Type: Volatile\n"
                recommendation += "Recommended Strategies:\n"
                recommendation += "1. Bollinger Bands\n"
                recommendation += "2. RSI\n"
                recommendation += "3. Mean Reversion\n"
            else:  # Range-bound
                recommendation += "Market Type: Range-bound\n"
                recommendation += "Recommended Strategies:\n"
                recommendation += "1. Mean Reversion\n"
                recommendation += "2. RSI\n"
                recommendation += "3. Support/Resistance\n"
            
            return recommendation
        
        except Exception as e:
            print(f"Error in strategy recommendation: {str(e)}")
            return "Error generating strategy recommendation"

    def create_analysis_widgets(self, parent):
        """Create all analysis widgets"""
        # Create analysis frame
        analysis_frame = ttk.Frame(parent)
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Risk Analysis
        risk_frame = ttk.LabelFrame(analysis_frame, text="Risk Analysis", padding="5")
        risk_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.risk_text = tk.Text(risk_frame, height=8, width=40, font=('Courier', 9))
        self.risk_text.pack(fill=tk.BOTH, expand=True)
        
        # Pattern Analysis
        pattern_frame = ttk.LabelFrame(analysis_frame, text="Pattern Analysis", padding="5")
        pattern_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.pattern_text = tk.Text(pattern_frame, height=8, width=40, font=('Courier', 9))
        self.pattern_text.pack(fill=tk.BOTH, expand=True)
        
        # Returns Analysis
        returns_frame = ttk.LabelFrame(analysis_frame, text="Returns Analysis", padding="5")
        returns_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.returns_text = tk.Text(returns_frame, height=8, width=40, font=('Courier', 9))
        self.returns_text.pack(fill=tk.BOTH, expand=True)

        # Simulation Statistics
        self.sim_stats_text = tk.Text(analysis_frame, height=8, width=40, font=('Courier', 9))
        self.sim_stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Monte Carlo Statistics
        self.mc_stats_text = tk.Text(analysis_frame, height=8, width=40, font=('Courier', 9))
        self.mc_stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def calculate_moving_averages(self, df, duration):
        """Calculate moving averages with periods adjusted for timeframe"""
        # Define MA periods based on duration
        if duration in ['1d', '5d']:
            short_period = 5
            long_period = 10
        elif duration in ['1mo', '3mo']:
            short_period = 20
            long_period = 50
        elif duration in ['6mo', '1y']:
            short_period = 50
            long_period = 200
        else:  # 2y, 5y, max
            short_period = 100
            long_period = 200
        
        # Calculate MAs
        df[f'SMA_{short_period}'] = df['Close'].rolling(window=short_period).mean()
        df[f'SMA_{long_period}'] = df['Close'].rolling(window=long_period).mean()
        
        return df, short_period, long_period

    def update_plots(self, df, ticker):
        """Update all plots with new data"""
        try:
            # Clear existing plots
            self.fig_main.clear()
            
            # Calculate moving averages based on duration
            df, short_ma, long_ma = self.calculate_moving_averages(df, self.duration_var.get())
            
            # Create subplot grid with specific positions and heights
            ax_candle = self.fig_main.add_subplot(411)
            ax_volume = self.fig_main.add_subplot(412)
            ax_ma = self.fig_main.add_subplot(413)
            ax_rsi = self.fig_main.add_subplot(414)
            
            # Set background color and style
            for ax in [ax_candle, ax_volume, ax_ma, ax_rsi]:
                ax.set_facecolor('#f0f0f0')
                ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', colors='#666666')
            
            # Plot candlestick chart
            up = df[df['Close'] >= df['Open']]
            down = df[df['Close'] < df['Open']]
            
            # Plot up candlesticks with better green color
            ax_candle.bar(up.index, up['Close'] - up['Open'], bottom=up['Open'], 
                         width=0.8, color='#26a69a', alpha=0.7)
            ax_candle.bar(up.index, up['High'] - up['Close'], bottom=up['Close'],
                         width=0.2, color='#26a69a', alpha=0.7)
            ax_candle.bar(up.index, up['Low'] - up['Open'], bottom=up['Open'],
                         width=0.2, color='#26a69a', alpha=0.7)
            
            # Plot down candlesticks with better red color
            ax_candle.bar(down.index, down['Close'] - down['Open'], bottom=down['Open'],
                         width=0.8, color='#ef5350', alpha=0.7)
            ax_candle.bar(down.index, down['High'] - down['Open'], bottom=down['Open'],
                         width=0.2, color='#ef5350', alpha=0.7)
            ax_candle.bar(down.index, down['Low'] - down['Close'], bottom=down['Close'],
                         width=0.2, color='#ef5350', alpha=0.7)
            
            # Plot volume with gradient colors
            colors = np.where(df['Close'] >= df['Open'], '#26a69a', '#ef5350')
            ax_volume.bar(df.index, df['Volume'], color=colors, alpha=0.7)
            ax_volume.set_ylabel('Volume', color='#666666')
            
            # Plot moving averages
            ax_ma.plot(df.index, df[f'SMA_{short_ma}'], 
                      label=f'SMA {short_ma}', color='#2196f3', linewidth=1.5)
            ax_ma.plot(df.index, df[f'SMA_{long_ma}'], 
                      label=f'SMA {long_ma}', color='#ff9800', linewidth=1.5)
            ax_ma.set_ylabel('Moving Averages', color='#666666')
            ax_ma.legend(loc='upper left', framealpha=0.9)
            
            # Plot RSI
            ax_rsi.plot(df.index, df['RSI'], label='RSI', color='#673ab7', linewidth=1.5)
            ax_rsi.axhline(y=70, color='#ef5350', linestyle='--', alpha=0.5)
            ax_rsi.axhline(y=30, color='#26a69a', linestyle='--', alpha=0.5)
            ax_rsi.fill_between(df.index, 70, 100, color='#ef5350', alpha=0.1)
            ax_rsi.fill_between(df.index, 0, 30, color='#26a69a', alpha=0.1)
            ax_rsi.set_ylabel('RSI', color='#666666')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.legend(loc='upper left', framealpha=0.9)
            
            # Format x-axis dates
            for ax in [ax_candle, ax_volume, ax_ma, ax_rsi]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='x', rotation=45)
            
            # Adjust spacing
            self.fig_main.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.3)
            
            # Add timestamp to title
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.fig_main.suptitle(f'{ticker} Technical Analysis\nLast Updated: {current_time}', 
                                 fontsize=12, y=0.98, color='#333333')
            
            # Update canvas
            self.canvas_main.draw()
            
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
            raise e

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def detect_ma_crossovers(self, df, short_period, long_period):
        """Detect moving average crossovers"""
        signals = {}
        short_ma = f'SMA_{short_period}'
        long_ma = f'SMA_{long_period}'  # Fixed variable name
        
        # Previous values
        prev_diff = df[short_ma].shift(1) - df[long_ma].shift(1)
        # Current values
        curr_diff = df[short_ma] - df[long_ma]
        
        # Detect crossovers
        for i in range(1, len(df)):
            if pd.notna(prev_diff.iloc[i]) and pd.notna(curr_diff.iloc[i]):
                if prev_diff.iloc[i] < 0 and curr_diff.iloc[i] > 0:
                    signals[df.index[i]] = 'buy'
                elif prev_diff.iloc[i] > 0 and curr_diff.iloc[i] < 0:
                    signals[df.index[i]] = 'sell'
        
        return signals

    def start_analysis(self):
        """Start the analysis with current settings"""
        ticker = self.ticker_var.get()
        if not ticker:
            messagebox.showerror("Error", "Please select a ticker")
            return
        
        try:
            # Show loading indicator
            self.loading_label.config(text="Loading data...")
            self.root.update()
            
            # Check cache first
            cache_key = f"{ticker}_{self.duration_var.get()}"
            if cache_key in self.data_cache:
                df = self.data_cache[cache_key]
            else:
                # Get data and calculate indicators using the analyzer
                df = self.analyzer.get_historical_data(ticker, self.duration_var.get())
                
                # Limit data based on duration
                duration = self.duration_var.get()
                if df is not None and len(df) > 0:
                    today = pd.Timestamp.now()
                    if duration == '1d':
                        df = df[df.index >= today.floor('D')]
                    elif duration == '5d':
                        df = df[df.index >= today - pd.Timedelta(days=5)]
                    elif duration == '1mo':
                        df = df[df.index >= today - pd.Timedelta(days=30)]
                    elif duration == '3mo':
                        df = df[df.index >= today - pd.Timedelta(days=90)]
                    elif duration == '6mo':
                        df = df[df.index >= today - pd.Timedelta(days=180)]
                    elif duration == '1y':
                        df = df[df.index >= today - pd.Timedelta(days=365)]
                    elif duration == '2y':
                        df = df[df.index >= today - pd.Timedelta(days=730)]
                    elif duration == '5y':
                        df = df[df.index >= today - pd.Timedelta(days=1825)]
                    # 'max' duration doesn't need limiting
                
                self.data_cache[cache_key] = df
            
            # Check if we got any data
            if df is None or len(df) == 0:
                self.loading_label.config(text="")
                messagebox.showwarning("No Data", f"No data available for {ticker} in selected timeframe")
                return
            
            # Check if we have enough data points for analysis
            min_periods = {
                '1d': 5,
                '5d': 5,
                '1mo': 20,
                '3mo': 20,
                '6mo': 50,
                '1y': 50,
                '2y': 100,
                '5y': 100,
                'max': 100
            }.get(self.duration_var.get(), 20)
            
            if len(df) < min_periods:
                self.loading_label.config(text="")
                messagebox.showwarning("Insufficient Data", 
                    f"Need at least {min_periods} data points for analysis. Got {len(df)}")
                return
            
            # Calculate technical indicators
            df = self.analyzer.calculate_technical_indicators(df)
            
            # Update plots and analysis
            self.loading_label.config(text="Updating plots...")
            self.root.update()
            self.update_plots(df, ticker)
            
            self.loading_label.config(text="Updating analysis...")
            self.root.update()
            self.update_analysis(df)
            
            # Clear loading indicator
            self.loading_label.config(text="")
            
        except Exception as e:
            self.loading_label.config(text="")
            print(f"Error in analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            raise e  # For debugging

    def get_tickers(self):
        """Get list of available tickers"""
        query = "SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker"
        return [ticker[0] for ticker in self.analyzer.conn.execute(query).fetchall()]

    def draw_trend_lines(self, ax, df):
        """Draw trend lines on the plot"""
        # Calculate highs and lows
        highs = df['High'].rolling(window=20).max()
        lows = df['Low'].rolling(window=20).min()
        
        # Find local maxima and minima
        high_points = []
        low_points = []
        
        for i in range(20, len(df)-20):
            if highs.iloc[i] == df['High'].iloc[i]:
                high_points.append((df.index[i], df['High'].iloc[i]))
            if lows.iloc[i] == df['Low'].iloc[i]:
                low_points.append((df.index[i], df['Low'].iloc[i]))
        
        # Draw resistance trend line
        if len(high_points) >= 2:
            x_high = mdates.date2num([x[0] for x in high_points])
            y_high = [x[1] for x in high_points]
            z_high = np.polyfit(x_high, y_high, 1)
            p_high = np.poly1d(z_high)
            ax.plot(df.index, p_high(mdates.date2num(df.index)), 
                   'r--', alpha=0.5, label='Resistance')
        
        # Draw support trend line
        if len(low_points) >= 2:
            x_low = mdates.date2num([x[0] for x in low_points])
            y_low = [x[1] for x in low_points]
            z_low = np.polyfit(x_low, y_low, 1)
            p_low = np.poly1d(z_low)
            ax.plot(df.index, p_low(mdates.date2num(df.index)), 
                   'g--', alpha=0.5, label='Support')

    def add_pattern_markers(self, ax, df):
        """Add pattern markers to the plot"""
        patterns = self.analyzer.identify_candlestick_patterns(df)
        for pattern in patterns:
            if pattern['action'] == 'Buy':
                marker = '^'
                color = 'g'
            elif pattern['action'] == 'Sell':
                marker = 'v'
                color = 'r'
            else:
                marker = 'o'
                color = 'y'
            
            size = pattern['strength'] * 15
            ax.plot(pattern['date'], df.loc[pattern['date'], 'Low'] * 0.99,
                   marker=marker, color=color, markersize=size, alpha=0.7,
                   label=f"{pattern['pattern']}")

    def plot_pattern_strength(self, ax, df):
        """Plot pattern strength analysis"""
        patterns = self.analyzer.identify_candlestick_patterns(df)
        if not patterns:
            return
        
        dates = [p['date'] for p in patterns]
        strengths = [p['strength'] for p in patterns]
        actions = [p['action'] for p in patterns]
        
        colors = ['g' if a == 'Buy' else 'r' if a == 'Sell' else 'y' for a in actions]
        
        ax.bar(dates, strengths, color=colors, alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_title('Pattern Strength Analysis')
        ax.tick_params(axis='x', rotation=45)

    def calculate_strategy_signals(self, df, strategy):
        """Calculate trading signals based on selected strategy"""
        signals = {'buy': [], 'sell': []}
        
        if strategy == "MA Crossover":
            # Get MA periods from current settings
            df, short_ma, long_ma = self.calculate_moving_averages(df, self.duration_var.get())
            crossover_signals = self.detect_ma_crossovers(df, short_ma, long_ma)
            signals['buy'] = [date for date, signal in crossover_signals.items() if signal == 'buy']
            signals['sell'] = [date for date, signal in crossover_signals.items() if signal == 'sell']
            
        elif strategy == "RSI":
            df['RSI'] = self.calculate_rsi(df['Close'])
            signals['buy'] = df[df['RSI'] < 30].index
            signals['sell'] = df[df['RSI'] > 70].index
            
        elif strategy == "MACD":
            # Add MACD calculation
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # MACD crossover signals
            signals['buy'] = df[macd > signal].index
            signals['sell'] = df[macd < signal].index
            
        elif strategy == "Bollinger Bands":
            # Calculate Bollinger Bands
            df['SMA'] = df['Close'].rolling(window=20).mean()
            df['STD'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['SMA'] + (df['STD'] * 2)
            df['Lower'] = df['SMA'] - (df['STD'] * 2)
            
            signals['buy'] = df[df['Close'] < df['Lower']].index
            signals['sell'] = df[df['Close'] > df['Upper']].index
        
        return signals

    def calculate_position_sizing(self, current_price, strategy_metrics=None):
        """Calculate optimal position size based on risk management rules"""
        try:
            # Get basic parameters
            capital = float(self.capital_var.get())
            risk_per_trade = float(self.risk_var.get()) / 100
            stop_loss = float(self.stop_loss_var.get()) / 100
            
            # Calculate base position size
            risk_amount = capital * risk_per_trade
            stop_loss_amount = current_price * stop_loss
            base_position_size = risk_amount / stop_loss_amount if stop_loss_amount > 0 else 0
            
            # Calculate max drawdown (default to stop loss if no metrics available)
            max_drawdown = risk_amount  # Default max drawdown is risk amount
            
            if strategy_metrics:
                # Use strategy metrics to adjust position size
                win_rate_factor = strategy_metrics['win_rate'] if strategy_metrics['win_rate'] > 0 else 0.5
                volatility_factor = 1 - abs(strategy_metrics['max_drawdown'])
                sharpe_factor = min(max(strategy_metrics['sharpe_ratio'], 0), 3) / 3
                
                # Calculate adjusted position size
                adjustment_factor = (win_rate_factor + volatility_factor + sharpe_factor) / 3
                adjusted_position_size = base_position_size * adjustment_factor
                
                # Update max drawdown from metrics
                max_drawdown = capital * abs(strategy_metrics['max_drawdown'])
                
                # Apply Kelly criterion
                if strategy_metrics['win_rate'] > 0 and strategy_metrics['avg_return'] > 0:
                    kelly_fraction = (strategy_metrics['win_rate'] - 
                                    ((1 - strategy_metrics['win_rate']) / 
                                     (strategy_metrics['avg_return'] / abs(strategy_metrics['max_drawdown']))))
                    kelly_fraction = max(0, min(kelly_fraction, 1))
                else:
                    kelly_fraction = 0.5
                
                final_position_size = adjusted_position_size * kelly_fraction
            else:
                final_position_size = base_position_size * 0.5  # Conservative sizing without metrics
            
            # Calculate position value and other metrics
            position_value = final_position_size * current_price
            max_position_value = capital * 0.25  # Maximum 25% of capital per position
            
            # Adjust if position value exceeds maximum
            if position_value > max_position_value:
                final_position_size = max_position_value / current_price
                position_value = max_position_value
            
            return {
                'position_size': final_position_size,
                'position_value': position_value,
                'risk_amount': risk_amount,
                'max_position': max_position_value,
                'stop_loss_price': current_price * (1 - stop_loss),
                'risk_reward_ratio': 1 / stop_loss if stop_loss > 0 else float('inf'),
                'capital_at_risk': (position_value / capital) * 100,
                'max_drawdown': max_drawdown  # Added max_drawdown to return dictionary
            }
            
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            return None

    def update_position_info(self, position_data):
        """Update position information display"""
        if not position_data:
            self.position_info.delete(1.0, tk.END)
            self.position_info.insert(tk.END, "Unable to calculate position size")
            return
        
        info_text = "Position Sizing Analysis\n"
        info_text += "=" * 40 + "\n\n"
        
        info_text += f"Shares to Trade: {position_data['position_size']:.2f}\n"
        info_text += f"Position Value: ${position_data['position_value']:.2f}\n"
        info_text += f"Risk Amount: ${position_data['risk_amount']:.2f}\n"
        info_text += f"Stop Loss Price: ${position_data['stop_loss_price']:.2f}\n"
        info_text += f"Risk/Reward Ratio: {position_data['risk_reward_ratio']:.2f}\n"
        info_text += f"Capital at Risk: {position_data['capital_at_risk']:.2f}%\n"
        
        self.position_info.delete(1.0, tk.END)
        self.position_info.insert(tk.END, info_text)

    def calculate_probability_of_ruin(self, df):
        """Calculate probability of ruin with weighted outcomes"""
        try:
            # Get trading parameters
            initial_capital = float(self.capital_var.get())
            risk_per_trade = float(self.risk_var.get()) / 100
            stop_loss = float(self.stop_loss_var.get()) / 100
            
            # Calculate trade statistics
            returns = df['Close'].pct_change().dropna()
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
            
            # Calculate edge (advantage per trade)
            edge = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Calculate average trade size based on current position sizing
            avg_trade_size = initial_capital * risk_per_trade
            
            # Calculate number of trades to ruin
            if edge > 0 and avg_loss > 0:
                n = int(initial_capital / (avg_trade_size * avg_loss))
            else:
                n = 0
            
            # Calculate classic probability of ruin
            try:
                if edge > 0 and win_rate > 0:
                    q = 1 - win_rate
                    p = win_rate
                    classic_ruin = ((q/p) ** n) if p != 0 else 1.0
                    classic_ruin = min(1.0, max(0.0, classic_ruin))  # Bound between 0 and 1
                else:
                    classic_ruin = 1.0
            except:
                classic_ruin = 1.0
            
            # Calculate extended probability of ruin (with Kelly criterion)
            try:
                if edge > 0:
                    extended_ruin = np.exp(-2 * edge * (initial_capital / avg_trade_size))
                    extended_ruin = min(1.0, max(0.0, extended_ruin))  # Bound between 0 and 1
                else:
                    extended_ruin = 1.0
            except:
                extended_ruin = 1.0
            
            # Calculate optimal f with safeguards
            if avg_loss > 0 and avg_win > 0:
                try:
                    optimal_f = (win_rate - ((1 - win_rate) / (avg_win/avg_loss)))
                    optimal_f = max(0, min(1, optimal_f))  # Bound between 0 and 1
                except:
                    optimal_f = 0
            else:
                optimal_f = 0
            
            # Create probability table with different weights
            weights = {
                'conservative': {
                    'win_rate': 0.4,
                    'edge': 0.3,
                    'capital': 0.3
                },
                'moderate': {
                    'win_rate': 0.33,
                    'edge': 0.34,
                    'capital': 0.33
                },
                'aggressive': {
                    'win_rate': 0.3,
                    'edge': 0.4,
                    'capital': 0.3
                }
            }
            
            weighted_outcomes = {}
            for style, weight in weights.items():
                weighted_ruin = (
                    classic_ruin * weight['win_rate'] +
                    extended_ruin * weight['edge'] +
                    (1 - optimal_f) * weight['capital']
                ) / 3
                weighted_outcomes[style] = round(weighted_ruin * 100, 2)
            
            return {
                'classic_ruin_probability': classic_ruin,
                'extended_ruin_probability': extended_ruin,
                'win_rate': win_rate,
                'edge_per_trade': edge,
                'optimal_f': optimal_f,
                'weighted_outcomes': weighted_outcomes,
                'risk_metrics': {
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'win_loss_ratio': avg_win/avg_loss if avg_loss > 0 else float('inf'),
                    'trades_to_ruin': n
                }
            }
            
        except Exception as e:
            print(f"Error calculating probability of ruin: {str(e)}")
            return None

    def update_probability_display(self, ruin_analysis):
        """Format probability analysis results into display text"""
        if not ruin_analysis:
            return "Probability analysis not available\n"
        
        try:
            text = "Probability Analysis:\n"
            text += "-" * 40 + "\n"
            
            # Classic probability of ruin
            text += f"Classic Ruin Probability: {ruin_analysis['classic_ruin_probability']:.2%}\n"
            
            # Extended probability of ruin
            text += f"Extended Ruin Probability: {ruin_analysis['extended_ruin_probability']:.2%}\n"
            
            # Win rate and edge
            text += f"Win Rate: {ruin_analysis['win_rate']:.2%}\n"
            text += f"Edge per Trade: {ruin_analysis['edge_per_trade']:.2%}\n"
            
            # Optimal f
            text += f"Optimal Position Size (f): {ruin_analysis['optimal_f']:.2%}\n"
            
            # Risk metrics
            text += "\nRisk Metrics:\n"
            text += f"Average Win: {ruin_analysis['risk_metrics']['avg_win']:.2%}\n"
            text += f"Average Loss: {ruin_analysis['risk_metrics']['avg_loss']:.2%}\n"
            text += f"Win/Loss Ratio: {ruin_analysis['risk_metrics']['win_loss_ratio']:.2f}\n"
            text += f"Trades to Ruin: {ruin_analysis['risk_metrics']['trades_to_ruin']}\n"
            
            # Weighted outcomes
            text += "\nWeighted Outcomes:\n"
            for style, prob in ruin_analysis['weighted_outcomes'].items():
                text += f"{style.capitalize()}: {prob:.2f}%\n"
            
            return text
            
        except Exception as e:
            print(f"Error formatting probability display: {str(e)}")
            return "Error displaying probability analysis\n"

    def run(self):
        try:
            self.root.mainloop()
        finally:
            plt.close('all')  # Ensure all plots are closed when the app exits

    def update_account_settings(self):
        """Update account settings with user input"""
        try:
            # Update analyzer account settings
            self.analyzer.account['balance'] = float(self.balance_var.get())
            self.analyzer.account['risk_per_trade'] = float(self.risk_var.get()) / 100
            self.analyzer.account['max_position_size'] = float(self.position_var.get()) / 100
            
            # Reset peak balance if initial balance changes
            self.analyzer.account['peak_balance'] = float(self.balance_var.get())
            self.analyzer.account['max_drawdown'] = 0.0
            
            messagebox.showinfo("Success", "Account settings updated successfully!")
            
            # If we have current analysis, update it
            if hasattr(self, 'current_df') and self.current_df is not None:
                self.update_analysis(self.current_df)
                
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numbers for all fields")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update account settings: {str(e)}")

    def analyze_patterns(self, df):
        """Analyze candlestick patterns and market patterns"""
        try:
            if df is None or df.empty:
                return "No data available for pattern analysis"
            
            analysis = "Pattern Analysis:\n"
            analysis += "=" * 40 + "\n\n"
            
            # Candlestick Patterns
            patterns = []
            
            # Bullish patterns
            if self.is_bullish_engulfing(df):
                patterns.append(("Bullish Engulfing", "Strong bullish reversal signal"))
            if self.is_hammer(df):
                patterns.append(("Hammer", "Potential bullish reversal"))
            if self.is_morning_star(df):
                patterns.append(("Morning Star", "Strong bullish reversal pattern"))
            
            # Bearish patterns
            if self.is_bearish_engulfing(df):
                patterns.append(("Bearish Engulfing", "Strong bearish reversal signal"))
            if self.is_shooting_star(df):
                patterns.append(("Shooting Star", "Potential bearish reversal"))
            if self.is_evening_star(df):
                patterns.append(("Evening Star", "Strong bearish reversal pattern"))
            
            # Continuation patterns
            if self.is_doji(df):
                patterns.append(("Doji", "Market indecision/potential reversal"))
            
            # Add identified patterns to analysis
            if patterns:
                analysis += "Candlestick Patterns:\n"
                for pattern, description in patterns:
                    analysis += f"- {pattern}: {description}\n"
                analysis += "\n"
            
            # Trend Analysis
            trend = self.analyze_trend(df)
            analysis += "Trend Analysis:\n"
            analysis += f"- {trend['description']}\n"
            analysis += f"- Strength: {trend['strength']:.2%}\n"
            analysis += f"- Duration: {trend['duration']} days\n\n"
            
            # Support/Resistance Levels
            levels = self.find_support_resistance(df)
            analysis += "Support/Resistance Levels:\n"
            current_price = df['Close'].iloc[-1]
            
            for level_type, level in levels.items():
                distance = abs(current_price - level) / current_price * 100
                analysis += f"- {level_type}: ${level:.2f} ({distance:.1f}% from current price)\n"
            
            analysis += "\n"
            
            # Volume Analysis
            volume_analysis = self.analyze_volume(df)
            analysis += "Volume Analysis:\n"
            analysis += f"- {volume_analysis['description']}\n"
            analysis += f"- Trend: {volume_analysis['trend']}\n"
            analysis += f"- Strength: {volume_analysis['strength']:.2%}\n\n"
            
            # Pattern Strength and Reliability
            reliability = self.calculate_pattern_reliability(df, patterns)
            analysis += "Pattern Reliability:\n"
            analysis += f"- Overall Strength: {reliability['strength']:.2%}\n"
            analysis += f"- Confidence Level: {reliability['confidence']}\n"
            analysis += f"- Historical Accuracy: {reliability['accuracy']:.2%}\n"
            
            return analysis
            
        except Exception as e:
            print(f"Error in pattern analysis: {str(e)}")
            return "Error performing pattern analysis"

    def is_bullish_engulfing(self, df):
        """Check for bullish engulfing pattern"""
        try:
            for i in range(1, len(df)):
                prev_day = df.iloc[i-1]
                current_day = df.iloc[i]
                
                if (prev_day['Close'] < prev_day['Open'] and  # Previous day is bearish
                    current_day['Close'] > current_day['Open'] and  # Current day is bullish
                    current_day['Open'] < prev_day['Close'] and  # Opens below previous close
                    current_day['Close'] > prev_day['Open']):  # Closes above previous open
                    return True
            return False
        except Exception as e:
            print(f"Error checking bullish engulfing: {str(e)}")
            return False

    def is_bearish_engulfing(self, df):
        """Check for bearish engulfing pattern"""
        try:
            for i in range(1, len(df)):
                prev_day = df.iloc[i-1]
                current_day = df.iloc[i]
                
                if (prev_day['Close'] > prev_day['Open'] and  # Previous day is bullish
                    current_day['Close'] < current_day['Open'] and  # Current day is bearish
                    current_day['Open'] > prev_day['Close'] and  # Opens above previous close
                    current_day['Close'] < prev_day['Open']):  # Closes below previous open
                    return True
            return False
        except Exception as e:
            print(f"Error checking bearish engulfing: {str(e)}")
            return False

    def analyze_trend(self, df):
        """Analyze price trend"""
        try:
            # Calculate short and long term trends
            short_ma = df['Close'].rolling(window=20).mean()
            long_ma = df['Close'].rolling(window=50).mean()
            
            # Calculate trend strength
            returns = df['Close'].pct_change()
            trend_strength = abs(returns.mean()) * np.sqrt(252)
            
            # Determine trend duration
            current_trend = 0
            for i in range(len(df)-1, 0, -1):
                if (df['Close'].iloc[i] > df['Close'].iloc[i-1] and 
                    df['Close'].iloc[i-1] > df['Close'].iloc[i-2]):
                    current_trend += 1
                else:
                    break
            
            # Determine trend type
            if short_ma.iloc[-1] > long_ma.iloc[-1]:
                trend_type = "Uptrend"
            elif short_ma.iloc[-1] < long_ma.iloc[-1]:
                trend_type = "Downtrend"
            else:
                trend_type = "Sideways"
            
            # Create trend description
            if trend_strength > 0.2:
                strength_desc = "Strong"
            elif trend_strength > 0.1:
                strength_desc = "Moderate"
            else:
                strength_desc = "Weak"
            
            description = f"{strength_desc} {trend_type}"
            
            return {
                'description': description,
                'strength': trend_strength,
                'duration': current_trend
            }
            
        except Exception as e:
            print(f"Error analyzing trend: {str(e)}")
            return {'description': "Error analyzing trend", 'strength': 0, 'duration': 0}

    def find_support_resistance(self, df):
        """Find support and resistance levels"""
        try:
            # Calculate pivot points
            pivot = (df['High'] + df['Low'] + df['Close']) / 3
            support1 = 2 * pivot - df['High']
            resistance1 = 2 * pivot - df['Low']
            
            current_price = df['Close'].iloc[-1]
            
            # Find closest support and resistance
            support_level = support1[support1 < current_price].max()
            resistance_level = resistance1[resistance1 > current_price].min()
            
            return {
                'Support': support_level,
                'Resistance': resistance_level
            }
            
        except Exception as e:
            print(f"Error finding support/resistance: {str(e)}")
            return {'Support': 0, 'Resistance': 0}

    def analyze_volume(self, df):
        """Analyze volume patterns"""
        try:
            # Calculate volume trend
            volume_ma = df['Volume'].rolling(window=20).mean()
            current_volume = df['Volume'].iloc[-1]
            
            # Calculate volume strength
            volume_strength = current_volume / volume_ma.mean()
            
            # Determine volume trend
            if current_volume > volume_ma.iloc[-1]:
                trend = "Increasing"
                if volume_strength > 1.5:
                    description = "Strong volume expansion"
                else:
                    description = "Moderate volume increase"
            else:
                trend = "Decreasing"
                if volume_strength < 0.5:
                    description = "Significant volume contraction"
                else:
                    description = "Moderate volume decrease"
            
            return {
                'description': description,
                'trend': trend,
                'strength': volume_strength
            }
            
        except Exception as e:
            print(f"Error analyzing volume: {str(e)}")
            return {'description': "Error", 'trend': "Unknown", 'strength': 0}

    def calculate_pattern_reliability(self, df, patterns):
        """Calculate the reliability of identified patterns"""
        try:
            if not patterns:
                return {'strength': 0, 'confidence': "Low", 'accuracy': 0}
            
            # Calculate pattern strength based on volume and price action
            volume_weight = 0.4
            price_weight = 0.6
            
            # Volume confirmation
            volume_confirm = df['Volume'].iloc[-1] > df['Volume'].rolling(window=20).mean().iloc[-1]
            
            # Price confirmation
            price_confirm = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1]) > df['Close'].std()
            
            # Calculate overall strength
            strength = (volume_confirm * volume_weight + price_confirm * price_weight)
            
            # Determine confidence level
            if strength > 0.8:
                confidence = "High"
            elif strength > 0.5:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Calculate historical accuracy (simplified)
            accuracy = len([p for p in patterns if p[0] in ['Bullish Engulfing', 'Bearish Engulfing']]) / len(patterns) if patterns else 0
            
            return {
                'strength': strength,
                'confidence': confidence,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"Error calculating pattern reliability: {str(e)}")
            return {'strength': 0, 'confidence': "Low", 'accuracy': 0}

    def is_hammer(self, df):
        """Check for hammer candlestick pattern"""
        try:
            for i in range(1, len(df)):
                current = df.iloc[i]
                
                # Calculate body and shadows
                body = abs(current['Open'] - current['Close'])
                upper_shadow = current['High'] - max(current['Open'], current['Close'])
                lower_shadow = min(current['Open'], current['Close']) - current['Low']
                
                # Hammer criteria:
                # 1. Small body
                # 2. Long lower shadow (2-3 times body)
                # 3. Very small or no upper shadow
                # 4. Appears in downtrend
                
                if (body > 0 and  # Ensure there is a body
                    lower_shadow > (2 * body) and  # Lower shadow 2-3 times the body
                    upper_shadow < (0.1 * body) and  # Very small upper shadow
                    df['Close'].iloc[i-5:i].mean() > current['Close']):  # In downtrend
                    return True
            return False
        except Exception as e:
            print(f"Error checking hammer pattern: {str(e)}")
            return False

    def is_shooting_star(self, df):
        """Check for shooting star candlestick pattern"""
        try:
            for i in range(1, len(df)):
                current = df.iloc[i]
                
                # Calculate body and shadows
                body = abs(current['Open'] - current['Close'])
                upper_shadow = current['High'] - max(current['Open'], current['Close'])
                lower_shadow = min(current['Open'], current['Close']) - current['Low']
                
                # Shooting Star criteria:
                # 1. Small body
                # 2. Long upper shadow (2-3 times body)
                # 3. Very small or no lower shadow
                # 4. Appears in uptrend
                
                if (body > 0 and  # Ensure there is a body
                    upper_shadow > (2 * body) and  # Upper shadow 2-3 times the body
                    lower_shadow < (0.1 * body) and  # Very small lower shadow
                    df['Close'].iloc[i-5:i].mean() < current['Close']):  # In uptrend
                    return True
            return False
        except Exception as e:
            print(f"Error checking shooting star pattern: {str(e)}")
            return False

    def is_doji(self, df):
        """Check for doji candlestick pattern"""
        try:
            for i in range(1, len(df)):
                current = df.iloc[i]
                
                # Calculate body and shadows
                body = abs(current['Open'] - current['Close'])
                total_range = current['High'] - current['Low']
                
                # Doji criteria:
                # 1. Very small body (less than 10% of total range)
                # 2. Upper and lower shadows should be significant
                
                if (total_range > 0 and  # Ensure there is a range
                    body <= (0.1 * total_range)):  # Body is very small compared to range
                    return True
            return False
        except Exception as e:
            print(f"Error checking doji pattern: {str(e)}")
            return False

    def is_morning_star(self, df):
        """Check for morning star candlestick pattern"""
        try:
            for i in range(2, len(df)):
                first = df.iloc[i-2]  # First day
                second = df.iloc[i-1]  # Second day
                third = df.iloc[i]    # Third day
                
                # Morning Star criteria:
                # 1. First day is bearish (long black body)
                # 2. Second day gaps down and has small body
                # 3. Third day is bullish and closes above midpoint of first day
                
                first_body = first['Open'] - first['Close']  # Bearish so open > close
                second_body = abs(second['Open'] - second['Close'])
                third_body = third['Close'] - third['Open']  # Bullish so close > open
                
                if (first_body > 0 and  # First day is bearish
                    second_body < (0.3 * first_body) and  # Second day has small body
                    second['High'] < first['Close'] and  # Gap down
                    third_body > 0 and  # Third day is bullish
                    third['Close'] > (first['Open'] + first['Close']) / 2):  # Closes above midpoint
                    return True
            return False
        except Exception as e:
            print(f"Error checking morning star pattern: {str(e)}")
            return False

    def is_evening_star(self, df):
        """Check for evening star candlestick pattern"""
        try:
            for i in range(2, len(df)):
                first = df.iloc[i-2]  # First day
                second = df.iloc[i-1]  # Second day
                third = df.iloc[i]    # Third day
                
                # Evening Star criteria:
                # 1. First day is bullish (long white body)
                # 2. Second day gaps up and has small body
                # 3. Third day is bearish and closes below midpoint of first day
                
                first_body = first['Close'] - first['Open']  # Bullish so close > open
                second_body = abs(second['Open'] - second['Close'])
                third_body = third['Open'] - third['Close']  # Bearish so open > close
                
                if (first_body > 0 and  # First day is bullish
                    second_body < (0.3 * first_body) and  # Second day has small body
                    second['Low'] > first['Close'] and  # Gap up
                    third_body > 0 and  # Third day is bearish
                    third['Close'] < (first['Open'] + first['Close']) / 2):  # Closes below midpoint
                    return True
            return False
        except Exception as e:
            print(f"Error checking evening star pattern: {str(e)}")
            return False

    def analyze_probability(self, df):
        """Analyze probability distributions and statistics"""
        try:
            if df is None or df.empty:
                return "No data available for probability analysis"
            
            analysis = "Probability Analysis:\n"
            analysis += "=" * 40 + "\n\n"
            
            # Calculate returns and statistics
            returns = df['Close'].pct_change().dropna()
            
            # Basic Statistics
            mean_return = returns.mean()
            std_return = returns.std()
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            analysis += "Return Statistics:\n"
            analysis += f"Mean Daily Return: {mean_return:.2%}\n"
            analysis += f"Standard Deviation: {std_return:.2%}\n"
            analysis += f"Annualized Volatility: {std_return * np.sqrt(252):.2%}\n"
            analysis += f"Skewness: {skewness:.2f}\n"
            analysis += f"Kurtosis: {kurtosis:.2f}\n\n"
            
            # Calculate Value at Risk (VaR)
            confidence_levels = [0.99, 0.95, 0.90]
            current_price = df['Close'].iloc[-1]
            
            analysis += "Value at Risk (VaR):\n"
            for confidence in confidence_levels:
                var = np.percentile(returns, (1 - confidence) * 100)
                dollar_var = current_price * abs(var)
                analysis += f"{confidence:.0%} VaR: ${dollar_var:.2f} ({abs(var):.2%})\n"
            analysis += "\n"
            
            # Calculate win/loss probability
            wins = len(returns[returns > 0])
            losses = len(returns[returns < 0])
            total_trades = len(returns)
            
            win_prob = wins / total_trades
            loss_prob = losses / total_trades
            
            analysis += "Win/Loss Probability:\n"
            analysis += f"Win Probability: {win_prob:.2%}\n"
            analysis += f"Loss Probability: {loss_prob:.2%}\n"
            analysis += f"Win/Loss Ratio: {win_prob/loss_prob:.2f}\n\n"
            
            # Calculate expected values
            avg_win = returns[returns > 0].mean()
            avg_loss = returns[returns < 0].mean()
            
            analysis += "Expected Values:\n"
            analysis += f"Average Win: {avg_win:.2%}\n"
            analysis += f"Average Loss: {avg_loss:.2%}\n"
            analysis += f"Expected Value: {(win_prob * avg_win + loss_prob * avg_loss):.2%}\n\n"
            
            # Calculate probability of reaching targets
            targets = [0.05, 0.10, 0.15]  # 5%, 10%, 15% returns
            days_forward = 20  # Looking forward 20 trading days
            
            analysis += "Probability of Reaching Targets (20 days):\n"
            for target in targets:
                z_score = (target - (mean_return * days_forward)) / (std_return * np.sqrt(days_forward))
                probability = 1 - stats.norm.cdf(z_score)
                analysis += f"+{target:.0%} Target: {probability:.1%}\n"
            
            # Calculate drawdown statistics
            rolling_max = df['Close'].expanding().max()
            drawdowns = (df['Close'] - rolling_max) / rolling_max
            
            analysis += "\nDrawdown Analysis:\n"
            analysis += f"Maximum Drawdown: {drawdowns.min():.2%}\n"
            analysis += f"Average Drawdown: {drawdowns.mean():.2%}\n"
            analysis += f"Drawdown Std Dev: {drawdowns.std():.2%}\n"
            
            # Calculate recovery probabilities
            current_drawdown = (df['Close'].iloc[-1] / rolling_max.iloc[-1] - 1)
            if current_drawdown < 0:
                analysis += f"Current Drawdown: {current_drawdown:.2%}\n"
                recovery_target = rolling_max.iloc[-1]
                days_to_recover = self.estimate_recovery_time(
                    current_price, 
                    recovery_target, 
                    mean_return, 
                    std_return
                )
                analysis += f"Estimated Recovery Time: {days_to_recover:.0f} trading days\n"
            
            # Calculate momentum probabilities
            momentum = self.calculate_momentum_probability(df)
            analysis += "\nMomentum Analysis:\n"
            analysis += f"Trend Strength: {momentum['strength']:.2%}\n"
            analysis += f"Trend Continuation Probability: {momentum['continuation_prob']:.1%}\n"
            analysis += f"Reversal Probability: {momentum['reversal_prob']:.1%}\n"
            
            return analysis
            
        except Exception as e:
            print(f"Error in probability analysis: {str(e)}")
            return "Error performing probability analysis"

    def estimate_recovery_time(self, current_price, target_price, mean_return, std_return):
        """Estimate the expected number of days to recover to a target price"""
        try:
            if current_price >= target_price:
                return 0
            
            # Calculate required return
            required_return = np.log(target_price / current_price)
            
            # Estimate time using continuous time random walk model
            time_estimate = required_return / (mean_return + 0.5 * std_return ** 2)
            
            return max(0, time_estimate)
            
        except Exception as e:
            print(f"Error estimating recovery time: {str(e)}")
            return float('inf')

    def calculate_momentum_probability(self, df):
        """Calculate momentum-based probabilities"""
        try:
            # Calculate returns and moving averages
            returns = df['Close'].pct_change()
            ma20 = df['Close'].rolling(window=20).mean()
            ma50 = df['Close'].rolling(window=50).mean()
            
            # Calculate trend strength
            current_price = df['Close'].iloc[-1]
            strength = abs(current_price - ma50.iloc[-1]) / ma50.iloc[-1]
            
            # Calculate directional movement
            direction = 1 if current_price > ma50.iloc[-1] else -1
            
            # Calculate probabilities based on historical patterns
            if direction > 0:
                # In uptrend
                continuation_prob = len(returns[(returns > 0) & (df['Close'] > ma20)]) / len(returns)
                reversal_prob = 1 - continuation_prob
            else:
                # In downtrend
                continuation_prob = len(returns[(returns < 0) & (df['Close'] < ma20)]) / len(returns)
                reversal_prob = 1 - continuation_prob
            
            return {
                'strength': strength,
                'continuation_prob': continuation_prob,
                'reversal_prob': reversal_prob
            }
            
        except Exception as e:
            print(f"Error calculating momentum probability: {str(e)}")
            return {'strength': 0, 'continuation_prob': 0.5, 'reversal_prob': 0.5}

    def analyze_returns_scenarios(self, df):
        """Analyze returns under different trading scenarios"""
        try:
            if df is None or df.empty:
                return "No data available for returns analysis"
            
            analysis = "Returns Scenario Analysis:\n"
            analysis += "=" * 40 + "\n\n"
            
            # Calculate basic returns
            returns = df['Close'].pct_change().dropna()
            current_price = df['Close'].iloc[-1]
            initial_price = df['Close'].iloc[0]
            
            # Buy and Hold Analysis
            hold_return = (current_price - initial_price) / initial_price
            annual_return = (1 + hold_return) ** (252 / len(df)) - 1
            
            analysis += "Buy and Hold Strategy:\n"
            analysis += f"Total Return: {hold_return:.2%}\n"
            analysis += f"Annualized Return: {annual_return:.2%}\n"
            
            # Calculate Sharpe Ratio
            risk_free_rate = 0.04  # Assume 4% risk-free rate
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
            
            analysis += f"Sharpe Ratio: {sharpe_ratio:.2f}\n\n"
            
            # Trading Scenarios
            analysis += "Trading Scenarios:\n"
            
            # Scenario 1: Regular Intervals
            interval_returns = self.calculate_interval_trading(df)
            analysis += "\nRegular Interval Trading:\n"
            analysis += f"Monthly Rebalance: {interval_returns['monthly']:.2%}\n"
            analysis += f"Weekly Rebalance: {interval_returns['weekly']:.2%}\n"
            analysis += f"Best Interval: {interval_returns['best_interval']} days\n"
            
            # Scenario 2: Moving Average Crossover
            ma_returns = self.calculate_ma_trading(df)
            analysis += "\nMoving Average Strategy:\n"
            analysis += f"MA Crossover Return: {ma_returns['total_return']:.2%}\n"
            analysis += f"Number of Trades: {ma_returns['num_trades']}\n"
            analysis += f"Win Rate: {ma_returns['win_rate']:.2%}\n"
            
            # Scenario 3: RSI Strategy
            rsi_returns = self.calculate_rsi_trading(df)
            analysis += "\nRSI Strategy:\n"
            analysis += f"RSI Return: {rsi_returns['total_return']:.2%}\n"
            analysis += f"Number of Trades: {rsi_returns['num_trades']}\n"
            analysis += f"Win Rate: {rsi_returns['win_rate']:.2%}\n"
            
            # Risk Metrics
            analysis += "\nRisk Metrics:\n"
            
            # Maximum Drawdown
            rolling_max = df['Close'].expanding().max()
            drawdowns = (df['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Sortino Ratio
            negative_returns = returns[returns < 0]
            downside_std = np.sqrt(252) * negative_returns.std()
            sortino_ratio = (annual_return - risk_free_rate) / downside_std
            
            analysis += f"Maximum Drawdown: {max_drawdown:.2%}\n"
            analysis += f"Sortino Ratio: {sortino_ratio:.2f}\n"
            
            # Position Sizing Scenarios
            analysis += "\nPosition Sizing Scenarios:\n"
            position_sizes = [0.25, 0.50, 0.75, 1.0]  # Percentage of capital
            
            for size in position_sizes:
                scenario = self.calculate_position_scenario(df, size)
                analysis += f"\n{size*100:.0f}% Position Size:\n"
                analysis += f"Expected Return: {scenario['expected_return']:.2%}\n"
                analysis += f"Maximum Risk: {scenario['max_risk']:.2%}\n"
                analysis += f"Risk-Adjusted Return: {scenario['risk_adjusted_return']:.2f}\n"
            
            return analysis
            
        except Exception as e:
            print(f"Error in returns scenario analysis: {str(e)}")
            return "Error performing returns analysis"

    def calculate_interval_trading(self, df):
        """Calculate returns for interval-based trading"""
        try:
            returns = {}
            
            # Monthly rebalancing
            monthly_returns = []
            for i in range(0, len(df), 21):  # Approximate trading days in a month
                if i + 21 < len(df):
                    ret = (df['Close'].iloc[i+21] - df['Close'].iloc[i]) / df['Close'].iloc[i]
                    monthly_returns.append(ret)
            returns['monthly'] = np.mean(monthly_returns) if monthly_returns else 0
            
            # Weekly rebalancing
            weekly_returns = []
            for i in range(0, len(df), 5):  # Trading days in a week
                if i + 5 < len(df):
                    ret = (df['Close'].iloc[i+5] - df['Close'].iloc[i]) / df['Close'].iloc[i]
                    weekly_returns.append(ret)
            returns['weekly'] = np.mean(weekly_returns) if weekly_returns else 0
            
            # Find best interval
            best_return = -np.inf
            best_interval = 0
            for interval in range(3, 30):
                interval_returns = []
                for i in range(0, len(df), interval):
                    if i + interval < len(df):
                        ret = (df['Close'].iloc[i+interval] - df['Close'].iloc[i]) / df['Close'].iloc[i]
                        interval_returns.append(ret)
                avg_return = np.mean(interval_returns) if interval_returns else -np.inf
                if avg_return > best_return:
                    best_return = avg_return
                    best_interval = interval
            
            returns['best_interval'] = best_interval
            
            return returns
            
        except Exception as e:
            print(f"Error calculating interval trading: {str(e)}")
            return {'monthly': 0, 'weekly': 0, 'best_interval': 0}

    def calculate_ma_trading(self, df):
        """Calculate returns for moving average crossover strategy"""
        try:
            # Calculate moving averages
            ma_short = df['Close'].rolling(window=20).mean()
            ma_long = df['Close'].rolling(window=50).mean()
            
            # Generate signals
            signals = pd.Series(0, index=df.index)
            signals[ma_short > ma_long] = 1  # Buy signal
            signals[ma_short < ma_long] = -1  # Sell signal
            
            # Calculate returns
            returns = df['Close'].pct_change()
            strategy_returns = signals.shift(1) * returns
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            trades = signals.diff().abs().sum() / 2
            winning_trades = len(strategy_returns[strategy_returns > 0])
            
            return {
                'total_return': total_return,
                'num_trades': trades,
                'win_rate': winning_trades / trades if trades > 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculating MA trading: {str(e)}")
            return {'total_return': 0, 'num_trades': 0, 'win_rate': 0}

    def calculate_rsi_trading(self, df):
        """Calculate returns for RSI strategy"""
        try:
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Generate signals
            signals = pd.Series(0, index=df.index)
            signals[rsi < 30] = 1  # Oversold - Buy
            signals[rsi > 70] = -1  # Overbought - Sell
            
            # Calculate returns
            returns = df['Close'].pct_change()
            strategy_returns = signals.shift(1) * returns
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            trades = signals.diff().abs().sum() / 2
            winning_trades = len(strategy_returns[strategy_returns > 0])
            
            return {
                'total_return': total_return,
                'num_trades': trades,
                'win_rate': winning_trades / trades if trades > 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculating RSI trading: {str(e)}")
            return {'total_return': 0, 'num_trades': 0, 'win_rate': 0}

    def calculate_position_scenario(self, df, position_size):
        """Calculate returns for different position sizes"""
        try:
            returns = df['Close'].pct_change().dropna()
            
            # Calculate expected return
            expected_return = returns.mean() * 252 * position_size
            
            # Calculate maximum risk
            var_99 = np.percentile(returns, 1)
            max_risk = abs(var_99) * position_size
            
            # Calculate risk-adjusted return
            risk_adjusted_return = expected_return / max_risk if max_risk != 0 else 0
            
            return {
                'expected_return': expected_return,
                'max_risk': max_risk,
                'risk_adjusted_return': risk_adjusted_return
            }
            
        except Exception as e:
            print(f"Error calculating position scenario: {str(e)}")
            return {'expected_return': 0, 'max_risk': 0, 'risk_adjusted_return': 0}

    def plot_data(self, df):
        """Plot stock data with projections and simulations"""
        try:
            if df is None or df.empty:
                return
            
            # Create figure with secondary y-axis
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            
            # Price and pattern subplot
            ax1 = plt.subplot(gs[0])
            ax1v = ax1.twinx()  # Volume overlay
            
            # Volume subplot
            ax2 = plt.subplot(gs[1], sharex=ax1)
            
            # Plot candlestick chart
            candlestick_ohlc(ax1, df[['Date', 'Open', 'High', 'Low', 'Close']].values,
                             width=0.6, colorup='green', colordown='red', alpha=0.8)
            
            # Plot volume bars
            ax1v.fill_between(df.index, df['Volume'], 0, alpha=0.3, color='lightgray')
            
            # Add Moving Averages with labels
            ma20 = df['Close'].rolling(window=20).mean()
            ma50 = df['Close'].rolling(window=50).mean()
            ma200 = df['Close'].rolling(window=200).mean()
            
            ax1.plot(df.index, ma20, label='MA20', color='blue', alpha=0.7)
            ax1.plot(df.index, ma50, label='MA50', color='orange', alpha=0.7)
            ax1.plot(df.index, ma200, label='MA200', color='red', alpha=0.7)
            
            # Add trend analysis and labels
            self.add_trend_analysis(df, ax1)
            
            # Mark patterns with labels
            self.mark_patterns_with_labels(df, ax1)
            
            # Add support and resistance levels with labels
            self.add_support_resistance_labels(df, ax1)
            
            # Plot RSI
            rsi = self.calculate_rsi(df)
            ax2.plot(df.index, rsi, color='purple', label='RSI')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            
            # Customize appearance
            ax1.set_title(f'{self.ticker_var.get()} Stock Price Analysis', fontsize=12, pad=20)
            ax1.set_ylabel('Price ($)', fontsize=10)
            ax1v.set_ylabel('Volume', fontsize=10)
            ax2.set_ylabel('RSI', fontsize=10)
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # Add legends
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.1))
            
            # Adjust layout
            plt.tight_layout()
            
            # Add projections and simulations
            self.add_projections_and_simulations(df, ax1)
            
            # Save and display plot
            plot_file = f'{self.ticker_var.get()}_analysis.png'
            plt.savefig(plot_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Update plot in GUI
            self.update_plot_display(plot_file)
            
        except Exception as e:
            print(f"Error plotting data: {str(e)}")

    def add_trend_analysis(self, df, ax):
        """Add trend analysis with labels"""
        try:
            # Calculate trend metrics
            current_price = df['Close'].iloc[-1]
            ma20_current = df['Close'].rolling(window=20).mean().iloc[-1]
            ma50_current = df['Close'].rolling(window=50).mean().iloc[-1]
            
            # Determine trend direction and strength
            short_trend = "Bullish" if current_price > ma20_current else "Bearish"
            medium_trend = "Bullish" if ma20_current > ma50_current else "Bearish"
            
            # Calculate trend strength
            trend_strength = abs((current_price - ma50_current) / ma50_current)
            strength_label = "Strong" if trend_strength > 0.05 else "Moderate" if trend_strength > 0.02 else "Weak"
            
            # Add trend labels
            y_pos = ax.get_ylim()[1]
            x_pos = df.index[-1]
            
            # Add trend annotations
            ax.annotate(f'Short-term Trend: {short_trend} ({strength_label})',
                       xy=(x_pos, y_pos),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                       fontsize=9)
            
            ax.annotate(f'Medium-term Trend: {medium_trend}',
                       xy=(x_pos, y_pos * 0.98),
                       xytext=(10, -10), textcoords='offset points',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                       fontsize=9)
            
            # Draw trend channels
            self.draw_trend_channels(df, ax)
            
        except Exception as e:
            print(f"Error adding trend analysis: {str(e)}")

    def draw_trend_channels(self, df, ax):
        """Draw trend channels with labels"""
        try:
            # Calculate upper and lower trend lines
            highs = df['High'].rolling(window=5).max()
            lows = df['Low'].rolling(window=5).min()
            
            # Linear regression for trend lines
            x = np.arange(len(df))
            
            # Upper channel
            z = np.polyfit(x, highs, 1)
            upper_line = np.poly1d(z)(x)
            
            # Lower channel
            z = np.polyfit(x, lows, 1)
            lower_line = np.poly1d(z)(x)
            
            # Plot channels
            ax.plot(df.index, upper_line, '--', color='gray', alpha=0.5, label='Upper Channel')
            ax.plot(df.index, lower_line, '--', color='gray', alpha=0.5, label='Lower Channel')
            
            # Add channel width label
            channel_width = ((upper_line[-1] - lower_line[-1]) / lower_line[-1]) * 100
            ax.annotate(f'Channel Width: {channel_width:.1f}%',
                       xy=(df.index[-1], upper_line[-1]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                       fontsize=9)
            
        except Exception as e:
            print(f"Error drawing trend channels: {str(e)}")

    def mark_patterns_with_labels(self, df, ax):
        """Mark patterns with descriptive labels"""
        try:
            patterns_found = []
            
            # Check for patterns
            for i in range(len(df)-1, max(len(df)-10, 0), -1):  # Look at last 10 candles
                current = df.iloc[i]
                
                # Pattern checks with labels
                if self.is_hammer(df.iloc[i-1:i+1]):
                    patterns_found.append({
                        'type': 'Hammer',
                        'position': (df.index[i], current['Low']),
                        'description': 'Potential reversal signal',
                        'color': 'green'
                    })
                
                if self.is_shooting_star(df.iloc[i-1:i+1]):
                    patterns_found.append({
                        'type': 'Shooting Star',
                        'position': (df.index[i], current['High']),
                        'description': 'Bearish reversal signal',
                        'color': 'red'
                    })
                
                if self.is_doji(df.iloc[i-1:i+1]):
                    patterns_found.append({
                        'type': 'Doji',
                        'position': (df.index[i], current['Close']),
                        'description': 'Market indecision',
                        'color': 'blue'
                    })
            
            # Add pattern markers and labels
            for pattern in patterns_found:
                ax.plot(pattern['position'][0], pattern['position'][1], 'o',
                       color=pattern['color'], markersize=8)
                ax.annotate(f"{pattern['type']}\n{pattern['description']}",
                           xy=pattern['position'],
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                           fontsize=8)
            
        except Exception as e:
            print(f"Error marking patterns: {str(e)}")

    def add_support_resistance_labels(self, df, ax):
        """Add support and resistance levels with labels"""
        try:
            levels = self.find_support_resistance(df)
            
            for level_type, level in levels.items():
                ax.axhline(y=level, color='purple', linestyle='--', alpha=0.5)
                
                # Add label with price and description
                ax.annotate(f'{level_type}: ${level:.2f}',
                           xy=(df.index[-1], level),
                           xytext=(10, 0), textcoords='offset points',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                           fontsize=9)
            
        except Exception as e:
            print(f"Error adding support/resistance labels: {str(e)}")

    def calculate_rsi(self, df, periods=14):
        """Calculate Relative Strength Index"""
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=df.index)

    def find_support_resistance(self, df):
        """Find support and resistance levels"""
        try:
            # Calculate price levels
            levels = {}
            
            # Current price
            current_price = df['Close'].iloc[-1]
            
            # Recent high and low
            recent_high = df['High'].rolling(window=20).max().iloc[-1]
            recent_low = df['Low'].rolling(window=20).min().iloc[-1]
            
            # Key levels
            levels['Support'] = recent_low
            levels['Resistance'] = recent_high
            
            return levels
            
        except Exception as e:
            print(f"Error finding support/resistance: {str(e)}")
            return {}

    def add_projections_and_simulations(self, df, ax):
        """Add price projections and trading simulations"""
        try:
            # Calculate projections
            last_price = df['Close'].iloc[-1]
            last_date = df.index[-1]
            projection_days = 5  # Project 5 days ahead
            
            # Monte Carlo simulation parameters
            n_simulations = 100
            volatility = df['Close'].pct_change().std()
            returns = df['Close'].pct_change().mean()
            
            # Generate date range for projections
            future_dates = pd.date_range(start=last_date, periods=projection_days + 1, freq='B')[1:]
            
            # Monte Carlo simulations
            simulations = np.zeros((n_simulations, projection_days))
            for i in range(n_simulations):
                prices = [last_price]
                for j in range(projection_days):
                    # Random walk with drift
                    shock = np.random.normal(returns, volatility)
                    price = prices[-1] * (1 + shock)
                    prices.append(price)
                simulations[i] = prices[1:]
            
            # Calculate confidence intervals
            lower_bound = np.percentile(simulations, 25, axis=0)
            upper_bound = np.percentile(simulations, 75, axis=0)
            median_projection = np.median(simulations, axis=0)
            
            # Plot projections
            ax.fill_between(future_dates, lower_bound, upper_bound, 
                           color='gray', alpha=0.2, label='Projection Range')
            ax.plot(future_dates, median_projection, '--', 
                    color='blue', label='Median Projection')
            
            # Add trading signals and simulated trades
            self.add_trading_signals(df, ax)
            
            # Add projection labels
            self.add_projection_labels(future_dates, median_projection, 
                                     lower_bound, upper_bound, ax)
            
        except Exception as e:
            print(f"Error adding projections: {str(e)}")

    def add_trading_signals(self, df, ax):
        """Add trading signals and simulated trades"""
        try:
            # Calculate trading signals
            signals = self.generate_trading_signals(df)
            
            # Simulate day trading
            balance = float(self.capital_var.get())
            position = 0
            trades = []
            
            for i in range(1, len(df)):
                date = df.index[i]
                price = df['Close'].iloc[i]
                
                # Check for buy signals
                if date in signals['buy'] and position == 0:
                    position = balance / price
                    trades.append({
                        'type': 'buy',
                        'date': date,
                        'price': price,
                        'shares': position
                    })
                    
                # Check for sell signals
                elif date in signals['sell'] and position > 0:
                    balance = position * price
                    trades.append({
                        'type': 'sell',
                        'date': date,
                        'price': price,
                        'shares': position
                    })
                    position = 0
            
            # Plot trades
            for trade in trades:
                color = 'green' if trade['type'] == 'buy' else 'red'
                marker = '^' if trade['type'] == 'buy' else 'v'
                
                ax.plot(trade['date'], trade['price'], marker,
                       color=color, markersize=10,
                       label=f"{trade['type'].capitalize()} Signal")
                
                # Add trade labels
                ax.annotate(f"{trade['type'].capitalize()}\n${trade['price']:.2f}\n{trade['shares']:.0f} shares",
                           xy=(trade['date'], trade['price']),
                           xytext=(10, 10 if trade['type'] == 'buy' else -10),
                           textcoords='offset points',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                           fontsize=8)
            
        except Exception as e:
            print(f"Error adding trading signals: {str(e)}")

    def add_projection_labels(self, future_dates, median_proj, lower_bound, upper_bound, ax):
        """Add labels for price projections"""
        try:
            # Add projection range labels
            last_date = future_dates[-1]
            
            # Upper bound label
            ax.annotate(f"Upper Target: ${upper_bound[-1]:.2f}",
                       xy=(last_date, upper_bound[-1]),
                       xytext=(10, 10),
                       textcoords='offset points',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                       fontsize=9)
            
            # Lower bound label
            ax.annotate(f"Lower Target: ${lower_bound[-1]:.2f}",
                       xy=(last_date, lower_bound[-1]),
                       xytext=(10, -10),
                       textcoords='offset points',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                       fontsize=9)
            
            # Add probability annotations
            prob_above = f"Probability Above: {self.calculate_probability_above(median_proj[-1]):.1%}"
            prob_below = f"Probability Below: {self.calculate_probability_below(median_proj[-1]):.1%}"
            
            ax.annotate(prob_above,
                       xy=(last_date, median_proj[-1]),
                       xytext=(10, 20),
                       textcoords='offset points',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                       fontsize=8)
            
            ax.annotate(prob_below,
                       xy=(last_date, median_proj[-1]),
                       xytext=(10, -20),
                       textcoords='offset points',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                       fontsize=8)
            
        except Exception as e:
            print(f"Error adding projection labels: {str(e)}")

    def calculate_probability_above(self, target_price):
        """Calculate probability of price moving above target"""
        try:
            # Simple probability calculation based on historical volatility
            return 0.5  # Placeholder - implement actual probability calculation
        except Exception as e:
            print(f"Error calculating probability above: {str(e)}")
            return 0.5

    def calculate_probability_below(self, target_price):
        """Calculate probability of price moving below target"""
        try:
            # Simple probability calculation based on historical volatility
            return 0.5  # Placeholder - implement actual probability calculation
        except Exception as e:
            print(f"Error calculating probability below: {str(e)}")
            return 0.5

    def create_simulation_controls(self, parent):
        """Create simulation control widgets"""
        sim_frame = ttk.LabelFrame(parent, text="Simulation Controls", padding="5")
        sim_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Simulation parameters
        param_frame = ttk.Frame(sim_frame)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Number of simulations
        ttk.Label(param_frame, text="Simulations:").grid(row=0, column=0, padx=5)
        self.sim_count_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.sim_count_var, width=10).grid(row=0, column=1, padx=5)
        
        # Projection period
        ttk.Label(param_frame, text="Extra Months:").grid(row=0, column=2, padx=5)
        self.proj_months_var = tk.StringVar(value="3")
        ttk.Entry(param_frame, textvariable=self.proj_months_var, width=10).grid(row=0, column=3, padx=5)
        
        # Volatility adjustment
        ttk.Label(param_frame, text="Volatility Factor:").grid(row=1, column=0, padx=5)
        self.vol_factor_var = tk.StringVar(value="1.0")
        ttk.Entry(param_frame, textvariable=self.vol_factor_var, width=10).grid(row=1, column=1, padx=5)
        
        # Trend bias
        ttk.Label(param_frame, text="Trend Bias:").grid(row=1, column=2, padx=5)
        self.trend_bias_var = tk.StringVar(value="0.0")
        ttk.Entry(param_frame, textvariable=self.trend_bias_var, width=10).grid(row=1, column=3, padx=5)
        
        # Simulation buttons
        btn_frame = ttk.Frame(sim_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Run Simulation", 
                   command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Results", 
                   command=self.save_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Results", 
                   command=self.load_simulation).pack(side=tk.LEFT, padx=5)

    def run_simulation(self):
        """Run price simulation with current parameters"""
        try:
            # Get simulation parameters
            n_sims = int(self.sim_count_var.get())
            extra_months = int(self.proj_months_var.get())
            vol_factor = float(self.vol_factor_var.get())
            trend_bias = float(self.trend_bias_var.get())
            
            # Get current data
            df = self.current_df.copy()
            
            # Calculate simulation parameters
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * vol_factor
            drift = returns.mean() + trend_bias
            
            # Generate future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(
                start=last_date,
                periods=extra_months * 21 + 1,  # Assuming 21 trading days per month
                freq='B'
            )[1:]
            
            # Run Monte Carlo simulation
            last_price = df['Close'].iloc[-1]
            simulations = np.zeros((n_sims, len(future_dates)))
            
            for i in range(n_sims):
                prices = [last_price]
                for t in range(len(future_dates)):
                    shock = np.random.normal(drift, volatility)
                    price = prices[-1] * (1 + shock)
                    prices.append(price)
                simulations[i] = prices[1:]
            
            # Store simulation results
            self.simulation_results = {
                'dates': future_dates,
                'simulations': simulations,
                'parameters': {
                    'n_sims': n_sims,
                    'extra_months': extra_months,
                    'vol_factor': vol_factor,
                    'trend_bias': trend_bias
                }
            }
            
            # Update plot with simulations
            self.plot_with_simulations(df)
            
            # Calculate and display statistics
            self.update_simulation_stats()
            
        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            messagebox.showerror("Error", "Failed to run simulation")

    def save_simulation(self):
        """Save simulation results to DuckDB"""
        try:
            if not hasattr(self, 'simulation_results'):
                messagebox.showwarning("Warning", "No simulation results to save")
                return
            
            # Connect to DuckDB
            conn = duckdb.connect('simulate.db')
            
            # Create tables if they don't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulation_params (
                    sim_id INTEGER PRIMARY KEY,
                    ticker VARCHAR,
                    sim_date TIMESTAMP,
                    n_sims INTEGER,
                    extra_months INTEGER,
                    vol_factor DOUBLE,
                    trend_bias DOUBLE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulation_results (
                    sim_id INTEGER,
                    date_idx INTEGER,
                    sim_idx INTEGER,
                    price DOUBLE,
                    projection_date DATE,
                    FOREIGN KEY (sim_id) REFERENCES simulation_params(sim_id)
                )
            """)
            
            # Insert simulation parameters
            params = self.simulation_results['parameters']
            conn.execute("""
                INSERT INTO simulation_params 
                (ticker, sim_date, n_sims, extra_months, vol_factor, trend_bias)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (self.ticker_var.get(), pd.Timestamp.now(), params['n_sims'],
                  params['extra_months'], params['vol_factor'], params['trend_bias']))
            
            # Get simulation ID
            sim_id = conn.execute("SELECT last_value FROM simulation_params_sim_id_seq").fetchone()[0]
            
            # Prepare simulation results
            results_data = []
            for sim_idx in range(len(self.simulation_results['simulations'])):
                for date_idx, date in enumerate(self.simulation_results['dates']):
                    results_data.append({
                        'sim_id': sim_id,
                        'date_idx': date_idx,
                        'sim_idx': sim_idx,
                        'price': self.simulation_results['simulations'][sim_idx][date_idx],
                        'projection_date': date
                    })
            
            # Insert simulation results
            results_df = pd.DataFrame(results_data)
            conn.execute("INSERT INTO simulation_results SELECT * FROM results_df")
            
            conn.close()
            messagebox.showinfo("Success", "Simulation results saved successfully")
            
        except Exception as e:
            print(f"Error saving simulation: {str(e)}")
            messagebox.showerror("Error", "Failed to save simulation results")

    def load_simulation(self):
        """Load simulation results from DuckDB"""
        try:
            conn = duckdb.connect('simulate.db')
            
            # Get list of available simulations
            sims = conn.execute("""
                SELECT sim_id, ticker, sim_date, n_sims, extra_months
                FROM simulation_params
                WHERE ticker = ?
                ORDER BY sim_date DESC
            """, [self.ticker_var.get()]).fetchall()
            
            if not sims:
                messagebox.showinfo("Info", "No saved simulations found for this ticker")
                return
            
            # Create selection dialog
            dialog = tk.Toplevel()
            dialog.title("Load Simulation")
            
            # Create listbox with simulations
            listbox = tk.Listbox(dialog, width=50)
            for sim in sims:
                listbox.insert(tk.END, f"ID: {sim[0]} - Date: {sim[2]} - Sims: {sim[3]} - Months: {sim[4]}")
            listbox.pack(padx=5, pady=5)
            
            def load_selected():
                selection = listbox.curselection()
                if selection:
                    sim_id = sims[selection[0]][0]
                    
                    # Load simulation results
                    results = conn.execute("""
                        SELECT date_idx, sim_idx, price, projection_date
                        FROM simulation_results
                        WHERE sim_id = ?
                        ORDER BY sim_idx, date_idx
                    """, [sim_id]).fetchall()
                    
                    # Reconstruct simulation results
                    dates = sorted(set(r[3] for r in results))
                    simulations = np.zeros((max(r[1] for r in results) + 1, len(dates)))
                    
                    for r in results:
                        simulations[r[1]][r[0]] = r[2]
                    
                    self.simulation_results = {
                        'dates': dates,
                        'simulations': simulations,
                        'parameters': conn.execute("""
                            SELECT n_sims, extra_months, vol_factor, trend_bias
                            FROM simulation_params
                            WHERE sim_id = ?
                        """, [sim_id]).fetchone()
                    }
                    
                    # Update plot and stats
                    self.plot_with_simulations(self.current_df)
                    self.update_simulation_stats()
                    
                    dialog.destroy()
            
            ttk.Button(dialog, text="Load", command=load_selected).pack(pady=5)
            
            conn.close()
            
        except Exception as e:
            print(f"Error loading simulation: {str(e)}")
            messagebox.showerror("Error", "Failed to load simulation results")

    def plot_with_simulations(self, df):
        """Update plot to include simulation results"""
        try:
            # Create base plot
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            
            # Price subplot
            ax1 = plt.subplot(gs[0])
            
            # Plot historical data
            ax1.plot(df.index, df['Close'], label='Historical', color='blue')
            
            # Plot simulations
            if hasattr(self, 'simulation_results'):
                sims = self.simulation_results['simulations']
                dates = self.simulation_results['dates']
                
                # Plot all simulations with low alpha
                for sim in sims:
                    ax1.plot(dates, sim, color='gray', alpha=0.1)
                
                # Plot confidence intervals
                percentiles = np.percentile(sims, [10, 25, 50, 75, 90], axis=0)
                
                ax1.fill_between(dates, percentiles[0], percentiles[4], 
                               color='gray', alpha=0.2, label='80% Confidence')
                ax1.fill_between(dates, percentiles[1], percentiles[3], 
                               color='gray', alpha=0.3, label='50% Confidence')
                ax1.plot(dates, percentiles[2], '--', 
                        color='red', label='Median Projection')
            
            # Customize plot
            ax1.set_title(f'{self.ticker_var.get()} Price Projection')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Save and display plot
            plt.tight_layout()
            plot_file = f'{self.ticker_var.get()}_simulation.png'
            plt.savefig(plot_file)
            plt.close()
            
            # Update plot in GUI
            self.update_plot_display(plot_file)
            
        except Exception as e:
            print(f"Error plotting simulations: {str(e)}")

    def update_simulation_stats(self):
        """Update simulation statistics display"""
        try:
            if not hasattr(self, 'simulation_results'):
                return
            
            sims = self.simulation_results['simulations']
            last_prices = sims[:, -1]
            current_price = self.current_df['Close'].iloc[-1]
            
            stats = "Simulation Statistics:\n"
            stats += "=" * 40 + "\n\n"
            
            # Price projections
            stats += "Price Projections:\n"
            percentiles = [10, 25, 50, 75, 90]
            for p in percentiles:
                price = np.percentile(last_prices, p)
                change = (price - current_price) / current_price
                stats += f"{p}th Percentile: ${price:.2f} ({change:.1%})\n"
            
            # Risk metrics
            stats += "\nRisk Metrics:\n"
            stats += f"Upside Potential (90th): {(np.percentile(last_prices, 90) - current_price) / current_price:.1%}\n"
            stats += f"Downside Risk (10th): {(np.percentile(last_prices, 10) - current_price) / current_price:.1%}\n"
            
            # Probability estimates
            prob_up = np.mean(last_prices > current_price)
            stats += f"\nProbability of Price Increase: {prob_up:.1%}\n"
            
            # Update stats display
            if hasattr(self, 'sim_stats_text'):
                self.sim_stats_text.delete(1.0, tk.END)
                self.sim_stats_text.insert(tk.END, stats)
            
        except Exception as e:
            print(f"Error updating simulation stats: {str(e)}")

    def create_monte_carlo_controls(self, parent):
        """Create Monte Carlo simulation control panel with scrollbar"""
        # Create main frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure canvas
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create Monte Carlo controls inside scrollable frame
        mc_frame = ttk.LabelFrame(scrollable_frame, text="Monte Carlo Simulation Controls", padding="5")
        mc_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Controls grid
        controls_frame = ttk.Frame(mc_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Simulation Parameters
        # Row 1
        ttk.Label(controls_frame, text="Number of Paths:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.mc_paths_var = tk.StringVar(value="1000")
        ttk.Entry(controls_frame, textvariable=self.mc_paths_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(controls_frame, text="Time Steps:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.mc_steps_var = tk.StringVar(value="252")
        ttk.Entry(controls_frame, textvariable=self.mc_steps_var, width=10).grid(row=0, column=3, padx=5)
        
        # Row 2
        ttk.Label(controls_frame, text="Volatility Adjust:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.mc_vol_var = tk.StringVar(value="1.0")
        ttk.Entry(controls_frame, textvariable=self.mc_vol_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(controls_frame, text="Drift Adjust:").grid(row=1, column=2, padx=5, pady=2, sticky="w")
        self.mc_drift_var = tk.StringVar(value="0.0")
        ttk.Entry(controls_frame, textvariable=self.mc_drift_var, width=10).grid(row=1, column=3, padx=5)
        
        # Row 3
        ttk.Label(controls_frame, text="Confidence Level:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.mc_confidence_var = tk.StringVar(value="95")
        ttk.Entry(controls_frame, textvariable=self.mc_confidence_var, width=10).grid(row=2, column=1, padx=5)
        
        ttk.Label(controls_frame, text="Seed:").grid(row=2, column=2, padx=5, pady=2, sticky="w")
        self.mc_seed_var = tk.StringVar(value="42")
        ttk.Entry(controls_frame, textvariable=self.mc_seed_var, width=10).grid(row=2, column=3, padx=5)
        
        # Scenario Selection
        scenario_frame = ttk.Frame(mc_frame)
        scenario_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(scenario_frame, text="Scenario:").pack(side=tk.LEFT, padx=5)
        self.mc_scenario_var = tk.StringVar(value="Base")
        scenario_combo = ttk.Combobox(scenario_frame, textvariable=self.mc_scenario_var, width=15)
        scenario_combo['values'] = ('Base', 'Bullish', 'Bearish', 'High Vol', 'Low Vol')
        scenario_combo.pack(side=tk.LEFT, padx=5)
        
        # Control Buttons
        button_frame = ttk.Frame(mc_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Run Monte Carlo", 
                   command=self.run_monte_carlo).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Scenario", 
                   command=self.save_monte_carlo).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Scenario", 
                   command=self.load_monte_carlo).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", 
                   command=self.reset_monte_carlo).pack(side=tk.LEFT, padx=5)

    def run_monte_carlo(self):
        """Execute Monte Carlo simulation with current parameters"""
        try:
            # Get parameters
            n_paths = int(self.mc_paths_var.get())
            n_steps = int(self.mc_steps_var.get())
            vol_adjust = float(self.mc_vol_var.get())
            drift_adjust = float(self.mc_drift_var.get())
            confidence = float(self.mc_confidence_var.get())
            seed = int(self.mc_seed_var.get())
            
            # Set random seed
            np.random.seed(seed)
            
            # Get historical data
            df = self.current_df.copy()
            returns = df['Close'].pct_change().dropna()
            
            # Calculate parameters
            mu = returns.mean() + drift_adjust
            sigma = returns.std() * vol_adjust
            S0 = df['Close'].iloc[-1]
            dt = 1/252  # Daily timesteps
            
            # Generate paths
            paths = np.zeros((n_paths, n_steps))
            for i in range(n_paths):
                prices = [S0]
                for t in range(n_steps-1):
                    dW = np.random.normal(0, np.sqrt(dt))
                    dS = prices[-1] * (mu * dt + sigma * dW)
                    prices.append(prices[-1] + dS)
                paths[i] = prices
            
            # Store results
            self.mc_results = {
                'paths': paths,
                'parameters': {
                    'n_paths': n_paths,
                    'n_steps': n_steps,
                    'vol_adjust': vol_adjust,
                    'drift_adjust': drift_adjust,
                    'confidence': confidence,
                    'seed': seed,
                    'scenario': self.mc_scenario_var.get()
                },
                'statistics': self.calculate_mc_statistics(paths, S0, confidence)
            }
            
            # Update display
            self.plot_monte_carlo()
            self.update_mc_statistics()
            
        except Exception as e:
            print(f"Error running Monte Carlo simulation: {str(e)}")
            messagebox.showerror("Error", "Failed to run Monte Carlo simulation")

    def calculate_mc_statistics(self, paths, initial_price, confidence):
        """Calculate statistics from Monte Carlo paths"""
        try:
            final_prices = paths[:, -1]
            returns = (final_prices - initial_price) / initial_price
            
            # Calculate percentiles
            conf_level = confidence / 100
            lower_percentile = (100 - confidence) / 2
            upper_percentile = 100 - lower_percentile
            
            stats = {
                'mean_return': returns.mean(),
                'median_return': np.median(returns),
                'std_return': returns.std(),
                'min_return': returns.min(),
                'max_return': returns.max(),
                f'lower_{confidence}': np.percentile(returns, lower_percentile),
                f'upper_{confidence}': np.percentile(returns, upper_percentile),
                'prob_positive': np.mean(returns > 0),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'var_95': np.percentile(returns, 5),
                'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating MC statistics: {str(e)}")
            return {}

    def plot_monte_carlo(self):
        """Plot Monte Carlo simulation results"""
        try:
            if not hasattr(self, 'mc_results'):
                return
            
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            
            # Price paths plot
            ax1 = plt.subplot(gs[0])
            paths = self.mc_results['paths']
            
            # Plot individual paths with low alpha
            for path in paths[::max(1, len(paths)//100)]:  # Plot subset of paths
                ax1.plot(path, color='gray', alpha=0.1)
                
            # Plot statistics
            mean_path = paths.mean(axis=0)
            std_path = paths.std(axis=0)
            conf = self.mc_results['parameters']['confidence']
            
            ax1.plot(mean_path, 'b-', label='Mean Path', linewidth=2)
            ax1.fill_between(range(len(mean_path)),
                            mean_path - 1.96*std_path,
                            mean_path + 1.96*std_path,
                            color='blue', alpha=0.1,
                            label=f'{conf}% Confidence')
            
            # Distribution plot
            ax2 = plt.subplot(gs[1])
            final_returns = (paths[:, -1] - paths[:, 0]) / paths[:, 0]
            ax2.hist(final_returns, bins=50, density=True, alpha=0.7)
            ax2.axvline(final_returns.mean(), color='r', linestyle='--', label='Mean Return')
            
            # Customize plots
            ax1.set_title(f'Monte Carlo Simulation - {self.mc_results["parameters"]["scenario"]} Scenario')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            
            ax2.set_xlabel('Return')
            ax2.set_ylabel('Density')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save and display
            plot_file = f'{self.ticker_var.get()}_monte_carlo.png'
            plt.savefig(plot_file)
            plt.close()
            
            self.update_plot_display(plot_file)
            
        except Exception as e:
            print(f"Error plotting Monte Carlo results: {str(e)}")

    def update_mc_statistics(self):
        """Update Monte Carlo statistics display"""
        try:
            if not hasattr(self, 'mc_results'):
                return
            
            stats = self.mc_results['statistics']
            params = self.mc_results['parameters']
            
            text = "Monte Carlo Analysis Results\n"
            text += "=" * 40 + "\n\n"
            
            text += f"Scenario: {params['scenario']}\n"
            text += f"Paths: {params['n_paths']}, Steps: {params['n_steps']}\n\n"
            
            text += "Return Statistics:\n"
            text += f"Mean Return: {stats['mean_return']:.2%}\n"
            text += f"Median Return: {stats['median_return']:.2%}\n"
            text += f"Return Std Dev: {stats['std_return']:.2%}\n"
            text += f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n\n"
            
            text += "Risk Metrics:\n"
            text += f"95% VaR: {abs(stats['var_95']):.2%}\n"
            text += f"95% CVaR: {abs(stats['cvar_95']):.2%}\n"
            text += f"Probability of Positive Return: {stats['prob_positive']:.1%}\n\n"
            
            text += f"Confidence Interval ({params['confidence']}%):\n"
            # Fixed f-string syntax
            lower_conf = stats[f"lower_{int(params['confidence'])}"]
            upper_conf = stats[f"upper_{int(params['confidence'])}"]
            text += f"Lower: {lower_conf:.2%}\n"
            text += f"Upper: {upper_conf:.2%}\n"
            
            # Update statistics display
            if hasattr(self, 'mc_stats_text'):
                self.mc_stats_text.delete(1.0, tk.END)
                self.mc_stats_text.insert(tk.END, text)
                
        except Exception as e:
            print(f"Error updating MC statistics: {str(e)}")

    def create_plot_area(self, parent):
        """Create scrollable plot area with matplotlib integration"""
        # Create main plot frame
        self.plot_frame = ttk.LabelFrame(parent, text="Analysis Charts", padding="5")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas with scrollbars
        self.plot_canvas = tk.Canvas(self.plot_frame)
        
        # Create scrollbars
        y_scrollbar = ttk.Scrollbar(self.plot_frame, orient="vertical", 
                                   command=self.plot_canvas.yview)
        x_scrollbar = ttk.Scrollbar(self.plot_frame, orient="horizontal", 
                                   command=self.plot_canvas.xview)
        
        # Create frame for plots
        self.scrollable_plot_frame = ttk.Frame(self.plot_canvas)
        
        # Configure canvas
        self.scrollable_plot_frame.bind(
            "<Configure>",
            lambda e: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
        )
        
        # Create canvas window
        self.plot_canvas.create_window((0, 0), window=self.scrollable_plot_frame, anchor="nw")
        self.plot_canvas.configure(yscrollcommand=y_scrollbar.set, 
                                 xscrollcommand=x_scrollbar.set)
        
        # Pack scrollbars and canvas
        y_scrollbar.pack(side="right", fill="y")
        x_scrollbar.pack(side="bottom", fill="x")
        self.plot_canvas.pack(side="left", fill="both", expand=True)
        
        # Create frame for individual plots
        self.price_plot_frame = ttk.Frame(self.scrollable_plot_frame)
        self.price_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.mc_plot_frame = ttk.Frame(self.scrollable_plot_frame)
        self.mc_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind mouse wheel
        self.plot_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.plot_canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)

    def _on_mousewheel(self, event):
        """Handle vertical scrolling"""
        self.plot_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_shift_mousewheel(self, event):
        """Handle horizontal scrolling"""
        self.plot_canvas.xview_scroll(int(-1*(event.delta/120)), "units")

    def update_plot_display(self, plot_file, plot_type='price'):
        """Update plot display with scrolling support"""
        try:
            # Load the image
            img = Image.open(plot_file)
            photo = ImageTk.PhotoImage(img)
            
            # Create or update label based on plot type
            if plot_type == 'price':
                if hasattr(self, 'price_plot_label'):
                    self.price_plot_label.destroy()
                self.price_plot_label = ttk.Label(self.price_plot_frame, image=photo)
                self.price_plot_label.image = photo  # Keep reference
                self.price_plot_label.pack(fill=tk.BOTH, expand=True)
            elif plot_type == 'monte_carlo':
                if hasattr(self, 'mc_plot_label'):
                    self.mc_plot_label.destroy()
                self.mc_plot_label = ttk.Label(self.mc_plot_frame, image=photo)
                self.mc_plot_label.image = photo  # Keep reference
                self.mc_plot_label.pack(fill=tk.BOTH, expand=True)
            
            # Update canvas scroll region
            self.plot_canvas.update_idletasks()
            self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
            
        except Exception as e:
            print(f"Error updating plot display: {str(e)}")

    def plot_data(self, df):
        """Plot stock data with scrollable display"""
        try:
            if df is None or df.empty:
                return
            
            # Create figure with adjusted size
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            
            # Price subplot
            ax1 = plt.subplot(gs[0])
            ax1v = ax1.twinx()  # Volume overlay
            
            # Volume subplot
            ax2 = plt.subplot(gs[1], sharex=ax1)
            
            # Plot data
            # ... (existing plotting code) ...
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            plot_file = f'{self.ticker_var.get()}_price.png'
            plt.savefig(plot_file, bbox_inches='tight', dpi=100)
            plt.close()
            
            # Update display with price plot
            self.update_plot_display(plot_file, 'price')
            
        except Exception as e:
            print(f"Error plotting data: {str(e)}")

    def plot_monte_carlo(self):
        """Plot Monte Carlo simulation with scrollable display"""
        try:
            if not hasattr(self, 'mc_results'):
                return
            
            # Create figure
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            
            # ... (existing Monte Carlo plotting code) ...
            
            # Save plot
            plot_file = f'{self.ticker_var.get()}_monte_carlo.png'
            plt.savefig(plot_file, bbox_inches='tight', dpi=100)
            plt.close()
            
            # Update display with Monte Carlo plot
            self.update_plot_display(plot_file, 'monte_carlo')
            
        except Exception as e:
            print(f"Error plotting Monte Carlo results: {str(e)}")

    # Update the GUI initialization
    def create_widgets(self):
        """Create all GUI widgets with scrollable plots"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Control panel (left side)
        control_frame = ttk.Frame(main_container)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create controls
        self.create_input_controls(control_frame)
        self.create_monte_carlo_controls(control_frame)
        
        # Plot area (right side)
        self.create_plot_area(main_container)

if __name__ == "__main__":
    try:
        # Create database if it doesn't exist
        db_path = 'stocks.db'
        if not os.path.exists(db_path):
            conn = duckdb.connect(db_path)
            conn.close()
        
        # Initialize analyzer with database
        analyzer = StockMarketAnalyzer(db_path)
        gui = StockAnalyzerGUI(analyzer)
        gui.root.mainloop()
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        raise e
