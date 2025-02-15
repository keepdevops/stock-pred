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

class StockMarketAnalyzer:
    def __init__(self, db_path):
        """Initialize the analyzer with database connection"""
        self.conn = duckdb.connect(db_path)
        
    def get_historical_data(self, ticker, duration):
        """Get historical data using DuckDB"""
        try:
            import duckdb
            import pandas as pd
            
            # Map duration to SQL interval
            duration_map = {
                '1d': 'INTERVAL 1 day',
                '1mo': 'INTERVAL 1 month',
                '3mo': 'INTERVAL 3 months',
                '6mo': 'INTERVAL 6 months',
                '1y': 'INTERVAL 1 year'
            }
            
            interval = duration_map.get(duration, 'INTERVAL 1 month')
            
            # Connect to DuckDB database
            self.conn = duckdb.connect('./stocks.db')
            
            query = f"""
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
            """
            
            df = self.conn.execute(query, [ticker]).df()
            
            # Convert date column to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            print(f"Retrieved {len(df)} records for {ticker} over {duration}")
            print("Columns in DataFrame:", df.columns.tolist())
            
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

    def calculate_strategy_performance(self, df, strategy):
        """Calculate performance metrics for each strategy"""
        try:
            signals = self.generate_trading_signals(df)
            performance = {
                'total_signals': len(signals['buy']) + len(signals['sell']),
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
            
            # Calculate returns for each signal
            returns = []
            for i, buy_date in enumerate(signals['buy']):
                if i < len(signals['sell']):
                    sell_date = signals['sell'][i]
                    if sell_date > buy_date:
                        entry_price = df.loc[buy_date, 'Close']
                        exit_price = df.loc[sell_date, 'Close']
                        returns.append((exit_price - entry_price) / entry_price)
            
            if returns:
                performance['win_rate'] = len([r for r in returns if r > 0]) / len(returns)
                performance['avg_return'] = np.mean(returns)
                performance['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
                performance['max_drawdown'] = min(returns) if returns else 0
            
            return performance
            
        except Exception as e:
            print(f"Error calculating strategy performance: {str(e)}")
            return None

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
                performance = self.calculate_strategy_performance(df, strategy_name)
                
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
        """Update strategy recommendation display"""
        recommendation = self.recommend_strategy(df)
        if not recommendation:
            self.strategy_text.delete(1.0, tk.END)
            self.strategy_text.insert(tk.END, "Unable to generate strategy recommendation")
            return
        
        # Format recommendation text
        text = "Market Analysis & Strategy Recommendation\n"
        text += "=" * 50 + "\n\n"
        
        # Market conditions
        conditions = recommendation['conditions']
        text += f"Market Type: {recommendation['market_type']}\n"
        text += f"Volatility: {conditions['volatility']:.2%}\n"
        text += f"Trend Strength: {conditions['trend_strength']:.2%}\n"
        text += f"Volume Trend: {conditions['volume_trend']:.2%}\n\n"
        
        # Strategy recommendation
        text += f"Recommended Strategy: {recommendation['recommended']}\n"
        text += f"Best For: {recommendation['scores'][recommendation['recommended']]['best_for']}\n\n"
        
        # Strategy rankings
        text += "Strategy Rankings:\n"
        text += "-" * 50 + "\n"
        sorted_strategies = sorted(
            recommendation['scores'].items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        for strategy, scores in sorted_strategies:
            text += f"\n{strategy}:\n"
            text += f"Total Score: {scores['total_score']:.2f}\n"
            text += f"Market Fit: {scores['market_score']:.2f}\n"
            text += f"Performance: {scores['performance_score']:.2f}\n"
            
            metrics = scores['performance_metrics']
            text += f"Win Rate: {metrics['win_rate']:.2%}\n"
            text += f"Avg Return: {metrics['avg_return']:.2%}\n"
            text += "-" * 25 + "\n"
        
        self.strategy_text.delete(1.0, tk.END)
        self.strategy_text.insert(tk.END, text)

class StockAnalyzerGUI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("Stock Market Analyzer")
        self.root.geometry("1600x1000")
        
        # Initialize data structures
        self.data_cache = {}
        self.strategy_metrics = {}
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure main container grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create left and right frames
        left_frame = ttk.Frame(self.main_container, padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew")
        
        right_frame = ttk.Frame(self.main_container, padding="5")
        right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Configure weights for the frames
        self.main_container.grid_columnconfigure(1, weight=3)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Create control panel in left frame
        self.create_control_panel(left_frame)
        
        # Create display panel in right frame
        self.create_display_panel(right_frame)

        # Initialize DuckDB connection
        try:
            self.db_conn = duckdb.connect('stock_data.db')
            self.create_stock_tables()
            print("Successfully connected to DuckDB")
        except Exception as e:
            print(f"Error connecting to DuckDB: {str(e)}")
            messagebox.showerror("Database Error", "Failed to connect to database")

    def create_stock_tables(self):
        """Create necessary database tables if they don't exist"""
        try:
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    ticker VARCHAR,
                    date DATE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    adj_close DOUBLE,
                    PRIMARY KEY (ticker, date)
                )
            """)
            self.db_conn.commit()
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
            raise e

    def save_to_database(self, df, ticker):
        """Save stock data to DuckDB"""
        try:
            # Prepare data for insertion
            df_to_save = df.reset_index()
            df_to_save['ticker'] = ticker
            df_to_save.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'ticker']
            
            # Insert data using UPSERT
            self.db_conn.execute("""
                INSERT OR REPLACE INTO stock_prices 
                SELECT * FROM df_to_save
            """)
            self.db_conn.commit()
            print(f"Saved {len(df)} records for {ticker}")
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            raise e

    def load_from_database(self, ticker, duration):
        """Load stock data from DuckDB"""
        try:
            # Calculate date range based on duration
            end_date = pd.Timestamp.now()
            if duration == '1d':
                start_date = end_date - pd.Timedelta(days=1)
            elif duration == '5d':
                start_date = end_date - pd.Timedelta(days=5)
            elif duration == '1mo':
                start_date = end_date - pd.Timedelta(days=30)
            elif duration == '3mo':
                start_date = end_date - pd.Timedelta(days=90)
            elif duration == '6mo':
                start_date = end_date - pd.Timedelta(days=180)
            elif duration == '1y':
                start_date = end_date - pd.Timedelta(days=365)
            elif duration == '2y':
                start_date = end_date - pd.Timedelta(days=730)
            elif duration == '5y':
                start_date = end_date - pd.Timedelta(days=1825)
            else:  # max
                start_date = pd.Timestamp.min

            # Query database
            query = """
                SELECT date, open, high, low, close, volume, adj_close
                FROM stock_prices
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            df = self.db_conn.execute(query, [ticker, start_date, end_date]).df()
            
            if len(df) > 0:
                # Set date as index
                df.set_index('date', inplace=True)
                return df
            return None
            
        except Exception as e:
            print(f"Error loading from database: {str(e)}")
            return None

    def create_control_panel(self, parent):
        """Create the control panel with all input widgets"""
        # Ticker selection
        ticker_frame = ttk.LabelFrame(parent, text="Stock Selection", padding="5")
        ticker_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Label(ticker_frame, text="Ticker:").grid(row=0, column=0, pady=5)
        self.ticker_var = tk.StringVar()
        self.ticker_combo = ttk.Combobox(ticker_frame, textvariable=self.ticker_var)
        self.ticker_combo.grid(row=0, column=1, pady=5, padx=5, sticky="ew")
        
        # Populate ticker dropdown from database
        try:
            query = "SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker"
            tickers = [row[0] for row in self.db_conn.execute(query).fetchall()]
            self.ticker_combo['values'] = tickers
            if tickers:
                self.ticker_combo.set(tickers[0])  # Set default value
        except Exception as e:
            print(f"Error loading tickers: {str(e)}")
            messagebox.showerror("Database Error", "Failed to load tickers from database")
        
        # Duration selection
        ttk.Label(ticker_frame, text="Duration:").grid(row=1, column=0, pady=5)
        self.duration_var = tk.StringVar(value="1mo")
        self.duration_combo = ttk.Combobox(ticker_frame, textvariable=self.duration_var,
                                         values=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
        self.duration_combo.grid(row=1, column=1, pady=5, padx=5, sticky="ew")
        
        # Trading system frame
        trading_frame = ttk.LabelFrame(parent, text="Trading System", padding="5")
        trading_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Capital input
        ttk.Label(trading_frame, text="Account Capital ($):").grid(row=0, column=0, pady=5)
        self.capital_var = tk.StringVar(value="10000")
        ttk.Entry(trading_frame, textvariable=self.capital_var).grid(row=0, column=1, pady=5, padx=5, sticky="ew")
        
        # Risk input
        ttk.Label(trading_frame, text="Risk per Trade (%):").grid(row=1, column=0, pady=5)
        self.risk_var = tk.StringVar(value="1")
        ttk.Entry(trading_frame, textvariable=self.risk_var).grid(row=1, column=1, pady=5, padx=5, sticky="ew")
        
        # Stop loss input
        ttk.Label(trading_frame, text="Stop Loss (%):").grid(row=2, column=0, pady=5)
        self.stop_loss_var = tk.StringVar(value="2")
        ttk.Entry(trading_frame, textvariable=self.stop_loss_var).grid(row=2, column=1, pady=5, padx=5, sticky="ew")
        
        # Position info
        self.position_info = tk.Text(trading_frame, height=4, width=40, font=('Courier', 9))
        self.position_info.grid(row=3, column=0, columnspan=2, pady=5, padx=5, sticky="ew")
        
        # Strategy frame
        strategy_frame = ttk.LabelFrame(parent, text="Strategy", padding="5")
        strategy_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        # Strategy selection
        ttk.Label(strategy_frame, text="Trading Strategy:").grid(row=0, column=0, pady=5)
        self.strategy_var = tk.StringVar(value="MA Crossover")
        self.strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.strategy_var,
                                         values=["MA Crossover", "RSI", "MACD", "Bollinger Bands"])
        self.strategy_combo.grid(row=0, column=1, pady=5, padx=5, sticky="ew")
        
        # Strategy info
        self.strategy_text = tk.Text(strategy_frame, height=10, width=40, font=('Courier', 9))
        self.strategy_text.grid(row=1, column=0, columnspan=2, pady=5, padx=5, sticky="ew")
        
        # Configure grid weights
        parent.grid_columnconfigure(0, weight=1)
        for frame in [ticker_frame, trading_frame, strategy_frame]:
            frame.grid_columnconfigure(1, weight=1)

    def create_display_panel(self, parent):
        """Create the display panel with charts and analysis"""
        # Create matplotlib figure
        self.fig_main = Figure(figsize=(12, 8))
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=parent)
        self.canvas_main.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Analysis text
        self.risk_text = tk.Text(parent, height=10, font=('Courier', 9))
        self.risk_text.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Configure grid weights
        parent.grid_rowconfigure(0, weight=3)
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def on_strategy_change(self, event=None):
        """Handle strategy change event"""
        if hasattr(self, 'current_df') and self.current_df is not None:
            self.update_analysis(self.current_df)

    def update_analysis(self, df):
        """Update all analysis components"""
        self.current_df = df
        recommendation = self.recommend_strategy(df)
        self.update_strategy_recommendation(recommendation)
        self.update_plots(df)

    def recommend_strategy(self, df):
        """Recommend the best strategy based on market conditions"""
        try:
            # Calculate market conditions
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            trend = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            volume_trend = (df['Volume'].iloc[-1] - df['Volume'].iloc[0]) / df['Volume'].iloc[0]
            
            # Define strategy characteristics and scoring
            strategies = {
                "MA Crossover": {
                    'volatility_weight': 0.3,
                    'trend_weight': 0.4,
                    'volume_weight': 0.3,
                    'best_for': 'trending markets'
                },
                "RSI": {
                    'volatility_weight': 0.4,
                    'trend_weight': 0.2,
                    'volume_weight': 0.4,
                    'best_for': 'ranging markets'
                },
                "MACD": {
                    'volatility_weight': 0.3,
                    'trend_weight': 0.5,
                    'volume_weight': 0.2,
                    'best_for': 'trending markets with momentum'
                },
                "Bollinger Bands": {
                    'volatility_weight': 0.5,
                    'trend_weight': 0.3,
                    'volume_weight': 0.2,
                    'best_for': 'volatile markets'
                }
            }
            
            # Score each strategy
            strategy_scores = {}
            for strategy_name, weights in strategies.items():
                market_score = (
                    weights['volatility_weight'] * (1 - volatility if volatility < 0.5 else 0.5 - volatility/2) +
                    weights['trend_weight'] * (trend if trend > 0 else -trend/2) +
                    weights['volume_weight'] * (volume_trend if volume_trend > 0 else 0)
                )
                
                strategy_scores[strategy_name] = {
                    'score': market_score,
                    'best_for': weights['best_for']
                }
            
            # Find best strategy
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1]['score'])
            
            return {
                'recommended': best_strategy[0],
                'scores': strategy_scores,
                'market_conditions': {
                    'volatility': volatility,
                    'trend': trend,
                    'volume_trend': volume_trend
                }
            }
            
        except Exception as e:
            print(f"Error in strategy recommendation: {str(e)}")
            return None

    def update_strategy_recommendation(self, recommendation):
        """Update the strategy recommendation display"""
        if not recommendation:
            self.strategy_text.delete(1.0, tk.END)
            self.strategy_text.insert(tk.END, "Unable to generate strategy recommendation")
            return
        
        text = "Strategy Recommendation\n"
        text += "=" * 40 + "\n\n"
        
        # Market conditions
        conditions = recommendation['market_conditions']
        text += "Market Conditions:\n"
        text += f"Volatility: {conditions['volatility']:.2%}\n"
        text += f"Trend: {conditions['trend']:.2%}\n"
        text += f"Volume Trend: {conditions['volume_trend']:.2%}\n\n"
        
        # Recommended strategy
        text += f"Recommended Strategy: {recommendation['recommended']}\n"
        text += f"Best For: {recommendation['scores'][recommendation['recommended']]['best_for']}\n\n"
        
        # Strategy rankings
        text += "Strategy Rankings:\n"
        sorted_strategies = sorted(
            recommendation['scores'].items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        for strategy, data in sorted_strategies:
            text += f"{strategy}: {data['score']:.2f}\n"
        
        self.strategy_text.delete(1.0, tk.END)
        self.strategy_text.insert(tk.END, text)

    def create_analysis_widgets(self, parent):
        """Create all analysis widgets"""
        # Risk Analysis
        risk_frame = ttk.LabelFrame(parent, text="Risk Analysis", padding="5")
        risk_frame.pack(fill=tk.X, padx=5, pady=5)
        self.risk_text = tk.Text(risk_frame, height=8, width=40, font=('Courier', 9))
        self.risk_text.pack(fill=tk.X)
        
        # Pattern Analysis
        pattern_frame = ttk.LabelFrame(parent, text="Pattern Analysis", padding="5")
        pattern_frame.pack(fill=tk.X, padx=5, pady=5)
        self.pattern_text = tk.Text(pattern_frame, height=8, width=40, font=('Courier', 9))
        self.pattern_text.pack(fill=tk.X)
        
        # Probability Analysis
        prob_frame = ttk.LabelFrame(parent, text="Probability Analysis", padding="5")
        prob_frame.pack(fill=tk.X, padx=5, pady=5)
        self.prob_text = tk.Text(prob_frame, height=8, width=40, font=('Courier', 9))
        self.prob_text.pack(fill=tk.X)
        
        # Returns Analysis
        returns_frame = ttk.LabelFrame(parent, text="Returns Analysis", padding="5")
        returns_frame.pack(fill=tk.X, padx=5, pady=5)
        self.returns_text = tk.Text(returns_frame, height=8, width=40, font=('Courier', 9))
        self.returns_text.pack(fill=tk.X)

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
            self.loading_label.config(text="Loading data...")
            self.root.update()
            
            # Try loading from database first
            df = self.load_from_database(ticker, self.duration_var.get())
            
            # If no data in database or data is old, fetch from API
            if df is None or len(df) == 0 or \
               (pd.Timestamp.now() - df.index.max()) > pd.Timedelta(days=1):
                df = self.analyzer.get_historical_data(ticker, self.duration_var.get())
                if df is not None and len(df) > 0:
                    self.save_to_database(df, ticker)
            
            if df is None or len(df) == 0:
                self.loading_label.config(text="")
                messagebox.showwarning("No Data", f"No data available for {ticker}")
                return
            
            # Continue with analysis...
            self.update_analysis(df)
            self.loading_label.config(text="")
            
        except Exception as e:
            self.loading_label.config(text="")
            print(f"Error in analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            raise e

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

    def update_analysis(self, df):
        """Update analysis with trading system calculations"""
        try:
            current_price = df['Close'].iloc[-1]
            
            # Calculate position sizing
            position_data = self.calculate_position_sizing(current_price)
            if position_data is None:
                return
            
            # Get selected strategy
            strategy = self.strategy_var.get()
            
            # Calculate strategy signals
            signals = self.detect_ma_crossovers(df, self.calculate_moving_averages(df, self.duration_var.get())[1], self.calculate_moving_averages(df, self.duration_var.get())[2])
            
            # Update analysis text
            self.update_analysis_text(df, position_data, signals)
            
        except Exception as e:
            print(f"Error updating analysis: {str(e)}")
            raise e

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

    def update_analysis_text(self, df, position_data, signals):
        """Update analysis text with trading system information and strategy recommendation"""
        current_price = df['Close'].iloc[-1]
        
        # Calculate probability of ruin
        ruin_analysis = self.calculate_probability_of_ruin(df)
        
        # Get strategy recommendation
        strategy_recommendation = self.recommend_strategy(df)
        
        # Format analysis text
        analysis = f"Current Price: ${current_price:.2f}\n\n"
        
        # Add position analysis
        analysis += "Position Analysis:\n"
        analysis += f"Position Size: {position_data['position_size']:.2f} shares\n"
        analysis += f"Position Value: ${position_data['position_value']:.2f}\n"
        analysis += f"Risk Amount: ${position_data['risk_amount']:.2f}\n"
        analysis += f"Max Drawdown: ${position_data['max_drawdown']:.2f}\n\n"
        
        # Add probability analysis
        if ruin_analysis:
            analysis += self.update_probability_display(ruin_analysis)
            analysis += "\n"
        
        # Add strategy analysis
        if strategy_recommendation:
            analysis += self.update_strategy_recommendation(df)
        
        # Update text widgets
        self.risk_text.delete(1.0, tk.END)
        self.risk_text.insert(tk.END, analysis)

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
            
            # Adjust position size based on strategy performance if available
            if strategy_metrics:
                # Adjust for win rate
                win_rate_factor = strategy_metrics['win_rate'] if strategy_metrics['win_rate'] > 0 else 0.5
                
                # Adjust for volatility (using max drawdown as proxy)
                volatility_factor = 1 - abs(strategy_metrics['max_drawdown'])
                
                # Adjust for Sharpe ratio
                sharpe_factor = min(max(strategy_metrics['sharpe_ratio'], 0), 3) / 3
                
                # Calculate adjusted position size
                adjustment_factor = (win_rate_factor + volatility_factor + sharpe_factor) / 3
                adjusted_position_size = base_position_size * adjustment_factor
                
                # Apply Kelly criterion
                if strategy_metrics['win_rate'] > 0 and strategy_metrics['avg_return'] > 0:
                    kelly_fraction = (strategy_metrics['win_rate'] - 
                                    ((1 - strategy_metrics['win_rate']) / 
                                     (strategy_metrics['avg_return'] / abs(strategy_metrics['max_drawdown']))))
                    kelly_fraction = max(0, min(kelly_fraction, 1))
                else:
                    kelly_fraction = 0.5
                
                # Final position size
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
                'capital_at_risk': (position_value / capital) * 100
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
        """Update probability analysis display"""
        if not ruin_analysis:
            self.prob_text.delete(1.0, tk.END)
            self.prob_text.insert(tk.END, "Unable to calculate probability analysis")
            return
        
        # Format probability text
        prob_text = "Probability of Ruin Analysis\n"
        prob_text += "=" * 40 + "\n\n"
        
        prob_text += "Risk Metrics:\n"
        prob_text += f"Win Rate: {ruin_analysis['win_rate']:.2%}\n"
        prob_text += f"Edge per Trade: {ruin_analysis['edge_per_trade']:.2%}\n"
        prob_text += f"Optimal Position Size: {ruin_analysis['optimal_f']:.2%}\n"
        prob_text += f"Trades to Ruin: {ruin_analysis['risk_metrics']['trades_to_ruin']}\n\n"
        
        prob_text += "Probability of Ruin by Style:\n"
        prob_text += "-" * 40 + "\n"
        prob_text += "Style      | Probability | Risk Level\n"
        prob_text += "-" * 40 + "\n"
        
        for style, prob in ruin_analysis['weighted_outcomes'].items():
            risk_level = "Low" if prob < 25 else "Medium" if prob < 50 else "High"
            prob_text += f"{style:<10} | {prob:>8.2f}% | {risk_level:>9}\n"
        
        prob_text += "-" * 40 + "\n"
        
        # Update display
        self.prob_text.delete(1.0, tk.END)
        self.prob_text.insert(tk.END, prob_text)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            plt.close('all')  # Ensure all plots are closed when the app exits

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db_conn'):
            try:
                self.db_conn.close()
            except:
                pass

if __name__ == "__main__":
    # Set matplotlib backend
    matplotlib.use('TkAgg')
    
    # Create analyzer and GUI
    analyzer = StockMarketAnalyzer("stocks.db")
    gui = StockAnalyzerGUI(analyzer)
    gui.run()
