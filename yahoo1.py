import yfinance as yf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class StockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Viewer")
        
        # Configure grid weights to allow resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Create menu bar
        self.menubar = tk.Menu(root)
        self.root.config(menu=self.menubar)
        
        # Create File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.open_file)
        self.file_menu.add_command(label="Save", command=self.save_file)
        self.file_menu.add_command(label="Save As", command=self.save_as_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.quit)
        
        # Create Help menu
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Technical Indicators", command=self.show_indicator_help)
        self.help_menu.add_command(label="Trading Signals", command=self.show_signal_help)
        self.help_menu.add_command(label="Band Settings", command=self.show_band_help)
        self.help_menu.add_command(label="About", command=self.show_about)
        self.help_menu.add_command(label="Day Trading", command=self.show_daytrading_help)
        self.help_menu.add_command(label="Margin Trading", command=self.show_margin_help)
        
        # Create Trading Strategies menu
        self.strategy_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Trading Strategies", menu=self.strategy_menu)
        
        # Add day trading strategy submenus
        self.day_menu = tk.Menu(self.strategy_menu, tearoff=0)
        self.strategy_menu.add_cascade(label="Day Trading", menu=self.day_menu)
        
        # Add specific timeframe strategies
        self.day_menu.add_command(label="1-Day Scalping", command=lambda: self.apply_strategy("1d_scalp"))
        self.day_menu.add_command(label="2-Day Momentum", command=lambda: self.apply_strategy("2d_momentum"))
        self.day_menu.add_command(label="3-Day Swing", command=lambda: self.apply_strategy("3d_swing"))
        self.day_menu.add_command(label="5-Day Trend", command=lambda: self.apply_strategy("5d_trend"))
        
        # Add strategy settings
        self.strategy_settings = {
            "1d_scalp": {
                "atr_period": "5",
                "sr_period": "10",
                "momentum_period": "5",
                "rsi_period": "7",
                "macd_fast": "6",
                "macd_slow": "13",
                "macd_signal": "4",
                "bb_std": "1.5",
                "buy_threshold": "0.3",
                "sell_threshold": "-0.3"
            },
            "2d_momentum": {
                "atr_period": "8",
                "sr_period": "15",
                "momentum_period": "8",
                "rsi_period": "10",
                "macd_fast": "8",
                "macd_slow": "17",
                "macd_signal": "6",
                "bb_std": "2.0",
                "buy_threshold": "0.4",
                "sell_threshold": "-0.4"
            },
            "3d_swing": {
                "atr_period": "10",
                "sr_period": "20",
                "momentum_period": "10",
                "rsi_period": "12",
                "macd_fast": "10",
                "macd_slow": "21",
                "macd_signal": "7",
                "bb_std": "2.2",
                "buy_threshold": "0.5",
                "sell_threshold": "-0.5"
            },
            "5d_trend": {
                "atr_period": "14",
                "sr_period": "30",
                "momentum_period": "14",
                "rsi_period": "14",
                "macd_fast": "12",
                "macd_slow": "26",
                "macd_signal": "9",
                "bb_std": "2.5",
                "buy_threshold": "0.6",
                "sell_threshold": "-0.6"
            }
        }
        
        # Store current filename
        self.current_file = None
        
        # Create left control panel frame
        control_frame = ttk.Frame(root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create input frame (now inside control_frame)
        input_frame = ttk.Frame(control_frame, padding="5")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Ticker input
        ttk.Label(input_frame, text="Ticker Symbol:").grid(row=0, column=0, sticky=tk.W)
        self.ticker_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.ticker_var).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Plot title input
        ttk.Label(input_frame, text="Plot Title:").grid(row=1, column=0, sticky=tk.W)
        self.title_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.title_var).grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Period selection
        ttk.Label(input_frame, text="Time Period:").grid(row=2, column=0, sticky=tk.W)
        self.period_var = tk.StringVar(value="1y")
        period_choices = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
        ttk.Combobox(input_frame, textvariable=self.period_var, values=period_choices).grid(row=2, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Band settings
        ttk.Label(input_frame, text="Band Settings:").grid(row=3, column=0, sticky=tk.W)
        band_frame = ttk.Frame(input_frame)
        band_frame.grid(row=3, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Bollinger Band settings
        ttk.Label(band_frame, text="BB STD:").pack(side=tk.LEFT)
        self.bb_std_var = tk.StringVar(value="2")
        ttk.Entry(band_frame, textvariable=self.bb_std_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # RSI settings
        ttk.Label(band_frame, text="RSI Period:").pack(side=tk.LEFT, padx=(10,0))
        self.rsi_period_var = tk.StringVar(value="14")
        ttk.Entry(band_frame, textvariable=self.rsi_period_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # MACD settings
        macd_frame = ttk.Frame(input_frame)
        macd_frame.grid(row=4, column=1, padx=5)
        ttk.Label(macd_frame, text="MACD (Fast/Slow/Signal):").pack(side=tk.LEFT)
        self.macd_fast_var = tk.StringVar(value="12")
        self.macd_slow_var = tk.StringVar(value="26")
        self.macd_signal_var = tk.StringVar(value="9")
        ttk.Entry(macd_frame, textvariable=self.macd_fast_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(macd_frame, textvariable=self.macd_slow_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(macd_frame, textvariable=self.macd_signal_var, width=4).pack(side=tk.LEFT, padx=2)
        
        # Signal threshold settings
        signal_frame = ttk.Frame(input_frame)
        signal_frame.grid(row=5, column=1, padx=5)
        ttk.Label(signal_frame, text="Signal Thresholds (Buy/Sell):").pack(side=tk.LEFT)
        self.buy_threshold_var = tk.StringVar(value="0.5")
        self.sell_threshold_var = tk.StringVar(value="-0.5")
        ttk.Entry(signal_frame, textvariable=self.buy_threshold_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(signal_frame, textvariable=self.sell_threshold_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # Add Margin Trading frame
        margin_frame = ttk.LabelFrame(input_frame, text="Margin Trading Settings", padding="5")
        margin_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Margin ratio setting
        ttk.Label(margin_frame, text="Margin Ratio:").grid(row=0, column=0, sticky=tk.W)
        self.margin_ratio_var = tk.StringVar(value="2.0")  # 2:1 margin ratio
        ttk.Entry(margin_frame, textvariable=self.margin_ratio_var, width=5).grid(row=0, column=1, padx=5)
        
        # Stop loss percentage
        ttk.Label(margin_frame, text="Stop Loss %:").grid(row=0, column=2, sticky=tk.W)
        self.stop_loss_var = tk.StringVar(value="2.0")
        ttk.Entry(margin_frame, textvariable=self.stop_loss_var, width=5).grid(row=0, column=3, padx=5)
        
        # Add Short Selling menu
        self.short_menu = tk.Menu(self.strategy_menu, tearoff=0)
        self.strategy_menu.add_cascade(label="Short Selling", menu=self.short_menu)
        
        # Add short selling strategies
        self.short_menu.add_command(label="Bearish Trend", command=lambda: self.apply_short_strategy("bearish_trend"))
        self.short_menu.add_command(label="Volatility Short", command=lambda: self.apply_short_strategy("volatility_short"))
        self.short_menu.add_command(label="Technical Short", command=lambda: self.apply_short_strategy("technical_short"))
        self.short_menu.add_command(label="Margin Settings", command=self.show_margin_settings)
        
        # Plot button
        ttk.Button(input_frame, text="Plot", command=self.plot_stock).grid(row=6, column=0, columnspan=2, pady=10)
        
        # Create plot frame (now in column 1)
        self.plot_frame = ttk.Frame(root, padding="10")
        self.plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        
        # Add separator between control panel and plot
        ttk.Separator(root, orient='vertical').grid(row=0, column=0, sticky=(tk.N, tk.S), padx=(2, 0))

    def plot_stock(self):
        ticker = self.ticker_var.get().upper()
        title = self.title_var.get() or f"{ticker} Stock Price"
        period = self.period_var.get()
        
        # Clear previous plot if it exists
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            # Get user-defined parameters
            bb_std = float(self.bb_std_var.get())
            rsi_period = int(self.rsi_period_var.get())
            macd_fast = int(self.macd_fast_var.get())
            macd_slow = int(self.macd_slow_var.get())
            macd_signal = int(self.macd_signal_var.get())
            buy_threshold = float(self.buy_threshold_var.get())
            sell_threshold = float(self.sell_threshold_var.get())
            
            # Calculate basic moving averages
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            
            # Calculate Bollinger Bands with custom STD
            hist['BB_middle'] = hist['Close'].rolling(window=20).mean()
            hist['BB_upper'] = hist['BB_middle'] + bb_std * hist['Close'].rolling(window=20).std()
            hist['BB_lower'] = hist['BB_middle'] - bb_std * hist['Close'].rolling(window=20).std()
            
            # Calculate RSI with custom period
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD with custom periods
            exp1 = hist['Close'].ewm(span=macd_fast, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=macd_slow, adjust=False).mean()
            hist['MACD'] = exp1 - exp2
            hist['Signal_Line'] = hist['MACD'].ewm(span=macd_signal, adjust=False).mean()
            hist['MACD_Histogram'] = hist['MACD'] - hist['Signal_Line']
            
            # Calculate Volume and Returns
            hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
            hist['Daily_Return'] = hist['Close'].pct_change()
            volatility = hist['Daily_Return'].std() * (252 ** 0.5)
            
            # Add new trading signals analysis
            # 1. Trend Analysis
            hist['Trend'] = np.where(hist['MA20'] > hist['MA50'], 1, -1)
            
            # 2. Bollinger Band Signals
            hist['BB_Position'] = (hist['Close'] - hist['BB_middle']) / (hist['BB_upper'] - hist['BB_lower'])
            hist['BB_Signal'] = np.where(hist['BB_Position'] < -0.8, 1, 
                                       np.where(hist['BB_Position'] > 0.8, -1, 0))
            
            # 3. RSI Signals
            hist['RSI_Signal'] = np.where(hist['RSI'] < 30, 1, 
                                        np.where(hist['RSI'] > 70, -1, 0))
            
            # 4. MACD Signals
            hist['MACD_Signal_Cross'] = np.where(
                (hist['MACD'] > hist['Signal_Line']) & (hist['MACD'].shift(1) <= hist['Signal_Line'].shift(1)), 1,
                np.where((hist['MACD'] < hist['Signal_Line']) & (hist['MACD'].shift(1) >= hist['Signal_Line'].shift(1)), -1, 0))
            
            # 5. Volume Analysis
            hist['Volume_Signal'] = np.where(hist['Volume'] > hist['Volume_MA20'] * 1.5, 1, 0)
            
            # Combine Signals
            hist['Combined_Signal'] = (
                hist['Trend'] * 0.3 +
                hist['BB_Signal'] * 0.2 +
                hist['RSI_Signal'] * 0.2 +
                hist['MACD_Signal_Cross'] * 0.2 +
                hist['Volume_Signal'] * 0.1
            )
            
            # Update trading signals with custom thresholds
            hist['Trading_Signal'] = np.where(hist['Combined_Signal'] >= buy_threshold, 'Buy',
                                            np.where(hist['Combined_Signal'] <= sell_threshold, 'Sell', 'Hold'))
            
            # Calculate hypothetical returns
            hist['Strategy_Position'] = np.where(hist['Trading_Signal'] == 'Buy', 1,
                                               np.where(hist['Trading_Signal'] == 'Sell', -1, 0))
            hist['Strategy_Returns'] = hist['Strategy_Position'].shift(1) * hist['Daily_Return']
            
            # Calculate cumulative returns
            hist['Cumulative_Market_Returns'] = (1 + hist['Daily_Return']).cumprod()
            hist['Cumulative_Strategy_Returns'] = (1 + hist['Strategy_Returns']).cumprod()
            
            # Calculate additional bands
            # Standard Error Bands for Returns
            hist['Return_MA'] = hist['Daily_Return'].rolling(window=20).mean()
            hist['Return_Std'] = hist['Daily_Return'].rolling(window=20).std()
            hist['Return_Upper'] = hist['Return_MA'] + 2 * hist['Return_Std']
            hist['Return_Lower'] = hist['Return_MA'] - 2 * hist['Return_Std']
            
            # RSI Bands
            hist['RSI_MA'] = hist['RSI'].rolling(window=20).mean()
            hist['RSI_Upper'] = hist['RSI_MA'] + hist['RSI'].rolling(window=20).std()
            hist['RSI_Lower'] = hist['RSI_MA'] - hist['RSI'].rolling(window=20).std()
            
            # MACD Bands
            hist['MACD_Std'] = hist['MACD'].rolling(window=20).std()
            hist['MACD_Upper'] = hist['MACD'] + hist['MACD_Std']
            hist['MACD_Lower'] = hist['MACD'] - hist['MACD_Std']
            
            # Short Selling Indicators
            # 1. Trend Strength
            hist['Trend_Strength'] = hist['Close'].rolling(window=20).std() / hist['Close'].rolling(window=20).mean()
            
            # 2. Short Squeeze Risk
            hist['Short_Interest'] = (hist['Volume'] / hist['Volume'].rolling(window=20).mean()) * (hist['Close'] / hist['Close'].shift(1))
            hist['Squeeze_Risk'] = hist['Short_Interest'].rolling(window=5).mean()
            
            # 3. Margin Call Risk
            margin_ratio = float(self.margin_ratio_var.get())
            stop_loss = float(self.stop_loss_var.get()) / 100
            hist['Margin_Risk'] = (hist['High'] - hist['Low']) / (hist['Close'] * margin_ratio)
            
            # Short Selling Signals
            hist['Short_Signal'] = np.where(
                (hist['Close'] < hist['BB_lower']) &  # Price below lower band
                (hist['RSI'] > 70) &                  # Overbought
                (hist['MACD'] < hist['Signal_Line']) & # Bearish MACD
                (hist['Squeeze_Risk'] < 1.5) &        # Low squeeze risk
                (hist['Margin_Risk'] < stop_loss),    # Within risk tolerance
                -1,  # Short signal
                np.where(
                    (hist['Close'] > hist['BB_middle']) |  # Price above middle band
                    (hist['RSI'] < 30) |                   # Oversold
                    (hist['Squeeze_Risk'] > 2),            # High squeeze risk
                    1,   # Cover signal
                    0    # Hold
                )
            )
            
            # Create figure with subplots
            fig = Figure(figsize=(12, 18))
            
            # Price and Bollinger Bands subplot
            ax1 = fig.add_subplot(611)
            ax1.plot(hist['Close'], label='Close', alpha=0.7, marker='x', linestyle='-')
            ax1.plot(hist['MA20'], label='20-day MA', alpha=0.8, linestyle='--', color='orange')
            ax1.plot(hist['MA50'], label='50-day MA', alpha=0.8, linestyle='--', color='red')
            ax1.plot(hist['BB_upper'], label='Upper BB', color='gray', alpha=0.5, linestyle=':')
            ax1.plot(hist['BB_lower'], label='Lower BB', color='gray', alpha=0.5, linestyle=':')
            ax1.fill_between(hist.index, hist['BB_upper'], hist['BB_lower'], alpha=0.1, color='gray')
            ax1.set_title(f"{title}\nAnnualized Volatility: {volatility:.2%}")
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Volume subplot with bands
            ax2 = fig.add_subplot(612, sharex=ax1)
            ax2.bar(hist.index, hist['Volume'], alpha=0.7, color='gray', label='Volume')
            ax2.plot(hist.index, hist['Volume_MA20'], color='red', label='20-day Volume MA')
            volume_std = hist['Volume'].rolling(window=20).std()
            ax2.fill_between(hist.index, 
                            hist['Volume_MA20'] - 2*volume_std,
                            hist['Volume_MA20'] + 2*volume_std,
                            alpha=0.1, color='red')
            ax2.set_ylabel('Volume')
            ax2.grid(True)
            ax2.legend()
            
            # RSI subplot with bands
            ax3 = fig.add_subplot(613, sharex=ax1)
            ax3.plot(hist.index, hist['RSI'], label='RSI', color='purple')
            ax3.plot(hist.index, hist['RSI_MA'], label='RSI MA', color='blue', alpha=0.5)
            ax3.fill_between(hist.index, hist['RSI_Lower'], hist['RSI_Upper'], 
                            alpha=0.1, color='blue', label='RSI Bands')
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax3.fill_between(hist.index, hist['RSI'], 70, where=(hist['RSI'] >= 70), color='r', alpha=0.3)
            ax3.fill_between(hist.index, hist['RSI'], 30, where=(hist['RSI'] <= 30), color='g', alpha=0.3)
            ax3.set_ylabel('RSI')
            ax3.set_ylim(0, 100)
            ax3.grid(True)
            ax3.legend()
            
            # MACD subplot with bands
            ax4 = fig.add_subplot(614, sharex=ax1)
            ax4.plot(hist.index, hist['MACD'], label='MACD', color='blue')
            ax4.plot(hist.index, hist['Signal_Line'], label='Signal Line', color='orange')
            ax4.fill_between(hist.index, hist['MACD_Lower'], hist['MACD_Upper'],
                            alpha=0.1, color='blue', label='MACD Bands')
            ax4.bar(hist.index, hist['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.3)
            ax4.set_ylabel('MACD')
            ax4.grid(True)
            ax4.legend()
            
            # Daily returns subplot with confidence bands
            ax5 = fig.add_subplot(615, sharex=ax1)
            ax5.plot(hist.index, hist['Daily_Return'], label='Daily Returns', color='blue', alpha=0.6)
            ax5.plot(hist.index, hist['Return_MA'], label='Returns MA', color='red', alpha=0.8)
            ax5.fill_between(hist.index, hist['Return_Lower'], hist['Return_Upper'],
                            alpha=0.1, color='red', label='Confidence Bands')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax5.fill_between(hist.index, hist['Daily_Return'], 0, 
                            where=(hist['Daily_Return'] >= 0), color='green', alpha=0.3)
            ax5.fill_between(hist.index, hist['Daily_Return'], 0, 
                            where=(hist['Daily_Return'] < 0), color='red', alpha=0.3)
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Daily Returns')
            ax5.grid(True)
            ax5.legend()
            
            # Plot updates - add signals to price chart
            ax1.scatter(hist[hist['Trading_Signal'] == 'Buy'].index,
                       hist[hist['Trading_Signal'] == 'Buy']['Close'],
                       marker='^', color='green', s=100, label='Buy Signal')
            ax1.scatter(hist[hist['Trading_Signal'] == 'Sell'].index,
                       hist[hist['Trading_Signal'] == 'Sell']['Close'],
                       marker='v', color='red', s=100, label='Sell Signal')
            
            # Add new subplot for strategy performance
            ax6 = fig.add_subplot(616, sharex=ax1)
            ax6.plot(hist.index, hist['Cumulative_Market_Returns'], 
                    label='Buy & Hold Returns', color='blue', alpha=0.7)
            ax6.plot(hist.index, hist['Cumulative_Strategy_Returns'],
                    label='Strategy Returns', color='green', alpha=0.7)
            ax6.set_ylabel('Cumulative Returns')
            ax6.grid(True)
            ax6.legend()
            
            # Update statistical summary
            summary_stats = (
                f"Summary Statistics:\n"
                f"Latest Close: ${hist['Close'].iloc[-1]:.2f}\n"
                f"Latest RSI: {hist['RSI'].iloc[-1]:.1f}\n"
                f"20-day MA: ${hist['MA20'].iloc[-1]:.2f}\n"
                f"50-day MA: ${hist['MA50'].iloc[-1]:.2f}\n"
                f"BB Upper: ${hist['BB_upper'].iloc[-1]:.2f}\n"
                f"BB Lower: ${hist['BB_lower'].iloc[-1]:.2f}\n"
                f"MACD: {hist['MACD'].iloc[-1]:.3f}\n"
                f"Signal: {hist['Signal_Line'].iloc[-1]:.3f}\n"
                f"Current Signal: {hist['Trading_Signal'].iloc[-1]}\n"
                f"Strategy Return: {(hist['Cumulative_Strategy_Returns'].iloc[-1] - 1):.2%}\n"
                f"Buy & Hold Return: {(hist['Cumulative_Market_Returns'].iloc[-1] - 1):.2%}"
            )
            
            # Add short selling subplot
            ax8 = fig.add_subplot(818, sharex=ax1)
            ax8.plot(hist.index, hist['Squeeze_Risk'], label='Squeeze Risk', color='red')
            ax8.plot(hist.index, hist['Margin_Risk'] * 100, label='Margin Risk %', color='orange')
            ax8.axhline(y=1.5, color='r', linestyle='--', alpha=0.5)
            ax8.set_ylabel('Short Risks')
            ax8.grid(True)
            ax8.legend()
            
            # Update statistical summary
            summary_stats += (
                f"\nShort Selling Metrics:\n"
                f"Margin Ratio: {margin_ratio:.1f}:1\n"
                f"Stop Loss: {stop_loss:.1%}\n"
                f"Squeeze Risk: {hist['Squeeze_Risk'].iloc[-1]:.2f}\n"
                f"Margin Risk: {hist['Margin_Risk'].iloc[-1]:.2%}\n"
                f"Trend Strength: {hist['Trend_Strength'].iloc[-1]:.3f}"
            )
            
            # Adjust figure size for new subplot
            fig = Figure(figsize=(12, 18))
            
            # Price and Bollinger Bands subplot
            ax1 = fig.add_subplot(611)
            ax1.plot(hist['Close'], label='Close', alpha=0.7, marker='x', linestyle='-')
            ax1.plot(hist['MA20'], label='20-day MA', alpha=0.8, linestyle='--', color='orange')
            ax1.plot(hist['MA50'], label='50-day MA', alpha=0.8, linestyle='--', color='red')
            ax1.plot(hist['BB_upper'], label='Upper BB', color='gray', alpha=0.5, linestyle=':')
            ax1.plot(hist['BB_lower'], label='Lower BB', color='gray', alpha=0.5, linestyle=':')
            ax1.fill_between(hist.index, hist['BB_upper'], hist['BB_lower'], alpha=0.1, color='gray')
            ax1.set_title(f"{title}\nAnnualized Volatility: {volatility:.2%}")
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Volume subplot with bands
            ax2 = fig.add_subplot(612, sharex=ax1)
            ax2.bar(hist.index, hist['Volume'], alpha=0.7, color='gray', label='Volume')
            ax2.plot(hist.index, hist['Volume_MA20'], color='red', label='20-day Volume MA')
            volume_std = hist['Volume'].rolling(window=20).std()
            ax2.fill_between(hist.index, 
                            hist['Volume_MA20'] - 2*volume_std,
                            hist['Volume_MA20'] + 2*volume_std,
                            alpha=0.1, color='red')
            ax2.set_ylabel('Volume')
            ax2.grid(True)
            ax2.legend()
            
            # RSI subplot with bands
            ax3 = fig.add_subplot(613, sharex=ax1)
            ax3.plot(hist.index, hist['RSI'], label='RSI', color='purple')
            ax3.plot(hist.index, hist['RSI_MA'], label='RSI MA', color='blue', alpha=0.5)
            ax3.fill_between(hist.index, hist['RSI_Lower'], hist['RSI_Upper'], 
                            alpha=0.1, color='blue', label='RSI Bands')
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax3.fill_between(hist.index, hist['RSI'], 70, where=(hist['RSI'] >= 70), color='r', alpha=0.3)
            ax3.fill_between(hist.index, hist['RSI'], 30, where=(hist['RSI'] <= 30), color='g', alpha=0.3)
            ax3.set_ylabel('RSI')
            ax3.set_ylim(0, 100)
            ax3.grid(True)
            ax3.legend()
            
            # MACD subplot with bands
            ax4 = fig.add_subplot(614, sharex=ax1)
            ax4.plot(hist.index, hist['MACD'], label='MACD', color='blue')
            ax4.plot(hist.index, hist['Signal_Line'], label='Signal Line', color='orange')
            ax4.fill_between(hist.index, hist['MACD_Lower'], hist['MACD_Upper'],
                            alpha=0.1, color='blue', label='MACD Bands')
            ax4.bar(hist.index, hist['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.3)
            ax4.set_ylabel('MACD')
            ax4.grid(True)
            ax4.legend()
            
            # Daily returns subplot with confidence bands
            ax5 = fig.add_subplot(615, sharex=ax1)
            ax5.plot(hist.index, hist['Daily_Return'], label='Daily Returns', color='blue', alpha=0.6)
            ax5.plot(hist.index, hist['Return_MA'], label='Returns MA', color='red', alpha=0.8)
            ax5.fill_between(hist.index, hist['Return_Lower'], hist['Return_Upper'],
                            alpha=0.1, color='red', label='Confidence Bands')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax5.fill_between(hist.index, hist['Daily_Return'], 0, 
                            where=(hist['Daily_Return'] >= 0), color='green', alpha=0.3)
            ax5.fill_between(hist.index, hist['Daily_Return'], 0, 
                            where=(hist['Daily_Return'] < 0), color='red', alpha=0.3)
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Daily Returns')
            ax5.grid(True)
            ax5.legend()
            
            # Plot updates - add signals to price chart
            ax1.scatter(hist[hist['Trading_Signal'] == 'Buy'].index,
                       hist[hist['Trading_Signal'] == 'Buy']['Close'],
                       marker='^', color='green', s=100, label='Buy Signal')
            ax1.scatter(hist[hist['Trading_Signal'] == 'Sell'].index,
                       hist[hist['Trading_Signal'] == 'Sell']['Close'],
                       marker='v', color='red', s=100, label='Sell Signal')
            
            # Add new subplot for strategy performance
            ax6 = fig.add_subplot(616, sharex=ax1)
            ax6.plot(hist.index, hist['Cumulative_Market_Returns'], 
                    label='Buy & Hold Returns', color='blue', alpha=0.7)
            ax6.plot(hist.index, hist['Cumulative_Strategy_Returns'],
                    label='Strategy Returns', color='green', alpha=0.7)
            ax6.set_ylabel('Cumulative Returns')
            ax6.grid(True)
            ax6.legend()
            
            # Adjust layout
            fig.tight_layout()
            
            # Embed plot
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
        except Exception as e:
            error_label = ttk.Label(self.plot_frame, text=f"Error: {str(e)}")
            error_label.grid(row=0, column=0)

    def open_file(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as file:
                    data = file.read().split('\n')
                    if len(data) >= 3:
                        self.ticker_var.set(data[0])
                        self.title_var.set(data[1])
                        self.period_var.set(data[2])
                        self.current_file = filename
                        self.plot_stock()
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to open file: {str(e)}")

    def save_file(self):
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_as_file()

    def save_as_file(self):
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            self._save_to_file(filename)
            self.current_file = filename

    def _save_to_file(self, filename):
        try:
            with open(filename, 'w') as file:
                file.write(f"{self.ticker_var.get()}\n")
                file.write(f"{self.title_var.get()}\n")
                file.write(f"{self.period_var.get()}\n")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save file: {str(e)}")

    def show_indicator_help(self):
        help_text = """
Technical Indicators Explained:

1. Moving Averages (MA)
   - 20-day MA: Short-term trend indicator
   - 50-day MA: Medium-term trend indicator
   - Crossovers signal potential trend changes

2. Bollinger Bands (BB)
   - Middle Band: 20-day moving average
   - Upper/Lower Bands: Middle ± (Standard Deviation × Multiplier)
   - Measures volatility and potential overbought/oversold levels
   - Default multiplier is 2, adjustable in settings

3. Relative Strength Index (RSI)
   - Momentum oscillator measuring speed/change of price movements
   - Range: 0-100
   - Traditional levels: 
     * Above 70: Overbought
     * Below 30: Oversold
   - Period is adjustable (default: 14 days)

4. Moving Average Convergence Divergence (MACD)
   - Trend-following momentum indicator
   - Components:
     * MACD Line: Difference between fast and slow EMAs
     * Signal Line: EMA of MACD line
     * Histogram: MACD Line - Signal Line
   - Parameters adjustable:
     * Fast period (default: 12)
     * Slow period (default: 26)
     * Signal period (default: 9)

5. Volume Analysis
   - Volume MA: 20-day average trading volume
   - High volume: Confirms price movements
   - Volume bands: Shows unusual trading activity
    """
        
        self.show_help_window("Technical Indicators Help", help_text)

    def show_signal_help(self):
        help_text = """
Trading Signals Explained:

1. Combined Signal Algorithm
   Weighted combination of multiple factors:
   - Trend (30%): Based on MA crossovers
   - Bollinger Bands (20%): Mean reversion
   - RSI (20%): Momentum
   - MACD (20%): Trend confirmation
   - Volume (10%): Trading activity confirmation

2. Signal Generation
   - Buy Signal: Combined signal ≥ Buy threshold
   - Sell Signal: Combined signal ≤ Sell threshold
   - Hold: Between thresholds
   
3. Signal Components:
   a) Trend Analysis
      - Positive: MA20 > MA50
      - Negative: MA20 < MA50
   
   b) Bollinger Band Signals
      - Buy: Price near lower band
      - Sell: Price near upper band
   
   c) RSI Signals
      - Buy: RSI < 30 (oversold)
      - Sell: RSI > 70 (overbought)
   
   d) MACD Signals
      - Buy: MACD crosses above Signal Line
      - Sell: MACD crosses below Signal Line
   
   e) Volume Confirmation
      - Significant: Volume > 1.5 × Volume MA
      - Normal: Volume ≤ 1.5 × Volume MA

4. Performance Metrics
   - Strategy Returns: Cumulative returns from signals
   - Buy & Hold Returns: Market benchmark
   - Risk-adjusted metrics included in summary
    """
        
        self.show_help_window("Trading Signals Help", help_text)

    def show_band_help(self):
        help_text = """
Band Settings Explained:

1. Bollinger Bands (BB)
   - STD Multiplier: Controls band width
   - Higher value = Wider bands = Fewer signals
   - Lower value = Narrower bands = More signals
   - Traditional value: 2

2. RSI Settings
   - Period: Number of days for calculation
   - Shorter period = More sensitive
   - Longer period = More stable
   - Traditional period: 14 days

3. MACD Parameters
   - Fast Period: Short-term EMA
   - Slow Period: Long-term EMA
   - Signal Period: MACD smoothing
   - Traditional values: 12/26/9

4. Signal Thresholds
   - Buy: Trigger level for buy signals
   - Sell: Trigger level for sell signals
   - Higher thresholds = More conservative
   - Lower thresholds = More aggressive

5. Band Applications
   - Price Bands: Volatility measurement
   - Volume Bands: Trading activity range
   - Return Bands: Expected price movement
   - RSI Bands: Momentum range
   - MACD Bands: Trend strength
    """
        
        self.show_help_window("Band Settings Help", help_text)

    def show_about(self):
        about_text = """
Stock Price Viewer

A technical analysis tool for stock market data.

Features:
- Real-time stock data retrieval
- Multiple technical indicators
- Custom trading signals
- Adjustable parameters
- Performance analysis

Data provided by Yahoo Finance
    """
        
        self.show_help_window("About", about_text)

    def show_daytrading_help(self):
        help_text = """
Day Trading Indicators and Algorithms:

1. Average True Range (ATR)
   - Measures volatility
   - Higher ATR = Higher volatility
   - Used for stop-loss placement
   - Adjustable period (default: 14)

2. Support/Resistance Levels
   - Dynamic price levels
   - Based on rolling high/low
   - Breakout signals on level breach
   - Volume confirmation required

3. Price Momentum
   - Measures price velocity
   - Includes confidence bands
   - Signals:
     * Above upper band: Strong upward momentum
     * Below lower band: Strong downward momentum
   - Mean reversion potential at extremes

4. Volatility Bands
   - Based on price standard deviation
   - Breakout signals on band breach
   - Volume confirmation required
   - Adjusts to market conditions

5. Intraday Momentum Index (IMI)
   - Range: 0-100
   - Above 70: Overbought
   - Below 30: Oversold
   - Confirms trend strength

Day Trading Signal Generation:
1. Volatility Breakout (40% weight)
   - Price breaks volatility bands
   - Volume confirmation required
   - Most significant for day trading

2. Support/Resistance Break (30% weight)
   - Price breaks S/R levels
   - Volume confirmation needed
   - Medium-term significance

3. Momentum Signal (30% weight)
   - Combines momentum and IMI
   - Confirms trend strength
   - Short-term focus

Risk Management:
- Use ATR for stop-loss placement
- Consider volatility for position sizing
- Monitor volume for confirmation
- Watch multiple timeframes
- Use confidence bands for entry/exit
    """
        
        self.show_help_window("Day Trading Help", help_text)

    def show_margin_help(self):
        help_text = """
Short Selling and Margin Trading Strategies:

1. Bearish Trend Strategy
   - For strong downtrends
   - Conservative margin (2:1)
   - Standard stop loss (2%)
   - Focuses on trend following
   Best for: Clear downtrend markets

2. Volatility Short Strategy
   - For high volatility stocks
   - Lower margin (1.5:1)
   - Tighter stops (1.5%)
   - Quick entry/exit
   Best for: Volatile, overextended stocks

3. Technical Short Strategy
   - Based on technical signals
   - Standard margin (2:1)
   - Wider stops (2.5%)
   - Multiple indicator confirmation
   Best for: Technical breakdowns

Risk Management:
1. Margin Requirements
   - Maintenance margin minimum
   - Available buying power
   - Margin call thresholds

2. Short Squeeze Risk
   - Volume analysis
   - Short interest monitoring
   - Squeeze risk indicator

3. Stop Loss Management
   - Percentage-based stops
   - ATR-based stops
   - Margin call prevention

4. Position Sizing
   - Account for margin requirements
   - Risk per trade calculation
   - Portfolio exposure limits

Key Metrics:
- Trend Strength: Volatility/Price ratio
- Squeeze Risk: Volume and price pressure
- Margin Risk: Position risk vs margin
- Technical Signals: Entry/exit timing

Best Practices:
1. Monitor borrowing costs
2. Watch for dividend dates
3. Keep margin cushion
4. Use strict stop losses
5. Monitor short interest
6. Plan exit strategies
    """
        
        self.show_help_window("Margin Trading Help", help_text)

    def show_help_window(self, title, text):
        help_window = tk.Toplevel(self.root)
        help_window.title(title)
        help_window.geometry("600x800")
        
        # Create text widget with scrollbar
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(help_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Insert help text
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)  # Make text read-only

    def apply_strategy(self, strategy_name):
        """Apply selected trading strategy settings"""
        if strategy_name in self.strategy_settings:
            settings = self.strategy_settings[strategy_name]
            
            # Update all settings variables
            self.bb_std_var.set(settings["bb_std"])
            self.rsi_period_var.set(settings["rsi_period"])
            self.macd_fast_var.set(settings["macd_fast"])
            self.macd_slow_var.set(settings["macd_slow"])
            self.macd_signal_var.set(settings["macd_signal"])
            self.buy_threshold_var.set(settings["buy_threshold"])
            self.sell_threshold_var.set(settings["sell_threshold"])
            
            # Update period for shorter timeframes
            if strategy_name == "1d_scalp":
                self.period_var.set("1d")
            elif strategy_name == "2d_momentum":
                self.period_var.set("2d")
            elif strategy_name == "3d_swing":
                self.period_var.set("3d")
            elif strategy_name == "5d_trend":
                self.period_var.set("5d")
            
            # Automatically update plot
            self.plot_stock()

    def show_strategy_help(self):
        help_text = """
Day Trading Strategy Presets:

1. 1-Day Scalping Strategy
   - Very short-term trades (minutes to hours)
   - Tight Bollinger Bands (1.5 STD)
   - Quick RSI (7 periods)
   - Fast MACD (6/13/4)
   - Aggressive thresholds
   Best for: High-volume, liquid stocks
   
2. 2-Day Momentum Strategy
   - Short-term momentum trades
   - Moderate bands (2.0 STD)
   - RSI (10 periods)
   - Modified MACD (8/17/6)
   - Moderate thresholds
   Best for: Trending stocks with good volume
   
3. 3-Day Swing Strategy
   - Multi-day position trades
   - Wider bands (2.2 STD)
   - Standard RSI (12 periods)
   - Balanced MACD (10/21/7)
   - Standard thresholds
   Best for: Stocks with clear support/resistance
   
4. 5-Day Trend Strategy
   - Week-long trend following
   - Conservative bands (2.5 STD)
   - Traditional RSI (14 periods)
   - Standard MACD (12/26/9)
   - Conservative thresholds
   Best for: Strong trending stocks

Strategy Components:
- ATR Period: Volatility measurement
- S/R Period: Support/Resistance levels
- Momentum Period: Price momentum
- RSI Period: Oversold/Overbought
- MACD Settings: Trend confirmation
- BB STD: Band width
- Thresholds: Signal sensitivity

Usage Tips:
1. Match strategy to market conditions
2. Consider stock volatility
3. Monitor volume for confirmation
4. Use appropriate position sizing
5. Set stops based on ATR
6. Confirm signals across timeframes
    """
        
        self.show_help_window("Trading Strategies Help", help_text)

    def apply_short_strategy(self, strategy_name):
        """Apply short selling strategy settings"""
        short_settings = {
            "bearish_trend": {
                "margin_ratio": "2.0",
                "stop_loss": "2.0",
                "atr_period": "14",
                "momentum_period": "10",
                "rsi_period": "14",
                "macd_fast": "12",
                "macd_slow": "26",
                "macd_signal": "9",
                "bb_std": "2.0",
                "sell_threshold": "-0.4",
                "buy_threshold": "0.4"  # For covering shorts
            },
            "volatility_short": {
                "margin_ratio": "1.5",
                "stop_loss": "1.5",
                "atr_period": "10",
                "momentum_period": "5",
                "rsi_period": "10",
                "macd_fast": "8",
                "macd_slow": "17",
                "macd_signal": "6",
                "bb_std": "2.5",
                "sell_threshold": "-0.5",
                "buy_threshold": "0.5"
            },
            "technical_short": {
                "margin_ratio": "2.0",
                "stop_loss": "2.5",
                "atr_period": "14",
                "momentum_period": "14",
                "rsi_period": "14",
                "macd_fast": "12",
                "macd_slow": "26",
                "macd_signal": "9",
                "bb_std": "2.2",
                "sell_threshold": "-0.6",
                "buy_threshold": "0.6"
            }
        }
        
        if strategy_name in short_settings:
            settings = short_settings[strategy_name]
            
            # Update margin settings
            self.margin_ratio_var.set(settings["margin_ratio"])
            self.stop_loss_var.set(settings["stop_loss"])
            
            # Update other settings
            self.bb_std_var.set(settings["bb_std"])
            self.rsi_period_var.set(settings["rsi_period"])
            self.macd_fast_var.set(settings["macd_fast"])
            self.macd_slow_var.set(settings["macd_slow"])
            self.macd_signal_var.set(settings["macd_signal"])
            self.buy_threshold_var.set(settings["buy_threshold"])
            self.sell_threshold_var.set(settings["sell_threshold"])
            
            self.plot_stock()

    def show_margin_settings(self):
        """Display margin settings configuration window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Margin Trading Settings")
        settings_window.geometry("400x500")
        
        # Create main frame
        main_frame = ttk.Frame(settings_window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Margin Requirements
        margin_frame = ttk.LabelFrame(main_frame, text="Margin Requirements", padding="5")
        margin_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(margin_frame, text="Initial Margin Ratio:").grid(row=0, column=0, sticky=tk.W)
        initial_margin_var = tk.StringVar(value=self.margin_ratio_var.get())
        ttk.Entry(margin_frame, textvariable=initial_margin_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(margin_frame, text="Maintenance Margin (%):").grid(row=1, column=0, sticky=tk.W)
        maintenance_margin_var = tk.StringVar(value="25.0")
        ttk.Entry(margin_frame, textvariable=maintenance_margin_var, width=8).grid(row=1, column=1, padx=5)
        
        # Risk Management
        risk_frame = ttk.LabelFrame(main_frame, text="Risk Management", padding="5")
        risk_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(risk_frame, text="Stop Loss (%):").grid(row=0, column=0, sticky=tk.W)
        stop_loss_var = tk.StringVar(value=self.stop_loss_var.get())
        ttk.Entry(risk_frame, textvariable=stop_loss_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(risk_frame, text="Max Position Size (%):").grid(row=1, column=0, sticky=tk.W)
        position_size_var = tk.StringVar(value="20.0")
        ttk.Entry(risk_frame, textvariable=position_size_var, width=8).grid(row=1, column=1, padx=5)
        
        ttk.Label(risk_frame, text="Max Portfolio Short (%):").grid(row=2, column=0, sticky=tk.W)
        portfolio_short_var = tk.StringVar(value="50.0")
        ttk.Entry(risk_frame, textvariable=portfolio_short_var, width=8).grid(row=2, column=1, padx=5)
        
        # Short Squeeze Protection
        squeeze_frame = ttk.LabelFrame(main_frame, text="Short Squeeze Protection", padding="5")
        squeeze_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(squeeze_frame, text="Max Short Interest Ratio:").grid(row=0, column=0, sticky=tk.W)
        short_interest_var = tk.StringVar(value="15.0")
        ttk.Entry(squeeze_frame, textvariable=short_interest_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(squeeze_frame, text="Volume Threshold:").grid(row=1, column=0, sticky=tk.W)
        volume_threshold_var = tk.StringVar(value="1.5")
        ttk.Entry(squeeze_frame, textvariable=volume_threshold_var, width=8).grid(row=1, column=1, padx=5)
        
        # Alert Settings
        alert_frame = ttk.LabelFrame(main_frame, text="Alert Settings", padding="5")
        alert_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(alert_frame, text="Margin Call Warning (%):").grid(row=0, column=0, sticky=tk.W)
        margin_warning_var = tk.StringVar(value="30.0")
        ttk.Entry(alert_frame, textvariable=margin_warning_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(alert_frame, text="Squeeze Alert Level:").grid(row=1, column=0, sticky=tk.W)
        squeeze_alert_var = tk.StringVar(value="2.0")
        ttk.Entry(alert_frame, textvariable=squeeze_alert_var, width=8).grid(row=1, column=1, padx=5)
        
        # Information text
        info_text = """
Risk Management Guidelines:
• Initial Margin: Minimum required to open position
• Maintenance Margin: Minimum to maintain position
• Stop Loss: Automatic exit point
• Position Size: Maximum single position
• Portfolio Short: Total short exposure limit
• Short Interest: Maximum market short ratio
• Volume Threshold: Unusual volume multiplier
• Margin Warning: Early warning threshold
• Squeeze Alert: Short squeeze risk level
    """
        
        info_label = ttk.Label(main_frame, text=info_text, wraplength=380, justify=tk.LEFT)
        info_label.grid(row=4, column=0, pady=10)
        
        def apply_settings():
            """Apply the margin settings"""
            self.margin_ratio_var.set(initial_margin_var.get())
            self.stop_loss_var.set(stop_loss_var.get())
            
            # Store other settings as instance variables
            self.maintenance_margin = float(maintenance_margin_var.get())
            self.max_position_size = float(position_size_var.get())
            self.max_portfolio_short = float(portfolio_short_var.get())
            self.max_short_interest = float(short_interest_var.get())
            self.volume_threshold = float(volume_threshold_var.get())
            self.margin_warning = float(margin_warning_var.get())
            self.squeeze_alert = float(squeeze_alert_var.get())
            
            # Update plot with new settings
            self.plot_stock()
            settings_window.destroy()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Make window modal
        settings_window.transient(self.root)
        settings_window.grab_set()
        self.root.wait_window(settings_window)

def main():
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()