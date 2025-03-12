import numpy as np
import datetime
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import stats
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pickle
import tensorflow as tf

def _create_ui(self):
    # ... existing code ...
    
    # Add a frame for visualization controls
    self.viz_control_frame = ttk.Frame(self)
    self.viz_control_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Add plot type selector
    ttk.Label(self.viz_control_frame, text="Visualization:").pack(side=tk.LEFT, padx=(0, 5))
    self.plot_type_var = tk.StringVar(value="Standard Prediction")
    self.plot_selector = ttk.Combobox(self.viz_control_frame, 
                                      textvariable=self.plot_type_var,
                                      state="readonly",
                                      width=25)
    self.plot_selector['values'] = [
        "Standard Prediction",
        "Confidence Intervals",
        "Candlestick with Prediction",
        "Trade Signals",
        "Volatility Forecast",
        "Multi-Scenario Analysis",
        "Comparative Benchmark"
    ]
    self.plot_selector.pack(side=tk.LEFT, padx=5)
    self.plot_selector.bind("<<ComboboxSelected>>", self._on_plot_type_changed)
    
    # Optional parameters frame for specific visualizations
    self.viz_params_frame = ttk.LabelFrame(self, text="Visualization Parameters")
    self.viz_params_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Initial parameters widgets
    self._setup_viz_parameters()
    
    # ... rest of existing UI code ...

def _setup_viz_parameters(self):
    # Clear existing widgets
    for widget in self.viz_params_frame.winfo_children():
        widget.destroy()
    
    plot_type = self.plot_type_var.get()
    
    if plot_type == "Confidence Intervals":
        ttk.Label(self.viz_params_frame, text="Confidence Level:").grid(row=0, column=0, padx=5, pady=5)
        self.conf_level_var = tk.DoubleVar(value=0.95)
        conf_spinner = ttk.Spinbox(self.viz_params_frame, from_=0.5, to=0.99, increment=0.01, 
                                  textvariable=self.conf_level_var, width=5)
        conf_spinner.grid(row=0, column=1, padx=5, pady=5)
    
    elif plot_type == "Trade Signals":
        ttk.Label(self.viz_params_frame, text="Signal Threshold (%):").grid(row=0, column=0, padx=5, pady=5)
        self.signal_threshold_var = tk.DoubleVar(value=1.5)
        threshold_spinner = ttk.Spinbox(self.viz_params_frame, from_=0.1, to=10.0, increment=0.1, 
                                      textvariable=self.signal_threshold_var, width=5)
        threshold_spinner.grid(row=0, column=1, padx=5, pady=5)
    
    elif plot_type == "Multi-Scenario Analysis":
        ttk.Label(self.viz_params_frame, text="Bull Case Factor:").grid(row=0, column=0, padx=5, pady=5)
        self.bull_factor_var = tk.DoubleVar(value=1.5)
        bull_spinner = ttk.Spinbox(self.viz_params_frame, from_=1.0, to=3.0, increment=0.1, 
                                 textvariable=self.bull_factor_var, width=5)
        bull_spinner.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.viz_params_frame, text="Bear Case Factor:").grid(row=0, column=2, padx=5, pady=5)
        self.bear_factor_var = tk.DoubleVar(value=0.5)
        bear_spinner = ttk.Spinbox(self.viz_params_frame, from_=0.1, to=1.0, increment=0.1, 
                                 textvariable=self.bear_factor_var, width=5)
        bear_spinner.grid(row=0, column=3, padx=5, pady=5)
    
    # Add an apply button
    ttk.Button(self.viz_params_frame, text="Apply", 
               command=self._on_plot_type_changed).grid(row=1, column=0, columnspan=4, pady=10)

def _on_plot_type_changed(self, event=None):
    """Handle changes to the plot type selector"""
    plot_type = self.plot_type_var.get()
    
    # Update parameter widgets for the selected visualization
    self._setup_viz_parameters()
    
    # If we have data, update the visualization
    if hasattr(self, 'prediction_data') and self.prediction_data is not None:
        self._display_prediction()

def _display_prediction(self):
    """Display prediction with the selected visualization type"""
    if not hasattr(self, 'prediction_data') or self.prediction_data is None:
        return
    
    # Clear previous plot
    for widget in self.plot_frame.winfo_children():
        widget.destroy()
    
    # Create figure and add subplot
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Get actual and predicted values from prediction data
    actual = self.prediction_data.get('actual', [])
    predicted = self.prediction_data.get('predicted', [])
    dates = self.prediction_data.get('dates', list(range(len(actual))))
    
    # Get the selected plot type and generate appropriate visualization
    plot_type = self.plot_type_var.get()
    
    if plot_type == "Standard Prediction":
        self._plot_standard_prediction(ax, dates, actual, predicted)
    
    elif plot_type == "Confidence Intervals":
        confidence = self.conf_level_var.get()
        self._plot_confidence_intervals(ax, dates, actual, predicted, confidence)
    
    elif plot_type == "Candlestick with Prediction":
        self._plot_candlestick_prediction(ax, dates, actual, predicted)
    
    elif plot_type == "Trade Signals":
        threshold = self.signal_threshold_var.get() / 100  # Convert % to decimal
        self._plot_trade_signals(ax, dates, actual, predicted, threshold)
    
    elif plot_type == "Volatility Forecast":
        self._plot_volatility_forecast(ax, dates, actual, predicted)
    
    elif plot_type == "Multi-Scenario Analysis":
        bull_factor = self.bull_factor_var.get()
        bear_factor = self.bear_factor_var.get()
        self._plot_multi_scenario(ax, dates, actual, predicted, bull_factor, bear_factor)
    
    elif plot_type == "Comparative Benchmark":
        self._plot_benchmark_comparison(ax, dates, actual, predicted)
    
    # Set title
    ax.set_title(f"{self.ticker_var.get()} - {plot_type}")
    
    # Format the x-axis to show dates nicely
    if isinstance(dates[0], (datetime.date, datetime.datetime)):
        fig.autofmt_xdate()
    
    # Add the figure to the UI
    canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add toolbar
    toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def _plot_standard_prediction(self, ax, dates, actual, predicted):
    """Plot standard prediction line chart"""
    ax.plot(dates[:len(actual)], actual, label='Actual', color='blue')
    ax.plot(dates[:len(predicted)], predicted, label='Predicted', color='red', linestyle='--')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Price')

def _plot_confidence_intervals(self, ax, dates, actual, predicted, confidence=0.95):
    """Plot prediction with confidence intervals"""
    # Calculate RMSE between actual and predicted for historical data
    historical_end = min(len(actual), len(predicted))
    historical_actual = actual[:historical_end]
    historical_predicted = predicted[:historical_end]
    
    if historical_end > 0:
        rmse = np.sqrt(np.mean((np.array(historical_actual) - np.array(historical_predicted))**2))
        
        # Set the confidence interval width based on RMSE and confidence level
        z_score = stats.norm.ppf((1 + confidence) / 2)
        ci_width = z_score * rmse
        
        # Plot actual and predicted
        ax.plot(dates[:len(actual)], actual, label='Actual', color='blue')
        ax.plot(dates[:len(predicted)], predicted, label='Predicted', color='red', linestyle='--')
        
        # Add confidence intervals
        upper_bound = [p + ci_width for p in predicted]
        lower_bound = [p - ci_width for p in predicted]
        
        ax.fill_between(dates[:len(predicted)], lower_bound, upper_bound, 
                        color='red', alpha=0.2, label=f'{int(confidence*100)}% Confidence Interval')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Price')
    else:
        ax.text(0.5, 0.5, "Insufficient data for confidence intervals", 
                ha='center', va='center', transform=ax.transAxes)

def _plot_trade_signals(self, ax, dates, actual, predicted, threshold=0.015):
    """Plot with buy/sell trade signals based on prediction"""
    # Plot actual and predicted data
    ax.plot(dates[:len(actual)], actual, label='Actual', color='blue')
    ax.plot(dates[:len(predicted)], predicted, label='Predicted', color='red', linestyle='--')
    
    # Calculate potential signals based on predicted price movements
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    
    for i in range(1, len(predicted)):
        # Calculate predicted percent change
        pct_change = (predicted[i] - predicted[i-1]) / predicted[i-1]
        
        # Buy signal: predicted increase exceeds threshold
        if pct_change > threshold:
            buy_dates.append(dates[i])
            buy_prices.append(predicted[i])
        
        # Sell signal: predicted decrease exceeds threshold
        elif pct_change < -threshold:
            sell_dates.append(dates[i])
            sell_prices.append(predicted[i])
    
    # Plot buy signals
    ax.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy Signal')
    
    # Plot sell signals
    ax.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell Signal')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Price')

def _plot_candlestick_prediction(self, ax, dates, actual, predicted):
    """Plot candlestick chart with prediction overlay"""
    # For candlestick, we need OHLC data
    # Check if we have OHLC data in prediction_data
    if all(k in self.prediction_data for k in ['open', 'high', 'low', 'close']):
        ohlc_data = []
        for i, date in enumerate(dates[:len(self.prediction_data['close'])]):
            # Convert date to matplotlib date number if it's a datetime
            if isinstance(date, (datetime.date, datetime.datetime)):
                date_num = mdates.date2num(date)
            else:
                date_num = date
                
            ohlc_data.append((date_num, 
                             self.prediction_data['open'][i],
                             self.prediction_data['high'][i], 
                             self.prediction_data['low'][i],
                             self.prediction_data['close'][i]))
        
        # Plot candlestick
        from mplfinance.original_flavor import candlestick_ohlc
        candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='green', colordown='red')
        
        # Plot prediction line
        ax.plot(dates[:len(predicted)], predicted, label='Predicted', color='blue', linestyle='--', linewidth=2)
        
        # Format x-axis as dates
        if isinstance(dates[0], (datetime.date, datetime.datetime)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Price')
    else:
        # If we don't have OHLC data, fall back to standard plot
        self._plot_standard_prediction(ax, dates, actual, predicted)
        ax.set_title("OHLC data not available - showing standard prediction")

def _plot_volatility_forecast(self, ax, dates, actual, predicted):
    """Plot price prediction with volatility forecast"""
    # Calculate historical volatility from actual prices
    if len(actual) > 5:  # Need a minimum number of points
        returns = np.diff(actual) / actual[:-1]
        historical_vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        # Calculate forecasted volatility (simple demonstration)
        # In a real model, this would come from a volatility prediction model
        forecast_vol = historical_vol * np.ones(len(predicted))
        
        # Adjust for trend - volatility tends to increase when price decreases
        for i in range(1, len(predicted)):
            if predicted[i] < predicted[i-1]:
                forecast_vol[i] = forecast_vol[i-1] * 1.05  # Increase by 5%
            else:
                forecast_vol[i] = max(forecast_vol[i-1] * 0.98, historical_vol * 0.5)  # Decrease but maintain minimum
        
        # Plot actual and predicted price
        ax.plot(dates[:len(actual)], actual, label='Actual', color='blue')
        ax.plot(dates[:len(predicted)], predicted, label='Predicted', color='red', linestyle='--')
        
        # Create a twin axis for volatility
        ax2 = ax.twinx()
        ax2.plot(dates[:len(forecast_vol)], forecast_vol * 100, label='Forecast Volatility (%)', 
                color='purple', alpha=0.7)
        ax2.set_ylabel('Volatility (%)', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.set_ylim(bottom=0)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Price')
    else:
        ax.text(0.5, 0.5, "Insufficient data for volatility forecast", 
                ha='center', va='center', transform=ax.transAxes)

def _plot_multi_scenario(self, ax, dates, actual, predicted, bull_factor=1.5, bear_factor=0.5):
    """Plot multiple prediction scenarios (bull, base, bear cases)"""
    # Base case is the standard prediction
    base_case = predicted
    
    # Create bull and bear cases by applying factors to the predicted changes
    bull_case = []
    bear_case = []
    
    for i in range(len(predicted)):
        if i == 0:
            # First point is the same in all scenarios
            bull_case.append(predicted[0])
            bear_case.append(predicted[0])
        else:
            # Calculate the change from the previous prediction
            change = predicted[i] - predicted[i-1]
            
            # Apply the factors to create bull and bear cases
            bull_change = change * bull_factor if change > 0 else change
            bear_change = change * bear_factor if change < 0 else change
            
            bull_case.append(bull_case[i-1] + bull_change)
            bear_case.append(bear_case[i-1] + bear_change)
    
    # Plot all scenarios
    ax.plot(dates[:len(actual)], actual, label='Actual', color='blue')
    ax.plot(dates[:len(base_case)], base_case, label='Base Case', color='green', linestyle='--')
    ax.plot(dates[:len(bull_case)], bull_case, label='Bull Case', color='darkgreen', linestyle=':')
    ax.plot(dates[:len(bear_case)], bear_case, label='Bear Case', color='darkred', linestyle=':')
    
    # Fill between bull and bear cases
    ax.fill_between(dates[:len(predicted)], bear_case, bull_case, color='gray', alpha=0.2)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Price')

def _plot_benchmark_comparison(self, ax, dates, actual, predicted):
    """Plot prediction compared to a benchmark (e.g., S&P 500 or sector index)"""
    # In a real implementation, you would fetch benchmark data
    # For demonstration, we'll generate synthetic benchmark data
    
    # Generate a benchmark that follows the general market trend with some variation
    benchmark = []
    market_bias = np.random.uniform(-0.1, 0.1)  # Random overall market direction
    
    for i in range(len(predicted)):
        if i == 0:
            benchmark.append(predicted[0])
        else:
            # Base change on actual data with some deviation
            market_change = ((i / len(predicted)) * market_bias) + np.random.uniform(-0.005, 0.005)
            benchmark.append(benchmark[i-1] * (1 + market_change))
    
    # Scale benchmark to match the starting price
    benchmark = np.array(benchmark) * (predicted[0] / benchmark[0])
    
    # Calculate relative performance (normalized to starting point)
    if len(actual) > 0 and len(predicted) > 0:
        normalized_actual = np.array(actual) / actual[0]
        normalized_predicted = np.array(predicted) / predicted[0]
        normalized_benchmark = np.array(benchmark) / benchmark[0]
        
        # Plot normalized values
        ax.plot(dates[:len(normalized_actual)], normalized_actual, 
                label='Actual (normalized)', color='blue')
        ax.plot(dates[:len(normalized_predicted)], normalized_predicted, 
                label='Predicted (normalized)', color='red', linestyle='--')
        ax.plot(dates[:len(normalized_benchmark)], normalized_benchmark, 
                label='Market Benchmark (normalized)', color='gray', linestyle='-.')
        
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Relative Performance (normalized)')
    else:
        ax.text(0.5, 0.5, "Insufficient data for benchmark comparison", 
                ha='center', va='center', transform=ax.transAxes)

def _create_prediction_controls(self):
    """Create the prediction controls section without a Save Model button"""
    controls_frame = ttk.LabelFrame(self, text="Prediction Controls")
    controls_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Ticker selection
    ticker_frame = ttk.Frame(controls_frame)
    ticker_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(ticker_frame, text="Ticker:").pack(side=tk.LEFT, padx=5)
    self.ticker_var = tk.StringVar()
    self.ticker_combo = ttk.Combobox(ticker_frame, textvariable=self.ticker_var, state="readonly")
    self.ticker_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Model selection
    model_frame = ttk.Frame(controls_frame)
    model_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
    self.model_var = tk.StringVar()
    self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
    self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    self.refresh_model_list()
    
    # Button frame for prediction controls
    button_frame = ttk.Frame(controls_frame)
    button_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Prediction button
    predict_btn = ttk.Button(button_frame, text="Predict", command=self._predict)
    predict_btn.pack(side=tk.LEFT, padx=5)
    
    # ... any other buttons ...

def _predict(self):
    """Generate predictions using the selected model"""
    try:
        # ... existing code ...
        
        # Load and store the model for later saving
        self.model = model  # Store the model instance
        self.model_type = model_type  # Store the model type
        
        # Try to load model history if it exists
        try:
            history_path = model_path.replace('.keras', '_history.pkl')
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    self.model_history = pickle.load(f)
            else:
                self.model_history = None
        except Exception as e:
            print(f"Could not load model history: {e}")
            self.model_history = None
            
        # ... rest of your prediction code ...
            
    except Exception as e:
        # ... existing error handling ... 

def add_save_model_button(self):
    """Add a Save Model button to the prediction panel"""
    try:
        # Find an appropriate parent widget
        # Look for existing button frames in the panel
        button_frames = [child for child in self.winfo_children() 
                        if isinstance(child, ttk.Frame) and any(isinstance(grandchild, ttk.Button)
                                                                for grandchild in child.winfo_children())]
        
        if button_frames:
            # Add to the first button frame found
            button_frame = button_frames[0]
        else:
            # Create a new frame if no button frames exist
            button_frame = ttk.Frame(self)
            button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create and add the button
        save_model_btn = ttk.Button(button_frame, text="Save Model", command=self._save_current_model)
        save_model_btn.pack(side=tk.LEFT, padx=5)
        print("Added Save Model button to prediction panel")
        
    except Exception as e:
        print(f"Error adding Save Model button: {e}") 