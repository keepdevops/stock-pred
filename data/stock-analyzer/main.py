#!/usr/bin/env python3
"""
Stock Market Analyzer - Main Entry Point
"""
import os
import sys
import tkinter as tk
from tkinter import ttk
import numpy as np
import scipy.stats as stats
import tensorflow as tf
from tensorflow.keras.models import Model
from model_save_patch import apply_model_save_patch
from model_view_patch import apply_view_model_patch
from training_panel_patch import apply_training_panel_patch

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set environment variable to silence Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Force CPU usage for TensorFlow to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_DISABLE_GRAPPLER'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.set_floatx('float32')

# Store the original save method
original_save = tf.keras.models.Model.save

# Create a wrapper function that forces .keras extension
def patched_save(self, filepath, *args, **kwargs):
    """Patched save function that ensures .keras extension"""
    import os
    # Force save_format to 'keras'
    kwargs['save_format'] = 'keras'
    
    # Ensure filepath has .keras extension
    if not filepath.endswith('.keras'):
        # If file has another extension, replace it
        if '.' in os.path.basename(filepath):
            filepath = os.path.splitext(filepath)[0] + '.keras'
        else:
            # If no extension, add .keras
            filepath = filepath + '.keras'
    
    print(f"\n>>> PATCHED SAVE: Saving model to: {filepath}")
    # Call the original save method with our modified arguments
    result = original_save(self, filepath, *args, **kwargs)
    print(f">>> PATCHED SAVE: Model saved successfully: {os.path.exists(filepath)}")
    return result

# Replace the save method with our patched version
tf.keras.models.Model.save = patched_save

print("\n=================================================")
print("| TensorFlow Model.save() patched successfully    |")
print("| All models will now be saved with .keras format |")
print("=================================================\n")

# Apply auto-save patch
apply_model_save_patch()

# Try to apply training panel patch (but don't crash if it fails)
try:
    apply_training_panel_patch()
except Exception as e:
    print(f"Warning: Could not apply training panel patch: {e}")
    print("Model saving will still work, but ticker detection might be limited.")

apply_view_model_patch()

from ui.main_window import StockAnalyzerApp
from controllers.model_controller import ModelController
from utils.event_bus import event_bus
from ui.visualization.training_results_panel import TrainingResultsPanel

def main():
    """Main entry point for the application"""
    app = StockAnalyzerApp()
    # Initialize model controller
    model_controller = ModelController(event_bus)
    print("Model controller initialized")
    app.run()

def plot_predictions_with_confidence(ax, dates, predictions, confidence=0.95):
    # Calculate confidence bands (this is simplified, actual calculation depends on your model)
    std_dev = predictions * 0.1  # Example: 10% of prediction value
    z_score = 1.96  # For 95% confidence
    upper_bound = predictions + (z_score * std_dev)
    lower_bound = predictions - (z_score * std_dev)
    
    # Plot prediction line with confidence interval
    ax.plot(dates, predictions, 'b-', label='Prediction')
    ax.fill_between(dates, lower_bound, upper_bound, color='b', alpha=0.2, 
                   label=f'{confidence*100}% Confidence Interval')

def plot_ensemble_predictions(ax, dates, predictions_dict):
    # Plot each model's predictions
    for model_name, preds in predictions_dict.items():
        ax.plot(dates, preds, label=f'{model_name} Prediction')
    
    # Plot ensemble average
    ensemble_avg = np.mean(list(predictions_dict.values()), axis=0)
    ax.plot(dates, ensemble_avg, 'k-', linewidth=2, label='Ensemble Average')

def plot_with_support_resistance(ax, dates, actuals, predictions, support_levels, resistance_levels):
    """Plot predictions with key support and resistance levels"""
    ax.plot(dates[:-len(predictions)], actuals[:len(actuals)-len(predictions)], 'k-', label='Historical')
    ax.plot(dates[-len(predictions):], predictions, 'b-', label='Prediction')
    
    # Plot support levels
    for level in support_levels:
        ax.axhline(y=level, linestyle='--', color='green', alpha=0.7)
        ax.text(dates[0], level, f'Support: {level:.2f}', verticalalignment='bottom')
    
    # Plot resistance levels
    for level in resistance_levels:
        ax.axhline(y=level, linestyle='--', color='red', alpha=0.7)
        ax.text(dates[0], level, f'Resistance: {level:.2f}', verticalalignment='top')
    
    ax.set_title('Prediction with Support/Resistance Levels')
    ax.legend(loc='best')

def plot_trading_signals(ax, dates, predictions, buy_threshold=0.02, sell_threshold=-0.02):
    """Visualize potential buy/sell signals based on predictions"""
    # Calculate daily returns
    pred_returns = [0] + [(predictions[i] - predictions[i-1])/predictions[i-1] 
                          for i in range(1, len(predictions))]
    
    # Plot predictions
    ax.plot(dates, predictions, 'b-', label='Prediction')
    
    # Find buy/sell signals
    buy_signals = [i for i, ret in enumerate(pred_returns) if ret > buy_threshold]
    sell_signals = [i for i, ret in enumerate(pred_returns) if ret < sell_threshold]
    
    # Plot signals
    for idx in buy_signals:
        ax.plot(dates[idx], predictions[idx], 'g^', markersize=10)
        
    for idx in sell_signals:
        ax.plot(dates[idx], predictions[idx], 'rv', markersize=10)
    
    if buy_signals:
        ax.plot([], [], 'g^', markersize=10, label='Buy Signal')
    if sell_signals:
        ax.plot([], [], 'rv', markersize=10, label='Sell Signal')
        
    ax.set_title('Price Prediction with Trading Signals')
    ax.legend(loc='best')

def plot_sector_comparison(ax, dates, ticker_pred, sector_preds):
    # Normalize predictions to same starting point
    norm_ticker = ticker_pred / ticker_pred[0]
    
    # Plot ticker prediction
    ax.plot(dates, norm_ticker, 'b-', linewidth=2, label=f'{ticker} (predicted)')
    
    # Plot sector predictions
    for sector, preds in sector_preds.items():
        norm_sector = preds / preds[0]
        ax.plot(dates, norm_sector, '--', alpha=0.7, label=f'{sector} (predicted)')
        
    ax.set_ylabel('Normalized Price')

def plot_volatility_forecast(ax, dates, price_predictions, volatility_predictions):
    """Plot price predictions alongside volatility predictions"""
    ax1 = ax
    ax1.plot(dates, price_predictions, 'b-', label='Price Prediction')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()  # Create second y-axis
    ax2.plot(dates, volatility_predictions, 'r-', label='Volatility')
    ax2.set_ylabel('Volatility', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_title('Price and Volatility Predictions')
    ax1.grid(True, alpha=0.3)

def plot_scenarios(ax, dates, predictions, bull_factor=1.10, bear_factor=0.90):
    """Plot bull/bear/base scenarios based on prediction"""
    base_case = predictions
    bull_case = predictions * bull_factor
    bear_case = predictions * bear_factor
    
    ax.plot(dates, base_case, 'b-', label='Base Case')
    ax.plot(dates, bull_case, 'g-', label='Bull Case')
    ax.plot(dates, bear_case, 'r-', label='Bear Case')
    
    # Fill between bull and bear cases
    ax.fill_between(dates, bear_case, bull_case, color='gray', alpha=0.2)
    
    ax.set_title('Prediction Scenarios')
    ax.legend(loc='best')

def update_visualization(self, event=None):
    viz_type = self.viz_var.get()
    ticker = self.ticker_var.get()
    
    if not ticker:
        return
        
    # Clear previous plot
    self.figure.clear()
    
    # Create new plot based on selected visualization
    if viz_type == "Basic Prediction":
        self._plot_basic_prediction(ticker)
    elif viz_type == "Confidence Intervals":
        self._plot_with_confidence(ticker)
    elif viz_type == "Ensemble Predictions":
        self._plot_ensemble(ticker)
    elif viz_type == "Multi-Horizon Prediction":
        self._plot_multi_horizon_prediction(ticker)
    # ... and so on for each visualization type
    
    # Refresh canvas
    self.canvas.draw()

def plot_multi_horizon_prediction(ax, dates, actual, predictions_dict):
    """Plot predictions at different time horizons (1-day, 5-day, 10-day, etc.)"""
    ax.plot(dates[:-len(predictions_dict['1-day'])], actual[:len(actual)-len(predictions_dict['1-day'])], 'k-', label='Historical')
    
    colors = ['blue', 'green', 'red', 'purple']
    for i, (horizon, preds) in enumerate(predictions_dict.items()):
        pred_dates = dates[-len(preds):]
        ax.plot(pred_dates, preds, color=colors[i % len(colors)], linestyle='--', label=f'{horizon} Prediction')
    
    ax.set_title('Multi-Horizon Predictions')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

def plot_prediction_with_confidence(ax, dates, actuals, predictions, std_devs, confidence=0.95):
    """Plot predictions with confidence intervals"""
    # Calculate z-score for confidence level
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate confidence bands
    upper_bound = predictions + (z * std_devs)
    lower_bound = predictions - (z * std_devs)
    
    # Plot
    ax.plot(dates[:-len(predictions)], actuals[:len(actuals)-len(predictions)], 'b-', label='Historical')
    ax.plot(dates[-len(predictions):], predictions, 'r-', label='Prediction')
    ax.fill_between(dates[-len(predictions):], lower_bound, upper_bound, 
                    color='red', alpha=0.2, label=f'{confidence*100:.0f}% Confidence')
    
    ax.set_title('Price Prediction with Confidence Intervals')
    ax.legend(loc='best')

def plot_drawdown_prediction(ax, dates, predictions, max_drawdowns):
    """Visualize predicted maximum drawdowns"""
    ax.plot(dates, predictions, 'b-', label='Prediction')
    
    # Create shaded regions for drawdown risk
    for i, drawdown in enumerate(max_drawdowns):
        bottom_line = predictions[i] * (1 - drawdown)
        ax.fill_between([dates[i]], [predictions[i]], [bottom_line], 
                        color='red', alpha=0.2)
    
    ax.set_title('Price Prediction with Max Drawdown Risk')
    ax.legend(loc='best')

def plot_vs_benchmark(ax, dates, stock_predictions, benchmark_predictions):
    """Compare stock predictions against benchmark (index/sector)"""
    # Normalize to same starting point
    norm_stock = stock_predictions / stock_predictions[0]
    norm_benchmark = benchmark_predictions / benchmark_predictions[0]
    
    ax.plot(dates, norm_stock, 'b-', label='Stock Prediction')
    ax.plot(dates, norm_benchmark, 'r-', label='Benchmark Prediction')
    
    # Calculate relative performance
    outperformance = ((norm_stock[-1] / norm_benchmark[-1]) - 1) * 100
    title = f'Relative Performance: {outperformance:.2f}% vs Benchmark'
    
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_ylabel('Normalized Price')

def _create_ui(self):
    # ... existing UI code ...
    
    # Add visualization selector
    self.viz_frame = ttk.LabelFrame(self, text="Prediction Visualization")
    self.viz_frame.pack(fill="x", padx=10, pady=5)
    
    self.viz_var = tk.StringVar(value="Basic Prediction")
    self.viz_combo = ttk.Combobox(
        self.viz_frame,
        textvariable=self.viz_var,
        values=[
            "Basic Prediction", 
            "Confidence Intervals",
            "Volatility Forecast",
            "Support/Resistance",
            "Trading Signals",
            "Scenarios",
            "Benchmark Comparison"
        ],
        state="readonly"
    )
    self.viz_combo.pack(side="left", padx=5)
    self.viz_combo.bind("<<ComboboxSelected>>", self._update_prediction_view)

def initialize_panels(self):
    # ... existing code ...
    self.prediction_panel = PredictionPanel(...)
    # If you need to add a save model button:
    if hasattr(self.prediction_panel, 'add_save_model_button'):
        self.prediction_panel.add_save_model_button()
    # ... rest of the method ...

def add_direct_save_button(self):
    """Add a direct save button to the main window"""
    import tkinter as tk
    from tkinter import ttk, messagebox
    
    print("Adding direct save button to main window...")
    
    # Create a frame at the top of the window
    save_frame = ttk.Frame(self)
    save_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    
    # Add the button
    save_btn = ttk.Button(
        save_frame,
        text="SAVE TRAINED MODEL (.KERAS)",
        command=self.save_trained_model
    )
    save_btn.pack(pady=5)
    
    print("Direct save button added to main window")

def save_trained_model(self):
    """Save the model from the training panel"""
    print("Attempting to save model from training panel...")
    
    # Try to access the training panel
    if hasattr(self, 'training_panel'):
        # If the panel has a direct save function, call it
        if hasattr(self.training_panel, 'direct_save_model'):
            self.training_panel.direct_save_model()
        else:
            # No save function, but maybe we can access the model directly
            from tkinter import messagebox
            import os
            import time
            
            if hasattr(self.training_panel, 'trained_model'):
                try:
                    # Get model
                    model = self.training_panel.trained_model
                    
                    # Save path
                    models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
                    os.makedirs(models_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    model_path = os.path.join(models_dir, f"SAVED_MODEL_{timestamp}.keras")
                    
                    # Save model
                    model.save(model_path, save_format='keras')
                    messagebox.showinfo("Save Successful", f"Model saved to: {model_path}")
                    
                except Exception as e:
                    messagebox.showerror("Save Error", f"Error saving model: {str(e)}")
            else:
                messagebox.showinfo("Save Error", "No trained model found in training panel")
    else:
        from tkinter import messagebox
        messagebox.showinfo("Error", "Training panel not found")

if __name__ == "__main__":
    main()
