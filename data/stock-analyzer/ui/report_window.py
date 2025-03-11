"""
Report window for displaying prediction reports and analysis
"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ReportWindow:
    """Window for displaying detailed prediction reports"""
    
    def __init__(self, parent, ticker, predictions, days, model_info):
        """Initialize the report window
        
        Args:
            parent: The parent window
            ticker: The ticker symbol
            predictions: Array of predicted prices
            days: Number of days predicted
            model_info: Dictionary containing model information
        """
        self.parent = parent
        self.ticker = ticker
        self.predictions = predictions
        self.days = days
        self.model_info = model_info
        
        # Configure the window
        self.parent.title(f"Prediction Report - {ticker}")
        self.parent.geometry("800x600")
        self.parent.minsize(600, 500)
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create header with summary information
        self._create_header()
        
        # Create report sections
        self._create_price_section()
        self._create_metrics_section()
        self._create_recommendation_section()
        
        # Button to save report
        self.save_button = ttk.Button(self.parent, text="Save Report", command=self._save_report)
        self.save_button.pack(side="bottom", pady=10)
    
    def _create_header(self):
        """Create the header section with summary information"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill="x", pady=10)
        
        # Title
        title = ttk.Label(header_frame, text=f"Prediction Report for {self.ticker}", 
                          font=("Helvetica", 16, "bold"))
        title.pack(anchor="w")
        
        # Date range
        today = datetime.now()
        end_date = today + timedelta(days=self.days-1)
        date_range = ttk.Label(header_frame, 
                              text=f"Period: {today.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({self.days} days)",
                              font=("Helvetica", 10))
        date_range.pack(anchor="w")
        
        # Model info
        if self.model_info:
            model_name = self.model_info.get('model').__class__.__name__
            model_info = ttk.Label(header_frame, text=f"Model: {model_name}", font=("Helvetica", 10))
            model_info.pack(anchor="w")
        
        # Summary metrics
        if len(self.predictions) > 0:
            start_price = self.predictions[0]
            end_price = self.predictions[-1]
            change = end_price - start_price
            pct_change = (change / start_price) * 100
            
            # Display prediction summary
            direction = "increase" if change >= 0 else "decrease"
            summary = ttk.Label(header_frame, 
                               text=f"Predicted to {direction} from ${start_price:.2f} to ${end_price:.2f} (${change:.2f}, {pct_change:.2f}%)",
                               font=("Helvetica", 12, "bold"))
            summary.pack(anchor="w", pady=(10, 0))
    
    def _create_price_section(self):
        """Create the price prediction chart section"""
        price_frame = ttk.LabelFrame(self.main_frame, text="Price Forecast")
        price_frame.pack(fill="both", expand=True, pady=10)
        
        # Create figure for price chart
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create date range for x-axis
        today = datetime.now()
        dates = [today + timedelta(days=i) for i in range(len(self.predictions))]
        
        # Calculate prediction uncertainty
        uncertainty = np.linspace(0.01, 0.10, len(self.predictions)) * np.mean(self.predictions)
        lower_bound = self.predictions - uncertainty
        upper_bound = self.predictions + uncertainty
        
        # Plot predicted prices
        ax.plot(dates, self.predictions, label='Predicted Price', color='blue', linewidth=2)
        
        # Add confidence interval
        ax.fill_between(dates, lower_bound, upper_bound, alpha=0.2, color='blue',
                       label='Prediction Uncertainty')
        
        # Add title and labels
        ax.set_title(f'{self.ticker} - Price Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.grid(True)
        ax.legend()
        
        # Format dates on x-axis
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, price_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, price_frame)
        toolbar.update()
    
    def _create_metrics_section(self):
        """Create the metrics section with prediction statistics"""
        metrics_frame = ttk.LabelFrame(self.main_frame, text="Prediction Metrics")
        metrics_frame.pack(fill="x", pady=10)
        
        # Create a grid for metrics
        for i in range(3):
            metrics_frame.columnconfigure(i, weight=1)
        
        # Calculate metrics
        if len(self.predictions) > 0:
            start_price = self.predictions[0]
            end_price = self.predictions[-1]
            high = max(self.predictions)
            low = min(self.predictions)
            avg = np.mean(self.predictions)
            
            # First row of metrics
            ttk.Label(metrics_frame, text="Starting Price:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            ttk.Label(metrics_frame, text=f"${start_price:.2f}").grid(row=0, column=0, padx=5, pady=5, sticky="e")
            
            ttk.Label(metrics_frame, text="Ending Price:").grid(row=0, column=1, padx=5, pady=5, sticky="w")
            ttk.Label(metrics_frame, text=f"${end_price:.2f}").grid(row=0, column=1, padx=5, pady=5, sticky="e")
            
            ttk.Label(metrics_frame, text="Change:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
            ttk.Label(metrics_frame, text=f"${end_price - start_price:.2f}").grid(row=0, column=2, padx=5, pady=5, sticky="e")
            
            # Second row of metrics
            ttk.Label(metrics_frame, text="Highest Price:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            ttk.Label(metrics_frame, text=f"${high:.2f}").grid(row=1, column=0, padx=5, pady=5, sticky="e")
            
            ttk.Label(metrics_frame, text="Lowest Price:").grid(row=1, column=1, padx=5, pady=5, sticky="w")
            ttk.Label(metrics_frame, text=f"${low:.2f}").grid(row=1, column=1, padx=5, pady=5, sticky="e")
            
            ttk.Label(metrics_frame, text="Average Price:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
            ttk.Label(metrics_frame, text=f"${avg:.2f}").grid(row=1, column=2, padx=5, pady=5, sticky="e")
    
    def _create_recommendation_section(self):
        """Create recommendation section with trading guidelines"""
        rec_frame = ttk.LabelFrame(self.main_frame, text="Trading Recommendation")
        rec_frame.pack(fill="x", pady=10)
        
        # Generate recommendation based on prediction trend
        if len(self.predictions) > 0:
            start_price = self.predictions[0]
            end_price = self.predictions[-1]
            change = end_price - start_price
            pct_change = (change / start_price) * 100
            
            # Simple recommendation based on predicted trend
            recommendation = ""
            if pct_change > 5:
                recommendation = "STRONG BUY: Model predicts significant price increase."
                color = "green"
            elif pct_change > 1:
                recommendation = "BUY: Model predicts moderate price increase."
                color = "green"
            elif pct_change > -1:
                recommendation = "HOLD: Model predicts minimal price movement."
                color = "black"
            elif pct_change > -5:
                recommendation = "SELL: Model predicts moderate price decrease."
                color = "red"
            else:
                recommendation = "STRONG SELL: Model predicts significant price decrease."
                color = "red"
            
            # Display recommendation
            rec_label = ttk.Label(rec_frame, text=recommendation, font=("Helvetica", 12, "bold"))
            rec_label.pack(anchor="w", padx=10, pady=10)
            
            # Additional guidance
            guidance = "This recommendation is based solely on the model prediction and should not be the only factor in making investment decisions. Always consider other fundamental and technical factors, and consult with a financial advisor."
            
            guidance_label = ttk.Label(rec_frame, text=guidance, wraplength=700)
            guidance_label.pack(anchor="w", padx=10, pady=(0, 10))
    
    def _save_report(self):
        """Save the prediction report as a file"""
        try:
            # This would be replaced with actual report saving functionality
            import os
            from datetime import datetime
            
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(os.getcwd(), "reports")
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)
            
            # Create filename based on ticker and current date
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.ticker}_prediction_report_{timestamp}.txt"
            filepath = os.path.join(reports_dir, filename)
            
            # Generate report content
            report_content = f"Prediction Report for {self.ticker}\n"
            report_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            report_content += f"Forecast Period: {self.days} days\n"
            report_content += f"Model Used: {self.model_info.get('model').__class__.__name__}\n\n"
            
            report_content += "Predicted Prices:\n"
            today = datetime.now()
            for i, price in enumerate(self.predictions):
                date = today + timedelta(days=i)
                report_content += f"{date.strftime('%Y-%m-%d')}: ${price:.2f}\n"
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(report_content)
            
            tk.messagebox.showinfo("Report Saved", f"Report saved to {filepath}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not save report: {str(e)}") 