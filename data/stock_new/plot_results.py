import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import json
from pathlib import Path
import logging

class PlotterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Results Plotter")
        self.root.geometry("1200x800")
        
        self.setup_logging()
        self.create_gui()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def create_gui(self):
        # Create main layout
        self.create_control_panel()
        self.create_plot_area()

    def create_control_panel(self):
        # Control panel on the left
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding="5")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Load data button
        ttk.Button(control_frame, text="Load Results", 
                  command=self.load_results).pack(fill=tk.X, padx=5, pady=5)
        
        # Plot type selection
        ttk.Label(control_frame, text="Plot Type:").pack(fill=tk.X, padx=5, pady=(10,0))
        self.plot_type = tk.StringVar(value="training")
        ttk.Radiobutton(control_frame, text="Training History", 
                       value="training", variable=self.plot_type).pack(fill=tk.X, padx=5)
        ttk.Radiobutton(control_frame, text="Predictions", 
                       value="predictions", variable=self.plot_type).pack(fill=tk.X, padx=5)
        
        # Plot button
        ttk.Button(control_frame, text="Generate Plot", 
                  command=self.generate_plot).pack(fill=tk.X, padx=5, pady=(10,5))
        
        # Save plot button
        ttk.Button(control_frame, text="Save Plot", 
                  command=self.save_plot).pack(fill=tk.X, padx=5, pady=5)

    def create_plot_area(self):
        # Plot area on the right
        plot_frame = ttk.LabelFrame(self.root, text="Plot", padding="5")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.plot = self.figure.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

    def load_results(self):
        """Load training results from file."""
        try:
            filename = filedialog.askopenfilename(
                title="Select Results File",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                self.logger.info(f"Loading results from {filename}")
                # Implement loading logic based on file type
                
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")

    def generate_plot(self):
        """Generate plot based on loaded data and selected plot type."""
        try:
            self.plot.clear()
            
            if self.plot_type.get() == "training":
                self.plot_training_history()
            else:
                self.plot_predictions()
                
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error generating plot: {e}")

    def plot_training_history(self):
        """Plot training history."""
        # Implement training history plotting
        self.plot.set_title("Training History")
        self.plot.set_xlabel("Epoch")
        self.plot.set_ylabel("Loss")

    def plot_predictions(self):
        """Plot predictions vs actual values."""
        # Implement predictions plotting
        self.plot.set_title("Predictions vs Actual")
        self.plot.set_xlabel("Date")
        self.plot.set_ylabel("Price")

    def save_plot(self):
        """Save current plot to file."""
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Plot",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                self.figure.savefig(filename, bbox_inches='tight', dpi=300)
                self.logger.info(f"Plot saved to {filename}")
                
        except Exception as e:
            self.logger.error(f"Error saving plot: {e}")

def main():
    root = tk.Tk()
    app = PlotterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 