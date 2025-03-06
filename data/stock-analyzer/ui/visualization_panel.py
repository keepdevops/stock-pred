"""
Visualization panel for the application
"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from config.settings import DARK_BG, LIGHT_TEXT

class VisualizationPanel:
    def __init__(self, parent):
        self.parent = parent
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Visualization")
        self.frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create visualization tab
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Chart")
        
        # Create output tab
        self.output_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.output_frame, text="Output")
        
        # Create figure for plotting
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#3E3E3E")
        self.ax.tick_params(colors=LIGHT_TEXT)
        for spine in self.ax.spines.values():
            spine.set_color(LIGHT_TEXT)
            
        # Create canvas for figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.viz_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Create output text
        self.output_text = tk.Text(self.output_frame, bg="#3E3E3E", fg=LIGHT_TEXT, wrap=tk.WORD)
        output_scrollbar = ttk.Scrollbar(self.output_frame, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scrollbar.set)
        output_scrollbar.pack(side="right", fill="y")
        self.output_text.pack(side="left", fill="both", expand=True)
        
    def clear_figure(self):
        """Clear the figure"""
        self.ax.clear()
        self.canvas.draw()
        
    def clear_output(self):
        """Clear the output text"""
        self.output_text.delete(1.0, tk.END)
        
    def add_output(self, text):
        """Add text to the output"""
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END) 