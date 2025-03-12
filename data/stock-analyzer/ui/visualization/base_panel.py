"""
Base class for visualization panels
"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class BasePanel(ttk.Frame):
    def __init__(self, parent, event_bus):
        super().__init__(parent)
        self.event_bus = event_bus
        
        # Create a figure and canvas for plotting
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        
    def clear_plot(self):
        """Clear the current plot"""
        self.figure.clear()
        self.canvas.draw()
        
    def show_error(self, message):
        """Show an error message in the plot"""
        self.clear_plot()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        ax.axis('off')
        self.canvas.draw() 