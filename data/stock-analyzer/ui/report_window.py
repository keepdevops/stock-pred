"""
Report window for displaying detailed analysis
"""
import tkinter as tk
from tkinter import ttk
from config.settings import DARK_BG, LIGHT_TEXT

class ReportWindow:
    def __init__(self, title="Analysis Report", width=800, height=600):
        self.window = tk.Toplevel()
        self.window.title(title)
        self.window.geometry(f"{width}x{height}")
        self.window.configure(bg=DARK_BG)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create text widget for report
        self.report_text = tk.Text(self.main_frame, bg="#3E3E3E", fg=LIGHT_TEXT, wrap=tk.WORD)
        report_scrollbar = ttk.Scrollbar(self.main_frame, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_scrollbar.set)
        report_scrollbar.pack(side="right", fill="y")
        self.report_text.pack(side="left", fill="both", expand=True)
        
    def set_content(self, content):
        """Set the content of the report"""
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, content)
        
    def add_content(self, content):
        """Add content to the report"""
        self.report_text.insert(tk.END, content)
        
    def show(self):
        """Show the window"""
        self.window.deiconify()
        self.window.lift()
        
    def hide(self):
        """Hide the window"""
        self.window.withdraw() 