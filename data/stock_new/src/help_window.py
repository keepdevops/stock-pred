import tkinter as tk
from tkinter import ttk
from typing import Dict

class HelpWindow(tk.Toplevel):
    def __init__(self, parent, title: str, definition: Dict):
        super().__init__(parent)
        self.title(f"Help: {title}")
        self.geometry("500x400")
        
        # Make window modal
        self.transient(parent)
        self.grab_set()
        
        self._create_widgets(definition)
        
    def _create_widgets(self, definition: Dict):
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame, 
            text=definition["title"],
            font=("Helvetica", 14, "bold")
        ).pack(pady=(0, 10))
        
        # Formula frame
        formula_frame = ttk.LabelFrame(main_frame, text="Formula", padding="5")
        formula_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            formula_frame,
            text=definition["formula"],
            font=("Courier", 12)
        ).pack()
        
        # Description
        desc_frame = ttk.LabelFrame(main_frame, text="Description", padding="5")
        desc_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            desc_frame,
            text=definition["description"],
            wraplength=400
        ).pack()
        
        # Interpretation
        interp_frame = ttk.LabelFrame(main_frame, text="Interpretation", padding="5")
        interp_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            interp_frame,
            text=definition["interpretation"],
            wraplength=400
        ).pack()
        
        # Close button
        ttk.Button(
            main_frame,
            text="Close",
            command=self.destroy
        ).pack(pady=(10, 0))

class HelpButton(ttk.Button):
    def __init__(self, parent, title: str, definition: Dict):
        super().__init__(
            parent,
            text="?",
            width=2,
            command=lambda: self._show_help(title, definition)
        )
        
        # Add tooltip
        self.tooltip = None
        self.bind('<Enter>', self._show_tooltip)
        self.bind('<Leave>', self._hide_tooltip)
        
    def _show_help(self, title: str, definition: Dict):
        HelpWindow(self.winfo_toplevel(), title, definition)
        
    def _show_tooltip(self, event):
        x, y, _, _ = self.bbox("insert")
        x += self.winfo_rootx() + 25
        y += self.winfo_rooty() + 20
        
        self.tooltip = tk.Toplevel(self)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(
            self.tooltip,
            text="Click for detailed explanation",
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            padding=(5, 2)
        )
        label.pack()
        
    def _hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None 