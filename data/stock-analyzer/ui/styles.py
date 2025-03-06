"""
UI styling for the application
"""
import tkinter as tk
from tkinter import ttk
from config.settings import DARK_BG, DARKER_BG, LIGHT_TEXT, ACCENT_COLOR

def configure_styles(root=None):
    """Configure all styles for the application"""
    style = ttk.Style()
    
    # Configure the main theme
    style.theme_use('clam')
    
    # Configure frame styles
    style.configure('TFrame', background=DARK_BG)
    style.configure('TLabelframe', background=DARK_BG, foreground=LIGHT_TEXT)
    style.configure('TLabelframe.Label', background=DARK_BG, foreground=LIGHT_TEXT)
    
    # Configure button styles
    style.configure('TButton', background=ACCENT_COLOR, foreground=LIGHT_TEXT)
    style.map('TButton',
              background=[('active', ACCENT_COLOR), ('pressed', DARKER_BG)],
              foreground=[('active', LIGHT_TEXT), ('pressed', LIGHT_TEXT)])
    
    # Configure label styles
    style.configure('TLabel', background=DARK_BG, foreground=LIGHT_TEXT)
    
    # Configure entry widgets to have black text
    style.map("TEntry", 
        foreground=[("active", "black"), ("disabled", "gray"), ("!disabled", "black")],
        fieldbackground=[("!disabled", "white")]
    )
    
    # Configure Combobox widgets to have black text
    style.map("TCombobox",
        foreground=[("active", "black"), ("disabled", "gray"), ("!disabled", "black")],
        fieldbackground=[("!disabled", "white")]
    )
    
    # Configure dropdown lists in comboboxes to have black text
    if root:
        root.option_add('*TCombobox*Listbox.foreground', 'black')
    
    # Make the selected text in comboboxes black
    style.map('TCombobox', 
              foreground=[('readonly', 'black'), ('active', 'black'), ('disabled', 'gray')],
              fieldbackground=[('readonly', 'white'), ('active', 'white'), ('disabled', 'gray')])
    
    # Configure Spinbox widgets to have black text
    style.configure("BlackText.TSpinbox", foreground="black")
    style.map("BlackText.TSpinbox",
              foreground=[('readonly', 'black'), ('disabled', 'gray'), ('active', 'black')],
              fieldbackground=[('readonly', 'white'), ('disabled', '#D3D3D3'), ('active', 'white')])
    
    return style
