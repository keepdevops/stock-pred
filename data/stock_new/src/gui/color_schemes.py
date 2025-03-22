from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from enum import Enum
import json
import os

class ColorScheme(Enum):
    LIGHT = "Light"
    DARK = "Dark"
    HIGH_CONTRAST = "High Contrast"
    DEUTERANOPIA = "Deuteranopia Friendly"
    PROTANOPIA = "Protanopia Friendly"
    TRITANOPIA = "Tritanopia Friendly"
    BLUE_LIGHT = "Blue Light Filter"
    CUSTOM = "Custom"

class ColorSchemeManager:
    def __init__(self):
        self.schemes = {
            ColorScheme.LIGHT: {
                'window': "#FFFFFF",
                'window_text': "#000000",
                'base': "#F0F0F0",
                'alternate_base': "#E0E0E0",
                'text': "#000000",
                'button': "#E0E0E0",
                'button_text': "#000000",
                'bright_text': "#FFFFFF",
                'link': "#0000FF",
                'highlight': "#308CC6",
                'highlight_text': "#FFFFFF",
                'table_header': "#E0E0E0",
                'table_row_odd': "#F5F5F5",
                'table_row_even': "#FFFFFF",
                'progress_bar': "#308CC6",
                'progress_bar_text': "#FFFFFF"
            },
            ColorScheme.DARK: {
                'window': "#2B2B2B",
                'window_text': "#FFFFFF",
                'base': "#323232",
                'alternate_base': "#383838",
                'text': "#FFFFFF",
                'button': "#454545",
                'button_text': "#FFFFFF",
                'bright_text': "#FFFFFF",
                'link': "#5294E2",
                'highlight': "#2979FF",
                'highlight_text': "#FFFFFF",
                'table_header': "#383838",
                'table_row_odd': "#2B2B2B",
                'table_row_even': "#323232",
                'progress_bar': "#2979FF",
                'progress_bar_text': "#FFFFFF"
            },
            ColorScheme.HIGH_CONTRAST: {
                'window': "#000000",
                'window_text': "#FFFFFF",
                'base': "#000000",
                'alternate_base': "#1A1A1A",
                'text': "#FFFFFF",
                'button': "#FFFFFF",
                'button_text': "#000000",
                'bright_text': "#FFFFFF",
                'link': "#00FF00",
                'highlight': "#FFFF00",
                'highlight_text': "#000000",
                'table_header': "#FFFFFF",
                'table_row_odd': "#000000",
                'table_row_even': "#1A1A1A",
                'progress_bar': "#FFFF00",
                'progress_bar_text': "#000000"
            },
            ColorScheme.DEUTERANOPIA: {
                'window': "#FFFFFF",
                'window_text': "#000000",
                'base': "#F0F0F0",
                'alternate_base': "#E0E0E0",
                'text': "#000000",
                'button': "#E0E0E0",
                'button_text': "#000000",
                'bright_text': "#FFFFFF",
                'link': "#0000FF",
                'highlight': "#FFA07A",  # Light Salmon
                'highlight_text': "#000000",
                'table_header': "#E0E0E0",
                'table_row_odd': "#F5F5F5",
                'table_row_even': "#FFFFFF",
                'progress_bar': "#FFA07A",
                'progress_bar_text': "#000000"
            },
            ColorScheme.PROTANOPIA: {
                'window': "#FFFFFF",
                'window_text': "#000000",
                'base': "#F0F0F0",
                'alternate_base': "#E0E0E0",
                'text': "#000000",
                'button': "#E0E0E0",
                'button_text': "#000000",
                'bright_text': "#FFFFFF",
                'link': "#0000FF",
                'highlight': "#87CEEB",  # Sky Blue
                'highlight_text': "#000000",
                'table_header': "#E0E0E0",
                'table_row_odd': "#F5F5F5",
                'table_row_even': "#FFFFFF",
                'progress_bar': "#87CEEB",
                'progress_bar_text': "#000000"
            },
            ColorScheme.TRITANOPIA: {
                'window': "#FFFFFF",
                'window_text': "#000000",
                'base': "#F0F0F0",
                'alternate_base': "#E0E0E0",
                'text': "#000000",
                'button': "#E0E0E0",
                'button_text': "#000000",
                'bright_text': "#FFFFFF",
                'link': "#FF0000",
                'highlight': "#FFB6C1",  # Light Pink
                'highlight_text': "#000000",
                'table_header': "#E0E0E0",
                'table_row_odd': "#F5F5F5",
                'table_row_even': "#FFFFFF",
                'progress_bar': "#FFB6C1",
                'progress_bar_text': "#000000"
            },
            ColorScheme.BLUE_LIGHT: {
                'window': "#F5F6F7",
                'window_text': "#2C3E50",
                'base': "#ECF0F1",
                'alternate_base': "#E0E6E8",
                'text': "#2C3E50",
                'button': "#D5DBDB",
                'button_text': "#2C3E50",
                'bright_text': "#2C3E50",
                'link': "#3498DB",
                'highlight': "#BDC3C7",
                'highlight_text': "#2C3E50",
                'table_header': "#D5DBDB",
                'table_row_odd': "#ECF0F1",
                'table_row_even': "#F5F6F7",
                'progress_bar': "#BDC3C7",
                'progress_bar_text': "#2C3E50"
            }
        }
        
        self.load_custom_schemes()

    def load_custom_schemes(self):
        """Load custom color schemes from file."""
        try:
            config_dir = os.path.expanduser("~/.stock_market_app")
            config_file = os.path.join(config_dir, "color_schemes.json")
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    custom_schemes = json.load(f)
                    self.schemes[ColorScheme.CUSTOM] = custom_schemes
        except Exception as e:
            print(f"Error loading custom color schemes: {e}")

    def save_custom_scheme(self, name: str, colors: dict):
        """Save a custom color scheme."""
        try:
            config_dir = os.path.expanduser("~/.stock_market_app")
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, "color_schemes.json")
            
            custom_schemes = {}
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    custom_schemes = json.load(f)
            
            custom_schemes[name] = colors
            
            with open(config_file, 'w') as f:
                json.dump(custom_schemes, f, indent=4)
                
        except Exception as e:
            print(f"Error saving custom color scheme: {e}")

    def apply_scheme(self, app, scheme: ColorScheme):
        """Apply a color scheme to the application."""
        if scheme not in self.schemes:
            return
            
        colors = self.schemes[scheme]
        palette = QPalette()
        
        # Set colors for different roles
        palette.setColor(QPalette.Window, QColor(colors['window']))
        palette.setColor(QPalette.WindowText, QColor(colors['window_text']))
        palette.setColor(QPalette.Base, QColor(colors['base']))
        palette.setColor(QPalette.AlternateBase, QColor(colors['alternate_base']))
        palette.setColor(QPalette.Text, QColor(colors['text']))
        palette.setColor(QPalette.Button, QColor(colors['button']))
        palette.setColor(QPalette.ButtonText, QColor(colors['button_text']))
        palette.setColor(QPalette.BrightText, QColor(colors['bright_text']))
        palette.setColor(QPalette.Link, QColor(colors['link']))
        palette.setColor(QPalette.Highlight, QColor(colors['highlight']))
        palette.setColor(QPalette.HighlightedText, QColor(colors['highlight_text']))
        
        app.setPalette(palette)
        
        # Apply additional styles
        app.setStyleSheet(f"""
            QHeaderView::section {{
                background-color: {colors['table_header']};
                color: {colors['text']};
                padding: 4px;
            }}
            
            QTableWidget {{
                alternate-background-color: {colors['table_row_odd']};
                background-color: {colors['table_row_even']};
                gridline-color: {colors['alternate_base']};
            }}
            
            QProgressBar {{
                background-color: {colors['base']};
                border: 1px solid {colors['alternate_base']};
                border-radius: 5px;
                text-align: center;
            }}
            
            QProgressBar::chunk {{
                background-color: {colors['progress_bar']};
                border-radius: 5px;
            }}
        """) 