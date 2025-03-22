from enum import Enum
from PyQt5.QtGui import QColor

class ColorScheme(Enum):
    LIGHT = "Light"
    DARK = "Dark"
    BLUE = "Blue"
    
    @staticmethod
    def get_colors(scheme):
        if scheme == ColorScheme.DARK:
            return {
                'background': '#2b2b2b',
                'text': '#ffffff',
                'button': '#404040',
                'button_hover': '#505050',
                'border': '#505050',
                'success': '#4CAF50',
                'error': '#f44336',
                'warning': '#ff9800'
            }
        elif scheme == ColorScheme.BLUE:
            return {
                'background': '#1e3d59',
                'text': '#ffffff',
                'button': '#2a5885',
                'button_hover': '#3a6fa5',
                'border': '#3a6fa5',
                'success': '#4CAF50',
                'error': '#f44336',
                'warning': '#ff9800'
            }
        else:  # Light theme (default)
            return {
                'background': '#ffffff',
                'text': '#000000',
                'button': '#e0e0e0',
                'button_hover': '#d0d0d0',
                'border': '#cccccc',
                'success': '#4CAF50',
                'error': '#f44336',
                'warning': '#ff9800'
            } 