import json
import os
from typing import Dict, Any
from pathlib import Path

class Settings:
    """Settings manager for the application."""
    
    def __init__(self):
        """Initialize settings."""
        self.settings_file = Path.home() / ".stock_market_analyzer" / "settings.json"
        self.settings = self._load_settings()
        
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file."""
        try:
            if not self.settings_file.exists():
                return self._get_default_settings()
                
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            return self._get_default_settings()
            
    def _save_settings(self):
        """Save settings to file."""
        try:
            # Ensure directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
            
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings."""
        return {
            'color_scheme': 'default',
            'color_schemes': {
                'default': {
                    'connected': 'green',
                    'disconnected': 'red',
                    'text': 'black',
                    'background': 'white'
                },
                'protanopia': {
                    'connected': '#0072B2',  # Blue
                    'disconnected': '#D55E00',  # Orange
                    'text': 'black',
                    'background': 'white'
                },
                'deuteranopia': {
                    'connected': '#56B4E9',  # Light blue
                    'disconnected': '#E69F00',  # Yellow
                    'text': 'black',
                    'background': 'white'
                },
                'tritanopia': {
                    'connected': '#009E73',  # Green
                    'disconnected': '#CC79A7',  # Pink
                    'text': 'black',
                    'background': 'white'
                },
                'high_contrast': {
                    'connected': '#FFFFFF',  # White
                    'disconnected': '#000000',  # Black
                    'text': '#FFFFFF',
                    'background': '#000000'
                }
            }
        }
        
    def get_color_scheme(self) -> str:
        """Get current color scheme."""
        return self.settings.get('color_scheme', 'default')
        
    def set_color_scheme(self, scheme: str):
        """Set color scheme."""
        self.settings['color_scheme'] = scheme
        self._save_settings()
        
    def get_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Get all color schemes."""
        return self.settings.get('color_schemes', self._get_default_settings()['color_schemes'])
        
    def add_color_scheme(self, name: str, scheme: Dict[str, str]):
        """Add a new color scheme."""
        self.settings['color_schemes'][name] = scheme
        self._save_settings()
        
    def remove_color_scheme(self, name: str):
        """Remove a color scheme."""
        if name in self.settings['color_schemes'] and name != 'default':
            del self.settings['color_schemes'][name]
            self._save_settings()
            
    def get_scheme_colors(self, scheme: str) -> Dict[str, str]:
        """Get colors for a specific scheme."""
        return self.settings['color_schemes'].get(scheme, self._get_default_settings()['color_schemes']['default']) 