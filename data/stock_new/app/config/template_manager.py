import json
from pathlib import Path
from typing import Dict, Any, Optional

class TemplateManager:
    def __init__(self, templates_dir: Path = None):
        self.templates_dir = templates_dir or Path(__file__).parent.parent.parent / "templates"
        self.current_config: Dict[str, Any] = {}
        self.load_default_config()
    
    def load_default_config(self) -> None:
        """Load the default configuration template."""
        default_path = self.templates_dir / "default_config.json"
        try:
            with open(default_path, 'r') as f:
                self.current_config = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Default config not found at {default_path}")
            self.current_config = {}
    
    def save_template(self, name: str, config: Dict[str, Any]) -> bool:
        """Save a new template configuration."""
        try:
            path = self.templates_dir / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving template: {e}")
            return False
    
    def load_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a specific template configuration."""
        try:
            path = self.templates_dir / f"{name}.json"
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Template not found: {name}")
            return None
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self.current_config.get("models", {}).get(model_name, {})
    
    def get_data_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.current_config.get("data_processing", {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.current_config.get("visualization", {})
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.current_config.get("trading", {}) 