import os
import logging
import json
from typing import List, Dict, Any
from datetime import datetime

class ModelManager:
    """Manages machine learning models for stock prediction."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.logger = logging.getLogger(__name__)
        self.models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.models_file = os.path.join(self.models_dir, "models.json")
        self.models = self._load_models()
        self.logger.info("Model manager initialized successfully")
        
    def _load_models(self) -> List[Dict[str, Any]]:
        """Load models from the models file."""
        try:
            if not os.path.exists(self.models_file):
                self.logger.warning("Models file not found, creating new one")
                os.makedirs(self.models_dir, exist_ok=True)
                with open(self.models_file, 'w') as f:
                    json.dump([], f)
                return []
                
            with open(self.models_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return []
            
    def _save_models(self):
        """Save models to the models file."""
        try:
            with open(self.models_file, 'w') as f:
                json.dump(self.models, f)
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            
    def get_trained_models(self) -> List[Dict[str, Any]]:
        """Get list of trained models."""
        return self.models
        
    def add_model(self, name: str, model_type: str, status: str = "trained"):
        """Add a new model to the list."""
        model = {
            "name": name,
            "type": model_type,
            "status": status,
            "last_updated": datetime.now().isoformat()
        }
        self.models.append(model)
        self._save_models()
        return model
        
    def update_model(self, name: str, updates: Dict[str, Any]):
        """Update an existing model."""
        for model in self.models:
            if model["name"] == name:
                model.update(updates)
                model["last_updated"] = datetime.now().isoformat()
                self._save_models()
                return model
        return None
        
    def delete_model(self, name: str):
        """Delete a model from the list."""
        self.models = [m for m in self.models if m["name"] != name]
        self._save_models() 