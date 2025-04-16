from typing import Dict, Any, Optional
import logging
from .transformer_model import TransformerStockPredictor
from .lstm_model import LSTMModel
from .xgboost_model import XGBoostModel

class ModelFactory:
    """Factory class for creating and managing different types of stock prediction models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Any] = {}
        
    def create_model(
        self,
        model_type: str,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a new model instance based on the specified type.
        
        Args:
            model_type: Type of model to create ('transformer', 'lstm', or 'xgboost')
            model_params: Optional dictionary of model parameters
            
        Returns:
            An instance of the specified model type
        """
        if model_params is None:
            model_params = {}
            
        if model_type.lower() == 'transformer':
            return TransformerStockPredictor(**model_params)
        elif model_type.lower() == 'lstm':
            return LSTMModel(**model_params)
        elif model_type.lower() == 'xgboost':
            return XGBoostModel(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get an existing model instance by ID."""
        return self.models.get(model_id)
        
    def save_model(self, model_id: str, model: Any, path: str):
        """Save a model instance to disk."""
        try:
            model.save_model(path)
            self.models[model_id] = model
            self.logger.info(f"Model {model_id} saved successfully to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model {model_id}: {e}")
            raise
            
    def load_model(self, model_id: str, model_type: str, path: str, model_params: Optional[Dict[str, Any]] = None):
        """Load a model instance from disk."""
        try:
            model = self.create_model(model_type, model_params)
            model.load_model(path)
            self.models[model_id] = model
            self.logger.info(f"Model {model_id} loaded successfully from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {e}")
            raise
            
    def remove_model(self, model_id: str):
        """Remove a model instance."""
        if model_id in self.models:
            del self.models[model_id]
            self.logger.info(f"Model {model_id} removed successfully")
            
    def list_models(self) -> Dict[str, str]:
        """List all available models and their types."""
        return {
            model_id: model.__class__.__name__
            for model_id, model in self.models.items()
        } 
