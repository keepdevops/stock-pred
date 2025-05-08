import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional
import logging
import joblib

class XGBoostModel:
    def __init__(self, **kwargs):
        """
        Initialize XGBoost model for stock prediction.
        
        Args:
            **kwargs: XGBoost parameters
        """
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Update parameters with provided values
        self.params.update(kwargs)
        
        self.model = xgb.XGBRegressor(**self.params)
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[list] = None) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            eval_set: Optional evaluation set for early stopping
            
        Returns:
            Dictionary containing training history
        """
        try:
            # Train model
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=True
            )
            
            # Get training history
            results = {
                'train_score': self.model.evals_result().get('validation_0', {}).get('rmse', []),
                'val_score': self.model.evals_result().get('validation_1', {}).get('rmse', []) if eval_set else []
            }
            
            self.logger.info("Model training completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return {'train_score': [], 'val_score': []}
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return np.array([])
            
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature indices to importance scores
        """
        try:
            importance = self.model.feature_importances_
            return {f"feature_{i}": float(score) for i, score in enumerate(importance)}
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {}
            
    def save(self, path: str):
        """Save model to file."""
        try:
            joblib.dump(self.model, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            
    def load(self, path: str):
        """Load model from file."""
        try:
            self.model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            
class XGBoostTrainer:
    def __init__(self, model: XGBoostModel):
        """
        Initialize XGBoost trainer.
        
        Args:
            model: XGBoostModel instance
        """
        self.model = model
        self.logger = logging.getLogger(__name__)
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Train the model with optional validation and early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Number of rounds for early stopping
            
        Returns:
            Dictionary containing training history
        """
        try:
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
                
            return self.model.fit(
                X_train, y_train,
                eval_set=eval_set if len(eval_set) > 1 else None
            )
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return {'train_score': [], 'val_score': []}
            
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            RMSE score
        """
        try:
            y_pred = self.model.predict(X)
            return np.sqrt(np.mean((y - y_pred) ** 2))
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            return float('inf') 