import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
from optuna import create_study, Trial
import optuna

class ModelTuner:
    """Class for tuning model hyperparameters using Optuna."""
    
    def __init__(self, model_type: str):
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type.lower()
        
        # Define parameter search spaces for different models
        self.param_spaces = {
            'transformer': {
                'd_model': (32, 256),
                'nhead': (2, 8),
                'num_layers': (1, 4),
                'dim_feedforward': (128, 512),
                'dropout': (0.0, 0.3),
                'learning_rate': (1e-4, 1e-2),
                'batch_size': (16, 128),
                'sequence_length': (5, 30)
            },
            'lstm': {
                'hidden_dim': (32, 256),
                'num_layers': (1, 4),
                'dropout': (0.0, 0.3),
                'learning_rate': (1e-4, 1e-2),
                'batch_size': (16, 128),
                'sequence_length': (5, 30)
            },
            'xgboost': {
                'n_estimators': (50, 300),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'min_child_weight': (1, 7),
                'subsample': (0.6, 0.9),
                'colsample_bytree': (0.6, 0.9)
            }
        }
        
    def objective(
        self,
        trial: Trial,
        data: np.ndarray,
        n_splits: int = 5
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            data: Training data
            n_splits: Number of splits for cross-validation
            
        Returns:
            Average validation loss across folds
        """
        # Get hyperparameters for this trial
        params = self._get_trial_params(trial)
        
        # Create time series cross-validation splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Store validation losses for each fold
        val_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
            # Split data
            train_data = data[train_idx]
            val_data = data[val_idx]
            
            # Create and train model with current parameters
            model = self._create_model(params)
            history = model.train(
                train_data,
                sequence_length=params.get('sequence_length', 10),
                epochs=50,  # Use fewer epochs during tuning
                batch_size=params.get('batch_size', 32)
            )
            
            # Get validation loss
            val_loss = history['val_loss'][-1]
            val_losses.append(val_loss)
            
            # Report intermediate value for pruning
            trial.report(val_loss, fold)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(val_losses)
        
    def tune(
        self,
        data: np.ndarray,
        n_trials: int = 100,
        n_splits: int = 5
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run hyperparameter optimization.
        
        Args:
            data: Training data
            n_trials: Number of optimization trials
            n_splits: Number of splits for cross-validation
            
        Returns:
            Tuple of (best parameters, best validation loss)
        """
        study = create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            lambda trial: self.objective(trial, data, n_splits),
            n_trials=n_trials
        )
        
        return study.best_params, study.best_value
        
    def _get_trial_params(self, trial: Trial) -> Dict[str, Any]:
        """Get hyperparameters for the current trial."""
        params = {}
        param_space = self.param_spaces[self.model_type]
        
        for param, (low, high) in param_space.items():
            if isinstance(low, int):
                params[param] = trial.suggest_int(param, low, high)
            else:
                params[param] = trial.suggest_float(param, low, high)
                
        return params
        
    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create a model instance with the given parameters."""
        if self.model_type == 'transformer':
            from .transformer_model import TransformerStockPredictor
            return TransformerStockPredictor(**params)
        elif self.model_type == 'lstm':
            from .lstm_model import LSTMStockPredictor
            return LSTMStockPredictor(**params)
        elif self.model_type == 'xgboost':
            from .xgboost_model import XGBoostStockPredictor
            return XGBoostStockPredictor(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}") 