"""
Training Panel Patch - Integrates with training functions to capture ticker
"""
import os
import importlib
import inspect
from model_save_patch import set_current_ticker

def monkey_patch_training_panel():
    """
    Patch the training panel to capture ticker information
    """
    try:
        # Import your training module
        training_module = importlib.import_module('ui.training_panel')
        
        # Find training-related classes
        training_classes = []
        for name, obj in inspect.getmembers(training_module):
            if inspect.isclass(obj) and hasattr(obj, '_train_model'):
                training_classes.append((name, obj))
        
        if not training_classes:
            print("No suitable training classes found in ui.training_panel")
            return
            
        print(f"Found training classes: {[name for name, _ in training_classes]}")
        
        # Patch each class
        for class_name, class_obj in training_classes:
            # Save original methods
            original_train_model = class_obj._train_model
            
            # Define patched methods
            def patched_train_model(self, *args, **kwargs):
                """Patched version of _train_model that captures ticker"""
                # Capture the ticker before training
                if hasattr(self, 'ticker_var') and hasattr(self.ticker_var, 'get'):
                    ticker = self.ticker_var.get()
                    print(f"Training model for ticker: {ticker}")
                    set_current_ticker(ticker)
                elif hasattr(self, 'selected_ticker'):
                    ticker = self.selected_ticker
                    print(f"Training model for selected ticker: {ticker}")
                    set_current_ticker(ticker)
                
                # Call the original training function
                result = original_train_model(self, *args, **kwargs)
                
                # Reset the ticker after training
                set_current_ticker(None)
                
                return result
            
            # Apply patches
            class_obj._train_model = patched_train_model
            print(f"Successfully patched {class_name}._train_model")
            
    except Exception as e:
        print(f"Failed to patch TrainingPanel: {e}")
        import traceback
        traceback.print_exc()

def apply_training_panel_patch():
    """Apply the training panel patch to the application"""
    print("\n=================================================")
    print("| Training Panel patch applied                    |")
    print("| Will capture ticker info for auto-save          |")
    print("=================================================\n")
    monkey_patch_training_panel() 