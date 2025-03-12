"""
Training Panel Patch - Integrates with training functions to capture ticker
"""
import os
from model_save_patch import set_current_ticker

def monkey_patch_training_panel():
    """
    Patch the TrainingPanel class to capture ticker information
    """
    try:
        # Import your training panel
        from ui.training_panel import TrainingPanel
        
        # Save original methods
        original_train_model = TrainingPanel._train_model
        
        # Define patched methods
        def patched_train_model(self, *args, **kwargs):
            """Patched version of _train_model that captures ticker"""
            # Capture the ticker before training
            if hasattr(self, 'ticker_var') and hasattr(self.ticker_var, 'get'):
                ticker = self.ticker_var.get()
                print(f"Training model for ticker: {ticker}")
                set_current_ticker(ticker)
            
            # Call the original training function
            result = original_train_model(self, *args, **kwargs)
            
            # Reset the ticker after training
            set_current_ticker(None)
            
            return result
        
        # Apply patches
        TrainingPanel._train_model = patched_train_model
        print("TrainingPanel._train_model successfully patched")
        
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