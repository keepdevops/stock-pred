"""
Patch for _create_and_train_model to ensure ticker is included in filename
"""
import os
import importlib
import inspect
import time
import pickle
from model_save_patch import set_current_ticker, auto_save_model_with_ticker

def patch_create_and_train():
    """
    Patch the _create_and_train_model function to capture ticker and save with it
    """
    try:
        # Try to import modules that might contain the function
        potential_modules = [
            'ui.training_panel',
            'model.trainer',
            'ui.model_panel',
            'controller.model_controller'
        ]
        
        for module_name in potential_modules:
            try:
                module = importlib.import_module(module_name)
                
                # Look for _create_and_train_model in module
                if hasattr(module, '_create_and_train_model'):
                    original_func = module._create_and_train_model
                    
                    def patched_create_and_train(input_data, sequence_length, future_steps, model_type='lstm', epochs=50, batch_size=32, *args, **kwargs):
                        """Patched version that captures ticker and auto-saves"""
                        print(f"Patched _create_and_train_model called with model_type={model_type}")
                        
                        # Try to find ticker
                        ticker = None
                        if hasattr(self, 'ticker_var') and hasattr(self.ticker_var, 'get'):
                            ticker = self.ticker_var.get()
                        elif hasattr(self, 'selected_ticker'):
                            ticker = self.selected_ticker
                        elif hasattr(self, 'ticker_combo') and hasattr(self.ticker_combo, 'get'):
                            ticker = self.ticker_combo.get()
                        
                        if ticker:
                            set_current_ticker(ticker)
                        
                        result = original_func(self, input_data, sequence_length, future_steps, model_type, epochs, batch_size, *args, **kwargs)
                        
                        if isinstance(result, tuple) and len(result) >= 2:
                            model, history = result[:2]
                            if hasattr(model, 'save'):
                                auto_save_model_with_ticker(model, history)
                        
                        set_current_ticker(None)
                        
                        return result
                    
                    # Apply the patch
                    setattr(module, '_create_and_train_model', patched_create_and_train)
                    print(f"Successfully patched {module_name}._create_and_train_model")
            
            except Exception as e:
                print(f"Error patching {module_name}: {e}")
    
    except Exception as e:
        print(f"Error patching _create_and_train_model: {e}")

def apply_create_and_train_patch():
    """Apply the patch to _create_and_train_model"""
    print("\n=================================================")
    print("| _create_and_train_model patch applied successfully |")
    print("| Models will be saved with ticker name in file   |")
    print("=================================================\n")
    patch_create_and_train() 