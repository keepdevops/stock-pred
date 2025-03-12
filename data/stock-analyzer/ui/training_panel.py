import time
import pickle
import tkinter as tk  # If not already imported
import tensorflow as tf  # If not already imported
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from force_save_models import force_save_trained_model

def __init__(self, parent, *args, **kwargs):
    # ... existing initialization code ...
    
    # Setup your UI elements
    # ... existing UI setup code ...
    
    # Add emergency save button
    self.add_emergency_save_button()
    
    # Direct call to create save button - add this at the END of __init__
    print("Creating direct save button...")
    self.create_direct_save_button()

def _save_trained_model(self):
    """Save the trained model to the models directory"""
    try:
        print("Save Model button clicked in Training Results panel")
        
        # Check if a model is available to save
        if not hasattr(self, 'trained_model') or self.trained_model is None:
            messagebox.showinfo("No Model", "No trained model available to save.")
            return
            
        # Get the ticker(s) used for training
        if hasattr(self, 'selected_tickers') and self.selected_tickers:
            if len(self.selected_tickers) == 1:
                ticker_prefix = self.selected_tickers[0]
            else:
                ticker_prefix = "multi_ticker"
        else:
            ticker_prefix = "unknown"
            
        # Get the model type
        model_type = getattr(self, 'model_type', 'custom')
        
        # Create a default filename
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"{ticker_prefix}_{model_type}_{timestamp}.keras"
        
        print(f"Default model filename: {default_name}")
        
        # Get the models directory
        try:
            if hasattr(self, 'main_app') and hasattr(self.main_app, 'models_dir'):
                models_dir = self.main_app.models_dir
            elif hasattr(self, 'parent') and hasattr(self.parent, 'models_dir'):
                models_dir = self.parent.models_dir
            else:
                # Default path
                models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
                
            # Make sure the directory exists
            import os
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                print(f"Created models directory: {models_dir}")
        except Exception as e:
            print(f"Error determining models directory: {e}")
            models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
            
        print(f"Models directory: {models_dir}")
        
        # Ask the user for the filename
        from tkinter import filedialog
        save_path = filedialog.asksaveasfilename(
            initialdir=models_dir,
            initialfile=default_name,
            defaultextension=".keras",
            filetypes=[("Keras Models", "*.keras"), ("All Files", "*.*")]
        )
        
        if not save_path:
            print("Save cancelled by user")
            return
            
        # Save the model
        print(f"Saving model to: {save_path}")
        self.trained_model.save(save_path, save_format='keras')

        # Save the model history if available
        if hasattr(self, 'training_history') and self.training_history is not None:
            try:
                import pickle
                history_path = save_path.replace('.keras', '_history.pkl')
                with open(history_path, 'wb') as f:
                    pickle.dump(self.training_history, f)
                print(f"Training history saved to: {history_path}")
            except Exception as e:
                print(f"Error saving training history: {e}")
        
        # Show success message with file location
        messagebox.showinfo("Model Saved", 
                           f"Model successfully saved to:\n{save_path}\n\n"
                           f"You can now select this model in the Model Selection section.")
        
        # Refresh the model list in the control panel
        self._refresh_model_list()
        
        # Update the training results view to show the saved model name
        self._update_results_display(f"Model saved as: {os.path.basename(save_path)}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"Failed to save model: {str(e)}")

def _refresh_model_list(self):
    """Refresh the model list in UI"""
    print("\n----- Refreshing Model List -----")
    
    try:
        import os
        
        # Get models directory
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        print(f"Models directory: {models_dir}")
        print(f"Directory exists: {os.path.exists(models_dir)}, writable: {os.access(models_dir, os.W_OK)}")
        
        # Find the model selection control (combobox or listbox)
        model_control = None
        if hasattr(self, 'model_listbox'):
            model_control = self.model_listbox
            print("Using model_listbox")
        elif hasattr(self, 'model_combobox'):
            model_control = self.model_combobox
            print("Using model_combobox")
        
        if model_control:
            # Clear existing items
            if hasattr(model_control, 'delete'):
                model_control.delete(0, 'end')
            elif hasattr(model_control, 'set'):
                model_control['values'] = []
                
            # Get list of models with .keras extension
            models = []
            for filename in os.listdir(models_dir):
                if filename.endswith('.keras') and not os.path.isdir(os.path.join(models_dir, filename)):
                    models.append(filename)
            
            # Add models to control
            for model in models:
                if hasattr(model_control, 'insert'):
                    model_control.insert('end', model)
                elif hasattr(model_control, 'set'):
                    values = list(model_control['values'])
                    values.append(model)
                    model_control['values'] = values
                    
            print(f"Added {len(models)} models to the selection list")
        else:
            print("No model selection control found")
            
    except Exception as e:
        print(f"Error refreshing model list: {e}")
        import traceback
        traceback.print_exc()
        
    print("----- Model List Refresh Complete -----\n")

def _update_results_display(self, message):
    """Update the training results display with a message"""
    try:
        # Find the results text widget
        if hasattr(self, 'results_text'):
            # Add the message to the text widget
            self.results_text.config(state="normal")
            self.results_text.insert("end", f"\n{message}\n")
            self.results_text.see("end")  # Scroll to show the new message
            self.results_text.config(state="disabled")
            print(f"Updated results display: {message}")
        else:
            print("Could not find results_text widget to update")
    except Exception as e:
        print(f"Error updating results display: {e}")

def _create_training_results(self):
    """Create the training results section with auto-save info"""
    results_frame = ttk.LabelFrame(self, text="Training Results")
    results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # Create a text widget to display results
    self.results_text = tk.Text(results_frame, wrap="word", height=10)
    self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    self.results_text.config(state="disabled")  # Make read-only initially
    
    # Add a scrollbar
    scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.results_text.config(yscrollcommand=scrollbar.set)
    
    # Add auto-save info message
    info_frame = ttk.Frame(results_frame)
    info_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(
        info_frame, 
        text="Models are automatically saved after training completes.",
        font=("Arial", 9, "italic"),
        foreground="gray"
    ).pack(side=tk.LEFT, padx=5)
    
    # Buttons frame
    button_frame = ttk.Frame(results_frame)
    button_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # View Latest Model button
    self.view_latest_model_btn = ttk.Button(
        button_frame, 
        text="View Latest Model",
        command=self._view_latest_model
    )
    self.view_latest_model_btn.pack(side=tk.LEFT, padx=5)
    self.view_latest_model_btn.config(state="disabled")  # Disabled until a model is trained
    
    # Add test button
    test_btn = ttk.Button(button_frame, text="Test Auto-Save", command=self._test_auto_save)
    test_btn.pack(side=tk.LEFT, padx=5)
    
    # Add train button
    train_btn = ttk.Button(button_frame, text="Train Model", command=self._train_model)
    train_btn.pack(side=tk.LEFT, padx=5)
    
    # Add refresh button
    refresh_btn = ttk.Button(button_frame, text="Refresh Models", command=self._refresh_model_list)
    refresh_btn.pack(side=tk.LEFT, padx=5)
    
    # Add any other buttons you need here
    
def _view_latest_model(self):
    """View the most recently trained model"""
    try:
        if not hasattr(self, 'latest_model_name') or not self.latest_model_name:
            messagebox.showinfo("No Model", "No model has been trained in this session.")
            return
            
        print(f"Viewing latest model: {self.latest_model_name}")
        
        # Use the existing view model functionality
        from ui.model_viewer import ModelViewer
        
        # Get models directory
        if hasattr(self, 'main_app') and hasattr(self.main_app, 'models_dir'):
            models_dir = self.main_app.models_dir
        elif hasattr(self, 'parent') and hasattr(self.parent, 'models_dir'):
            models_dir = self.parent.models_dir
        else:
            models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
            
        # Call the model viewer
        ModelViewer.view_model(models_dir, self.latest_model_name)
            
    except Exception as e:
        print(f"Error viewing latest model: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("View Error", f"Could not view latest model: {str(e)}")

def _on_training_complete(self, model, history):
    """Show dialog when training completes with option to save model"""
    import tkinter as tk
    from tkinter import ttk, messagebox
    import os
    import time
    import pickle
    
    print("\n===== TRAINING COMPLETE DIALOG =====")
    
    # Create dialog
    dialog = tk.Toplevel(self)
    dialog.title("Training Complete")
    dialog.geometry("400x280")
    dialog.transient(self)
    dialog.grab_set()
    
    # Main frame
    frame = ttk.Frame(dialog, padding=20)
    frame.pack(fill="both", expand=True)
    
    # Success message
    ttk.Label(
        frame, 
        text="Model Training Completed Successfully!", 
        font=("Arial", 12, "bold")
    ).pack(pady=10)
    
    # Get ticker and model type
    ticker = self.ticker_var.get() if hasattr(self, 'ticker_var') else "UNKNOWN"
    model_type = self.model_type_var.get() if hasattr(self, 'model_type_var') else "MODEL"
    
    # Create filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{model_type}_{timestamp}.keras"
    
    # Save info
    info_frame = ttk.Frame(frame)
    info_frame.pack(fill="x", pady=10)
    
    ttk.Label(info_frame, text="Save Details:").pack(anchor="w")
    ttk.Label(info_frame, text=f"Ticker: {ticker}").pack(anchor="w", padx=10)
    ttk.Label(info_frame, text=f"Model Type: {model_type}").pack(anchor="w", padx=10)
    ttk.Label(info_frame, text=f"Filename: {filename}").pack(anchor="w", padx=10)
    
    # Save option
    save_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        frame, 
        text="Save model to models directory", 
        variable=save_var
    ).pack(anchor="w", pady=5)
    
    # Button actions
    def on_save():
        if save_var.get():
            try:
                # Paths
                models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, filename)
                history_path = model_path.replace('.keras', '_history.pkl')
                
                # Save model
                print(f"Saving model to: {model_path}")
                model.save(model_path, save_format='keras')
                
                # Save history
                print(f"Saving history to: {history_path}")
                history_data = history.history if hasattr(history, 'history') else history
                with open(history_path, 'wb') as f:
                    pickle.dump(history_data, f)
                    
                print(f"Model and history saved successfully")
                
                # Refresh model list
                if hasattr(self, '_refresh_model_list'):
                    self._refresh_model_list()
                    
                messagebox.showinfo("Save Complete", f"Model saved as:\n{filename}")
            except Exception as e:
                print(f"Error saving model: {e}")
                messagebox.showerror("Save Error", f"Error saving model: {str(e)}")
        
        dialog.destroy()
        
    def on_cancel():
        dialog.destroy()
    
    # Buttons
    button_frame = ttk.Frame(frame)
    button_frame.pack(fill="x", pady=10)
    
    save_btn = ttk.Button(button_frame, text="Save & Close", command=on_save)
    save_btn.pack(side="right", padx=5)
    
    cancel_btn = ttk.Button(button_frame, text="Cancel", command=on_cancel)
    cancel_btn.pack(side="right", padx=5)
    
    # Center dialog
    dialog.update_idletasks()
    x = self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2
    y = self.winfo_rooty() + (self.winfo_height() - dialog.winfo_height()) // 2
    dialog.geometry(f"+{x}+{y}")
    
    # Make dialog modal
    dialog.wait_window()
    
    print("===== TRAINING DIALOG CLOSED =====\n")

def _train_model(self):
    """Train a model with the selected parameters and guarantee saving"""
    try:
        # Get selected parameters
        ticker = self.ticker_var.get()
        if not ticker:
            messagebox.showinfo("Input Error", "Please select a ticker.")
            return
            
        model_type = self.model_type_var.get()
        if not model_type:
            messagebox.showinfo("Input Error", "Please select a model type.")
            return
            
        # Additional training parameters
        sequence_length = int(self.sequence_length_var.get())
        epochs = int(self.epochs_var.get())
        
        # Update results to show training has started
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Training {model_type} model for {ticker}...\n")
        self.results_text.config(state="disabled")
        self.update_idletasks()  # Force UI update
        
        # Get data for the selected ticker
        df = self._get_ticker_data(ticker)
        if df is None or df.empty:
            self.results_text.config(state="normal")
            self.results_text.insert(tk.END, "No data available for the selected ticker.\n")
            self.results_text.config(state="disabled")
            return
            
        # Prepare data for training
        X_train, X_test, y_train, y_test = self._prepare_training_data(df, sequence_length)
        
        # Train the model
        model, history = self._create_and_train_model(X_train, y_train, X_test, y_test, model_type, epochs)
        
        # FORCE SAVE MODEL - add these lines
        force_save_trained_model(model, history, ticker, model_type)
        
        # Store the model and history 
        self.trained_model = model
        self.trained_history = history
        
        # Update UI
        self.results_text.config(state="normal")
        self.results_text.insert(tk.END, f"Training complete!\n")
        self.results_text.config(state="disabled")
        
        # Show the save dialog
        self._on_training_complete(model, history)
        
        # Refresh model list
        self._refresh_model_list()
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        self.results_text.config(state="normal")
        self.results_text.insert(tk.END, f"Error during training: {str(e)}\n")
        self.results_text.config(state="disabled")

def _get_models_directory(self):
    """Get the models directory path"""
    try:
        if hasattr(self, 'main_app') and hasattr(self.main_app, 'models_dir'):
            return self.main_app.models_dir
        elif hasattr(self, 'parent') and hasattr(self.parent, 'models_dir'):
            return self.parent.models_dir
        else:
            return "/Users/moose/stock-pred/data/stock-analyzer/models"
    except Exception as e:
        print(f"Error getting models directory: {e}")
        return "/Users/moose/stock-pred/data/stock-analyzer/models"

def _create_and_train_model(self, X_train, y_train, X_test, y_test, model_type, epochs):
    """Create and train a model with auto-save functionality"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    
    # Import the SaveModelCallback
    from save_after_training import SaveModelCallback
    
    # Get ticker
    ticker = self.ticker_var.get() if hasattr(self, 'ticker_var') else "UNKNOWN"
    
    # Create a callback to save the model after training
    save_callback = SaveModelCallback(ticker=ticker, model_type=model_type)
    
    # Create the model
    model = Sequential()
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Add layers based on model type
    if model_type == 'LSTM':
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
    elif model_type == 'GRU':
        model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(units=50))
        model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model - ADD THE CALLBACK HERE
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[save_callback]  # This will save the model after training
    )
    
    # Store the model and history for later use
    self.trained_model = model
    self.trained_history = history
    
    # Return model and history
    return model, history

def _test_auto_save(self):
    """Test function to verify auto-save functionality"""
    import tensorflow as tf
    import os
    import time
    import pickle
    
    print("\n===== TESTING AUTO-SAVE FUNCTIONALITY =====")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Create dummy history
    history = {'loss': [0.1, 0.05, 0.02], 'val_loss': [0.2, 0.15, 0.1]}
    
    # Get ticker and model type
    ticker = "TEST"
    model_type = "LSTM"
    
    # Generate timestamp for filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_filename = f"{ticker}_{model_type}_{timestamp}.keras"
    
    # Get models directory
    models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
    
    # Ensure directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, model_filename)
    print(f"Saving test model to: {model_path}")
    model.save(model_path, save_format='keras')
    
    # Save history
    history_path = model_path.replace('.keras', '_history.pkl')
    print(f"Saving test history to: {history_path}")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Check if files exist
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"History file exists: {os.path.exists(history_path)}")
    
    # Refresh model list
    print("Refreshing model list...")
    self._refresh_model_list()
    
    # Update results text if available
    if hasattr(self, 'results_text'):
        self.results_text.config(state="normal")
        self.results_text.insert(tk.END, f"\nTest model saved as: {model_filename}\n")
        self.results_text.config(state="disabled")
    
    print("===== AUTO-SAVE TEST COMPLETE =====\n")
    
    return model_path, history_path 

def ensure_model_is_saved(self, model, history, ticker, model_type):
    """Guaranteed model saving function - call this after training"""
    import os
    import time
    import pickle
    import tensorflow as tf
    
    print("\n===== DIRECT MODEL SAVE FUNCTION =====")
    
    try:
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"{ticker}_{model_type}_{timestamp}.keras"
        
        # Get models directory - hardcoded for reliability
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        
        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Full paths
        model_path = os.path.join(models_dir, model_filename)
        history_path = model_path.replace('.keras', '_history.pkl')
        
        # Save model
        print(f"Saving model to: {model_path}")
        model.save(model_path, save_format='keras')
        
        # Check if model was saved
        if os.path.exists(model_path):
            print(f"✓ Model saved successfully ({os.path.getsize(model_path)/1024/1024:.2f} MB)")
        else:
            print("✗ Model save failed - file doesn't exist")
            
        # Save history
        print(f"Saving history to: {history_path}")
        history_data = history.history if hasattr(history, 'history') else history
        with open(history_path, 'wb') as f:
            pickle.dump(history_data, f)
            
        # Check if history was saved
        if os.path.exists(history_path):
            print(f"✓ History saved successfully ({os.path.getsize(history_path)/1024:.2f} KB)")
        else:
            print("✗ History save failed - file doesn't exist")
            
        # Return the paths
        return model_path, history_path
        
    except Exception as e:
        print(f"ERROR saving model: {e}")
        import traceback
        traceback.print_exc()
        return None, None 

def direct_save_model(self):
    """Directly save model with .keras extension"""
    import os
    import time
    import pickle
    import tkinter as tk
    from tkinter import messagebox
    
    print("\n===== DIRECT MODEL SAVE =====")
    
    # Check if model exists
    if not hasattr(self, 'trained_model') or self.trained_model is None:
        msg = "No trained model available. Please train a model first."
        print(msg)
        messagebox.showinfo("Save Error", msg)
        return
    
    try:
        # Get basic info
        ticker = getattr(self, 'ticker_var', tk.StringVar(value="UNKNOWN")).get()
        model_type = getattr(self, 'model_type_var', tk.StringVar(value="MODEL")).get()
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"{ticker}_{model_type}_{timestamp}.keras"
        
        # Models directory
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save paths
        model_path = os.path.join(models_dir, model_filename)
        history_path = model_path.replace('.keras', '_history.pkl')
        
        # Save model
        print(f"Saving model to: {model_path}")
        self.trained_model.save(model_path, save_format='keras')
        
        # Save history if available
        if hasattr(self, 'trained_history'):
            print(f"Saving history to: {history_path}")
            history_data = self.trained_history.history if hasattr(self.trained_history, 'history') else self.trained_history
            with open(history_path, 'wb') as f:
                pickle.dump(history_data, f)
        
        # Show success message
        success_msg = f"Model saved successfully as:\n{model_filename}"
        print(success_msg)
        messagebox.showinfo("Save Successful", success_msg)
        
        # Refresh model list if possible
        if hasattr(self, '_refresh_model_list'):
            try:
                self._refresh_model_list()
                print("Model list refreshed")
            except Exception as e:
                print(f"Error refreshing model list: {e}")
        
    except Exception as e:
        print(f"ERROR saving model: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Save Error", f"Error saving model: {str(e)}")
    
    print("===== DIRECT SAVE COMPLETE =====\n")

def guaranteed_save_trained_model(self, model, history):
    """Guaranteed model saving function that works independently"""
    import os
    import time
    import pickle
    
    print("\n===== GUARANTEED MODEL SAVE =====")
    
    try:
        # Get ticker from UI
        ticker = "UNKNOWN"
        if hasattr(self, 'ticker_var'):
            ticker = self.ticker_var.get()
            print(f"Using ticker: {ticker}")
        
        # Get model type from UI
        model_type = "MODEL"
        if hasattr(self, 'model_type_var'):
            model_type = self.model_type_var.get()
            print(f"Using model type: {model_type}")
        
        # Create timestamp for unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"{ticker}_{model_type}_{timestamp}.keras"
        print(f"Generated filename: {model_filename}")
        
        # Hardcoded models directory for reliability
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        print(f"Models directory: {models_dir}")
        
        # Ensure directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Save paths
        model_path = os.path.join(models_dir, model_filename)
        history_path = model_path.replace('.keras', '_history.pkl')
        
        # Save model
        print(f"Saving model to: {model_path}")
        model.save(model_path, save_format='keras')
        print(f"Model saved successfully: {os.path.exists(model_path)}")
        
        # Save history
        print(f"Saving history to: {history_path}")
        history_data = history.history if hasattr(history, 'history') else history
        with open(history_path, 'wb') as f:
            pickle.dump(history_data, f)
        print(f"History saved successfully: {os.path.exists(history_path)}")
        
        # Update UI if possible
        if hasattr(self, 'results_text'):
            try:
                import tkinter as tk
                self.results_text.config(state=tk.NORMAL)
                self.results_text.insert(tk.END, f"\nModel automatically saved as: {model_filename}\n")
                self.results_text.config(state=tk.DISABLED)
                print("Updated results text")
            except Exception as e:
                print(f"Error updating results text: {e}")
        
        # Return the model path
        return model_path
        
    except Exception as e:
        print(f"ERROR in guaranteed save: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        print("===== SAVE PROCESS COMPLETE =====\n") 

def add_emergency_save_button(self):
    """Add an emergency save button to the training panel"""
    import tkinter as tk
    from tkinter import ttk, messagebox
    
    # Find a good place to add the button - look for existing button frames
    # or create a new one
    try:
        # Try to find existing button frame
        button_frame = None
        for child in self.winfo_children():
            if isinstance(child, ttk.Frame) and any(isinstance(c, ttk.Button) for c in child.winfo_children()):
                button_frame = child
                break
        
        # If no button frame found, create a new one
        if button_frame is None:
            button_frame = ttk.Frame(self)
            button_frame.pack(fill="x", pady=10, padx=10)
        
        # Create the emergency save button with a distinctive style
        save_btn = ttk.Button(
            button_frame, 
            text="EMERGENCY SAVE MODEL (.keras)", 
            command=self.emergency_save_model
        )
        save_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        print("Emergency save button added successfully")
        
    except Exception as e:
        print(f"Error adding emergency save button: {e}")

def emergency_save_model(self):
    """Save the last trained model with .keras extension"""
    import os
    import time
    import pickle
    import tkinter as tk
    from tkinter import messagebox
    
    print("\n===== EMERGENCY MODEL SAVE =====")
    
    # Check if model is available
    if not hasattr(self, 'trained_model') or self.trained_model is None:
        messagebox.showinfo("Save Error", "No trained model available. Please train a model first.")
        print("No model available to save")
        return
    
    try:
        # Get ticker and model type
        ticker = "UNKNOWN"
        if hasattr(self, 'ticker_var'):
            ticker = self.ticker_var.get()
        
        model_type = "MODEL" 
        if hasattr(self, 'model_type_var'):
            model_type = self.model_type_var.get()
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"{ticker}_{model_type}_{timestamp}.keras"
        
        # Models directory
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Full paths
        model_path = os.path.join(models_dir, model_filename)
        history_path = model_path.replace('.keras', '_history.pkl')
        
        # Save model
        print(f"Saving model to: {model_path}")
        self.trained_model.save(model_path, save_format='keras')
        
        # Save history if available
        if hasattr(self, 'trained_history'):
            print(f"Saving history to: {history_path}")
            history_data = self.trained_history.history if hasattr(self.trained_history, 'history') else self.trained_history
            with open(history_path, 'wb') as f:
                pickle.dump(history_data, f)
        
        # Show success message
        messagebox.showinfo("Save Successful", f"Model saved as:\n{model_filename}")
        
        # Update UI
        if hasattr(self, 'results_text'):
            self.results_text.config(state="normal")
            self.results_text.insert(tk.END, f"\nModel saved as: {model_filename}\n")
            self.results_text.config(state="disabled")
        
        # Refresh model list
        if hasattr(self, '_refresh_model_list'):
            self._refresh_model_list()
        
        print(f"Model saved successfully: {model_path}")
        
    except Exception as e:
        print(f"ERROR saving model: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Save Error", f"Error saving model: {str(e)}")
    
    print("===== EMERGENCY SAVE COMPLETE =====\n") 

def create_direct_save_button(self):
    """Create a direct emergency save button that will definitely show up"""
    import tkinter as tk
    from tkinter import ttk, messagebox
    
    print("\n===== CREATING DIRECT SAVE BUTTON =====")
    
    try:
        # Create a new top-level frame for our button to ensure it's visible
        save_frame = ttk.Frame(self)
        
        # Use pack with expand and fill to make sure it appears
        save_frame.pack(side=tk.TOP, fill=tk.X, expand=True, pady=10, before=self.winfo_children()[0] if self.winfo_children() else None)
        
        # Create a distinctive button
        save_btn = ttk.Button(
            save_frame, 
            text="💾 SAVE MODEL WITH .KERAS EXTENSION",
            command=self.direct_save_model
        )
        
        # Make button very visible with some padding
        save_btn.pack(pady=10, padx=20, fill=tk.X)
        
        # Add a label explaining what it does
        ttk.Label(
            save_frame,
            text="Click this button to save the last trained model with .keras extension",
            foreground="blue"
        ).pack(pady=5)
        
        print("Direct save button created successfully!")
        
        # Force layout update
        self.update_idletasks()
        
    except Exception as e:
        print(f"ERROR creating direct save button: {e}")
        import traceback
        traceback.print_exc()

def _save_current_model(self):
    """Manual save function for the current model"""
    print("\n=========== STARTING MANUAL MODEL SAVE ===========")
    
    # Check if model exists
    if not hasattr(self, 'trained_model') or self.trained_model is None:
        print("❌ No trained model available to save!")
        print("=========== MANUAL SAVE FAILED ===========\n")
        return False
    
    try:
        import os
        import time
        import pickle
        from tkinter import messagebox
        
        # Get model information
        ticker = self.ticker_var.get() if hasattr(self, 'ticker_var') else "UNKNOWN"
        model_type = self.model_type_var.get() if hasattr(self, 'model_type_var') else "MODEL"
        print(f"Ticker: {ticker}, Model type: {model_type}")
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"{ticker}_{model_type}_{timestamp}.keras"
        print(f"Generated filename: {model_filename}")
        
        # Get models directory
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        os.makedirs(models_dir, exist_ok=True)
        print(f"Models directory: {models_dir}")
        
        # Save paths
        model_path = os.path.join(models_dir, model_filename)
        history_path = model_path.replace('.keras', '_history.pkl')
        print(f"Full model path: {model_path}")
        print(f"Full history path: {history_path}")
        
        # Save model
        print(f"Saving model to: {model_path}")
        self.trained_model.save(model_path, save_format='keras')
        
        # Verify model save
        if os.path.exists(model_path):
            print(f"✅ Model saved successfully! Size: {os.path.getsize(model_path)/1024/1024:.2f} MB")
        else:
            print("❌ ERROR: Model file was not created!")
            
        # Save history if available
        if hasattr(self, 'trained_history'):
            print(f"Saving history to: {history_path}")
            history_data = self.trained_history.history if hasattr(self.trained_history, 'history') else self.trained_history
            with open(history_path, 'wb') as f:
                pickle.dump(history_data, f)
                
            # Verify history save
            if os.path.exists(history_path):
                print(f"✅ History saved successfully! Size: {os.path.getsize(history_path)/1024:.2f} KB")
            else:
                print("❌ ERROR: History file was not created!")
        else:
            print("No history available to save")
            
        # Show message to user
        messagebox.showinfo("Save Complete", f"Model saved as:\n{model_filename}")
        
        # Refresh model list
        if hasattr(self, '_refresh_model_list'):
            print("Refreshing model list...")
            self._refresh_model_list()
            print("Model list refreshed")
            
        print("=========== MANUAL SAVE COMPLETED ===========\n")
        return True
        
    except Exception as e:
        print(f"❌ ERROR saving model: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Save Error", f"Error saving model: {str(e)}")
        print("=========== MANUAL SAVE FAILED ===========\n")
        return False

def load_model(model_path):
    """Load a model with proper extension handling"""
    import tensorflow as tf
    import os
    
    # If path doesn't end with .keras, try adding it
    if not model_path.endswith('.keras') and not os.path.exists(model_path):
        keras_path = model_path + '.keras'
        if os.path.exists(keras_path):
            model_path = keras_path
            
    return tf.keras.models.load_model(model_path)

def show_training_complete_dialog(self, model, history):
    """Show training complete dialog with OK button that saves the model"""
    import tkinter as tk
    from tkinter import ttk
    import os
    import time
    import pickle
    
    print("\n===== SHOWING TRAINING COMPLETE DIALOG =====")
    
    # Create dialog window
    dialog = tk.Toplevel(self)
    dialog.title("Training Complete")
    dialog.geometry("400x300")
    dialog.transient(self)  # Make dialog modal
    dialog.grab_set()
    
    # Configure dialog
    dialog.columnconfigure(0, weight=1)
    main_frame = ttk.Frame(dialog, padding=20)
    main_frame.grid(sticky="nsew")
    
    # Success message
    ttk.Label(
        main_frame, 
        text="Model Training Completed Successfully!", 
        font=("Arial", 12, "bold")
    ).pack(pady=10)
    
    # Training info
    info_frame = ttk.Frame(main_frame)
    info_frame.pack(fill="x", pady=10)
    
    # Get training information
    ticker = self.ticker_var.get() if hasattr(self, 'ticker_var') else "UNKNOWN"
    model_type = self.model_type_var.get() if hasattr(self, 'model_type_var') else "MODEL"
    
    # Get final loss values
    train_loss = history.history['loss'][-1] if hasattr(history, 'history') and 'loss' in history.history else "N/A"
    val_loss = history.history['val_loss'][-1] if hasattr(history, 'history') and 'val_loss' in history.history else "N/A"
    
    # Display training information
    ttk.Label(info_frame, text=f"Ticker: {ticker}").pack(anchor="w")
    ttk.Label(info_frame, text=f"Model Type: {model_type}").pack(anchor="w")
    ttk.Label(info_frame, text=f"Final Training Loss: {train_loss:.6f}" if train_loss != "N/A" else "Final Training Loss: N/A").pack(anchor="w")
    ttk.Label(info_frame, text=f"Final Validation Loss: {val_loss:.6f}" if val_loss != "N/A" else "Final Validation Loss: N/A").pack(anchor="w")
    
    # Save options
    save_frame = ttk.LabelFrame(main_frame, text="Save Options")
    save_frame.pack(fill="x", pady=10)
    
    # Filename preview
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{model_type}_{timestamp}.keras"
    ttk.Label(save_frame, text=f"Model will be saved as:").pack(anchor="w", padx=5, pady=2)
    ttk.Label(save_frame, text=filename, font=("Arial", 9, "italic")).pack(anchor="w", padx=5, pady=2)
    
    # Checkbox to auto-save
    auto_save_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        save_frame, 
        text="Save model when clicking OK", 
        variable=auto_save_var
    ).pack(anchor="w", padx=5, pady=5)
    
    # Function to handle OK button click
    def on_ok_click():
        if auto_save_var.get():
            print("OK clicked with save option enabled")
            try:
                # Generate paths
                models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, filename)
                history_path = model_path.replace('.keras', '_history.pkl')
                
                # Save model
                print(f"Saving model to: {model_path}")
                model.save(model_path, save_format='keras')
                print(f"Model saved successfully: {os.path.exists(model_path)}")
                
                # Save history
                if hasattr(history, 'history'):
                    print(f"Saving history to: {history_path}")
                    with open(history_path, 'wb') as f:
                        pickle.dump(history.history, f)
                    print(f"History saved successfully: {os.path.exists(history_path)}")
                
                # Store for future reference
                self.trained_model = model
                self.trained_history = history
                self.latest_model_path = model_path
                
                # Refresh model list
                if hasattr(self, '_refresh_model_list'):
                    print("Refreshing model list")
                    self._refresh_model_list()
                
                print("Model saved on OK button click")
            except Exception as e:
                print(f"Error saving model on OK click: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("OK clicked but save option was disabled")
            
        # Close dialog
        dialog.destroy()
    
    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill="x", pady=10)
    
    ok_button = ttk.Button(button_frame, text="OK", command=on_ok_click, width=10)
    ok_button.pack(side="right", padx=5)
    
    # Center dialog on parent window
    dialog.update_idletasks()
    dialog_width = dialog.winfo_width()
    dialog_height = dialog.winfo_height()
    parent_x = self.winfo_rootx()
    parent_y = self.winfo_rooty()
    parent_width = self.winfo_width()
    parent_height = self.winfo_height()
    x = parent_x + (parent_width - dialog_width) // 2
    y = parent_y + (parent_height - dialog_height) // 2
    dialog.geometry(f"+{x}+{y}")
    
    # Wait for dialog to close
    dialog.wait_window()
    
    print("===== TRAINING DIALOG CLOSED =====\n")