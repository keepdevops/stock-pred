"""
Panel for displaying training results visualization
"""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from ui.visualization.base_panel import BasePanel
from ui.ui_utils import apply_listbox_style

class TrainingResultsPanel(BasePanel):
    def __init__(self, parent, event_bus):
        super().__init__(parent, event_bus)
        
        # Initialize model data
        self.current_model = None
        self.current_ticker = None
        
        # Create UI components
        self._create_ui()
        
        # Subscribe to events
        event_bus.subscribe("training_completed", self._on_training_completed)
        event_bus.subscribe("show_model_details", self._on_show_model_details)
        event_bus.subscribe("model_loaded", self._on_model_loaded)
        
    def _create_ui(self):
        """Create the UI components"""
        # Create a frame for controls
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Create ticker/model selection dropdown
        ttk.Label(controls_frame, text="Model:").pack(side="left", padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(controls_frame, textvariable=self.model_var, state="readonly")
        self.model_combo.pack(side="left", padx=5)
        
        # Bind model selection event
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        
        # Create view selection dropdown
        ttk.Label(controls_frame, text="View:").pack(side="left", padx=5)
        self.view_var = tk.StringVar(value="Loss")
        self.view_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.view_var,
            values=["Loss", "Accuracy", "Test Predictions", "Model Summary"],
            state="readonly"
        )
        self.view_combo.current(0)
        self.view_combo.pack(side="left", padx=5)
        
        # Bind view selection event
        self.view_combo.bind("<<ComboboxSelected>>", self._on_view_selected)
        
        # Create a frame for the main content (plot or text)
        content_frame = ttk.Frame(self)
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Pack the matplotlib canvas into the content frame
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Status label
        self.status_var = tk.StringVar(value="No training results available")
        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.status_label.pack(fill="x", padx=10, pady=5)
        
        # Initial message
        self.show_error("Select a trained model to view results")
        
    def _on_training_completed(self, data):
        """Handle training completed event"""
        tickers = data.get('tickers', [])
        models = data.get('models', {})
        
        if not models:
            self.status_var.set("Training completed but no models were generated")
            return
            
        # Update model dropdown
        self.model_combo['values'] = list(models.keys())
        self.model_combo.current(0)
        
        # Store the current ticker
        self.current_ticker = self.model_var.get()
        
        # Get the training service to access the model
        training_service = self._get_training_service()
        if training_service:
            self.current_model = training_service.get_training_model(self.current_ticker)
            
        # Display the first model's results
        self._display_training_results()
        
        self.status_var.set(f"Training completed for {', '.join(tickers)}")
        
    def _on_model_selected(self, event):
        """Handle model selection"""
        self.current_ticker = self.model_var.get()
        
        # Get training service to access model
        training_service = self._get_training_service()
        if not training_service:
            self.show_error("Could not access training service")
            return
            
        self.current_model = training_service.get_training_model(self.current_ticker)
        if not self.current_model or not self.current_model.has_results():
            self.show_error(f"No results available for {self.current_ticker}")
            return
            
        self._display_training_results()
        
    def _on_view_selected(self, event):
        """Handle view selection"""
        self._display_training_results()
        
    def _on_show_model_details(self, data):
        """Handle show model details event"""
        model_name = data.get('model_name')
        model_path = data.get('model_path')
        metadata = data.get('metadata', {})
        
        if not model_name:
            return
            
        # Try to get ticker from metadata or filename
        ticker = metadata.get('ticker')
        if not ticker:
            # Try to extract from filename
            ticker = model_name.split('_')[0]  # Assuming format: TICKER_TIMESTAMP.keras
            
        # Update the UI
        if ticker:
            # Update dropdown if needed
            values = list(self.model_combo['values'])
            if ticker not in values:
                values.append(ticker)
                self.model_combo['values'] = values
                
            self.model_var.set(ticker)
            self.current_ticker = ticker
            
        # Set view to model summary
        self.view_var.set("Model Summary")
        
        # Display model information
        self.clear_plot()
        ax = self.figure.add_subplot(111)
        ax.axis('off')
        
        # Format metadata
        info_text = f"Model: {model_name}\n"
        info_text += f"Path: {model_path}\n\n"
        
        for key, value in metadata.items():
            if key not in ['name', 'model_path']:
                info_text += f"{key}: {value}\n"
                
        ax.text(0.5, 0.5, info_text, 
               ha='center', va='center', fontsize=12, 
               transform=ax.transAxes)
               
        self.canvas.draw()
        self.status_var.set(f"Showing details for {model_name}")
        
    def _on_model_loaded(self, data):
        """Handle model loaded event"""
        ticker = data.get('ticker')
        if not ticker:
            return
            
        # Update the model list
        values = list(self.model_combo['values'])
        if ticker not in values:
            values.append(ticker)
            self.model_combo['values'] = values
            
        # Set current model
        self.model_var.set(ticker)
        self.current_ticker = ticker
        
        # Get training service to access model
        training_service = self._get_training_service()
        if training_service:
            self.current_model = training_service.get_training_model(ticker)
            
        # Display results
        self.view_var.set("Model Summary")
        self._display_training_results()
        
    def _display_training_results(self):
        """Display training results for the current model"""
        if not self.current_model or not self.current_model.has_results():
            self.show_error(f"No training results available for {self.current_ticker}")
            return
            
        view_type = self.view_var.get()
        self.clear_plot()
        
        try:
            if view_type == "Loss":
                self._plot_loss()
            elif view_type == "Accuracy":
                self._plot_accuracy()
            elif view_type == "Test Predictions":
                self._plot_test_predictions()
            elif view_type == "Model Summary":
                self._show_model_summary()
            else:
                self.show_error(f"Unknown view type: {view_type}")
                
            self.status_var.set(f"Displaying {view_type} for {self.current_ticker}")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.show_error(f"Error displaying results: {str(e)}")
    
    def _plot_loss(self):
        """Plot training and validation loss"""
        if not self.current_model.history:
            self.show_error("No training history available")
            return
            
        ax = self.figure.add_subplot(111)
        
        # Plot training & validation loss values
        if 'loss' in self.current_model.history.history:
            ax.plot(self.current_model.history.history['loss'], label='Training Loss')
        
        if 'val_loss' in self.current_model.history.history:
            ax.plot(self.current_model.history.history['val_loss'], label='Validation Loss')
            
        ax.set_title(f'Model Loss for {self.current_ticker}')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        ax.grid(True)
        
        self.canvas.draw()
        
    def _plot_accuracy(self):
        """Plot training and validation accuracy if available"""
        if not self.current_model.history:
            self.show_error("No training history available")
            return
            
        # Check if we have accuracy metrics (not always available in regression)
        has_accuracy = False
        for key in self.current_model.history.history.keys():
            if 'acc' in key:
                has_accuracy = True
                break
                
        if not has_accuracy:
            self.show_error("Accuracy metrics not available for this model")
            return
            
        ax = self.figure.add_subplot(111)
        
        # Plot training & validation accuracy values
        if 'accuracy' in self.current_model.history.history:
            ax.plot(self.current_model.history.history['accuracy'], label='Training Accuracy')
        elif 'acc' in self.current_model.history.history:
            ax.plot(self.current_model.history.history['acc'], label='Training Accuracy')
        
        if 'val_accuracy' in self.current_model.history.history:
            ax.plot(self.current_model.history.history['val_accuracy'], label='Validation Accuracy')
        elif 'val_acc' in self.current_model.history.history:
            ax.plot(self.current_model.history.history['val_acc'], label='Validation Accuracy')
            
        ax.set_title(f'Model Accuracy for {self.current_ticker}')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(loc='lower right')
        ax.grid(True)
        
        self.canvas.draw()
        
    def _plot_test_predictions(self):
        """Plot test predictions vs actual values"""
        if not hasattr(self.current_model, 'X_test') or self.current_model.X_test is None or \
           not hasattr(self.current_model, 'y_test') or self.current_model.y_test is None:
            self.show_error("Test data not available")
            return
            
        ax = self.figure.add_subplot(111)
        
        try:
            # Get predictions on test data
            predictions = self.current_model.model.predict(self.current_model.X_test)
            
            # Plot actual vs predictions
            ax.plot(self.current_model.y_test, label='Actual', color='blue')
            ax.plot(predictions, label='Predicted', color='red', linestyle='--')
            
            ax.set_title(f'Test Predictions vs Actual for {self.current_ticker}')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Value')
            ax.legend(loc='best')
            ax.grid(True)
            
            self.canvas.draw()
        except Exception as e:
            self.show_error(f"Error plotting test predictions: {str(e)}")
        
    def _show_model_summary(self):
        """Show model summary"""
        if not self.current_model.model:
            self.show_error("No model available")
            return
            
        # Get model summary as a string
        from io import StringIO
        summary_io = StringIO()
        self.current_model.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        summary_string = summary_io.getvalue()
        
        # Display summary on plot
        ax = self.figure.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, summary_string, 
               ha='center', va='center', fontsize=9,
               transform=ax.transAxes, family='monospace')
               
        self.canvas.draw()
        
    def _get_training_service(self):
        """Get a reference to the training service"""
        try:
            # Try to find the main app window
            import tkinter as tk
            root = None
            for window in tk._default_root.winfo_children():
                if hasattr(window, 'training_service'):
                    return window.training_service
                    
            # If not found, try creating a new instance
            from services.training_service import TrainingService
            from services.data_service import DataService
            
            return TrainingService(DataService())
        except Exception as e:
            print(f"Error getting training service: {str(e)}")
            return None 