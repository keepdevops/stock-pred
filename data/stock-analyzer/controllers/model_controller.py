"""
Controller for model operations - bridges UI and services
"""
import os
import datetime
import traceback
from models.model_factory import ModelFactory
from utils.event_bus import event_bus

class ModelController:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Subscribe to events
        self.event_bus.subscribe("save_model", self.handle_save_model)
        self.event_bus.subscribe("load_model", self.handle_load_model)
        self.event_bus.subscribe("view_model", self.handle_view_model)
        
        print(f"ModelController initialized. Models directory: {self.models_dir}")
        
    def handle_save_model(self, data):
        """Handle save model event"""
        print(f"Handling save model event with data: {data}")
        ticker = data.get('model_name')
        if not ticker:
            self.event_bus.publish("model_save_error", {'message': "No ticker provided"})
            return
            
        try:
            # Get the training service to access the trained model
            training_service = self._get_training_service()
            if not training_service:
                self.event_bus.publish("model_save_error", 
                                      {'message': "Could not access training service"})
                return
            
            # Get the training model for this ticker
            training_model = training_service.get_training_model(ticker)
            if not training_model or not training_model.model:
                self.event_bus.publish("model_save_error", 
                                      {'message': f"No trained model found for {ticker}"})
                return
                
            # Create a filename for the model
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_{timestamp}.keras"
            model_path = os.path.join(self.models_dir, filename)
            
            # Create metadata
            metadata = {
                'ticker': ticker,
                'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_type': training_model.model.__class__.__name__,
                'performance': training_model.performance if hasattr(training_model, 'performance') else None
            }
            
            # Save the model
            ModelFactory.save_model(training_model.model, model_path, metadata)
            
            # Publish success event
            self.event_bus.publish("model_saved", {
                'model_path': model_path,
                'ticker': ticker
            })
            
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Error saving model: {str(e)}\n{traceback_str}")
            self.event_bus.publish("model_save_error", {'message': str(e)})
    
    def handle_load_model(self, data):
        """Handle load model event"""
        print(f"Handling load model event with data: {data}")
        model_path = data.get('model_path')
        if not model_path:
            self.event_bus.publish("model_load_error", {'message': "No model path provided"})
            return
            
        try:
            # Load the model using the ModelFactory
            model, metadata = ModelFactory.load_model(model_path)
            
            # Get ticker from metadata or filename
            ticker = metadata.get('ticker')
            if not ticker:
                # Try to extract from filename
                filename = os.path.basename(model_path)
                ticker = filename.split('_')[0]  # Assuming format: TICKER_TIMESTAMP.keras
            
            # Get training service to register the model
            training_service = self._get_training_service()
            if training_service:
                # Create or update training model for this ticker
                if ticker not in training_service.training_models:
                    from models.training_model import TrainingModel
                    training_service.training_models[ticker] = TrainingModel(ticker)
                
                # Update model
                training_service.training_models[ticker].model = model
                
                # Update other attributes if available in metadata
                if 'performance' in metadata and metadata['performance'] is not None:
                    training_service.training_models[ticker].performance = metadata['performance']
            
            # Publish success event
            self.event_bus.publish("model_loaded", {
                'model_path': model_path, 
                'metadata': metadata,
                'ticker': ticker
            })
            
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Error loading model: {str(e)}\n{traceback_str}")
            self.event_bus.publish("model_load_error", {'message': str(e)})
    
    def handle_view_model(self, data):
        """Handle view model event"""
        print(f"Handling view model event with data: {data}")
        model_name = data.get('model_name')
        if not model_name:
            return
            
        # Find the model file
        try:
            model_path = None
            if os.path.exists(os.path.join(self.models_dir, model_name)):
                model_path = os.path.join(self.models_dir, model_name)
            
            if not model_path:
                self.event_bus.publish("model_view_error", 
                                      {'message': f"Model file {model_name} not found"})
                return
                
            # Load the model to get metadata
            _, metadata = ModelFactory.load_model(model_path)
            
            # Publish event to show model details
            self.event_bus.publish("show_model_details", {
                'model_name': model_name,
                'model_path': model_path,
                'metadata': metadata
            })
            
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Error viewing model: {str(e)}\n{traceback_str}")
            self.event_bus.publish("model_view_error", {'message': str(e)})
    
    def _get_training_service(self):
        """Get a reference to the training service"""
        # Look for main application window with training service
        try:
            # This is a heuristic approach - you might need to adjust based on
            # how your application manages services
            import tkinter as tk
            root = tk.Tk.winfo_get_all()[0]
            
            # Try to find main app attributes
            if hasattr(root, 'training_service'):
                return root.training_service
                
            # If we can't find it directly, create a new instance
            from services.training_service import TrainingService
            from services.data_service import DataService
            
            data_service = DataService()
            return TrainingService(data_service)
            
        except Exception as e:
            print(f"Error getting training service: {str(e)}")
            return None 