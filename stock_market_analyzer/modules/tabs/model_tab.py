class ModelTab(BaseTab):
    """Model tab for managing and training machine learning models."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.model_cache = {}
        self.pending_requests = {}  # Track pending requests
        self.available_models = []  # List of trained models
        
    def setup_ui(self):
        """Setup the model tab UI."""
        main_layout = QVBoxLayout(self)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Model type selection
        controls_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "Linear Regression",
            "Random Forest",
            "LSTM",
            "XGBoost"
        ])
        controls_layout.addWidget(self.model_type_combo)
        
        # Ticker selection
        controls_layout.addWidget(QLabel("Ticker:"))
        self.ticker_combo = QComboBox()
        controls_layout.addWidget(self.ticker_combo)
        
        # Train button
        train_button = QPushButton("Train Model")
        train_button.clicked.connect(self.train_model)
        controls_layout.addWidget(train_button)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_models)
        controls_layout.addWidget(refresh_button)
        
        main_layout.addLayout(controls_layout)
        
        # Splitter for model list and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Model list
        model_list_widget = QWidget()
        model_list_layout = QVBoxLayout(model_list_widget)
        
        self.model_list = QListWidget()
        self.model_list.itemSelectionChanged.connect(self.on_model_selected)
        model_list_layout.addWidget(self.model_list)
        
        splitter.addWidget(model_list_widget)
        
        # Model details
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        self.details_text = QLabel("Model details will appear here")
        self.details_text.setWordWrap(True)
        details_layout.addWidget(self.details_text)
        
        splitter.addWidget(details_widget)
        
        # Set initial sizes
        splitter.setSizes([400, 400])
        
        main_layout.addWidget(splitter)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
    def train_model(self):
        """Train a new model."""
        try:
            model_type = self.model_type_combo.currentText()
            ticker = self.ticker_combo.currentText()
            
            if not all([model_type, ticker]):
                self.status_label.setText("Please select model type and ticker")
                return
                
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                'type': 'train',
                'model_type': model_type,
                'ticker': ticker,
                'timestamp': datetime.now(),
                'status': 'pending'
            }
            
            # Request data from Data tab
            self.message_bus.publish(
                "Model",
                "data_request",
                {
                    'request_id': request_id,
                    'ticker': ticker,
                    'purpose': 'training'
                }
            )
            
            self.status_label.setText(f"Requested data for training {model_type} model")
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def refresh_models(self):
        """Refresh the list of available models."""
        try:
            # Request model list from storage
            self.message_bus.publish(
                "Model",
                "model_list_request",
                {}
            )
            
        except Exception as e:
            self.logger.error(f"Error refreshing models: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def on_model_selected(self):
        """Handle model selection change."""
        try:
            selected_items = self.model_list.selectedItems()
            if not selected_items:
                return
                
            model_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if model_id in self.model_cache:
                self.update_model_details(self.model_cache[model_id])
                
        except Exception as e:
            self.logger.error(f"Error handling model selection: {e}")
            
    def process_message(self, sender: str, message_type: str, data: Any):
        """Process incoming messages."""
        try:
            if message_type == "data_response":
                self.handle_data_response(sender, data)
            elif message_type == "prediction_request":
                self.handle_prediction_request(sender, data)
            elif message_type == "model_list_response":
                self.handle_model_list_response(sender, data)
            elif message_type == "error":
                self.handle_error(sender, data)
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def handle_data_response(self, sender: str, data: Any):
        """Handle data response from Data tab."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                if request['type'] == 'train':
                    # Train model with received data
                    self.train_model_with_data(request, data.get('data'))
                elif request['type'] == 'predict':
                    # Make prediction with received data
                    self.make_prediction_with_data(request, data.get('data'))
                    
        except Exception as e:
            self.logger.error(f"Error handling data response: {e}")
            
    def handle_prediction_request(self, sender: str, data: Any):
        """Handle prediction request from Predictions tab."""
        try:
            request_id = data.get('request_id')
            ticker = data.get('ticker')
            model = data.get('model')
            horizon = data.get('horizon')
            
            if not all([request_id, ticker, model, horizon]):
                self.logger.error("Invalid prediction request")
                return
                
            # Store request
            self.pending_requests[request_id] = {
                'type': 'predict',
                'ticker': ticker,
                'model': model,
                'horizon': horizon,
                'timestamp': datetime.now(),
                'status': 'pending'
            }
            
            # Request data from Data tab
            self.message_bus.publish(
                "Model",
                "data_request",
                {
                    'request_id': request_id,
                    'ticker': ticker,
                    'purpose': 'prediction'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling prediction request: {e}")
            
    def handle_model_list_response(self, sender: str, data: Any):
        """Handle model list response from storage."""
        try:
            models = data.get('models', [])
            self.available_models = models
            
            # Update model list
            self.model_list.clear()
            for model in models:
                item = QListWidgetItem(f"{model['type']} - {model['ticker']}")
                item.setData(Qt.ItemDataRole.UserRole, model['id'])
                self.model_list.addItem(item)
                
            # Publish updated model list
            self.message_bus.publish(
                "Model",
                "model_list",
                {
                    'models': [model['type'] for model in models]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling model list response: {e}")
            
    def handle_error(self, sender: str, data: Any):
        """Handle error messages."""
        try:
            request_id = data.get('request_id')
            error_message = data.get('error')
            
            if request_id in self.pending_requests:
                self.pending_requests[request_id]['status'] = 'error'
                self.pending_requests[request_id]['error'] = error_message
                
                # Forward error to requester
                request = self.pending_requests[request_id]
                if request['type'] == 'predict':
                    self.message_bus.publish(
                        "Model",
                        "error",
                        {
                            'request_id': request_id,
                            'error': error_message
                        }
                    )
                    
            self.status_label.setText(f"Error: {error_message}")
            
        except Exception as e:
            self.logger.error(f"Error handling error message: {e}")
            
    def train_model_with_data(self, request: Dict, data: pd.DataFrame):
        """Train model with received data."""
        try:
            model_type = request['model_type']
            ticker = request['ticker']
            
            # Train model (implementation depends on model type)
            model = self.train_model_by_type(model_type, data)
            
            # Store trained model
            model_id = str(uuid.uuid4())
            self.model_cache[model_id] = {
                'id': model_id,
                'type': model_type,
                'ticker': ticker,
                'model': model,
                'timestamp': datetime.now(),
                'metrics': self.evaluate_model(model, data)
            }
            
            # Update model list
            self.refresh_models()
            
            self.status_label.setText(f"Trained {model_type} model for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def make_prediction_with_data(self, request: Dict, data: pd.DataFrame):
        """Make prediction with received data."""
        try:
            model = request['model']
            horizon = request['horizon']
            
            # Get model from cache
            model_data = next((m for m in self.model_cache.values() if m['type'] == model), None)
            if not model_data:
                raise ValueError(f"Model {model} not found")
                
            # Make predictions
            predictions = self.predict_with_model(model_data['model'], data, horizon)
            
            # Send predictions to Predictions tab
            self.message_bus.publish(
                "Model",
                "prediction_response",
                {
                    'request_id': request['request_id'],
                    'results': {
                        'predictions': predictions,
                        'details': {
                            'model': model,
                            'horizon': horizon,
                            'accuracy': model_data['metrics']['accuracy'],
                            'last_training': model_data['timestamp'],
                            'features': model_data['metrics']['features']
                        }
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            self.message_bus.publish(
                "Model",
                "error",
                {
                    'request_id': request['request_id'],
                    'error': str(e)
                }
            )
            
    def train_model_by_type(self, model_type: str, data: pd.DataFrame):
        """Train model based on type."""
        # Implementation depends on model type
        # This is a placeholder for actual model training code
        return None
        
    def evaluate_model(self, model: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance."""
        # Implementation depends on model type
        # This is a placeholder for actual model evaluation code
        return {
            'accuracy': 0.0,
            'features': []
        }
        
    def predict_with_model(self, model: Any, data: pd.DataFrame, horizon: int) -> List[Dict]:
        """Make predictions using the model."""
        # Implementation depends on model type
        # This is a placeholder for actual prediction code
        return []
        
    def update_model_details(self, model_data: Dict):
        """Update model details display."""
        try:
            text = []
            text.append(f"Type: {model_data['type']}")
            text.append(f"Ticker: {model_data['ticker']}")
            text.append(f"Trained: {model_data['timestamp']}")
            text.append(f"Accuracy: {model_data['metrics']['accuracy']:.2%}")
            text.append(f"Features: {', '.join(model_data['metrics']['features'])}")
            
            self.details_text.setText("\n".join(text))
            
        except Exception as e:
            self.logger.error(f"Error updating model details: {e}")
            
    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        self.model_cache.clear()
        self.pending_requests.clear()

def main():
    """Main function for the model tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting model tab process")
    
    # Create and show the model tab
    window = ModelTab()
    window.setWindowTitle("Model Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 