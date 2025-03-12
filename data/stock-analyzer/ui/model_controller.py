import os
import pickle
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time

class ModelController:
    def view_model(self, model_name):
        """View model with proper .keras extension handling"""
        print(f"View Model button clicked for: {model_name}")
        
        # Ensure we're looking for .keras files
        if not model_name.endswith('.keras'):
            model_path = os.path.join(self.models_dir, model_name + '.keras')
            print(f"Adding .keras extension, looking at: {model_path}")
        else:
            model_path = os.path.join(self.models_dir, model_name)
            print(f"Model already has .keras extension: {model_path}")
        
        # Check if the model exists
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            # Try the original path without extension as fallback
            model_path = os.path.join(self.models_dir, model_name)
            print(f"Trying fallback path: {model_path}")
        
        # Load model
        print(f"Loading model from: {model_path}")
        
        try:
            print(f"Attempting to view model: {model_name}")
            
            # Check for history file - try multiple possible naming patterns
            possible_history_paths = [
                model_path.replace('.keras', '_history.pkl'),
                model_path.replace('.keras', '.history'),
                model_path + '.history',
                os.path.join(self.models_dir, model_name.split('.')[0] + '_history.pkl')
            ]
            
            history_path = None
            for path in possible_history_paths:
                print(f"Checking for history at: {path}")
                if os.path.exists(path):
                    history_path = path
                    print(f"History file found at: {history_path}")
                    break
            
            # If no history file is found, create a simple window with model info
            if not history_path:
                print("No history file found for model")
                self._show_model_info_without_history(model_name, model_path)
                return
                
            # Try to load history
            try:
                print(f"Loading history from: {history_path}")
                with open(history_path, 'rb') as f:
                    history = pickle.load(f)
                print(f"History keys: {list(history.keys()) if isinstance(history, dict) else 'not a dict'}")
                
                # Ensure history is a dictionary
                if not isinstance(history, dict):
                    print(f"History is not a dictionary, it's a {type(history)}")
                    if hasattr(history, 'history'):
                        history = history.history
                        print(f"Extracted history from object, keys: {list(history.keys())}")
                    else:
                        print("Could not extract history dictionary")
                        self._show_model_info_without_history(model_name, model_path)
                        return
                        
            except Exception as e:
                print(f"Error loading history file: {e}")
                import traceback
                traceback.print_exc()
                self._show_model_info_without_history(model_name, model_path)
                return
            
            # Now create the visualization window
            self._create_history_visualization(model_name, history)
            
        except Exception as e:
            print(f"Error viewing model: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to view model: {str(e)}")

    def _show_model_info_without_history(self, model_name, model_path):
        """Display basic model information when history is not available"""
        try:
            # Create a simple window with model info
            info_window = tk.Toplevel()
            info_window.title(f"Model Information: {model_name}")
            info_window.geometry("500x300")
            info_window.lift()
            info_window.focus_force()
            
            # Add some basic information
            tk.Label(info_window, text=f"Model: {model_name}", font=("Arial", 12, "bold")).pack(pady=10)
            tk.Label(info_window, text=f"Path: {model_path}", font=("Arial", 10)).pack(pady=5)
            
            # Try to get file size and creation date
            if os.path.exists(model_path):
                size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
                created = time.ctime(os.path.getctime(model_path))
                modified = time.ctime(os.path.getmtime(model_path))
                
                tk.Label(info_window, text=f"Size: {size:.2f} MB", font=("Arial", 10)).pack(pady=5)
                tk.Label(info_window, text=f"Created: {created}", font=("Arial", 10)).pack(pady=5)
                tk.Label(info_window, text=f"Last Modified: {modified}", font=("Arial", 10)).pack(pady=5)
            
            # Add a message about missing history
            message = tk.Label(
                info_window, 
                text="Training history not available for this model.\n"
                     "To see training plots, ensure history is saved during training.",
                font=("Arial", 10),
                fg="red"
            )
            message.pack(pady=20)
            
            # Add a close button
            close_btn = tk.Button(info_window, text="Close", command=info_window.destroy)
            close_btn.pack(pady=10)
            
            print("Displayed model info window without history")
            
        except Exception as e:
            print(f"Error showing model info: {e}")
            import traceback
            traceback.print_exc()
            
    def _create_history_visualization(self, model_name, history):
        """Create a visualization window for the model training history"""
        try:
            print("Creating history visualization")
            
            # Create window
            plot_window = tk.Toplevel()
            plot_window.title(f"Training Results: {model_name}")
            plot_window.geometry("800x600")
            plot_window.lift()
            plot_window.focus_force()
            
            # Initialize matplotlib figure
            import matplotlib
            matplotlib.use("TkAgg")  # Explicitly set backend
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            
            print("Creating figure")
            fig = Figure(figsize=(10, 8), dpi=100)
            
            # First, determine which metrics are available
            available_metrics = []
            if 'loss' in history:
                available_metrics.append('loss')
            if 'val_loss' in history:
                available_metrics.append('val_loss')
            if 'accuracy' in history:
                available_metrics.append('accuracy')
            if 'val_accuracy' in history:
                available_metrics.append('val_accuracy')
                
            print(f"Available metrics: {available_metrics}")
            
            # Plot loss
            if 'loss' in available_metrics:
                print("Plotting loss")
                ax1 = fig.add_subplot(211)
                epochs = range(1, len(history['loss']) + 1)
                
                ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
                if 'val_loss' in available_metrics:
                    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
                    
                ax1.set_title('Model Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True)
                
            # Plot accuracy if available
            if 'accuracy' in available_metrics:
                print("Plotting accuracy")
                ax2 = fig.add_subplot(212)
                epochs = range(1, len(history['accuracy']) + 1)
                
                ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
                if 'val_accuracy' in available_metrics:
                    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
                    
                ax2.set_title('Model Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                ax2.grid(True)
            
            fig.tight_layout()
            
            # Create frame for the figure
            frame = tk.Frame(plot_window)
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Create canvas and add it to the frame
            print("Creating canvas")
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Add toolbar
            print("Adding toolbar")
            toolbar_frame = tk.Frame(plot_window)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            print("History visualization complete")
            
        except Exception as e:
            print(f"Error creating history visualization: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Visualization Error", f"Failed to create visualization: {str(e)}") 

    def view_model_simple(self, model_name):
        """A simplified version of view_model that just opens a basic dialog"""
        print(f"*** Simple view_model function called with model: {model_name} ***")
        
        try:
            model_path = os.path.join(self.models_dir, model_name)
            
            # Create a simple window
            window = tk.Toplevel()
            window.title(f"Model: {model_name}")
            window.geometry("400x300")
            window.lift()
            window.focus_force()
            
            # Add some basic information
            tk.Label(window, text=f"Model: {model_name}", font=("Arial", 14, "bold")).pack(pady=10)
            tk.Label(window, text=f"Path: {model_path}", font=("Arial", 10)).pack(pady=5)
            
            if os.path.exists(model_path):
                size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
                modified = time.ctime(os.path.getmtime(model_path))
                tk.Label(window, text=f"Size: {size:.2f} MB", font=("Arial", 10)).pack(pady=5)
                tk.Label(window, text=f"Last Modified: {modified}", font=("Arial", 10)).pack(pady=5)
                status_text = "Model file exists"
            else:
                status_text = "Model file does not exist"
            
            tk.Label(window, text=status_text, font=("Arial", 10), 
                     fg="green" if os.path.exists(model_path) else "red").pack(pady=10)
            
            # Close button
            tk.Button(window, text="Close", command=window.destroy).pack(pady=20)
            
            print("Simple model info window displayed")
            
        except Exception as e:
            print(f"Error in simple view: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to view model: {str(e)}") 