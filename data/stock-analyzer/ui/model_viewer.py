import os
import time
import pickle
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class ModelViewer:
    @staticmethod
    def view_model(models_dir, model_name):
        """Static method to view a model - can be called from anywhere"""
        print(f"ModelViewer.view_model called with model: {model_name}")
        
        try:
            # Create a new toplevel window
            window = tk.Toplevel()
            window.title(f"Model: {model_name}")
            window.geometry("500x400")
            window.lift()
            window.focus_force()
            
            # Model path
            model_path = os.path.join(models_dir, model_name)
            
            # Add basic model information
            info_frame = tk.Frame(window)
            info_frame.pack(fill=tk.X, padx=10, pady=10)
            
            tk.Label(info_frame, text=f"Model: {model_name}", font=("Arial", 14, "bold")).pack(anchor=tk.W)
            tk.Label(info_frame, text=f"Path: {model_path}", font=("Arial", 10)).pack(anchor=tk.W)
            
            if os.path.exists(model_path):
                size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
                modified = time.ctime(os.path.getmtime(model_path))
                tk.Label(info_frame, text=f"Size: {size:.2f} MB", font=("Arial", 10)).pack(anchor=tk.W)
                tk.Label(info_frame, text=f"Last Modified: {modified}", font=("Arial", 10)).pack(anchor=tk.W)
                status_text = "Model file exists"
            else:
                status_text = "Model file does not exist"
            
            status_label = tk.Label(info_frame, text=status_text, font=("Arial", 10), 
                     fg="green" if os.path.exists(model_path) else "red")
            status_label.pack(anchor=tk.W, pady=5)
            
            # Look for history file
            history_paths = [
                model_path.replace('.keras', '_history.pkl'),
                model_path.replace('.keras', '.history'),
                os.path.join(models_dir, model_name.split('.')[0] + '_history.pkl')
            ]
            
            history_path = None
            for path in history_paths:
                print(f"Checking for history at: {path}")
                if os.path.exists(path):
                    history_path = path
                    print(f"History file found at: {history_path}")
                    break
            
            if history_path:
                tk.Label(info_frame, text=f"History: {os.path.basename(history_path)}", 
                         font=("Arial", 10)).pack(anchor=tk.W)
                
                try:
                    # Try to load and display history plots
                    with open(history_path, 'rb') as f:
                        history = pickle.load(f)
                    
                    if isinstance(history, dict):
                        # Create plot frame
                        plot_frame = tk.Frame(window)
                        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                        
                        # Create figure
                        fig = Figure(figsize=(6, 4), dpi=100)
                        
                        # Plot loss if available
                        if 'loss' in history:
                            ax1 = fig.add_subplot(111)
                            ax1.plot(history['loss'], label='Training Loss')
                            if 'val_loss' in history:
                                ax1.plot(history['val_loss'], label='Validation Loss')
                            ax1.set_title('Model Loss')
                            ax1.set_xlabel('Epoch')
                            ax1.set_ylabel('Loss')
                            ax1.legend()
                            ax1.grid(True)
                            
                            # Add canvas to window
                            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                            canvas.draw()
                            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                            
                            # Add toolbar
                            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                            toolbar.update()
                    else:
                        tk.Label(info_frame, text="History file format not recognized", 
                                 font=("Arial", 10), fg="red").pack(anchor=tk.W)
                except Exception as e:
                    print(f"Error plotting history: {e}")
                    tk.Label(info_frame, text=f"Error plotting history: {e}", 
                             font=("Arial", 10), fg="red").pack(anchor=tk.W)
            else:
                tk.Label(info_frame, text="No history file found", 
                         font=("Arial", 10), fg="orange").pack(anchor=tk.W)
            
            # Add close button
            tk.Button(window, text="Close", command=window.destroy).pack(pady=10)
            
            print("Model viewer window displayed successfully")
            
        except Exception as e:
            print(f"Error in model viewer: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to view model: {str(e)}") 