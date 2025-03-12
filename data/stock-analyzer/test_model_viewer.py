"""
Test script for Model Viewer
"""
import tkinter as tk
from ui.model_viewer import open_model_viewer_window

def main():
    """Run a standalone test of the model viewer"""
    root = tk.Tk()
    root.title("Model Viewer Test")
    root.geometry("200x100")
    
    # Create a button to open the model viewer
    btn = tk.Button(root, text="Open Model Viewer", 
                   command=lambda: open_model_viewer_window(root))
    btn.pack(padx=20, pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    main() 