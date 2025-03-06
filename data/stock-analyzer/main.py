#!/usr/bin/env python3
"""
Stock Market Analyzer - Main Entry Point
"""
import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set environment variable to silence Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Force CPU usage for TensorFlow to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_DISABLE_GRAPPLER'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.set_floatx('float32')

from ui.main_window import StockAnalyzerApp

def main():
    """Main entry point for the application"""
    app = StockAnalyzerApp()
    app.run()

if __name__ == "__main__":
    main()
