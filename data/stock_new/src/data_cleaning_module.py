import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional
import re

class DataCleaner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def clean_csv_files(self, input_dir: str = "data/raw", output_dir: str = "data/clean") -> None:
        """Clean all CSV files in the input directory"""
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process all raw CSV files
            for csv_file in input_path.glob("raw_*.csv"):
                try:
                    self._clean_single_file(csv_file, 