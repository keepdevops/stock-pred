import os
import shutil
from datetime import datetime
import logging
from pathlib import Path

def organize_csv_files():
    """
    Move all CSV files from stock_new/data to stock_new/data/csv directory.
    Creates a backup of existing files with timestamp.
    """
    try:
        # Set up paths
        base_dir = os.path.abspath(os.path.join(os.getcwd(), 'data'))
        csv_dir = os.path.join(base_dir, 'csv')
        
        # Create csv directory if it doesn't exist
        os.makedirs(csv_dir, exist_ok=True)
        
        # Get current timestamp for backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Find all CSV files in data directory and its subdirectories
        moved_files = 0
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.csv'):
                    source_path = os.path.join(root, file)
                    
                    # Skip if file is already in csv directory
                    if root == csv_dir:
                        continue
                    
                    # Create new filename with timestamp if file already exists
                    dest_path = os.path.join(csv_dir, file)
                    if os.path.exists(dest_path):
                        name, ext = os.path.splitext(file)
                        dest_path = os.path.join(csv_dir, f"{name}_{timestamp}{ext}")
                    
                    # Move the file
                    shutil.move(source_path, dest_path)
                    moved_files += 1
                    print(f"Moved: {source_path} -> {dest_path}")
        
        print(f"\nSuccessfully moved {moved_files} CSV files to {csv_dir}")
        return True
        
    except Exception as e:
        print(f"Error organizing CSV files: {str(e)}")
        return False

def organize_json_files():
    """
    Move all JSON files from stock_new/data to stock_new/data/json directory.
    Creates a backup of existing files with timestamp.
    """
    try:
        # Set up paths
        base_dir = Path('data')
        json_dir = base_dir / 'json'
        
        # Create json directory if it doesn't exist
        json_dir.mkdir(exist_ok=True)
        
        # Get current timestamp for backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Move all JSON files
        moved_files = 0
        for json_file in base_dir.rglob('*.json'):
            # Skip if file is already in json directory
            if json_dir in json_file.parents:
                continue
                
            # Create new filename with timestamp if file already exists
            new_path = json_dir / json_file.name
            if new_path.exists():
                new_path = json_dir / f"{json_file.stem}_{timestamp}{json_file.suffix}"
            
            # Move the file
            shutil.move(str(json_file), str(new_path))
            moved_files += 1
            print(f"Moved: {json_file.name} -> {new_path.name}")
        
        print(f"\nSuccessfully moved {moved_files} JSON files to {json_dir}")
        return True
        
    except Exception as e:
        print(f"Error organizing JSON files: {str(e)}")
        return False

if __name__ == "__main__":
    organize_csv_files()
    organize_json_files() 