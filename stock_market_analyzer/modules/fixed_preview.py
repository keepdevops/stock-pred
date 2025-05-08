def preview_import_data(self):
    """Preview the imported data."""
    try:
        file_path = self.file_path.text()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select a file")
            return
            
        format_type = self.file_format.currentText()
        self.logger.info(f"Previewing data from {file_path} with format {format_type}")
        
        # Initialize data as None
        data = None
        
        # Auto-detect format if needed
        if format_type == "Auto-detect":
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                format_type = "CSV"
            elif file_ext == '.json':
                format_type = "JSON"
            elif file_ext in ['.xlsx', '.xls']:
                format_type = "Excel"
            elif file_ext == '.duckdb':
                format_type = "DuckDB"
            elif file_ext in ['.db', '.sqlite', '.sqlite3']:
                format_type = "SQLite"
            elif file_ext == '.h5':
                format_type = "HDF5"
            elif file_ext == '.parquet':
                format_type = "Parquet"
            else:
                raise ValueError(f"Could not auto-detect format for file: {file_path}")
            self.logger.info(f"Auto-detected format: {format_type}")
        
        # Load data based on format
        if format_type == "CSV":
            data = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded CSV data with {len(data)} rows")
        elif format_type == "JSON":
            data = pd.read_json(file_path)
            self.logger.info(f"Successfully loaded JSON data with {len(data)} rows")
        elif format_type == "DuckDB":
            try:
                import duckdb
                conn = duckdb.connect(file_path)
                # Get list of tables
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if not tables:
                    raise ValueError("No tables found in DuckDB file")
                
                # Get the first table's data
                table_name = tables[0][0]
                data = conn.execute(f"SELECT * FROM {table_name}").df()
                
                # Close the connection
                conn.close()
                self.logger.info(f"Successfully loaded DuckDB data with {len(data)} rows")
                
                # Check if we got any data
                if data.empty:
                    raise ValueError(f"No data found in table '{table_name}'")
                    
            except Exception as e:
                self.logger.error(f"Error reading DuckDB file: {str(e)}")
                raise ValueError(f"Error reading DuckDB file: {str(e)}")
        elif format_type == "Excel":
            data = pd.read_excel(file_path)
            self.logger.info(f"Successfully loaded Excel data with {len(data)} rows")
        elif format_type == "SQLite":
            import sqlite3
            conn = sqlite3.connect(file_path)
            # Get list of tables
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if tables.empty:
                conn.close()
                raise ValueError("No tables found in SQLite database")
            
            # Get the first table's data
            table_name = tables.iloc[0, 0]
            data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            self.logger.info(f"Successfully loaded SQLite data with {len(data)} rows")
        elif format_type == "HDF5":
            data = pd.read_hdf(file_path)
            self.logger.info(f"Successfully loaded HDF5 data with {len(data)} rows")
        elif format_type == "Parquet":
            data = pd.read_parquet(file_path)
            self.logger.info(f"Successfully loaded Parquet data with {len(data)} rows")
        else:
            raise ValueError(f"Unsupported file format: {format_type}")
        
        # Check if data was loaded successfully
        if data is None:
            raise ValueError(f"Failed to load data for format: {format_type}")
            
        # Verify we have data
        if data.empty:
            raise ValueError("No data was found in the file")
        
        # Update import data table
        if hasattr(self, 'import_data_table'):
            self.import_data_table.setRowCount(len(data))
            self.import_data_table.setColumnCount(len(data.columns))
            self.import_data_table.setHorizontalHeaderLabels(data.columns)
            
            for i, (_, row) in enumerate(data.iterrows()):
                for j, (col, value) in enumerate(row.items()):
                    self.import_data_table.setItem(i, j, QTableWidgetItem(str(value)))
            
            # Update chart if we have price data
            if all(col in data.columns for col in ['date', 'close']):
                self.import_chart.update_data(data)
        
        # Update symbol entry with filename
        symbol = os.path.splitext(os.path.basename(file_path))[0]
        # Extract ticker from filename if possible
        match = re.search(r'_([A-Z]{1,5})(?:\.csv|\.json|\.db|$)', symbol)
        if match:
            symbol = match.group(1)
        
        # If we have a symbol override field, set it with detected symbol
        if hasattr(self, 'symbol_override'):
            self.symbol_override.setText(symbol)
        
        self.logger.info(f"Preview completed successfully for {file_path}")
        
    except Exception as e:
        self.logger.error(f"Error previewing import data: {e}")
        import traceback
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        QMessageBox.critical(self, "Error", f"Failed to preview data: {str(e)}") 