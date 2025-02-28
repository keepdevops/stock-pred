import tkinter as tk
from tkinter import ttk
import duckdb
from tkinter import messagebox
from ticker_plotter import TickerPlotter
from ai_agent import TickerAIAgent
from ticker_ai_agent import TickerAIAgent as SimpleAIAgent
import tensorflow as tf
import numpy as np
from predictions_plotter import PredictionsPlotter
import glob
import os

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind('<Enter>', self.show_tooltip)
        self.widget.bind('<Leave>', self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        # Create tooltip window
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(self.tooltip, text=self.text, 
                         justify=tk.LEFT, background="#ffffe0", 
                         relief=tk.SOLID, borderwidth=1,
                         wraplength=300, padding=(5, 5))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class TickerSelector:
    def __init__(self, root):
        self.root = root
        self.setup_variables()
        self.setup_ui()
        self.refresh_databases()

    def setup_variables(self):
        """Initialize variables"""
        self.db_var = tk.StringVar()
        self.table_var = tk.StringVar()
        self.sector_var = tk.StringVar()
        self.selected_tickers = []
        self.selected_fields = []
        
        # Create instances of plotters
        self.plotter = None
        self.predictions_plotter = None

    def create_plots(self):
        """Create plots for selected data"""
        if not self.selected_tickers or not self.selected_fields:
            messagebox.showwarning("Selection Required", "Please select tickers and fields to plot")
            return
            
        if not hasattr(self, 'predictions_plotter') or self.predictions_plotter is None:
            self.predictions_plotter = PredictionsPlotter(
                self.plot_frame,
                self.table_var.get(),
                self.selected_tickers,
                self.selected_fields,
                self.conn
            )
        
        self.predictions_plotter.create_plots()

    def setup_ui(self):
        """Set up the user interface"""
        self.root.title("Data Selector")
        self.root.grid_columnconfigure(1, weight=1)  # Plot area gets the extra space
        self.root.grid_rowconfigure(0, weight=1)  # All rows expand equally
        
        # Create left panel for controls
        self.left_panel = ttk.Frame(self.root, padding="5")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Initialize variables
        self.duration_var = tk.StringVar(value="1y")  # Default duration
        self.interval_var = tk.StringVar(value="1d")  # Default interval
        self.available_tables = []
        
        # Create database selection frame
        self.db_frame = ttk.LabelFrame(self.left_panel, text="Database Selection", padding="5")
        self.db_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Database selection
        self.db_combo = ttk.Combobox(self.db_frame, textvariable=self.db_var)
        self.db_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Refresh button
        self.refresh_btn = ttk.Button(self.db_frame, text="🔄", width=3, command=self.refresh_databases)
        self.refresh_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Create table selection frame
        self.table_frame = ttk.LabelFrame(self.left_panel, text="Table Selection", padding="5")
        self.table_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Table selection
        self.table_combo = ttk.Combobox(self.table_frame, textvariable=self.table_var)
        self.table_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.table_combo.bind('<<ComboboxSelected>>', self.on_table_change)
        
        # Create control panel
        self.control_panel = ttk.LabelFrame(self.left_panel, text="Data Controls", padding="5")
        self.control_panel.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create sector selection
        sector_frame = ttk.LabelFrame(self.control_panel, text="Sector Selection", padding="5")
        sector_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.sector_combo = ttk.Combobox(sector_frame, textvariable=self.sector_var)
        self.sector_combo.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.sector_combo.bind('<<ComboboxSelected>>', self.on_sector_change)
        
        # Create ticker selection
        ticker_frame = ttk.LabelFrame(self.control_panel, text="Ticker Selection", padding="5")
        ticker_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.ticker_listbox = tk.Listbox(ticker_frame, height=10)
        self.ticker_listbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ticker_scrollbar = ttk.Scrollbar(ticker_frame, orient="vertical", command=self.ticker_listbox.yview)
        ticker_scrollbar.grid(row=0, column=1, sticky="ns")
        self.ticker_listbox.configure(yscrollcommand=ticker_scrollbar.set)
        
        # Create fields selection
        fields_frame = ttk.LabelFrame(self.control_panel, text="Fields Selection", padding="5")
        fields_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        self.fields_listbox = tk.Listbox(fields_frame, height=10, selectmode=tk.MULTIPLE)
        self.fields_listbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        fields_scrollbar = ttk.Scrollbar(fields_frame, orient="vertical", command=self.fields_listbox.yview)
        fields_scrollbar.grid(row=0, column=1, sticky="ns")
        self.fields_listbox.configure(yscrollcommand=fields_scrollbar.set)
        
        # Create time controls frame
        time_frame = ttk.LabelFrame(self.control_panel, text="Time Controls", padding="5")
        time_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Duration selection
        ttk.Label(time_frame, text="Duration:").grid(row=0, column=0, padx=5, pady=2)
        self.duration_combo = ttk.Combobox(time_frame, textvariable=self.duration_var, width=5)
        self.duration_combo['values'] = ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
        self.duration_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Interval selection
        ttk.Label(time_frame, text="Interval:").grid(row=1, column=0, padx=5, pady=2)
        self.interval_combo = ttk.Combobox(time_frame, textvariable=self.interval_var, width=5)
        self.interval_combo['values'] = ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        self.interval_combo.grid(row=1, column=1, padx=5, pady=2)
        
        # Create analysis controls frame
        analysis_frame = ttk.LabelFrame(self.control_panel, text="Analysis Controls", padding="5")
        analysis_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        # Technical indicators
        self.ma_var = tk.BooleanVar(value=False)
        self.rsi_var = tk.BooleanVar(value=False)
        self.macd_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(analysis_frame, text="Moving Average", variable=self.ma_var).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(analysis_frame, text="RSI", variable=self.rsi_var).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(analysis_frame, text="MACD", variable=self.macd_var).grid(row=2, column=0, sticky="w", padx=5, pady=2)
        
        # Add plot button
        self.plot_button = ttk.Button(self.control_panel, text="Plot Data", command=self.plot_data)
        self.plot_button.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        
        # Create plot frame on the right
        self.plot_frame = ttk.Frame(self.root, padding="5")
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights for plot frame
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        
        # Create the plotter
        self.plotter = TickerPlotter(self.plot_frame)
        
        # Load initial data
        self.refresh_databases()
        self.refresh_tables()

    def create_tables(self):
        """Create tables if they don't exist - now empty as we're not creating tables"""
        pass  # Remove table creation as we're using existing database schema

    def setup_widgets(self):
        """Create and setup all widgets"""
        # Add database combobox binding
        self.db_combo.bind('<<ComboboxSelected>>', self.on_database_change)
        
        # Create plot frame
        self.plot_frame = ttk.Frame(self.left_panel)
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        
        # Create the plotter
        self.plotter = TickerPlotter(self.plot_frame)
        
        # Create control panel
        self.control_panel = ttk.LabelFrame(self.left_panel, text="Controls", padding="5")
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create sector selection
        sector_frame = ttk.LabelFrame(self.control_panel, text="Sector Selection", padding="5")
        sector_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.sector_combo = ttk.Combobox(sector_frame, textvariable=self.sector_var)
        self.sector_combo.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.sector_combo.bind('<<ComboboxSelected>>', self.on_sector_change)
        
        # Create ticker selection
        ticker_frame = ttk.LabelFrame(self.control_panel, text="Ticker Selection", padding="5")
        ticker_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.ticker_listbox = tk.Listbox(ticker_frame, height=10)
        self.ticker_listbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create fields selection
        fields_frame = ttk.LabelFrame(self.control_panel, text="Fields Selection", padding="5")
        fields_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        self.fields_listbox = tk.Listbox(fields_frame, height=10, selectmode=tk.MULTIPLE)
        self.fields_listbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Add plot button
        self.plot_button = ttk.Button(self.control_panel, text="Plot Data", command=self.plot_data)
        self.plot_button.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

    def on_table_change(self, event):
        """Handle table selection change"""
        try:
            # Verify table exists
            current_table = self.table_var.get()
            if current_table not in self.available_tables:
                raise ValueError(f"Selected table '{current_table}' does not exist")
            
            # Reload fields for new table
            self.load_sectors()
            self.setup_fields()
        except Exception as e:
            print(f"Error changing table: {e}")
            messagebox.showerror("Error", f"Failed to change table: {e}")

    def load_sectors(self):
        """Load sectors for the current table"""
        try:
            current_table = self.table_var.get()
            if not current_table:
                return
            
            # Get column names
            columns = self.conn.execute(f"SELECT * FROM {current_table} LIMIT 1").df().columns
            
            # Check for sector or category column
            if 'sector' in columns:
                sector_col = 'sector'
            elif 'category' in columns:
                sector_col = 'category'
            else:
                print(f"No sector/category column found in {current_table}")
                return
            
            # Get unique sectors
            query = f"""
                SELECT DISTINCT {sector_col}
                FROM {current_table}
                WHERE {sector_col} IS NOT NULL
                ORDER BY {sector_col}
            """
            sectors = self.conn.execute(query).fetchall()
            
            # Update sector combobox
            if hasattr(self, 'sector_combo'):
                self.sector_combo['values'] = [sector[0] for sector in sectors]
                if sectors:
                    self.sector_combo.set(sectors[0][0])
                
            print(f"Loaded sectors for table {current_table}")
            
        except Exception as e:
            print(f"Error loading sectors: {e}")
            if hasattr(self, 'sector_combo'):
                self.sector_combo.set('')
                self.sector_combo['values'] = []

    def load_tickers(self, sector=None):
        """Load tickers based on sector and search term"""
        try:
            current_table = self.table_var.get()
            search_term = self.search_var.get().strip().upper()
            
            # Get all column names from the table
            columns = self.conn.execute(f"SELECT * FROM {current_table} LIMIT 0").df().columns
            
            # Use 'pair' if we're in the historical_forex table and it exists
            identifier_col = None
            if current_table == 'historical_forex' and 'pair' in columns:
                identifier_col = 'pair'
                print("Using 'pair' column for historical_forex table.")
            else:
                # Try to find appropriate identifier column for other tables
                for possible_col in ['symbol', 'ticker', 'stock_symbol']:
                    if possible_col in columns:
                        identifier_col = possible_col
                        break
            
            if not identifier_col:
                messagebox.showerror("Error",
                                     f"No ticker/symbol column found in table {current_table}")
                return
            
            # Build the query
            query = f"""
                SELECT DISTINCT {identifier_col}
                FROM {current_table}
                WHERE 1=1
            """
            params = []
            
            if sector and 'sector' in columns:
                query += " AND sector = ?"
                params.append(sector)
            
            if search_term:
                query += f" AND {identifier_col} LIKE ?"
                params.append(f"%{search_term}%")
            
            query += f" ORDER BY {identifier_col}"
            
            # Execute query and update listbox
            tickers = self.conn.execute(query, params).fetchall()
            
            self.ticker_listbox.delete(0, tk.END)
            for ticker in tickers:
                self.ticker_listbox.insert(tk.END, ticker[0])
            
            self.update_statistics()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load tickers: {e}")

    def update_listbox_label(self, identifier_col):
        """Update the listbox label based on the identifier column"""
        # Add this method to show what type of data is being displayed
        label_text = identifier_col.replace('_', ' ').title()
        if hasattr(self, 'listbox_label'):
            self.listbox_label.config(text=f"Available {label_text}s:")
        else:
            self.listbox_label = ttk.Label(self.left_panel, text=f"Available {label_text}s:")
            self.listbox_label.grid(row=3, column=0, sticky=tk.W, pady=(5,0))

    def on_sector_change(self, event):
        """Handle sector selection change"""
        try:
            # Update tickers based on new sector
            self.update_tickers()
        except Exception as e:
            print(f"Error handling sector change: {e}")

    def on_search_change(self, *args):
        """Handle search text change"""
        self.load_tickers(self.sector_var.get())

    def select_all(self):
        """Select all tickers in the listbox"""
        self.ticker_listbox.select_set(0, tk.END)
        self.update_statistics()

    def clear_selection(self):
        """Clear all selections"""
        self.ticker_listbox.selection_clear(0, tk.END)
        self.update_statistics()

    def get_selected(self):
        """Handle the selection of tickers and fields for plotting"""
        selected_indices = self.ticker_listbox.curselection()
        selected_tickers = [self.ticker_listbox.get(i) for i in selected_indices]
        
        if selected_tickers:
            # Get selected fields using actual database column names
            selected_fields = [field for field, var in self.field_vars.items() if var.get()]
            
            # Ensure 'date' is included in the selected fields
            if 'date' not in selected_fields:
                selected_fields.append('date')
            
            # Verify that selected fields exist in the table
            available_fields = self.get_available_fields(self.table_var.get())
            selected_fields = [field for field in selected_fields if field in available_fields]
            
            if selected_fields:
                try:
                    plotter = TickerPlotter(self.root, selected_tickers, self.table_var.get(), connection=self.conn, fields=selected_fields)
                    plotter.create_plot()  # Ensure this method is called to initialize plotting
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to plot data: {e}")
            else:
                messagebox.showerror("Error", "No valid fields selected for plotting")
        else:
            messagebox.showwarning("Warning", "Please select at least one ticker")

    def update_statistics(self):
        """Update statistics label"""
        total_items = self.ticker_listbox.size()
        selected_items = len(self.ticker_listbox.curselection())
        self.stats_label.config(
            text=f"Total Items: {total_items} | Selected: {selected_items}"
        )

    def show_predictions(self):
        """Show predictions for selected tickers"""
        selected_indices = self.ticker_listbox.curselection()
        selected_tickers = [self.ticker_listbox.get(i) for i in selected_indices]
        
        if selected_tickers:
            # Get selected fields using actual database column names
            selected_fields = [field for field, var in self.field_vars.items() if var.get()]
            
            # Ensure 'date' is included in the selected fields
            if 'date' not in selected_fields:
                selected_fields.append('date')
            
            if selected_fields:
                try:
                    # Create an instance of TickerAIAgent with the required arguments
                    parameters = {
                        'learning_rate': self.learning_rate_var.get(),
                        'epochs': self.epochs_var.get(),
                        'batch_size': self.batch_size_var.get()
                    }
                    model_type = self.model_type_var.get()
                    agent = TickerAIAgent(tickers=selected_tickers, fields=selected_fields, connection=self.conn, parameters=parameters, model_type=model_type)
                    
                    # Assuming you have a model and data prepared for predictions
                    model = agent.model
                    data = np.random.rand(100, 10)  # Example input data
                    predictions_plotter = PredictionsPlotter(model, data, prediction_days=self.prediction_days_var.get())
                    predictions_plotter.make_predictions()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to show predictions: {e}")
            else:
                messagebox.showerror("Error", "No fields selected for predictions")
        else:
            messagebox.showwarning("Warning", "Please select at least one ticker")

    def get_field_tooltip(self, field):
        """Get tooltip text for a given field"""
        tooltips = {
            'value': "Numerical value for the selected metric",
            'sector': "Business sector classification",
            'id': "Unique identifier for the record",
            # Add more tooltips as needed for your specific fields
        }
        return tooltips.get(field, f"Value of {field.replace('_', ' ').title()}")

    def start_predictions(self):
        """Start predictions with selected tickers and fields"""
        # Get selected tickers
        selected_tickers = [self.ticker_listbox.get(i) for i in self.ticker_listbox.curselection()]
        
        # Get selected fields using actual database column names
        selected_fields = [field for field, var in self.field_vars.items() 
                          if var.get()]
        
        if not selected_tickers:
            messagebox.showwarning("Warning", "Please select at least one ticker")
            return
        
        if not selected_fields:
            messagebox.showwarning("Warning", "Please select at least one field")
            return

        try:
            # Create predictions plotter with actual database column names
            predictions_plotter = PredictionsPlotter(
                self.root,
                selected_tickers,
                self.table_var.get(),
                selected_fields
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start predictions: {e}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            import threading
            # Only close the connection if we're in the main thread
            if threading.current_thread() is threading.main_thread():
                self.conn.close()
                print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {e}")

    def setup_fields(self):
        """Setup fields for the current table"""
        try:
            current_table = self.table_var.get()
            fields = self.conn.execute(f"SELECT * FROM {current_table} LIMIT 0").df().columns
            
            # Filter out non-numeric and special fields
            excluded_fields = ['id', 'symbol', 'industry', 'date', 'updated_at']
            available_fields = [field for field in fields if field not in excluded_fields]
            
            # Clear existing field frame if it exists
            if hasattr(self, 'field_frame'):
                self.field_frame.destroy()
            
            # Create new field frame
            self.field_frame = ttk.LabelFrame(self.left_panel, text="Select Fields", padding="5")
            self.field_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Create checkboxes for fields
            self.field_vars = {}
            for i, field in enumerate(available_fields):
                display_name = field.replace('_', ' ').title()
                var = tk.BooleanVar(value=field in ['value', 'sector'])
                self.field_vars[field] = var
                cb = ttk.Checkbutton(self.field_frame, text=display_name,
                                    variable=var)
                cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=5, pady=2)
                
                # Add tooltip for the field
                field_tooltip = self.get_field_tooltip(field)
                ToolTip(cb, field_tooltip)
                
        except Exception as e:
            print(f"Error setting up fields: {e}")
            messagebox.showerror("Error", f"Failed to setup fields: {e}")

    def create_metrics_selector(self, table_name):
        columns = get_table_columns(self.conn, table_name)
        self.metrics_var = tk.StringVar(value=columns)
        self.metrics_listbox = tk.Listbox(self.root, listvariable=self.metrics_var, selectmode='multiple')
        self.metrics_listbox.pack()

    def plot_selected_metrics(self):
        selected_indices = self.metrics_listbox.curselection()
        selected_metrics = [self.metrics_listbox.get(i) for i in selected_indices]
        # Use selected_metrics for plotting
        for metric in selected_metrics:
            print(f"Plotting data for {metric}")
            # Add your plotting logic here

    def get_available_fields(self, table_name):
        """Get available fields for the selected table"""
        query = f"PRAGMA table_info({table_name})"
        df = self.conn.execute(query).df()
        return df['name'].tolist()

    def detect_ticker_column(self):
        """Detect which column to use for ticker identification and rename if necessary"""
        try:
            current_table = self.table_var.get() if hasattr(self, 'table_var') else self.available_tables[0]
            columns = self.conn.execute(f"SELECT * FROM {current_table} LIMIT 0").df().columns
            print(f"Detecting ticker column for table {current_table}")
            print(f"Available columns: {columns.tolist()}")
            
            # Check for ticker identifier column (pair, symbol, or ticker)
            possible_ticker_columns = ['pair', 'symbol', 'ticker', 'currency_pair']
            
            for col in possible_ticker_columns:
                if col in columns:
                    if col == 'pair':
                        # Rename 'pair' to 'ticker'
                        print("Renaming 'pair' column to 'ticker'...")
                        self.conn.execute(f"""
                            ALTER TABLE {current_table}
                            RENAME COLUMN pair TO ticker;
                        """)
                        self.ticker_column = 'ticker'
                    elif col == 'symbol':
                        # Rename 'symbol' to 'ticker'
                        print("Renaming 'symbol' column to 'ticker'...")
                        self.conn.execute(f"""
                            ALTER TABLE {current_table}
                            RENAME COLUMN symbol TO ticker;
                        """)
                        self.ticker_column = 'ticker'
                    else:
                        self.ticker_column = col
                    print(f"Using '{self.ticker_column}' as ticker identifier")
                    return
            
            # If no standard column found, use the first string column that might be an identifier
            for col in columns:
                if col not in ['date', 'sector', 'industry', 'updated_at', 'value']:
                    self.ticker_column = col
                    print(f"Using '{self.ticker_column}' as ticker identifier")
                    return
            
            raise ValueError(f"No suitable ticker column found in table {current_table}")
            
        except Exception as e:
            print(f"Error detecting ticker column: {e}")
            messagebox.showerror(
                "Database Error",
                f"Failed to detect ticker column: {e}\n\nPlease ensure the table has a suitable identifier column."
            )
            raise

    def refresh_databases(self):
        """Refresh the list of available DuckDB databases"""
        try:
            # Store current selection
            current_db = self.db_var.get()
            
            # Search in current directory and subdirectories for .db files
            db_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.db'):
                        db_path = os.path.join(root, file)
                        # Convert to relative path
                        rel_path = os.path.relpath(db_path, '.')
                        db_files.append(rel_path)
            
            # Filter for DuckDB databases
            duckdb_files = []
            for db_file in db_files:
                try:
                    # Try to open each file with DuckDB
                    with duckdb.connect(db_file, read_only=True) as test_conn:
                        # Verify it's a DuckDB database by attempting a simple query
                        test_conn.execute('SELECT 1').fetchone()
                    duckdb_files.append(db_file)
                    print(f"Found valid DuckDB database: {db_file}")
                except Exception as e:
                    print(f"Skipping {db_file}: {str(e)}")
                    continue
            
            if not duckdb_files:
                print("No DuckDB databases found")
                messagebox.showwarning("Warning", "No DuckDB databases found in the current directory or subdirectories")
            
            # Update database combobox
            self.db_combo['values'] = duckdb_files
            
            # Restore previous selection if valid, otherwise select first database
            if current_db in duckdb_files:
                self.db_combo.set(current_db)
            elif duckdb_files:
                self.db_combo.set(duckdb_files[0])
                self.on_database_change()
            
            print(f"Available databases: {duckdb_files}")
            
        except Exception as e:
            print(f"Error refreshing databases: {e}")
            messagebox.showerror("Error", f"Failed to refresh databases: {e}")

    def switch_database(self, db_name):
        """Switch to a different database and update all dependent controls"""
        try:
            # Close existing connection if any
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
            
            # Open new connection
            self.conn = duckdb.connect(db_name)
            print(f"Connected to database: {db_name}")
            
            # Refresh tables
            self.refresh_tables()
            
            # Clear existing selections
            self.clear_selections()
            
        except Exception as e:
            print(f"Error switching database: {e}")
            messagebox.showerror("Error", f"Failed to switch to database {db_name}: {e}")

    def clear_selections(self):
        """Clear all selection controls"""
        self.sector_combo.set('')
        self.sector_combo['values'] = []
        self.ticker_listbox.delete(0, tk.END)
        self.fields_listbox.delete(0, tk.END)

    def refresh_tables(self):
        """Refresh the list of available tables and update dependent controls"""
        try:
            # Get list of tables from database schema
            tables = self.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """).fetchall()
            
            # Extract table names
            table_names = [table[0] for table in tables]
            print(f"Retrieved tables: {table_names}")
            
            # Update table combobox
            if hasattr(self, 'table_combo'):
                self.table_combo['values'] = table_names
                if table_names:
                    self.table_combo.set(table_names[0])
                    # Automatically load sectors for first table
                    self.load_table_data(table_names[0])
        
            print(f"Found tables: {table_names}")
            
        except Exception as e:
            print(f"Error refreshing tables: {e}")
            if hasattr(self, 'table_combo'):
                self.table_combo.set('')
                self.table_combo['values'] = []

    def load_table_data(self, table_name):
        """Load all data associated with a table"""
        try:
            print(f"\nLoading data for table: {table_name}")
            
            # Get table columns
            columns = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 1").df().columns
            print(f"Available columns: {columns}")
            
            # Update fields listbox
            self.fields_listbox.delete(0, tk.END)
            for col in columns:
                self.fields_listbox.insert(tk.END, col)
            
            # Check for sector/category column and load sectors
            if 'sector' in columns or 'category' in columns:
                sector_col = 'sector' if 'sector' in columns else 'category'
                sectors = self.conn.execute(f"""
                    SELECT DISTINCT {sector_col}
                    FROM {table_name}
                    WHERE {sector_col} IS NOT NULL
                    ORDER BY {sector_col}
                """).fetchall()
                
                # Update sector combobox
                sector_values = [sector[0] for sector in sectors]
                self.sector_combo['values'] = sector_values
                if sector_values:
                    self.sector_combo.set(sector_values[0])
                    # Load tickers for first sector
                    self.load_tickers(table_name, sector_values[0], sector_col)
            else:
                # If no sector column, just load all tickers
                self.load_tickers(table_name)
                
            print(f"Successfully loaded data for table {table_name}")
            
        except Exception as e:
            print(f"Error loading table data: {e}")
            self.clear_selections()

    def load_tickers(self, table_name, sector=None, sector_col=None):
        """Load tickers based on table and optional sector"""
        try:
            if sector and sector_col:
                query = f"""
                    SELECT DISTINCT ticker
                    FROM {table_name}
                    WHERE {sector_col} = ?
                    ORDER BY ticker
                """
                tickers = self.conn.execute(query, [sector]).fetchall()
            else:
                query = f"""
                    SELECT DISTINCT ticker
                    FROM {table_name}
                    ORDER BY ticker
                """
                tickers = self.conn.execute(query).fetchall()
            
            # Update ticker listbox
            self.ticker_listbox.delete(0, tk.END)
            for ticker in tickers:
                self.ticker_listbox.insert(tk.END, ticker[0])
            
            print(f"Loaded tickers for table {table_name}" + 
                  (f" and sector {sector}" if sector else ""))
            
        except Exception as e:
            print(f"Error loading tickers: {e}")
            self.ticker_listbox.delete(0, tk.END)

    def on_table_change(self, event=None):
        """Handle table selection change"""
        selected_table = self.table_var.get()
        if selected_table:
            self.load_table_data(selected_table)

    def on_sector_change(self, event=None):
        """Handle sector selection change"""
        try:
            selected_table = self.table_var.get()
            selected_sector = self.sector_var.get()
            
            if selected_table and selected_sector:
                # Determine sector column name
                columns = self.conn.execute(f"SELECT * FROM {selected_table} LIMIT 1").df().columns
                sector_col = 'sector' if 'sector' in columns else 'category'
                
                # Load tickers for selected sector
                self.load_tickers(selected_table, selected_sector, sector_col)
                
        except Exception as e:
            print(f"Error handling sector change: {e}")

    def on_database_change(self, event=None):
        """Handle database selection change"""
        selected_db = self.db_var.get()
        if selected_db:
            self.switch_database(selected_db)

    def update_fields(self):
        """Update available fields based on current table"""
        try:
            current_table = self.table_var.get()
            if not current_table:
                return
            
            # Verify table exists before querying
            table_exists_query = f"""
                SELECT count(*) 
                FROM information_schema.tables 
                WHERE table_name = '{current_table}'
            """
            if not self.conn.execute(table_exists_query).fetchone()[0]:
                print(f"Table {current_table} does not exist")
                return
            
            # Get column names from current table
            columns = self.conn.execute(f"SELECT * FROM {current_table} LIMIT 0").description
            column_names = [col[0] for col in columns]
            
            # Update fields listbox
            if hasattr(self, 'fields_listbox'):
                self.fields_listbox.delete(0, tk.END)
                for col in column_names:
                    self.fields_listbox.insert(tk.END, col)
            
            print(f"Updated fields for table {current_table}: {column_names}")
            
        except Exception as e:
            print(f"Error updating fields: {e}")
            # Clear the fields listbox on error
            if hasattr(self, 'fields_listbox'):
                self.fields_listbox.delete(0, tk.END)

    def update_tickers(self):
        """Update available tickers based on current table and sector"""
        try:
            current_table = self.table_var.get()
            current_sector = self.sector_var.get()
            
            if not current_table:
                return
            
            # Verify table exists before querying
            table_exists_query = f"""
                SELECT count(*) 
                FROM information_schema.tables 
                WHERE table_name = '{current_table}'
            """
            if not self.conn.execute(table_exists_query).fetchone()[0]:
                print(f"Table {current_table} does not exist")
                return
            
            # Get column names directly from a SELECT statement
            columns = self.conn.execute(f"SELECT * FROM {current_table} LIMIT 1").df().columns
            print(f"Available columns in {current_table}: {columns}")
            
            # Build query based on available columns
            if current_sector and ('sector' in columns or 'category' in columns):
                sector_col = 'sector' if 'sector' in columns else 'category'
                query = f"""
                    SELECT DISTINCT ticker 
                    FROM {current_table} 
                    WHERE {sector_col} = '{current_sector}'
                    ORDER BY ticker
                """
                tickers = self.conn.execute(query).fetchall()
            else:
                # If no sector column or no sector selected, get all tickers
                query = f"""
                    SELECT DISTINCT ticker 
                    FROM {current_table} 
                    ORDER BY ticker
                """
                tickers = self.conn.execute(query).fetchall()
            
            # Update ticker listbox
            if hasattr(self, 'ticker_listbox'):
                self.ticker_listbox.delete(0, tk.END)
                for ticker in tickers:
                    self.ticker_listbox.insert(tk.END, ticker[0])
            
            print(f"Updated tickers for table {current_table}, sector {current_sector}")
            print(f"Query executed: {query}")
            
        except Exception as e:
            print(f"Error updating tickers: {e}")
            # Clear the ticker listbox on error
            if hasattr(self, 'ticker_listbox'):
                self.ticker_listbox.delete(0, tk.END)

    def on_table_change(self, event=None):
        """Handle table selection change"""
        try:
            # Load sectors for new table
            self.load_sectors()
            
            # Update fields
            self.update_fields()
            
            # Update tickers
            self.update_tickers()
            
        except Exception as e:
            print(f"Error handling table change: {e}")
            self.clear_all_fields()

    def load_database(self, db_name):
        """Load database and update all associated UI elements"""
        try:
            print(f"\nLoading database: {db_name}")
            
            # Switch database connection
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
            self.conn = duckdb.connect(db_name)
            print(f"Connected to database: {db_name}")
            
            # Update tables
            self.refresh_tables()
            
            # Load sectors for the current table
            self.load_sectors()
            
            # Update fields
            self.update_fields()
            
            # Update tickers
            self.update_tickers()
            
            print(f"Database {db_name} loaded successfully")
            
        except Exception as e:
            print(f"Error loading database: {e}")
            self.clear_all_fields()

    def clear_all_fields(self):
        """Clear all UI elements"""
        if hasattr(self, 'fields_listbox'):
            self.fields_listbox.delete(0, tk.END)
        if hasattr(self, 'ticker_listbox'):
            self.ticker_listbox.delete(0, tk.END)
        if hasattr(self, 'sector_combo'):
            self.sector_combo.set('')
        if hasattr(self, 'table_combo'):
            self.table_combo.set('')

    def plot_data(self):
        """Plot the selected data"""
        try:
            # Get selected ticker and fields
            selected_ticker_idx = self.ticker_listbox.curselection()
            selected_field_indices = self.fields_listbox.curselection()
            
            if not selected_ticker_idx or not selected_field_indices:
                print("Please select a ticker and at least one field")
                return
            
            selected_ticker = self.ticker_listbox.get(selected_ticker_idx[0])
            selected_fields = [self.fields_listbox.get(idx) for idx in selected_field_indices]
            current_table = self.table_var.get()
            
            # Build and execute query
            fields_str = ", ".join(selected_fields)
            query = f"""
                SELECT date, {fields_str}
                FROM {current_table}
                WHERE ticker = '{selected_ticker}'
                ORDER BY date
            """
            
            # Execute query and get data
            data = self.conn.execute(query).df()
            
            # Plot the data
            self.plotter.plot_data(data, selected_ticker, selected_fields)
            print(f"Plotted data for {selected_ticker}: {selected_fields}")
            
        except Exception as e:
            print(f"Failed to plot data: {e}")
            messagebox.showerror("Error", f"Failed to plot data: {e}")

# Define the function outside of any loops
@tf.function
def train_step(model, inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = compute_loss(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def main():
    root = tk.Tk()
    
    # Open the database connection without read-only mode
    try:
        conn = duckdb.connect('historical_market_data.db')  # Remove read_only=True
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        messagebox.showerror(
            "Database Error",
            f"Failed to connect to database: {e}\n\nPlease ensure the database file exists and has proper permissions."
        )
        return
    
    app = TickerSelector(root)
    
    # Configure window with larger size
    root.geometry("800x1000")
    root.minsize(600, 800)
    
    # Setup cleanup on window close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    finally:
        # Ensure cleanup happens even if the mainloop is interrupted
        app.cleanup()

if __name__ == "__main__":
    main() 