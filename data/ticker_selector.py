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
    def __init__(self, root, connection):
        self.root = root
        self.root.title("Data Selector")
        self.conn = connection
        
        # Initialize variables
        self.table_var = tk.StringVar()
        self.sector_var = tk.StringVar()
        self.available_tables = []
        
        # Create database selection frame
        self.db_frame = ttk.LabelFrame(self.root, text="Database Selection", padding="5")
        self.db_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Database selection
        self.db_var = tk.StringVar()
        self.db_combo = ttk.Combobox(self.db_frame, textvariable=self.db_var)
        self.db_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Refresh button
        self.refresh_btn = ttk.Button(self.db_frame, text="🔄", width=3, command=self.refresh_databases)
        self.refresh_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Create table selection frame
        self.table_frame = ttk.LabelFrame(self.root, text="Table Selection", padding="5")
        self.table_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Table selection
        self.table_combo = ttk.Combobox(self.table_frame, textvariable=self.table_var)
        self.table_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.table_combo.bind('<<ComboboxSelected>>', self.on_table_change)
        
        # Load initial databases and tables
        self.refresh_databases()
        self.refresh_tables()  # This will populate available_tables
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create and pack widgets
        self.setup_widgets()
        
        # Load initial data
        self.load_sectors()

    def create_tables(self):
        """Create all required tables if they don't exist"""
        tables = [
            'balance_sheets',
            'financial_ratios',
            'historical_commodities',
            'historical_exchanges',
            'historical_forex',
            'historical_indices',
            'historical_prices',
            'income_statements',
            'industry_metrics',
            'market_sentiment',
            'sector_financials',
            'sector_sentiment',
            'stock_metrics'
        ]
        
        for table in tables:
            try:
                # First drop the table if it exists
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
                
                # Create table with a simplified structure
                self.conn.execute(f"""
                    CREATE TABLE {table} (
                        ticker VARCHAR,
                        date TIMESTAMP,
                        sector VARCHAR,
                        industry VARCHAR,
                        value DOUBLE
                    )
                """)
                print(f"Created table: {table}")
                
                # Now insert sample data
                if table == 'historical_prices':
                    self.conn.execute(f"""
                        INSERT INTO {table} (ticker, date, sector, industry, value)
                        VALUES 
                        ('AAPL', '2024-02-21', 'Technology', 'Consumer Electronics', 180.5),
                        ('MSFT', '2024-02-21', 'Technology', 'Software', 410.2),
                        ('GOOGL', '2024-02-21', 'Technology', 'Internet Services', 138.4),
                        ('AMZN', '2024-02-21', 'Consumer Cyclical', 'Internet Retail', 175.6),
                        ('TSLA', '2024-02-21', 'Consumer Cyclical', 'Auto Manufacturers', 202.8)
                    """)
                else:
                    self.conn.execute(f"""
                        INSERT INTO {table} (ticker, date, sector, industry, value)
                        VALUES 
                        ('SAMPLE1', '2024-02-21', 'Technology', 'Software', 100.0),
                        ('SAMPLE2', '2024-02-21', 'Healthcare', 'Biotechnology', 50.0),
                        ('SAMPLE3', '2024-02-21', 'Finance', 'Banking', 75.0)
                    """)
                print(f"Added sample data to {table}")
                
            except Exception as e:
                print(f"Error with table {table}: {e}")
                messagebox.showerror(
                    "Database Error",
                    f"Failed to setup table {table}: {e}"
                )

    def setup_widgets(self):
        # Add database combobox binding
        self.db_combo.bind('<<ComboboxSelected>>', self.on_database_change)
        
        # Table selection
        ttk.Label(self.main_frame, text="Table:").grid(row=0, column=0, sticky=tk.W)
        self.table_combo.bind('<<ComboboxSelected>>', self.on_table_change)
        
        # Add sector selection
        ttk.Label(self.main_frame, text="Sector:").grid(row=1, column=0, sticky=tk.W)
        self.sector_combo = ttk.Combobox(self.main_frame,
                                        textvariable=self.sector_var,
                                        state="readonly")
        self.sector_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.sector_combo.bind('<<ComboboxSelected>>', self.on_sector_change)
        
        # Set default table to first available table
        if self.available_tables:
            self.table_combo.set(self.available_tables[0])
        
        # Create field selection variables and checkboxes
        self.field_vars = {}
        
        try:
            # Get columns for the selected table
            current_table = self.table_var.get()
            fields = self.conn.execute(f"SELECT * FROM {current_table} LIMIT 0").df().columns
            
            # Filter out non-numeric and special fields
            excluded_fields = ['id', 'symbol', 'industry', 'date', 'updated_at']
            available_fields = [field for field in fields if field not in excluded_fields]
            
            # Create field frame
            field_frame = ttk.LabelFrame(self.main_frame, text="Select Fields", padding="5")
            field_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Create checkboxes for fields
            for i, field in enumerate(available_fields):
                display_name = field.replace('_', ' ').title()
                var = tk.BooleanVar(value=field in ['value', 'sector'])  # Default selected fields
                self.field_vars[field] = var
                cb = ttk.Checkbutton(field_frame, text=display_name,
                                    variable=var)
                cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=5, pady=2)
                
                # Add tooltip for the field
                field_tooltip = self.get_field_tooltip(field)
                ToolTip(cb, field_tooltip)
                
        except Exception as e:
            print(f"Error setting up fields: {e}")
            messagebox.showerror("Error", f"Failed to setup fields: {e}")
            raise

        # Add a label above the listbox to show what type of data is being displayed
        self.listbox_label = ttk.Label(self.main_frame, text="Available Items:")
        self.listbox_label.grid(row=3, column=0, sticky=tk.W, pady=(5,0))

        # Search frame (move after listbox label)
        search_frame = ttk.LabelFrame(self.main_frame, text="Search", padding="5")
        search_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search_change)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(fill=tk.X, padx=5)

        # Ticker listbox with scrollbar
        listbox_frame = ttk.Frame(self.main_frame)
        listbox_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.ticker_listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, height=15)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.ticker_listbox.yview)
        self.ticker_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.ticker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Statistics frame
        stats_frame = ttk.LabelFrame(self.main_frame, text="Selection Statistics", padding="5")
        stats_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_label = ttk.Label(stats_frame, text="")
        self.stats_label.pack(fill=tk.X, padx=5)

        # Buttons frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Plot Data", command=self.get_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show Predictions", command=self.show_predictions).pack(side=tk.LEFT, padx=5)

        # AI Tuning Parameters Frame
        tuning_frame = ttk.LabelFrame(self.main_frame, text="AI Tuning Parameters", padding="5")
        tuning_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Learning Rate
        ttk.Label(tuning_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Entry(tuning_frame, textvariable=self.learning_rate_var).grid(row=0, column=1, sticky=(tk.W, tk.E))

        # Epochs
        ttk.Label(tuning_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W)
        self.epochs_var = tk.IntVar(value=10)
        ttk.Entry(tuning_frame, textvariable=self.epochs_var).grid(row=1, column=1, sticky=(tk.W, tk.E))

        # Batch Size
        ttk.Label(tuning_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(tuning_frame, textvariable=self.batch_size_var).grid(row=2, column=1, sticky=(tk.W, tk.E))

        # Model Type
        ttk.Label(tuning_frame, text="Model Type:").grid(row=3, column=0, sticky=tk.W)
        self.model_type_var = tk.StringVar(value='simple')
        model_type_options = ['simple', 'lstm', 'gru', 'deep', 'bidirectional', 'transformer', 'cnn_lstm', 'attention']
        model_type_menu = ttk.OptionMenu(tuning_frame, self.model_type_var, *model_type_options)
        model_type_menu.grid(row=3, column=1, sticky=(tk.W, tk.E))

        # Prediction Days
        ttk.Label(tuning_frame, text="Prediction Days:").grid(row=4, column=0, sticky=tk.W)
        self.prediction_days_var = tk.IntVar(value=30)
        prediction_days_options = [30, 60, 90, 120]  # Example range of days
        prediction_days_menu = ttk.OptionMenu(tuning_frame, self.prediction_days_var, *prediction_days_options)
        prediction_days_menu.grid(row=4, column=1, sticky=(tk.W, tk.E))

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
            self.listbox_label = ttk.Label(self.main_frame, text=f"Available {label_text}s:")
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
            self.field_frame = ttk.LabelFrame(self.main_frame, text="Select Fields", padding="5")
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
            
            # Find all .db files in current directory
            db_files = glob.glob('*.db')
            
            # Update combobox values
            self.db_combo['values'] = db_files
            
            # Restore previous selection or select first available
            if current_db in db_files:
                self.db_combo.set(current_db)
            elif db_files:
                self.db_combo.set(db_files[0])
                self.switch_database(db_files[0])
            
            print(f"Found databases: {db_files}")
            
        except Exception as e:
            print(f"Error refreshing databases: {e}")
            messagebox.showerror("Error", f"Failed to refresh databases: {e}")

    def switch_database(self, db_name):
        """Switch to a different database"""
        try:
            # Close existing connection if any
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
            
            # Open new connection
            self.conn = duckdb.connect(db_name)
            print(f"Connected to database: {db_name}")
            
            # Refresh tables and related data
            self.refresh_tables()
            self.update_fields()
            self.update_tickers()
            
        except Exception as e:
            print(f"Error switching database: {e}")
            messagebox.showerror("Error", f"Failed to switch to database {db_name}: {e}")

    def refresh_tables(self):
        """Refresh the list of available tables"""
        try:
            # Get list of tables
            tables = self.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name != 'sqlite_sequence'
            """).fetchall()
            
            # Extract table names and update available_tables
            self.available_tables = [table[0] for table in tables]
            print(f"Retrieved tables: {self.available_tables}")
            
            # Update table combobox
            if hasattr(self, 'table_combo'):
                self.table_combo['values'] = self.available_tables
                if self.available_tables:
                    self.table_combo.set(self.available_tables[0])
                    self.on_table_change()
            
            print(f"Found tables: {self.available_tables}")
            
        except Exception as e:
            print(f"Error refreshing tables: {e}")
            messagebox.showerror("Error", f"Failed to refresh tables: {e}")

    def on_database_change(self, event=None):
        """Handle database selection change"""
        selected_db = self.db_var.get()
        if selected_db:
            self.load_database(selected_db)

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
    
    app = TickerSelector(root, conn)
    
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