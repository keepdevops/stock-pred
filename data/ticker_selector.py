import tkinter as tk
from tkinter import ttk
import duckdb
from tkinter import messagebox
from ticker_plotter import TickerPlotter
from ai_agent import TickerAIAgent as AdvancedAIAgent
from ticker_ai_agent import TickerAIAgent as SimpleAIAgent

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
        self.conn = connection  # Use the passed connection
        
        # Define available tables
        self.available_tables = [
            'balance_sheets', 'financial_ratios', 'historical_commodities',
            'historical_exchanges', 'historical_forex', 'historical_indices', 
            'historical_prices', 'income_statements', 'industry_metrics', 
            'market_sentiment', 'sector_financials', 'stock_metrics'
        ]
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
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
                # Create table with a basic structure if it doesn't exist
                self.conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id INTEGER PRIMARY KEY,
                        symbol VARCHAR,
                        date TIMESTAMP,
                        sector VARCHAR,
                        industry VARCHAR,
                        value DOUBLE,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                print(f"Created or verified table: {table}")
                
                # Add sample data if table is empty
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count == 0:
                    # Insert sample data
                    if table == 'historical_prices':
                        self.conn.execute(f"""
                            INSERT INTO {table} (symbol, date, sector, industry, value)
                            VALUES 
                            ('AAPL', '2024-02-21', 'Technology', 'Consumer Electronics', 180.5),
                            ('MSFT', '2024-02-21', 'Technology', 'Software', 410.2),
                            ('GOOGL', '2024-02-21', 'Technology', 'Internet Services', 138.4),
                            ('AMZN', '2024-02-21', 'Consumer Cyclical', 'Internet Retail', 175.6),
                            ('TSLA', '2024-02-21', 'Consumer Cyclical', 'Auto Manufacturers', 202.8)
                        """)
                    elif table == 'sector_sentiment':
                        self.conn.execute(f"""
                            INSERT INTO {table} (symbol, date, sector, industry, value)
                            VALUES 
                            ('Technology', '2024-02-21', 'Technology', NULL, 0.75),
                            ('Healthcare', '2024-02-21', 'Healthcare', NULL, 0.62),
                            ('Finance', '2024-02-21', 'Finance', NULL, 0.58),
                            ('Energy', '2024-02-21', 'Energy', NULL, 0.45),
                            ('Materials', '2024-02-21', 'Materials', NULL, 0.52)
                        """)
                    else:
                        self.conn.execute(f"""
                            INSERT INTO {table} (symbol, date, sector, industry, value)
                            VALUES 
                            ('SAMPLE1', '2024-02-21', 'Technology', 'Software', 100.0),
                            ('SAMPLE2', '2024-02-21', 'Healthcare', 'Biotechnology', 50.0),
                            ('SAMPLE3', '2024-02-21', 'Finance', 'Banking', 75.0)
                        """)
                    print(f"Added sample data to {table}")
                
            except Exception as e:
                print(f"Error creating table {table}: {e}")
                messagebox.showerror(
                    "Database Error",
                    f"Failed to create table {table}: {e}"
                )

    def setup_widgets(self):
        # Table selection
        ttk.Label(self.main_frame, text="Table:").grid(row=0, column=0, sticky=tk.W)
        self.table_var = tk.StringVar()
        self.table_combo = ttk.Combobox(self.main_frame, 
                                       textvariable=self.table_var,
                                       values=self.available_tables)
        self.table_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Add sector selection
        ttk.Label(self.main_frame, text="Sector:").grid(row=1, column=0, sticky=tk.W)
        self.sector_var = tk.StringVar()
        self.sector_combo = ttk.Combobox(self.main_frame,
                                        textvariable=self.sector_var,
                                        state="readonly")
        self.sector_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.sector_combo.bind('<<ComboboxSelected>>', self.on_sector_change)
        
        # Set default table to first available table
        if self.available_tables:
            self.table_combo.set(self.available_tables[0])
        
        self.table_combo.bind('<<ComboboxSelected>>', self.on_table_change)
        
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

        # Add model selection control
        self.model_type_var = tk.StringVar(value='simple')
        ttk.Label(self.main_frame, text="Model Type:").grid(row=8, column=0, sticky=tk.W)
        model_type_frame = ttk.Frame(self.main_frame)
        model_type_frame.grid(row=8, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Radiobutton(model_type_frame, text="Simple", variable=self.model_type_var, value='simple').pack(side=tk.LEFT)
        ttk.Radiobutton(model_type_frame, text="Advanced", variable=self.model_type_var, value='advanced').pack(side=tk.LEFT)

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
        """Load sectors from database"""
        try:
            current_table = self.table_var.get()
            
            # Check if the table has a sector column
            columns = self.conn.execute(f"SELECT * FROM {current_table} LIMIT 0").df().columns
            if 'sector' in columns:
                sectors = self.conn.execute(f"""
                    SELECT DISTINCT sector 
                    FROM {current_table}
                    WHERE sector IS NOT NULL 
                    ORDER BY sector
                """).fetchall()
                
                sector_list = [row[0] for row in sectors]
                self.sector_combo['values'] = sector_list
                
                if sector_list:
                    self.sector_combo.set(sector_list[0])
                    self.load_tickers(sector_list[0])
                else:
                    self.sector_combo.set('')
                    self.load_tickers(None)
            else:
                self.sector_combo.set('')
                self.sector_combo['values'] = []
                self.load_tickers(None)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sectors: {e}")

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
        selected_sector = self.sector_var.get()
        self.load_tickers(selected_sector)

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
        """Get selected tickers and create plot"""
        selected_indices = self.ticker_listbox.curselection()
        selected_tickers = [self.ticker_listbox.get(i) for i in selected_indices]
        
        if selected_tickers:
            print("\nSelected Tickers:")
            for ticker in selected_tickers:
                print(ticker)
            
            # Determine which AI agent to use
            if self.model_type_var.get() == 'simple':
                agent_class = SimpleAIAgent
            else:
                agent_class = AdvancedAIAgent
            
            # Create plot window with a valid connection
            TickerPlotter(self.root, selected_tickers, self.table_var.get(), connection=self.conn, agent_class=agent_class)
        else:
            print("No tickers selected")

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
            
            if selected_fields:
                # Determine which AI agent to use
                if self.model_type_var.get() == 'simple':
                    agent_class = SimpleAIAgent
                else:
                    agent_class = AdvancedAIAgent
                
                from predictions_plotter import PredictionsPlotter
                PredictionsPlotter(self.root, selected_tickers, self.table_var.get(), selected_fields, agent_class=agent_class)
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

def main():
    root = tk.Tk()
    
    # Open the database connection once
    try:
        conn = duckdb.connect('historical_market_data.db', read_only=True)
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