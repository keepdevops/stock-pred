# Stock Market Analyzer (`main.py`)

Tkinter desktop app that wires together database, data loading, AI agent, and trading agent with a single GUI. Use it to browse tickers, load historical data, run analysis, and train models.

## What it does

- **Config** – Loads `config/system_config.json` (or creates a default if missing). Uses `data_processing.database.path`, `data_collection` (period/interval), and optional `gui_settings` (window size, theme).
- **Database** – Connects to DuckDB at the path from config (e.g. `data/market_data.duckdb`), creates `stock_data` if needed.
- **Data loader** – Fetches historical data (yfinance) using config period/interval. “Load Historical” in the GUI downloads and **saves** data for the selected ticker into the DB.
- **AI agent** – “Analyze” runs predictions for the selected ticker (requires data for that ticker; load historical first if you see “No price data”).
- **Trading agent** – Backing for trading-related features in the GUI.
- **GUI** – Database list, ticker list (from NASDAQ screener CSV or default tickers), Load Historical, Refresh, Train Model, Make Prediction, Analyze, and trading toggle.

## Requirements

- **Python** 3.10+
- **tkinter** (usually with Python)
- **pandas**, **numpy**, **yfinance**, **duckdb**, **psutil**

From `stock_new`:

```bash
pip install pandas numpy yfinance duckdb psutil
```

Or use the same Conda env as `app.py` (see main [README.md](README.md)).

## How to run

Run from the **`stock_new`** directory:

```bash
cd stock_new
python3 main.py
```

From the repo root:

```bash
python3 data/stock_new/main.py
```

Config, DB, and logs are resolved relative to the current working directory (intended: `stock_new`).

## Config file

- **Path** – `config/system_config.json` (created with defaults if missing).
- **Minimal example** – `{"db_path":"data/market_data.duckdb","data_period":"2y","data_interval":"1d","log_level":"INFO"}`.  
  The app also supports a full structure with `data_processing.database.path`, `data_collection`, and `gui_settings` (see `create_default_config` in `main.py`).

## Typical workflow

1. Start the app: `python3 main.py` from `stock_new`.
2. Select a ticker (e.g. AAPL) in the list. If the list is short or “default tickers”, run `python3 src/data/download_nasdaq.py` once to get a NASDAQ screener CSV (optional).
3. Click **Load Historical** to download and save price data for that ticker.
4. Click **Analyze** to run the AI prediction for that ticker.
5. Use **Train Model** / **Make Prediction** as needed.

## Directory layout (used by main.py)

```
stock_new/
├── main.py                 # This entry point
├── config/
│   └── system_config.json  # Config (path, period, interval, gui, etc.)
├── data/
│   ├── market_data.duckdb  # DuckDB DB (from config path)
│   └── nasdaq_screener_*.csv  # Optional; for full ticker list
├── logs/
│   └── app.log
├── modules/                # GUI, database, data_loader, stock_ai_agent, trading
└── config/                # config_manager (ConfigurationManager)
```

## Other entry points in this folder

- **`app.py`** – PyQt5 stock market app (different GUI, same general idea). See [README.md](README.md).
- **`run.py`** – Data Collector GUI (Tkinter).
- **`run_analyzer.py`** – Runs `stock_market_analyzer.main.main()` (separate package under `stock_market_analyzer/`).

For a full list of entry points, see the project **ENTRY_POINTS.md**.
