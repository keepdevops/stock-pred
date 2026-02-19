# Stock Market Analyzer (most complete app)

Tkinter desktop app that ties together **config**, **database**, **data loading**, **AI agent**, and **trading agent** in one GUI. Use it to load historical data, run analysis, train models, and manage tickers.

**Entry point:** `main.py`

---

## Quick start

```bash
cd stock_new
pip install pandas numpy yfinance duckdb psutil
python3 download_sample_data.py --source stooq   # optional: preload data (or use --from-csv)
python3 main.py
```

Then in the GUI: select a ticker → **Load Historical** (if needed) → **Analyze** or **Train Model**.

---

## What it does

| Piece | Role |
|-------|------|
| **Config** | `config/system_config.json` — DB path, data period/interval, GUI size/theme. Created with defaults if missing. |
| **Database** | DuckDB at path from config (e.g. `data/market_data.duckdb`). Table: `stock_data` (date, ticker, OHLCV, adj_close). |
| **Data loader** | Fetches history via yfinance. **Load Historical** in the GUI downloads and **saves** into the DB. |
| **AI agent** | **Analyze** runs predictions for the selected ticker (needs data; load historical first if you see “No price data”). **Train Model** / **Make Prediction** use the same agent. |
| **Trading agent** | Backing for trading-related features in the GUI. |
| **GUI** | Database list, ticker list (NASDAQ screener or defaults), Load Historical, Refresh, Train Model, Make Prediction, Analyze, trading toggle. |

---

## Requirements

- **Python** 3.10+
- **tkinter** (usually bundled with Python)
- **pandas**, **numpy**, **yfinance**, **duckdb**, **psutil**

```bash
pip install pandas numpy yfinance duckdb psutil
```

Optional: use `environment.yml` in this folder for a Conda env (includes PyQt5 for `app.py`).

---

## Getting data

The app needs price data in the DB before **Analyze** or **Train Model** work.

### Option 1: Script (recommended)

From `stock_new`:

```bash
# Stooq (usually not rate limited)
python3 download_sample_data.py --source stooq

# Or Yahoo (often rate limited)
python3 download_sample_data.py

# Or from your own CSV(s)
python3 download_sample_data.py --from-csv data/seed
python3 download_sample_data.py --from-csv ~/Downloads/AAPL.csv --ticker AAPL
```

Uses the same DB as `main.py` (see `config/system_config.json` or default `data/market_data.duckdb`).

### Option 2: Inside the app

1. Run `python3 main.py`.
2. Select a ticker (e.g. AAPL).
3. Click **Load Historical** to fetch and save data for that ticker.

### Optional: full ticker list

For a full NASDAQ-style list instead of a few defaults:

```bash
python3 src/data/download_nasdaq.py
```

Then restart the app; it will pick up `nasdaq_screener_*.csv` from the current directory or `data/`.

---

## Config

- **File:** `config/system_config.json` (created with defaults if missing).
- **Minimal:**

```json
{
  "db_path": "data/market_data.duckdb",
  "data_period": "2y",
  "data_interval": "1d",
  "log_level": "INFO"
}
```

The app also accepts a full structure with `data_processing.database.path`, `data_collection`, and `gui_settings` (see `create_default_config` in `main.py`).

---

## How to run

From **`stock_new`** (recommended):

```bash
cd stock_new
python3 main.py
```

From repo root:

```bash
python3 data/stock_new/main.py
```

Config, DB, and logs are relative to the current working directory; run from `stock_new` so paths match.

---

## Typical workflow

1. **Data:** `python3 download_sample_data.py --source stooq` (or Load Historical in the app).
2. **Start app:** `python3 main.py`.
3. **Select ticker** in the list.
4. **Analyze** (or **Train Model** / **Make Prediction** as needed).

---

## Directory layout

```
stock_new/
├── README.md              # This file
├── main.py                # Main entry (Stock Market Analyzer)
├── download_sample_data.py # Load data: Yahoo / Stooq / CSV
├── config/
│   └── system_config.json
├── data/
│   ├── market_data.duckdb   # Default DB (from config)
│   └── seed/                # Optional: AAPL.csv etc. for --from-csv
├── logs/
│   └── app.log
├── modules/                  # GUI, database, data_loader, stock_ai_agent, trading
└── config/                   # config_manager (ConfigurationManager)
```

---

## Troubleshooting

- **“No price data for …”**  
  Load data first: run `download_sample_data.py` or use **Load Historical** for that ticker.

- **Yahoo “rate limited”**  
  Use Stooq: `python3 download_sample_data.py --source stooq`, or CSV: `--from-csv path/to/file.csv --ticker AAPL`.

- **NASDAQ screener “not found”**  
  App still runs with default tickers (e.g. AAPL, GOOG, MSFT, AMZN, META). Optional: run `python3 src/data/download_nasdaq.py` for a full list.

---

## Other apps in this folder

| Entry point | Description |
|-------------|-------------|
| **`app.py`** | PyQt5 stock app (different GUI, DB at `data/db/stock_market.db`). |
| **`run.py`** | Data Collector GUI (Tkinter). |
| **`run_analyzer.py`** | Launches `stock_market_analyzer` package. |

For all entry points in the repo, see the project **ENTRY_POINTS.md**.
