#!/usr/bin/env python3
"""
Download sample historical data into the app database.
Run from stock_new:  python download_sample_data.py

Options:
  python download_sample_data.py                      # Yahoo (often rate limited)
  python download_sample_data.py --source stooq       # Stooq (usually no rate limit)
  python download_sample_data.py --from-csv FILE      # load from CSV (no API)
  python download_sample_data.py --from-csv DIR       # load all *.csv in directory

CSV format: Date, Open, High, Low, Close, Adj Close, Volume (Yahoo export).
Optional column: Symbol or Ticker. Or use --ticker AAPL for a single-ticker file.

When Yahoo rate limits you, use Stooq or CSV:
  --source stooq   uses stooq.com (US tickers get .US suffix, e.g. AAPL.US)
  Or download from https://finance.yahoo.com/quote/AAPL/history → Download
  Then:  python download_sample_data.py --from-csv ~/Downloads/AAPL.csv --ticker AAPL
"""
import sys
import argparse
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import time

# Default tickers
DEFAULT_TICKERS = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
PERIOD = "2y"
INTERVAL = "1d"
# Stooq: delay between requests (seconds) to be polite
STOOQ_PAUSE = 0.5


def get_db_path():
    """Read DB path from config or use default."""
    config_path = _root / "config" / "system_config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        path = cfg.get("db_path")
        if not path and isinstance(cfg.get("data_processing"), dict):
            path = cfg["data_processing"].get("database", {}).get("path")
        return path or "data/market_data.duckdb"
    return "data/market_data.duckdb"


def csv_to_standard(df, symbol=None):
    """Normalize CSV columns to: date, ticker, open, high, low, close, adj_close, volume."""
    df = df.copy()
    # Normalize to lowercase for matching (Stooq and others may vary)
    df.columns = df.columns.str.strip()
    renames = {
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
        "Adj Close": "adj_close", "Adj. Close": "adj_close",
        "Symbol": "ticker", "Ticker": "ticker",
    }
    # Also accept lowercase column names (e.g. from Stooq)
    for k, v in list(renames.items()):
        if k != v and k.lower() not in renames:
            renames[k.lower()] = v
    df = df.rename(columns={c: renames[c] for c in renames if c in df.columns})
    if "date" not in df.columns and "Date" in df.columns:
        df["date"] = df["Date"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"].astype(float)
    if symbol:
        df["ticker"] = symbol
    if "ticker" not in df.columns and symbol:
        df["ticker"] = symbol
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def load_from_csv(path: Path, ticker_override=None):
    """
    Load one or more CSVs. Returns list of (symbol, df).
    path: file or directory. If dir, glob *.csv. Ticker from filename (e.g. AAPL.csv) or column or --ticker.
    """
    path = Path(path).resolve()
    if not path.exists():
        return []

    files = [path] if path.is_file() else list(path.glob("*.csv"))
    if not files:
        print(f"No CSV files in {path}")
        return []

    results = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Skip {f.name}: {e}")
            continue
        if df.empty:
            continue
        # Ticker: override > column > filename stem
        sym = ticker_override
        if sym is None and "ticker" in df.columns:
            sym = df["ticker"].iloc[0] if len(df) else None
        if sym is None and "Symbol" in df.columns:
            sym = df["Symbol"].iloc[0] if len(df) else None
        if sym is None:
            sym = f.stem.upper()
        df = csv_to_standard(df, symbol=sym)
        if not df.empty:
            results.append((sym, df))
    return results


def fetch_stooq(tickers):
    """
    Fetch tickers from Stooq (direct CSV URLs). Usually not rate limited.
    US symbols get .US suffix (e.g. AAPL → aapl.us). Returns list of (symbol, df).
    """
    import urllib.request
    from datetime import datetime, timedelta
    if not tickers:
        return []
    end = datetime.now()
    start = end - timedelta(days=730)  # ~2 years
    d1 = start.strftime("%Y%m%d")
    d2 = end.strftime("%Y%m%d")
    results = []
    for i, sym in enumerate(tickers):
        if i > 0:
            time.sleep(STOOQ_PAUSE)
        # Stooq US symbols: aapl.us, msft.us, etc.
        symbol_stooq = sym.strip().upper()
        if "." not in symbol_stooq:
            symbol_stooq = f"{symbol_stooq}.US"
        symbol_stooq = symbol_stooq.lower()
        url = f"https://stooq.com/q/d/l/?s={symbol_stooq}&d1={d1}&d2={d2}&i=d"
        print(f"Fetching {sym} from Stooq...", end=" ")
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                df = pd.read_csv(r)
        except Exception as e:
            print(f"  {e}")
            continue
        if df is None or df.empty or len(df.columns) < 5:
            print("  no data")
            continue
        # Stooq CSV: Date, Open, High, Low, Close, Volume (no Adj Close)
        df = csv_to_standard(df, symbol=sym.strip().upper())
        if df.empty:
            print("  no rows")
            continue
        results.append((sym.strip().upper(), df))
        print(f"  {len(df)} rows")
    return results


def fetch_batch(tickers):
    """Fetch tickers via yfinance (one batch). Returns list of (symbol, df)."""
    import yfinance as yf
    if not tickers:
        return []
    print("Downloading via Yahoo (may be rate limited)...")
    try:
        data = yf.download(
            tickers, period=PERIOD, interval=INTERVAL,
            group_by="ticker", auto_adjust=False, threads=False, progress=False,
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return []
    if data is None or data.empty:
        return []

    results = []
    if len(tickers) == 1:
        df = _to_standard_format(data, tickers[0])
        if df is not None and not df.empty:
            results.append((tickers[0], df))
        return results
    for sym in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if sym not in data.columns.get_level_values(0):
                    continue
                sub = data[sym].copy()
            else:
                sub = data.copy()
            df = _to_standard_format(sub, sym)
            if df is not None and not df.empty:
                results.append((sym, df))
        except Exception as e:
            print(f"  {sym}: {e}")
    return results


def _to_standard_format(df, symbol):
    df = df.copy()
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume", "Adj Close": "adj_close",
    })
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df["ticker"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    return df[[c for c in cols if c in df.columns]]


def save_to_db(conn, results):
    import duckdb
    for symbol, df in results:
        print(f"Saving {symbol}... {len(df)} rows")
        conn.register("_df", df)
        conn.execute("""
            INSERT INTO stock_data SELECT * FROM _df
            ON CONFLICT (date, ticker) DO NOTHING
        """)
        conn.unregister("_df")


def main():
    ap = argparse.ArgumentParser(description="Load sample stock data into app DB")
    ap.add_argument("--from-csv", metavar="PATH", help="Load from CSV file or directory (no API)")
    ap.add_argument("--ticker", help="Ticker symbol for single-ticker CSV (e.g. AAPL)")
    ap.add_argument("--source", choices=("yahoo", "stooq"), default="yahoo",
                    help="Download source when not using --from-csv (default: yahoo)")
    ap.add_argument("tickers", nargs="*", default=None,
                    help="Ticker symbols (default: AAPL, GOOG, MSFT, AMZN, META)")
    args = ap.parse_args()

    db_path = get_db_path()
    db_file = _root / db_path if not Path(db_path).is_absolute() else Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Database: {db_file}")

    if args.from_csv:
        path = Path(args.from_csv)
        if not path.is_absolute():
            path = (_root / path).resolve()
        print(f"Loading from CSV: {path}")
        results = load_from_csv(path, ticker_override=args.ticker)
    else:
        tickers = args.tickers or DEFAULT_TICKERS
        print(f"Tickers: {tickers}  (source: {args.source})")
        if args.source == "stooq":
            results = fetch_stooq(tickers)
        else:
            results = fetch_batch(tickers)

    if not results:
        if args.from_csv:
            print("No data loaded. Check CSV has columns: Date, Open, High, Low, Close, Volume (and optionally Adj Close, Symbol).")
        else:
            print("No data fetched.")
            if args.source == "yahoo":
                print("Yahoo is likely rate limiting. Try:  python download_sample_data.py --source stooq")
            print("Or use CSV:  python download_sample_data.py --from-csv ~/Downloads/AAPL.csv --ticker AAPL")
        return

    print()
    import duckdb
    conn = duckdb.connect(str(db_file))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            date DATE NOT NULL, ticker VARCHAR NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, adj_close DOUBLE, volume BIGINT,
            PRIMARY KEY (date, ticker)
        )
    """)
    save_to_db(conn, results)
    conn.close()
    print("Done. Start the app with:  python main.py")


if __name__ == "__main__":
    main()
