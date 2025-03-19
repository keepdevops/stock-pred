MARKET_DEFINITIONS = {
    "returns": {
        "simple_return": {
            "title": "Simple Returns",
            "formula": "(Pt - Pt-1) / Pt-1",
            "description": "Percentage change in price between two periods.",
            "interpretation": "Direct measure of price change. A value of 0.05 means 5% increase."
        },
        "log_return": {
            "title": "Logarithmic Returns",
            "formula": "ln(Pt/Pt-1)",
            "description": "Natural logarithm of the price ratio between two periods.",
            "interpretation": "Better for statistical analysis and combining returns over time."
        },
        "excess_return": {
            "title": "Excess Returns",
            "formula": "Return - Risk_Free_Rate",
            "description": "Return above the risk-free rate (usually Treasury bill rate).",
            "interpretation": "Measures additional return earned over risk-free investment."
        }
    },
    "volume": {
        "vwap": {
            "title": "Volume Weighted Average Price (VWAP)",
            "formula": "Σ(Price × Volume) / Σ(Volume)",
            "description": "Average price weighted by trading volume.",
            "interpretation": "Benchmark for trading efficiency and price impact."
        },
        "relative_volume": {
            "title": "Relative Volume",
            "formula": "Current_Volume / Average_Volume(20 days)",
            "description": "Current volume compared to recent average.",
            "interpretation": "Values > 1 indicate higher than average volume."
        },
        "dollar_volume": {
            "title": "Dollar Volume",
            "formula": "Price × Volume",
            "description": "Total value of shares traded.",
            "interpretation": "Measures trading activity in monetary terms."
        }
    },
    "volatility": {
        "garch": {
            "title": "GARCH Volatility",
            "formula": "Complex statistical model",
            "description": "Generalized AutoRegressive Conditional Heteroskedasticity model.",
            "interpretation": "Captures volatility clustering and time-varying risk."
        },
        "parkinson": {
            "title": "Parkinson Volatility",
            "formula": "√(1/4ln(2)) × ln(High/Low)²",
            "description": "Volatility estimate using high-low range.",
            "interpretation": "More efficient than close-to-close volatility."
        },
        "realized": {
            "title": "Realized Volatility",
            "formula": "√(Σ(returns²))",
            "description": "Historical volatility based on actual returns.",
            "interpretation": "Measures observed price variation."
        }
    },
    "technical": {
        "rsi": {
            "title": "Relative Strength Index (RSI)",
            "formula": "100 - (100 / (1 + RS))",
            "description": "Momentum oscillator measuring speed of price changes.",
            "interpretation": "Values > 70 suggest overbought, < 30 oversold."
        },
        "macd": {
            "title": "MACD (Moving Average Convergence Divergence)",
            "formula": "12-day EMA - 26-day EMA",
            "description": "Trend-following momentum indicator.",
            "interpretation": "Signal line crossovers indicate trend changes."
        },
        "bollinger": {
            "title": "Bollinger Bands",
            "formula": "20-day MA ± (2 × 20-day std)",
            "description": "Price channels based on moving average and volatility.",
            "interpretation": "Price touching bands indicates potential reversal."
        }
    },
    "market_impact": {
        "amihud": {
            "title": "Amihud Illiquidity Ratio",
            "formula": "|Return| / Dollar_Volume",
            "description": "Price impact per dollar of trading volume.",
            "interpretation": "Higher values indicate lower liquidity."
        },
        "kyle_lambda": {
            "title": "Kyle's Lambda",
            "formula": "ΔPrice / ΔVolume",
            "description": "Price sensitivity to order flow.",
            "interpretation": "Measures market depth and resilience."
        }
    }
} 