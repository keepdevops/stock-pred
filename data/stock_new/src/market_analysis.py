def analyze_market_metrics(ticker: str, start_date: str, end_date: str) -> Dict:
    """
    Comprehensive market analysis with normalized metrics
    """
    try:
        # Initialize components
        mixer = TickerMixer()
        normalizer = MarketNormalizer()
        
        # Get data
        data = mixer.execute_combination({
            "name": "single_ticker",
            "tickers": [ticker],
            "fields": ["open", "high", "low", "close", "volume"],
            "date_range": {"start": start_date, "end": end_date}
        })
        
        # Apply normalizations
        results = {}
        
        # Returns analysis
        returns_data = normalizer.normalize_returns(
            df=data,
            methods=["simple", "log", "excess"]
        )
        
        # Volume analysis
        volume_data = normalizer.normalize_volume(
            df=returns_data,
            methods=["vwap", "relative", "dollar"]
        )
        
        # Volatility analysis
        volatility_data = normalizer.normalize_volatility(
            df=volume_data,
            methods=["garch", "parkinson", "realized"]
        )
        
        # Price level analysis
        price_data = normalizer.normalize_price_levels(
            df=volatility_data,
            methods=["ma", "bb", "atr"]
        )
        
        # Momentum analysis
        momentum_data = normalizer.normalize_momentum(
            df=price_data,
            methods=["rsi", "macd", "roc"]
        )
        
        # Market impact analysis
        final_data = normalizer.normalize_market_impact(
            df=momentum_data,
            methods=["amihud", "kyle_lambda"]
        )
        
        # Calculate correlations between normalized metrics
        numeric_cols = [col for col in final_data.columns 
                       if col.startswith(("normalized_", "price_", "relative_"))]
        corr_matrix = final_data.select(numeric_cols).corr()
        
        # Generate summary statistics
        summary_stats = {
            "returns": {
                "mean": float(final_data["close_simple_return"].mean()),
                "volatility": float(final_data["realized_volatility"].mean()),
                "sharpe": float(final_data["close_excess_return"].mean() / 
                              final_data["realized_volatility"].std())
            },
            "volume": {
                "avg_relative_volume": float(final_data["relative_volume"].mean()),
                "dollar_volume_trend": float(np.polyfit(
                    range(len(final_data)), 
                    final_data["dollar_volume_normalized"].to_numpy(), 
                    1
                )[0])
            },
            "momentum": {
                "avg_rsi": float(final_data["normalized_rsi"].mean()),
                "macd_trend": float(final_data["normalized_macd_hist"].mean())
            },
            "liquidity": {
                "avg_illiquidity": float(final_data["normalized_illiquidity"].mean()),
                "price_impact": float(final_data["price_impact"].mean())
            }
        }
        
        return {
            "normalized_data": final_data,
            "correlation_matrix": corr_matrix,
            "summary_stats": summary_stats
        }
        
    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        raise 