from setuptools import setup, find_packages

# Package setup configuration
setup(
    name="stock_market_analyzer",
    version="1.0.0",
    packages=find_packages(include=['stock_market_analyzer', 'stock_market_analyzer.*']),
    package_dir={'': '.'},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'duckdb',
        'polars',
        'yfinance',
        'lightgbm',
        # Add other dependencies
    ],
    python_requires='>=3.9',
) 