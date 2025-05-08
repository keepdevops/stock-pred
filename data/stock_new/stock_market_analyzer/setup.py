from setuptools import setup, find_packages

setup(
    name="stock_market_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "PyQt5",
        "duckdb",
        "yfinance",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.9",
)
