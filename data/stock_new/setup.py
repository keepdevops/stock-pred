from setuptools import setup, find_packages

setup(
    name="stock_market_analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'duckdb',
        # Add other dependencies
    ],
) 