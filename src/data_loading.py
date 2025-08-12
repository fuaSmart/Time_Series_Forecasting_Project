# src/data_loading.py

import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date, file_path):
    """
    Fetches historical financial data from Yahoo Finance and saves it to a CSV file.

    Args:
        tickers (list): List of stock tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        file_path (str): Path to save the raw data CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the raw financial data.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path)
    print(f"Raw data saved to {file_path}")
    return data

def preprocess_data(df):
    """
    Cleans and preprocesses the financial data.

    Args:
        df (pd.DataFrame): The raw financial data.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned adjusted close prices.
    """
    print("Preprocessing data...")
    # Select the 'Close' prices, which are now auto-adjusted by yfinance.
    adj_close = df['Close'].copy()

    # Check for missing values
    print("Missing values before cleaning:")
    print(adj_close.isnull().sum())

    # Handle missing values using forward-fill. This propagates the last valid observation forward.
    adj_close.fillna(method='ffill', inplace=True)

    print("\nMissing values after cleaning:")
    print(adj_close.isnull().sum())

    # Ensure all columns are numeric
    for col in adj_close.columns:
        adj_close[col] = pd.to_numeric(adj_close[col], errors='coerce')
    
    # Drop any rows that might still have NaNs after coercion
    adj_close.dropna(inplace=True)

    print("Data preprocessing complete.")
    return adj_close