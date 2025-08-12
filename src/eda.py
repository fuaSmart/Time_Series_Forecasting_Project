# src/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os

# Ensure the reports directory exists
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

def plot_closing_prices(data, title='Adjusted Closing Prices', filename='closing_prices.png'):
    """Visualizes and saves the closing prices."""
    plt.figure(figsize=(14, 7))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_DIR, filename))
    plt.show()

def calculate_and_plot_daily_returns(data, title='Daily Percentage Change', filename='daily_returns.png'):
    """Calculates, visualizes, and saves daily returns."""
    daily_returns = data.pct_change().dropna()
    plt.figure(figsize=(14, 7))
    for column in daily_returns.columns:
        plt.plot(daily_returns.index, daily_returns[column], label=column, alpha=0.8)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_DIR, filename))
    plt.show()
    return daily_returns

def calculate_and_plot_rolling_volatility(daily_returns, window=30, title='30-Day Rolling Volatility', filename='rolling_volatility.png'):
    """Calculates, visualizes, and saves rolling volatility."""
    rolling_volatility = daily_returns.rolling(window=window).std()
    plt.figure(figsize=(14, 7))
    for column in rolling_volatility.columns:
        plt.plot(rolling_volatility.index, rolling_volatility[column], label=f'{column} {window}-Day Rolling Volatility')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatility (Standard Deviation)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_DIR, filename))
    plt.show()

def perform_adf_test(series, series_name):
    """Performs and prints the results of the Augmented Dickey-Fuller test."""
    print(f'\n--- Augmented Dickey-Fuller Test for {series_name} ---')
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    if result[1] <= 0.05:
        print("Conclusion: Reject the null hypothesis. The data is likely stationary.")
    else:
        print("Conclusion: Fail to reject the null hypothesis. The data is likely non-stationary.")

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    """
    Calculates the annualized Sharpe Ratio for each asset.

    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        risk_free_rate (float): The annual risk-free rate.

    Returns:
        pd.Series: A Series containing the Sharpe Ratio for each asset.
    """
    print("\n--- Calculating Annualized Sharpe Ratio ---")
    # The number of trading days in a year is typically 252.
    trading_days = 252
    daily_risk_free_rate = (1 + risk_free_rate)**(1/trading_days) - 1
    
    excess_returns = daily_returns - daily_risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()
    
    sharpe_ratio = (mean_excess_return / std_dev_excess_return) * np.sqrt(trading_days)
    
    print("Annualized Sharpe Ratios:")
    print(sharpe_ratio)
    return sharpe_ratio

def calculate_var(daily_returns, confidence_level=0.95):
    """
    Calculates the historical Value at Risk (VaR).

    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        confidence_level (float): The confidence level for VaR (e.g., 0.95 for 95%).
    """
    print(f"\n--- Calculating Value at Risk (VaR) at {confidence_level:.0%} confidence level ---")
    # VaR is the quantile of the returns distribution.
    # For a 95% confidence level, we look at the 5th percentile.
    var = daily_returns.quantile(1 - confidence_level)
    print("Historical VaR:")
    for ticker, value in var.items():
        print(f"{ticker}: There is a {(1-confidence_level):.0%} chance that the daily loss will be {abs(value):.2%} or more.")
    return var