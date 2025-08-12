# GMF Time Series Forecasting and Portfolio Optimization

This project is a solution to the "Guide Me in Finance (GMF) Investments" challenge. It involves fetching financial data, performing exploratory data analysis, building time series forecasting models, and optimizing a portfolio based on the results.

## Business Objective

The goal is to leverage time series forecasting on historical financial data to enhance portfolio management strategies for GMF Investments. This involves analyzing data for TSLA, BND, and SPY, building predictive models, and recommending portfolio adjustments to optimize returns while managing risk.

## Project Structure

The project is organized into a modular structure to ensure clarity and maintainability.

## Setup and Installation

To set up and run this project on your local machine, follow these steps:

1.  **Clone the repository (once it's on GitHub):**

    ```bash
    git clone <your-repo-url>
    cd GMF_TimeSeries_Project
    ```

2.  **Create and activate a Python virtual environment:**

    ```bash
    python3 -m venv gmf_env
    source gmf_env/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Project

The analysis is performed sequentially through the Jupyter notebooks.

1.  **Task 1: Data Exploration & Analysis**

    - Open and run the cells in `notebooks/Task1_Analysis.ipynb`.
    - This notebook will fetch the data, clean it, and perform the initial exploratory data analysis and risk calculations.

2.  **Task 2: ARIMA Forecasting**
    - Open and run the cells in `notebooks/Task2_ARIMA_Modeling.ipynb`.
    - This notebook builds and evaluates an ARIMA model for forecasting TSLA stock prices.

## Tasks Completed

- [x] **Task 1:** Data Preprocessing, EDA, and Risk Metrics (VaR, Sharpe Ratio)
- [x] **Task 2:** Time Series Forecasting (ARIMA model implemented)
- [ ] **Task 2:** Time Series Forecasting (LSTM model)
- [ ] **Task 3:** Forecast Future Market Trends
- [ ] **Task 4:** Portfolio Optimization
- [ ] **Task 5:** Strategy Backtesting
