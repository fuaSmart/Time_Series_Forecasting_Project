# src/modeling.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
# Ensure the reports directory exists
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)


def build_and_train_lstm(X_train, y_train, epochs=50, batch_size=32):
    """
    Builds, compiles, and trains the LSTM model.
    """
    print("\nBuilding and training LSTM model...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    print("LSTM model training complete.")
    return model

# ... (the plot_lstm_forecast function) ...

def create_lstm_dataset(dataset, look_back=60):
    """
    Creates sequences for LSTM model training.
    """
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def split_data(series, train_ratio=0.8):
    """
    Splits time series data into training and testing sets chronologically.

    Args:
        series (pd.Series): The time series data to split.
        train_ratio (float): The proportion of data to use for training.

    Returns:
        tuple: A tuple containing train_data and test_data.
    """
    print(f"Splitting data into {train_ratio*100}% training and {(1-train_ratio)*100}% testing...")
    split_index = int(len(series) * train_ratio)
    train_data = series[:split_index]
    test_data = series[split_index:]
    
    print(f"Training data points: {len(train_data)}")
    print(f"Testing data points: {len(test_data)}")
    
    return train_data, test_data

def train_auto_arima(train_data):
    """
    Trains an ARIMA model using auto_arima to find the best parameters.

    Args:
        train_data (pd.Series): The training time series data.

    Returns:
        pmdarima.ARIMA: The fitted ARIMA model.
    """
    print("\nTraining Auto-ARIMA model...")

    model = auto_arima(train_data,
                       start_p=1, start_q=1,
                       test='adf',       
                       max_p=5, max_q=5, 
                       m=1,              
                       d=None,           
                       seasonal=False,   
                       start_P=0,
                       D=0,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    print("\nARIMA model summary:")
    print(model.summary())
    return model

def evaluate_forecast(test_data, forecast, model_name="ARIMA"):
    """
    Calculates and prints evaluation metrics for the forecast.

    Args:
        test_data (pd.Series): The true values.
        forecast (np.array): The predicted values.
    """
    print(f"\n--- {model_name} Model Evaluation ---")
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


def plot_lstm_forecast(train_data, test_data, forecast, model_name="LSTM"):
    """
    Visualizes the LSTM forecast against the historical data.
    """
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(14, 7))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Actual Prices (Test Set)', color='orange')
    plt.plot(test_data.index, forecast, label='Forecasted Prices', color='green')
    plt.title(f'{model_name} Forecast vs Actuals for TSLA')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price (USD)')
    plt.legend()
    plt.grid(True)
    
    REPORTS_DIR = 'reports'
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, f'{model_name}_forecast.png'))
    plt.show()
    
    
    plt.title(f'{model_name} Forecast vs Actuals for TSLA')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_DIR, f'{model_name}_forecast.png'))
    plt.show()