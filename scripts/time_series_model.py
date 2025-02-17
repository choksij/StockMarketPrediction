import pandas as pd
import numpy as np
import logging
import os
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import matplotlib.pyplot as plt
import pmdarima as pm

def load_data(file_path='data/processed/engineered_stock_data.parquet'):
    """Load processed data for time series forecasting with error handling."""
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        data = pd.read_parquet(file_path)
        if data.empty:
            logging.error("Loaded Parquet file is empty.")
            raise ValueError("Loaded Parquet file is empty.")
    except Exception as e:
        logging.error(f"Error reading Parquet file: {e}")
        raise ValueError(f"Error reading Parquet file: {e}")
    
    return data[['Date', 'Close']]

def ensure_model_directory():
    """Ensure that the models directory exists."""
    if not os.path.exists('models'):
        os.makedirs('models')
        logging.info("Created 'models' directory.")

def train_arima_model(data):
    """Train an ARIMA model with automatic order selection."""
    ensure_model_directory()
    try:
        auto_arima_model = pm.auto_arima(data['Close'], seasonal=False, stepwise=True, trace=True)
        order = auto_arima_model.order
        logging.info(f"Selected ARIMA order: {order}")
        
        model = ARIMA(data['Close'], order=order)
        model_fit = model.fit()
        model_fit.save('models/arima_model.pkl')
        logging.info("ARIMA model saved.")
        return model_fit
    except Exception as e:
        logging.error(f"Error training ARIMA model: {e}")
        raise RuntimeError(f"ARIMA training failed: {e}")

def train_sarima_model(data):
    """Train a SARIMA model with dynamic hyperparameter selection."""
    ensure_model_directory()
    try:
        auto_sarima_model = pm.auto_arima(data['Close'], seasonal=True, m=12, stepwise=True, trace=True)
        order = auto_sarima_model.order
        seasonal_order = auto_sarima_model.seasonal_order
        logging.info(f"Selected SARIMA order: {order}, seasonal order: {seasonal_order}")
        
        model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        model_fit.save('models/sarima_model.pkl')
        logging.info("SARIMA model saved.")
        return model_fit
    except Exception as e:
        logging.error(f"Error training SARIMA model: {e}")
        raise RuntimeError(f"SARIMA training failed: {e}")

def train_prophet_model(data):
    """Train a Prophet model with additional regressors and hyperparameter tuning."""
    ensure_model_directory()
    try:
        df = data.rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet()
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='weekly', period=7, fourier_order=3)
        model.fit(df)
        with open('models/prophet_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        logging.info("Prophet model saved with additional seasonalities.")
        return model
    except Exception as e:
        logging.error(f"Error training Prophet model: {e}")
        raise RuntimeError(f"Prophet training failed: {e}")

def run_time_series_models():
    """Run all time series forecasting models."""
    logging.info("Starting time series modeling...")
    data = load_data()
    
    arima_model = train_arima_model(data)
    sarima_model = train_sarima_model(data)
    prophet_model = train_prophet_model(data)
    
    logging.info("Time series modeling completed.")

if __name__ == "__main__":
    run_time_series_models()
