import pandas as pd
import numpy as np
import logging
import os

def calculate_moving_averages(data, windows=[5, 10, 20]):
    """Add moving average features for different window sizes."""
    for window in windows:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    logging.info("Moving average features added.")
    return data

def calculate_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)  # Prevent division by zero
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].clip(0, 100)  # Clip RSI values between 0 and 100
    logging.info("RSI feature added.")
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculate the Moving Average Convergence Divergence (MACD) indicator."""
    if len(data) < long_window:
        logging.warning("Not enough data points to compute MACD.")
        return data
    
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    logging.info("MACD features added.")
    return data

def generate_features(input_file='data/processed/processed_stock_data.csv', output_file='data/processed/engineered_stock_data.parquet'):
    """Execute the full feature engineering pipeline."""
    if not os.path.exists(input_file):
        logging.error(f"File not found: {input_file}")
        raise FileNotFoundError(f"File not found: {input_file}")
    
    logging.info("Starting feature engineering...")
    data = pd.read_csv(input_file)
    
    if 'Date' not in data.columns:
        logging.error("Missing 'Date' column in dataset.")
        raise KeyError("Missing 'Date' column in dataset.")
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    data = calculate_moving_averages(data)
    data = calculate_rsi(data)
    data = calculate_macd(data)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data.to_parquet(output_file, compression='gzip')
    logging.info("Feature engineering complete and saved.")
    return data

if __name__ == "__main__":
    generate_features()
