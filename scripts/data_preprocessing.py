import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
    """Load raw stock market data from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        if 'Date' not in data.columns:
            logging.error("Missing 'Date' column in dataset.")
            raise KeyError("Missing 'Date' column in dataset.")
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def handle_missing_values(data, drop_threshold=0.3):
    """Fill or drop missing values in the dataset based on a threshold."""
    missing_ratio = data.isnull().sum() / len(data)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    
    if columns_to_drop:
        data.drop(columns=columns_to_drop, inplace=True)
        logging.info(f"Dropped columns with excessive missing values: {columns_to_drop}")
    
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    logging.info("Remaining missing values handled.")
    return data

def normalize_data(data, columns, method='standard'):
    """Normalize numerical columns using StandardScaler or MinMaxScaler."""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid normalization method. Choose 'standard' or 'minmax'.")
    
    for col in columns:
        if not np.issubdtype(data[col].dtype, np.number):
            logging.error(f"Non-numeric column detected: {col}")
            raise ValueError(f"Column {col} contains non-numeric values and cannot be scaled.")
    
    data[columns] = scaler.fit_transform(data[columns])
    logging.info(f"Data normalized using {method} scaling.")
    return data

def preprocess_data(raw_file_path='data/raw/historical_stock_data.csv', processed_file_path='data/processed/processed_stock_data.csv'):
    """Execute the full preprocessing pipeline with parameterized file paths."""
    logging.info("Starting data preprocessing...")
    data = load_data(raw_file_path)
    data = handle_missing_values(data)
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = normalize_data(data, numeric_columns)
    
    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
    data.to_csv(processed_file_path)
    logging.info("Data preprocessing complete and saved.")
    return data

if __name__ == "__main__":
    preprocess_data()
