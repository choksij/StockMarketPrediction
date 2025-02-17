import pandas as pd
import numpy as np
import logging
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path='data/processed/engineered_stock_data.parquet'):
    """Load and preprocess data for LSTM model training."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_parquet(file_path)
    return data

def preprocess_data(data, sequence_length=60):
    """Prepare data for LSTM by normalizing and structuring sequences."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.drop(columns=['Date']))  # Scale all features except date
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predicting the first column (assumed 'Close' is first)
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def build_lstm_model(input_shape):
    """Construct an LSTM model for stock price prediction."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model():
    """Train the LSTM model with early stopping and learning rate scheduling."""
    logging.info("Starting LSTM model training...")
    data = load_data()
    X, y, scaler = preprocess_data(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping, lr_scheduler])
    
    model_path = 'models/lstm_model.h5'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    
    architecture_path = 'models/lstm_model_architecture.json'
    optimizer_path = 'models/lstm_model_optimizer.h5'
    with open(architecture_path, 'w') as f:
        f.write(model.to_json())
    model.save_weights(optimizer_path)
    
    logging.info(f"LSTM model training complete. Best epoch stopped at: {len(history.history['loss']) - early_stopping.patience}")
    
    return model, scaler

if __name__ == "__main__":
    train_lstm_model()
