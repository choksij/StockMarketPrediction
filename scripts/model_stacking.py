import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path='data/processed/engineered_stock_data.parquet'):
    """Load processed data for model stacking with error handling."""
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        data = pd.read_parquet(file_path)
        if data.empty:
            logging.warning("Loaded Parquet file is empty. Attempting to fill missing values.")
            data.fillna(method='ffill', inplace=True)  # Forward fill missing values
            if data.empty:
                logging.error("Dataset remains empty after handling missing values.")
                raise ValueError("Processed dataset is empty.")
    except Exception as e:
        logging.error(f"Error reading Parquet file: {e}")
        raise ValueError(f"Error reading Parquet file: {e}")
    
    return data

def ensure_model_directory():
    """Ensure that the models directory exists."""
    if not os.path.exists('models'):
        os.makedirs('models')
        logging.info("Created 'models' directory.")

def train_stacked_model(data):
    """Train a stacked model using multiple regressors with hyperparameter tuning."""
    ensure_model_directory()
    feature_columns = [col for col in data.columns if col != 'Close']
    X = data[feature_columns]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'rf__n_estimators': [50, 100, 150],
        'gb__n_estimators': [50, 100, 150]
    }
    
    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42))
    ]
    
    meta_model = Ridge()
    stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    
    grid_search = GridSearchCV(stacked_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    model_metadata = {
        'model_version': '1.0',
        'base_models': ['RandomForestRegressor', 'GradientBoostingRegressor'],
        'meta_model': 'Ridge',
        'parameters': grid_search.best_params_
    }
    
    with open('models/stacked_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'metadata': model_metadata}, f)
    logging.info("Optimized stacked model with metadata saved.")
    
    y_pred = best_model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2_Score': r2_score(y_test, y_pred)
    }
    logging.info(f"Optimized Stacked Model Evaluation: {metrics}")
    return best_model

def run_model_stacking():
    """Run the model stacking process."""
    logging.info("Starting model stacking...")
    data = load_data()
    stacked_model = train_stacked_model(data)
    logging.info("Model stacking completed.")

if __name__ == "__main__":
    run_model_stacking()
