import pandas as pd
import numpy as np
import logging
import pickle
import os
import json
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def load_data(file_path='data/processed/engineered_stock_data.parquet'):
    """Load processed data for model building with error handling for corrupt files."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data = pd.read_parquet(file_path)
    except Exception as e:
        logging.error(f"Error reading Parquet file: {e}")
        raise ValueError(f"Error reading Parquet file: {e}")
    
    return data

def objective(trial):
    """Objective function for hyperparameter tuning."""
    data = load_data()
    X = data.drop(columns=['Close'])
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_type = trial.suggest_categorical('model_type', ['random_forest', 'gradient_boosting'])
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 200),
            max_depth=trial.suggest_int('max_depth', 5, 20),
            random_state=42
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 200),
            learning_rate=trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            random_state=42
        )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

def build_models():
    """Build and train multiple models with hyperparameter tuning."""
    logging.info("Starting model training...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    logging.info(f"Best hyperparameters: {best_params}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logging.info("Model training complete with hyperparameter tuning.")
    return best_params

if __name__ == "__main__":
    build_models()
