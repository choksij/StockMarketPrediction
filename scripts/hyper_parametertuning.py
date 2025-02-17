import optuna
import xgboost as xgb
import logging
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(file_path='data/processed/engineered_stock_data.parquet'):
    """Load processed stock data for hyperparameter tuning."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data = pd.read_parquet(file_path)
    except Exception as e:
        logging.error(f"Error reading Parquet file: {e}")
        raise
    
    return data

def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    data = load_data()
    if 'Close' not in data.columns:
        logging.error("Missing 'Close' column in dataset.")
        raise KeyError("Missing 'Close' column in dataset.")
    
    feature_columns = [col for col in data.columns if col != 'Close']
    if not feature_columns:
        logging.error("No features left after dropping 'Close'. Ensure necessary predictors are retained.")
        raise ValueError("No features available for training after dropping 'Close'.")
    
    X = data[feature_columns]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 0.001, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    }
    
    model = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    return mse

def tune_hyperparameters(n_trials=50, output_file='models/xgboost_best_params.yaml'):
    """Run hyperparameter tuning with Optuna and save the best parameters."""
    logging.info("Starting hyperparameter tuning...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    logging.info(f"Best hyperparameters found: {best_params}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    logging.info("Hyperparameter tuning complete and parameters saved.")
    
    return best_params

if __name__ == "__main__":
    tune_hyperparameters()
