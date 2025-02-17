import pandas as pd
import numpy as np
import logging
import os
import pickle
import concurrent.futures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def load_model(model_path):
    """Load a trained model from the given path."""
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data(file_path='data/processed/engineered_stock_data.parquet'):
    """Load processed data for evaluation with error handling."""
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        data = pd.read_parquet(file_path)
    except Exception as e:
        logging.error(f"Error reading Parquet file: {e}")
        raise ValueError(f"Error reading Parquet file: {e}")
    
    if data.isnull().sum().sum() > 0:
        logging.warning("Missing values detected in dataset. Consider handling missing data before evaluation.")
    
    return data

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using various metrics with error handling."""
    try:
        predictions = model.predict(X_test)
    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        raise RuntimeError(f"Error during model prediction: {e}")
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    logging.info(f"Model Evaluation Metrics: MAE={mae}, MSE={mse}, RMSE={rmse}, R^2={r2}, MAPE={mape}")
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2_Score': r2,
        'MAPE': mape
    }

def run_evaluation(model_path):
    """Run the evaluation process."""
    logging.info(f"Evaluating model: {model_path}")
    model = load_model(model_path)
    data = load_data()
    
    feature_columns = [col for col in data.columns if col != 'Close']
    X = data[feature_columns]
    y = data['Close']
    
    metrics = evaluate_model(model, X, y)
    logging.info(f"Evaluation completed for model: {model_path}")
    
    return {model_path: metrics}

if __name__ == "__main__":
    model_paths = ['models/xgboost_model.pkl', 'models/stacked_model.pkl', 'models/gradient_boosting_model.pkl']
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_evaluation, [path for path in model_paths if os.path.exists(path)]))
    
    for result in results:
        logging.info(f"Parallel evaluation results: {result}")
