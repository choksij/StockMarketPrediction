import pandas as pd
import numpy as np
import logging
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error

def load_data(file_path='data/processed/engineered_stock_data.parquet'):
    """Load processed data for XGBoost model training."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_parquet(file_path)
    
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        logging.warning("Missing values detected in dataset. Consider handling missing data before training.")
        data = data.dropna()
    
    return data

def train_xgboost_model():
    """Train the XGBoost model with optimized parameters."""
    logging.info("Starting XGBoost model training...")
    data = load_data()
    
    X = data.drop(columns=['Close'])
    y = data['Close']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=True)
    
    for i, eval_result in enumerate(model.evals_result()['validation_0']['rmse']):
        logging.info(f"Epoch {i+1}: RMSE = {eval_result}")
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    logging.info(f"XGBoost Model - Test MSE: {mse}")
    
    model_path = 'models/xgboost_model.pkl'
    feature_importance_path = 'models/xgboost_feature_importance.npy'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    np.save(feature_importance_path, model.feature_importances_)
    logging.info("XGBoost model training complete and saved with feature importance.")
    return model

if __name__ == "__main__":
    train_xgboost_model()
