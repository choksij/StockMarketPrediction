import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor

def build_ensemble_model(filepath):
    data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    X = data[['Open', 'High', 'Low', 'Volume', '20_SMA', '50_SMA', 'Volatility']]
    y = data['Close']
    
    # Load individual models
    reg_model = joblib.load("models/regression_model.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")
    
    ensemble_model = VotingRegressor(estimators=[
        ('regression', reg_model),
        ('xgboost', xgb_model)
    ])
    ensemble_model.fit(X, y)
    joblib.dump(ensemble_model, "models/ensemble_model.pkl")
    
    predictions = ensemble_model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"Ensemble Model MSE: {mse}")

if __name__ == "__main__":
    filepath = "C:/Users/choks/stock_market_prediction/data/processed/engineered_stock_data.csv"
    build_ensemble_model(filepath)
