import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import logging
import os
import pickle
import matplotlib.pyplot as plt
import concurrent.futures

def load_model(model_path):
    """Load a trained model from the given path."""
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data(file_path='data/processed/engineered_stock_data.parquet'):
    """Load processed data for explainability analysis with error handling."""
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        data = pd.read_parquet(file_path)
    except Exception as e:
        logging.error(f"Error reading Parquet file: {e}")
        raise ValueError(f"Error reading Parquet file: {e}")
    
    if data.isnull().sum().sum() > 0:
        logging.warning("Missing or corrupted data detected. Consider handling missing values before analysis.")
    
    return data

def shap_analysis(model, X):
    """Perform SHAP analysis for model interpretability with error handling."""
    try:
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X)
        plt.savefig('reports/shap_summary_plot.png')
        logging.info("SHAP summary plot saved.")
    except Exception as e:
        logging.error(f"Error during SHAP analysis: {e}")
        raise RuntimeError(f"Error during SHAP analysis: {e}")

def lime_analysis(model, X):
    """Perform LIME analysis for feature importance explanation with single instance prediction handling."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns,
        mode='regression'
    )
    
    instance = X.iloc[0].values.reshape(1, -1)  # Ensure the input shape matches model's predict method
    explanation = explainer.explain_instance(instance[0], lambda x: model.predict(x.reshape(1, -1)))
    explanation.save_to_file('reports/lime_explanation.html')
    logging.info("LIME explanation saved.")

def explain_model(model_path):
    """Run explainability analysis using SHAP and LIME."""
    logging.info(f"Explaining model: {model_path}")
    model = load_model(model_path)
    data = load_data()
    
    feature_columns = [col for col in data.columns if col != 'Close']
    X = data[feature_columns]
    
    shap_analysis(model, X)
    lime_analysis(model, X)
    
    logging.info(f"Explanation completed for model: {model_path}")

if __name__ == "__main__":
    model_paths = ['models/xgboost_model.pkl', 'models/stacked_model.pkl', 'models/gradient_boosting_model.pkl']
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(explain_model, [path for path in model_paths if os.path.exists(path)])
