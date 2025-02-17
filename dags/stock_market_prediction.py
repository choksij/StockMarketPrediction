from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
import logging
import os
from scripts.data_collection import fetch_stock_data
from scripts.data_preprocessing import preprocess_data
from scripts.feature_engineering import generate_features
from scripts.model_building import train_models
from scripts.model_evaluation import evaluate_models

EMAIL_RECIPIENTS = os.getenv("AIRFLOW_EMAIL_RECIPIENTS", "admin@example.com").split(',')
DEFAULT_TICKER = os.getenv("STOCK_TICKER", "AAPL")
START_DATE = os.getenv("START_DATE", "2023-01-01")
END_DATE = os.getenv("END_DATE", "2024-01-01")
SCHEDULE_INTERVAL = os.getenv("SCHEDULE_INTERVAL", "@daily")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.today(),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': EMAIL_RECIPIENTS,
}

def run_pipeline():
    logging.info("Starting stock data collection...")
    fetch_stock_data(DEFAULT_TICKER, START_DATE, END_DATE)
    logging.info("Stock data collection complete.")
    
    logging.info("Starting data preprocessing...")
    preprocess_data()
    logging.info("Data preprocessing complete.")
    
    logging.info("Generating features...")
    generate_features()
    logging.info("Feature engineering complete.")
    
    logging.info("Starting model training...")
    train_models()
    logging.info("Model training complete.")
    
    logging.info("Evaluating models...")
    evaluate_models()
    logging.info("Model evaluation complete.")

dag = DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    schedule_interval=SCHEDULE_INTERVAL,
)

task_run_pipeline = PythonOperator(
    task_id='run_pipeline',
    python_callable=run_pipeline,
    dag=dag,
)

logging.info("Setting up failure email notification task.")
task_email_notification = EmailOperator(
    task_id='send_failure_email',
    to=EMAIL_RECIPIENTS,
    subject='Stock Prediction DAG Failed',
    html_content='The stock prediction pipeline has failed. Please check the logs.',
    trigger_rule='one_failed',
    dag=dag,
)

task_run_pipeline >> task_email_notification
