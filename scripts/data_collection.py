import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data using yfinance API with input validation."""
    if not isinstance(ticker, str) or not ticker.isalpha():
        logging.error("Invalid ticker symbol. It must be a string containing only letters.")
        raise ValueError("Invalid ticker symbol. It must be a string containing only letters.")
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        logging.error("Start and end dates must be strings in 'YYYY-MM-DD' format.")
        raise ValueError("Start and end dates must be strings in 'YYYY-MM-DD' format.")
    
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            logging.warning(f"No data found for {ticker}")
            return None
        os.makedirs("data/raw", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stock_data.to_csv(f"data/raw/{ticker}_historical_stock_data_{timestamp}.csv")
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        raise RuntimeError(f"Error fetching data for {ticker}: {e}")
