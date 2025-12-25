# data_loader.py
import os
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker, start_date, end_date, save_path):
    # Download historical stock data from Yahoo Finance and save it as a CSV file.
    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        raise ValueError("No data downloaded. Please check ticker or date range.")

    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save data to CSV
    data.to_csv(save_path)

    return data


if __name__ == "__main__":
    # Project parameters
    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2024-12-31"
    SAVE_PATH = "data/raw/AAPL_stock_data.csv"

    # Fetch and save data
    df = fetch_stock_data(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        save_path=SAVE_PATH
    )

    # Display basic information
    print("Data download completed successfully!")
    print(f"Total records: {len(df)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
