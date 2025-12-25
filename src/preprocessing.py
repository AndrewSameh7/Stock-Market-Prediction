import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(df, feature_col="Close", window_size=60,
                    fit_scaler=True, scaler=None):
    # Preprocess the stock data for LSTM model.
    data = df[[feature_col]].ffill().values

    if fit_scaler:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y), scaler


def train_test_split(X, y, train_ratio=0.8):
    # Split the data into training and testing sets.
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("data/raw/AAPL_stock_data.csv", header=[0, 1], index_col=0)
    df.columns = df.columns.get_level_values(0)

    # Preprocess data
    X, y, scaler = preprocess_data(df, feature_col='Close', window_size=60)

    # Save processed data
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)
    pd.DataFrame(X.reshape(X.shape[0], -1)).to_csv(f"{processed_path}/X_processed.csv", index=False)
    pd.DataFrame(y).to_csv(f"{processed_path}/y_processed.csv", index=False)
    print("[INFO] Processed data saved in 'data/processed' folder")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.8)

    print("Preprocessing completed successfully!")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
