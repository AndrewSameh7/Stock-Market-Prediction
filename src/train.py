import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocessing import preprocess_data, train_test_split
from model import build_lstm_model
from utils import plot_predictions, evaluate_model

# Paths
DATA_PATH = "data/raw/AAPL_stock_data.csv"
MODEL_PATH = "models/LSTM_AAPL_model.h5"
SCALER_PATH = "models/scaler.pkl"
PLOTS_PATH = "outputs/plots"
PRED_PATH = "outputs/predictions"

# Make sure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(PRED_PATH, exist_ok=True)

# Load Data
df = pd.read_csv(DATA_PATH, header=[0, 1], index_col=0)
df.columns = df.columns.get_level_values(0)

# Preprocessing
X, y, scaler = preprocess_data(df, fit_scaler=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.8)

# Build Model
model = build_lstm_model(input_shape=(X_train.shape[1], 1))

# Callbacks to prevent overfitting
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-5,
    verbose=1
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# TRAIN Evaluation
train_pred = model.predict(X_train)
train_pred_prices = scaler.inverse_transform(train_pred)
train_actual_prices = scaler.inverse_transform(y_train.reshape(-1, 1))

train_metrics = evaluate_model(train_actual_prices, train_pred_prices)
train_accuracy = (1 - train_metrics["NRMSE"]) * 100

print("\n TRAIN RESULTS ")
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")
print(f"Train Accuracy: {train_accuracy:.2f}%")

# Save plot
plot_predictions(
    train_actual_prices,
    train_pred_prices,
    save_path=f"{PLOTS_PATH}/train_predictions.png"
)

# Save predictions CSV
train_pred_df = pd.DataFrame({
    "Actual": train_actual_prices.flatten(),
    "Predicted": train_pred_prices.flatten()
})
train_pred_df.to_csv(f"{PRED_PATH}/train_predictions.csv", index=False)
print(f"[INFO] Train predictions saved at: {PRED_PATH}/train_predictions.csv")

# TEST Evaluation
test_pred = model.predict(X_test)
test_pred_prices = scaler.inverse_transform(test_pred)
test_actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

test_metrics = evaluate_model(test_actual_prices, test_pred_prices)
test_accuracy = (1 - test_metrics["NRMSE"]) * 100

print("\n TEST RESULTS ")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save plot
plot_predictions(
    test_actual_prices,
    test_pred_prices,
    save_path=f"{PLOTS_PATH}/test_predictions.png"
)

# Save predictions CSV
test_pred_df = pd.DataFrame({
    "Actual": test_actual_prices.flatten(),
    "Predicted": test_pred_prices.flatten()
})
test_pred_df.to_csv(f"{PRED_PATH}/test_predictions.csv", index=False)
print(f"[INFO] Test predictions saved at: {PRED_PATH}/test_predictions.csv")

# Save Model & Scaler
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print("\n Model and scaler saved successfully!")
print("Training completed successfully!")
