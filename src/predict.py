import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from preprocessing import preprocess_data
from utils import plot_predictions, evaluate_model

# Paths
DATA_PATH = "data/raw/AAPL_stock_data.csv"
MODEL_PATH = "models/LSTM_AAPL_model.h5"
SCALER_PATH = "models/scaler.pkl"
PLOTS_PATH = "outputs/plots"
PRED_PATH = "outputs/predictions"

# Ensure folders exist
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(PRED_PATH, exist_ok=True)

# Load Model & Scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load New Data
df = pd.read_csv(DATA_PATH, header=[0, 1], index_col=0)
df.columns = df.columns.get_level_values(0)

# Preprocessing (use existing scaler)
X, y, _ = preprocess_data(df, fit_scaler=False, scaler=scaler)

# Prediction
predictions = model.predict(X)
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Evaluation
metrics = evaluate_model(actual_prices, predicted_prices)
accuracy = (1 - metrics["NRMSE"]) * 100

print("\n PREDICTION RESULTS ")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print(f"Prediction Accuracy: {accuracy:.2f}%")

# Save plot
plot_predictions(
    actual_prices,
    predicted_prices,
    save_path=f"{PLOTS_PATH}/predict_results.png"
)
print(f"[INFO] Plot saved at: {PLOTS_PATH}/predict_results.png")

# Save predictions CSV
pred_df = pd.DataFrame({
    "Actual": actual_prices.flatten(),
    "Predicted": predicted_prices.flatten()
})
pred_csv_path = f"{PRED_PATH}/predict_results.csv"
pred_df.to_csv(pred_csv_path, index=False)
print(f"[INFO] Predictions saved at: {pred_csv_path}")

print("Prediction completed successfully!")
