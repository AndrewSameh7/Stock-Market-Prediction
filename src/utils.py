import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)

    # Normalized RMSE
    nrmse = rmse / (np.max(actual) - np.min(actual))

    # R2 Score
    r2 = r2_score(actual, predicted)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE": nrmse,
        "R2": r2
    }

def plot_predictions(actual, predicted, save_path="outputs/plots/prediction_plot.png"):
    # Plot actual vs predicted prices and save the figure
    # Create directories if they do not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Price", linewidth=2)
    plt.plot(predicted, label="Predicted Price", linewidth=2)

    plt.title("Stock Price Prediction", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show plot
    plt.show()

    print(f"[INFO] Plot saved at: {save_path}")
