# Stock Market Prediction Using LSTM :

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices using historical data. It includes full data preprocessing, model training, evaluation, and prediction on new data. The project also generates visualizations and saves predictions for analysis.

============================================================================================================

## Project Structure :

stock-market-prediction/
│
├─ data/
│  └─ raw/                      # Raw stock data CSV files
│
├─ models/                       # Trained models and scalers
│  ├─ LSTM_AAPL_model.h5
│  └─ scaler.pkl
│
├─ outputs/                       # Results & analysis
│  ├─ plots/                     # Graphs for train, test, and predictions
│  │  ├─ train_predictions.png
│  │  ├─ test_predictions.png
│  │  └─ predict_results.png
│  └─ predictions/               # CSV files for predictions
│     ├─ train_predictions.csv
│     ├─ test_predictions.csv
│     └─ predict_results.csv
│
├─ src/                           # Source code
│  ├─ train.py                    # Script to train the model
│  ├─ predict.py                  # Script to predict new data
│  ├─ preprocessing.py            # Data preprocessing functions
│  ├─ model.py                    # LSTM model definition
│  └─ utils.py                    # Evaluation & plotting utilities
│
└─ README.md                      # Project overview and 
|_License                         # License file (MIT License)
|_ requirements.txt               # Project dependencies
|_ gitignore                      # Git ignore file to exclude unnecessary files/folders

============================================================================================================

## Installation :

1. Clone the repository:
git clone https://github.com/AndrewSameh7/stock-market-prediction.git
cd stock-market-prediction

============================================================================================================

## Usage :
1. Train the Model

Run the training script to preprocess the data, train the LSTM model, evaluate, and save outputs:
python src/train.py


Outputs:
Trained model: models/LSTM_AAPL_model.h5
Scaler: models/scaler.pkl
Training and test plots: outputs/plots/
Predictions CSV: outputs/predictions/
Metrics printed in console: MSE, RMSE, MAE, NRMSE, R², and accuracy.

2. Predict New Data

Run the prediction script to generate predictions on new stock data:
python src/predict.py


Outputs:
Predictions plot: outputs/plots/predict_results.png
Predictions CSV: outputs/predictions/predict_results.csv
Metrics printed in console: MSE, RMSE, MAE, NRMSE, R², and accuracy.

============================================================================================================

## Results :

1-Training Results
MSE: 10.4280
RMSE: 3.2292
MAE: 2.0397
NRMSE: 0.0205
R²: 0.9956
Train Accuracy: 97.95%

2-Testing Results
MSE: 60.5984
RMSE: 7.7845
MAE: 6.2590
NRMSE: 0.0624
R²: 0.9164
Test Accuracy: 93.76%

3-Prediction Results
MSE: 20.4621
RMSE: 4.5235
MAE: 2.8836
NRMSE: 0.0191
R²: 0.9952
Prediction Accuracy: 98.09%
Note: The accuracy is calculated based on the Normalized Root Mean Squared Error (NRMSE).

============================================================================================================

## Features :

Preprocessing of raw stock data.
LSTM model for time series prediction.
Training with early stopping and learning rate scheduler to prevent overfitting.
Evaluation metrics and accuracy calculation.
Graphical visualization of predictions vs actual prices.
Save predictions and plots for further analysis.

============================================================================================================

## Requirements :

Python 3.11.9
TensorFlow 2.x
Keras
pandas
numpy
matplotlib
seabon
scikit-learn
joblib
jupyter

Install dependencies using:
pip install -r requirements.txt

============================================================================================================

## License :

This project is licensed under the MIT License.
