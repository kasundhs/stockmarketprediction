import pandas as pd
import numpy as np
import os
import argparse
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot predicted vs actual closing prices for a given stock.')
parser.add_argument('--stock', required=True, help='Stock name (e.g., TSLA)')
args = parser.parse_args()

# Paths
model_path = os.path.join('results', 'lstm_model.h5')
data_path = os.path.join('results', 'merged_dataset.parquet')

# Load model
try:
    model = load_model(model_path)
except Exception as e:
    print(f'Error loading trained model: {e}')
    print('Please run train_lstm.py first.')
    exit(1)

# Load integrated data
try:
    df = pd.read_parquet(data_path).dropna()
except Exception as e:
    print(f'Error loading integrated data: {e}')
    exit(1)

# Filter for the given stock
if 'Stock Name' not in df.columns:
    print('No "Stock Name" column found in data.')
    print('Available columns:', df.columns.tolist())
    exit(1)
stock_df = df[df['Stock Name'] == args.stock].copy()
if len(stock_df) < 100:
    print(f'Not enough data for stock {args.stock} (need at least 100 rows).')
    exit(1)

# Features used for training
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_sentiment']
if not all(col in stock_df.columns for col in feature_cols):
    print('Missing one or more required feature columns in the data.')
    print('Available columns:', stock_df.columns.tolist())
    exit(1)

# Scale features as in training
scaler = MinMaxScaler()
scaled = scaler.fit_transform(stock_df[feature_cols].values)

# Rolling window prediction (using only actual data)
window_size = 60
num_windows = len(scaled) - window_size
predicted = []
actual = []
dates = []
for i in range(num_windows):
    X_input = scaled[i:i+window_size]
    X_input = np.expand_dims(X_input, axis=0)
    prediction = model.predict(X_input, verbose=0)
    # Inverse transform to get real closing price
    last_row = stock_df[feature_cols].values[i+window_size-1].copy()
    last_row[3] = prediction[0][0]
    pred_real = scaler.inverse_transform([last_row])[0][3]
    predicted.append(pred_real)
    # Actual closing price for the next time step
    actual.append(stock_df[feature_cols].values[i+window_size][3])
    # Date for the next time step
    if 'Date' in stock_df.columns:
        dates.append(stock_df['Date'].iloc[i+window_size])
    elif 'date_only' in stock_df.columns:
        dates.append(stock_df['date_only'].iloc[i+window_size])
    else:
        dates.append(i+window_size)

# Plot
plt.figure(figsize=(14,6))
plt.plot(dates, actual, label='Actual Close', color='blue')
plt.plot(dates, predicted, label='Predicted Close', color='orange')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title(f'Predicted vs Actual Closing Price for {args.stock}')
plt.legend()
plt.tight_layout()
plt.show() 