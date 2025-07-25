import pandas as pd
import numpy as np
import os
import argparse
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Predict next closing price for a given stock.')
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
stock_df = df[df['Stock Name'] == args.stock]
if len(stock_df) < 60:
    print(f'Not enough data for stock {args.stock} (need at least 60 rows).')
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

# Get last 60 time steps
X_input = scaled[-60:]
X_input = np.expand_dims(X_input, axis=0)  # shape (1, 60, 6)

# Predict
prediction = model.predict(X_input, verbose=0)
last_row = stock_df[feature_cols].values[-1].copy()
last_row[3] = prediction[0][0]  # Replace 'Close' with predicted value
real_value = scaler.inverse_transform([last_row])[0][3]
print(f'Predicted next closing price for {args.stock}: {real_value}') 