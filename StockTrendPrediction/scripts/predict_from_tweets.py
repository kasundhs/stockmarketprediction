import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

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

# Features used for training
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_sentiment']
if not all(col in df.columns for col in feature_cols):
    print('Missing one or more required feature columns in the data.')
    print('Available columns:', df.columns.tolist())
    exit(1)

# Scale features as in training
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[feature_cols].values)

# Rolling window prediction
window_size = 60
num_windows = len(scaled) - window_size + 1
if num_windows < 1:
    print('Not enough data for rolling window prediction.')
    exit(1)

scaled_preds = []
real_preds = []
window_indices = []
for i in range(num_windows):
    X_input = scaled[i:i+window_size]
    X_input = np.expand_dims(X_input, axis=0)
    prediction = model.predict(X_input, verbose=0)
    scaled_preds.append(prediction[0][0])
    last_row = df[feature_cols].values[i+window_size-1].copy()
    last_row[3] = prediction[0][0]
    real_value = scaler.inverse_transform([last_row])[0][3]
    real_preds.append(real_value)
    window_indices.append(i+window_size-1)

# Calculate averages
avg_scaled = np.mean(scaled_preds)
avg_real = np.mean(real_preds)
print(f'Average predicted closing price (scaled): {avg_scaled}')
print(f'Average predicted closing price (real scale): {avg_real}')

# Plot results
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(window_indices, scaled_preds, color='skyblue')
    plt.title('Predicted Closing Price (Scaled)')
    plt.xlabel('Window End Index')
    plt.ylabel('Scaled Value')
    plt.subplot(1,2,2)
    plt.plot(window_indices, real_preds, color='salmon')
    plt.title('Predicted Closing Price (Real)')
    plt.xlabel('Window End Index')
    plt.ylabel('Real Value')
    plt.tight_layout()
    plt.show()
except ImportError:
    print('matplotlib not installed, skipping plot.') 