import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

df = pd.read_parquet("results/merged_dataset.parquet").dropna()
print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns)
print(df.head())
features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'avg_sentiment']].values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i, 3])
X, y = np.array(X), np.array(y)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)