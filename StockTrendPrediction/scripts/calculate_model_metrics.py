"""
Stock Trend Prediction Model Performance Metrics Calculator

This script loads the merged dataset, recreates the LSTM model training process,
generates predictions, and calculates performance metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Correlation (r)

Usage: python calculate_model_metrics.py --stock TSLA

Author: Generated for Stock Trend Prediction Analysis
"""

import pandas as pd
import numpy as np
import argparse
import sys
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# FIX: Set random seeds for reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
# -----------------------------

def load_and_prepare_data(stock_symbol):
    """Load and prepare the merged dataset for analysis"""
    print(f"Loading merged dataset for stock: {stock_symbol}")
    
    try:
        df = pd.read_parquet("results/merged_dataset.parquet")
        print(f"Loaded parquet file with shape: {df.shape}")
    except Exception as e:
        print(f"Could not load parquet file: {e}")
        print("Reconstructing dataset from CSV files...")
        
        sentiment_df = pd.read_csv('results/sentiment_output.csv')
        stock_df = pd.read_csv('data/stock_data.csv')
        
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        
        daily_sentiment = sentiment_df.groupby(sentiment_df['Date'].dt.date)['sentiment'].mean().reset_index()
        daily_sentiment.columns = ['date_only', 'avg_sentiment']
        daily_sentiment['date_only'] = pd.to_datetime(daily_sentiment['date_only'])
        
        stock_df['date_only'] = stock_df['Date'].dt.date
        stock_df['date_only'] = pd.to_datetime(stock_df['date_only'])
        
        df = stock_df.merge(daily_sentiment, on='date_only', how='left')
        print(f"Reconstructed dataset with shape: {df.shape}")
    
    if 'Stock Name' in df.columns:
        df = df[df['Stock Name'] == stock_symbol]
        print(f"After filtering for {stock_symbol}: {df.shape}")
    elif 'Stock_Name' in df.columns:
        df = df[df['Stock_Name'] == stock_symbol]
        print(f"After filtering for {stock_symbol}: {df.shape}")
    else:
        print(f"Warning: No stock name column found. Processing all data.")
    
    if len(df) == 0:
        raise ValueError(f"No data found for stock symbol: {stock_symbol}")
    
    df = df.dropna()
    print(f"After removing NaN values: {df.shape}")
    
    return df

def prepare_lstm_data(df, sequence_length=60):
    """Prepare data for LSTM model training and testing"""
    print(f"Preparing LSTM data with sequence length: {sequence_length}")
    
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_sentiment']
    features = df[feature_columns].values
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(scaled_features[i, 3])
    
    X, y = np.array(X), np.array(y)
    
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler
def save_results_to_csv(stock_symbol, train_metrics, test_metrics, y_true_train, y_pred_train, 
                       y_true_test, y_pred_test):
    """Save detailed results to CSV files"""
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('output', exist_ok=True)
    
    # Save formatted metrics in the requested format (using test set metrics)
    formatted_metrics = {
        'Metric': [
            f'Mean Absolute Error (MAE) = ~${test_metrics["MAE"]:.2f}',
            f'Root Mean Squared Error (RMSE) = ~${test_metrics["RMSE"]:.2f}',
            f'R² Score = ~{test_metrics["R2"]:.2f}',
            f'Correlation (r) = ~{test_metrics["Correlation"]:.2f}'
        ]
    }
    
    formatted_df = pd.DataFrame(formatted_metrics)
    formatted_df.to_csv(f'output/{stock_symbol}_performance_metrics.csv', index=False)
    print(f"Formatted metrics saved to: output/{stock_symbol}_performance_metrics.csv")
    
    # Save metrics summary (both training and test)
    metrics_df = pd.DataFrame({
        'Dataset': ['Training', 'Test'],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'R2_Score': [train_metrics['R2'], test_metrics['R2']],
        'Correlation': [train_metrics['Correlation'], test_metrics['Correlation']],
        'P_value': [train_metrics['P_value'], test_metrics['P_value']]
    })
    
    # metrics_df.to_csv(f'output/{stock_symbol}_detailed_metrics.csv', index=False)
    # print(f"Detailed metrics summary saved to: output/{stock_symbol}_detailed_metrics.csv")
    
    # Save detailed predictions
    train_predictions_df = pd.DataFrame({
        'Actual_Price': y_true_train,
        'Predicted_Price': y_pred_train,
        'Residual': y_true_train - y_pred_train,
        'Dataset': 'Training'
    })
    
    test_predictions_df = pd.DataFrame({
        'Actual_Price': y_true_test,
        'Predicted_Price': y_pred_test,
        'Residual': y_true_test - y_pred_test,
        'Dataset': 'Test'
    })
    
    all_predictions_df = pd.concat([train_predictions_df, test_predictions_df], ignore_index=True)
    # all_predictions_df.to_csv(f'output/{stock_symbol}_predictions.csv', index=False)
    # print(f"Detailed predictions saved to: output/{stock_symbol}_predictions.csv")

def create_and_train_lstm_model(X_train, y_train, X_test, y_test):
    """Create and train the LSTM model"""
    print("Creating and training LSTM model...")
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Training model...")
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    print("Generating predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return model, y_pred_train, y_pred_test, history

def inverse_transform_predictions(scaler, y_true, y_pred, feature_columns):
    dummy_features = np.zeros((len(y_true), len(feature_columns)))
    dummy_features[:, 3] = y_pred.flatten()
    y_pred_original = scaler.inverse_transform(dummy_features)[:, 3]
    
    dummy_features[:, 3] = y_true.flatten()
    y_true_original = scaler.inverse_transform(dummy_features)[:, 3]
    
    return y_true_original, y_pred_original

def calculate_metrics(y_true, y_pred, dataset_name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    correlation, p_value = pearsonr(y_true, y_pred)
    
    return {
        'Dataset': dataset_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'P_value': p_value
    }

# (Remaining functions print_metrics_table, plot_predictions, save_results_to_csv, parse_arguments stay unchanged)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Calculate performance metrics for stock trend prediction model'
    )
    parser.add_argument('--stock', type=str, required=True, help='Stock symbol')
    parser.add_argument('--sequence_length', type=int, default=60, help='LSTM sequence length')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    return parser.parse_args()

def main():
    args = parse_arguments()
    stock_symbol = args.stock.upper()
    
    print("=== Stock Trend Prediction Model Performance Analysis ===")
    print(f"Analyzing stock: {stock_symbol}\n")
    
    try:
        df = load_and_prepare_data(stock_symbol)
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, args.sequence_length)
        model, y_pred_train_scaled, y_pred_test_scaled, history = create_and_train_lstm_model(X_train, y_train, X_test, y_test)
        
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_sentiment']
        y_true_train_orig, y_pred_train_orig = inverse_transform_predictions(scaler, y_train, y_pred_train_scaled, feature_columns)
        y_true_test_orig, y_pred_test_orig = inverse_transform_predictions(scaler, y_test, y_pred_test_scaled, feature_columns)
        
        train_metrics = calculate_metrics(y_true_train_orig, y_pred_train_orig, "Training")
        test_metrics = calculate_metrics(y_true_test_orig, y_pred_test_orig, "Test")
        
        # print_metrics_table(stock_symbol, train_metrics, test_metrics)
        # plot_predictions(y_true_train_orig, y_pred_train_orig, f"{stock_symbol} Training Set", f"results/{stock_symbol}_training_predictions_plot.png")
        # plot_predictions(y_true_test_orig, y_pred_test_orig, f"{stock_symbol} Test Set", f"results/{stock_symbol}_test_predictions_plot.png")
        save_results_to_csv(stock_symbol, train_metrics, test_metrics, y_true_train_orig, y_pred_train_orig, y_true_test_orig, y_pred_test_orig)
        
        return train_metrics, test_metrics
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    train_metrics, test_metrics = main()
