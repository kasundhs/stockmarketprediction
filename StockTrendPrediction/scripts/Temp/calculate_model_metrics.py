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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(stock_symbol):
    """Load and prepare the merged dataset for analysis"""
    print(f"Loading merged dataset for stock: {stock_symbol}")
    
    # Try to load parquet first, fallback to CSV reconstruction if needed
    try:
        df = pd.read_parquet("results/merged_dataset.parquet")
        print(f"Loaded parquet file with shape: {df.shape}")
    except Exception as e:
        print(f"Could not load parquet file: {e}")
        print("Reconstructing dataset from CSV files...")
        
        # Load individual datasets
        sentiment_df = pd.read_csv('results/sentiment_output.csv')
        stock_df = pd.read_csv('data/stock_data.csv')
        
        # Convert dates
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        
        # Aggregate sentiment by date
        daily_sentiment = sentiment_df.groupby(sentiment_df['Date'].dt.date)['sentiment'].mean().reset_index()
        daily_sentiment.columns = ['date_only', 'avg_sentiment']
        daily_sentiment['date_only'] = pd.to_datetime(daily_sentiment['date_only'])
        
        # Merge with stock data
        stock_df['date_only'] = stock_df['Date'].dt.date
        stock_df['date_only'] = pd.to_datetime(stock_df['date_only'])
        
        df = stock_df.merge(daily_sentiment, on='date_only', how='left')
        print(f"Reconstructed dataset with shape: {df.shape}")
    
    # Filter by stock symbol if provided
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
    
    # Remove rows with missing sentiment data
    df = df.dropna()
    print(f"After removing NaN values: {df.shape}")
    
    return df

def prepare_lstm_data(df, sequence_length=60):
    """Prepare data for LSTM model training and testing"""
    print(f"Preparing LSTM data with sequence length: {sequence_length}")
    
    # Select features for training
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_sentiment']
    features = df[feature_columns].values
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Prepare sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(scaled_features[i, 3])  # Close price index
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets (80-20 split)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

def create_and_train_lstm_model(X_train, y_train, X_test, y_test):
    """Create and train the LSTM model"""
    print("Creating and training LSTM model...")
    
    # Create model architecture
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Generate predictions
    print("Generating predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return model, y_pred_train, y_pred_test, history

def inverse_transform_predictions(scaler, y_true, y_pred, feature_columns):
    """Inverse transform scaled predictions back to original scale"""
    # Create dummy arrays with the same shape as original features
    dummy_features = np.zeros((len(y_true), len(feature_columns)))
    
    # Set the Close price column (index 3) with our predictions/actual values
    dummy_features[:, 3] = y_pred.flatten()
    y_pred_original = scaler.inverse_transform(dummy_features)[:, 3]
    
    dummy_features[:, 3] = y_true.flatten()
    y_true_original = scaler.inverse_transform(dummy_features)[:, 3]
    
    return y_true_original, y_pred_original

def calculate_metrics(y_true, y_pred, dataset_name=""):
    """Calculate performance metrics"""
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # R² Score
    r2 = r2_score(y_true, y_pred)
    
    # Correlation coefficient
    correlation, p_value = pearsonr(y_true, y_pred)
    
    return {
        'Dataset': dataset_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'P_value': p_value
    }

def print_metrics_table(stock_symbol, train_metrics, test_metrics):
    """Print metrics in a formatted table"""
    print(f"\n{'='*80}")
    print(f"STOCK TREND PREDICTION PERFORMANCE METRICS - {stock_symbol}")
    print(f"{'='*80}")
    
    # Create table data
    headers = ["Dataset", "MAE ($)", "RMSE ($)", "R² Score", "Correlation (r)"]
    
    train_row = [
        "Training",
        f"{train_metrics['MAE']:.4f}",
        f"{train_metrics['RMSE']:.4f}", 
        f"{train_metrics['R2']:.4f}",
        f"{train_metrics['Correlation']:.4f}"
    ]
    
    test_row = [
        "Test",
        f"{test_metrics['MAE']:.4f}",
        f"{test_metrics['RMSE']:.4f}",
        f"{test_metrics['R2']:.4f}", 
        f"{test_metrics['Correlation']:.4f}"
    ]
    
    # Calculate column widths
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, train_row, test_row)]
    col_widths = [max(width, 12) for width in col_widths]  # Minimum width of 12
    
    # Print table
    def print_row(row_data, widths):
        row = "|"
        for i, (data, width) in enumerate(zip(row_data, widths)):
            row += f" {str(data).center(width)} |"
        print(row)
    
    def print_separator(widths):
        row = "+"
        for width in widths:
            row += "-" * (width + 2) + "+"
        print(row)
    
    # Print table
    print_separator(col_widths)
    print_row(headers, col_widths)
    print_separator(col_widths)
    print_row(train_row, col_widths)
    print_row(test_row, col_widths)
    print_separator(col_widths)
    
    # Summary section
    print(f"\nKEY INSIGHTS:")
    print(f"• Best performance metric: R² = {max(train_metrics['R2'], test_metrics['R2']):.4f}")
    print(f"• Model generalization: {'Good' if abs(train_metrics['R2'] - test_metrics['R2']) < 0.1 else 'Needs improvement'}")
    print(f"• Prediction accuracy: {'High' if test_metrics['Correlation'] > 0.7 else 'Moderate' if test_metrics['Correlation'] > 0.5 else 'Low'}")
    print(f"{'='*80}\n")

def plot_predictions(y_true, y_pred, title, save_path=None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title} - Model Performance Analysis', fontsize=16)
    
    # 1. Time series plot
    ax1.plot(y_true, label='Actual', alpha=0.7)
    ax1.plot(y_pred, label='Predicted', alpha=0.7)
    ax1.set_title('Actual vs Predicted Stock Prices')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Stock Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.6)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.set_title('Actual vs Predicted Scatter Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals plot
    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.6)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Predicted Price ($)')
    ax3.set_ylabel('Residuals ($)')
    ax3.set_title('Residuals Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution of residuals
    ax4.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Residuals ($)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

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
    
    metrics_df.to_csv(f'output/{stock_symbol}_detailed_metrics.csv', index=False)
    print(f"Detailed metrics summary saved to: output/{stock_symbol}_detailed_metrics.csv")
    
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
    all_predictions_df.to_csv(f'output/{stock_symbol}_predictions.csv', index=False)
    print(f"Detailed predictions saved to: output/{stock_symbol}_predictions.csv")

def get_available_stocks():
    """Get list of available stock symbols from the dataset"""
    try:
        stock_df = pd.read_csv('data/stock_data.csv')
        if 'Stock Name' in stock_df.columns:
            available_stocks = stock_df['Stock Name'].unique()
        elif 'Stock_Name' in stock_df.columns:
            available_stocks = stock_df['Stock_Name'].unique()
        else:
            print("Warning: No stock name column found in data.")
            return []
        
        print(f"Found {len(available_stocks)} available stocks: {', '.join(available_stocks)}")
        return available_stocks.tolist()
    except Exception as e:
        print(f"Error reading stock data: {e}")
        return []

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Calculate performance metrics for stock trend prediction model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_model_metrics.py --stock TSLA
  python calculate_model_metrics.py --all-stocks
  python calculate_model_metrics.py --stock AAPL --epochs 20
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--stock', 
        type=str,
        help='Stock symbol to analyze (e.g., TSLA, AAPL, GOOGL)'
    )
    
    group.add_argument(
        '--all-stocks',
        action='store_true',
        help='Analyze all available stocks and generate summary CSV'
    )
    
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=60,
        help='LSTM sequence length (default: 60)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    
    return parser.parse_args()

def analyze_single_stock(stock_symbol, args):
    """Analyze a single stock and return metrics"""
    print(f"\n{'='*80}")
    print(f"ANALYZING STOCK: {stock_symbol}")
    print(f"{'='*80}")
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(stock_symbol)
        
        # Check if we have enough data
        if len(df) < args.sequence_length + 10:  # Need minimum data for LSTM
            print(f"⚠️  Insufficient data for {stock_symbol} (only {len(df)} rows). Skipping...")
            return None
        
        # Prepare LSTM data
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, args.sequence_length)
        
        # Train model and get predictions
        model, y_pred_train_scaled, y_pred_test_scaled, history = create_and_train_lstm_model(
            X_train, y_train, X_test, y_test
        )
        
        # Inverse transform predictions to original scale
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_sentiment']
        
        y_true_train_orig, y_pred_train_orig = inverse_transform_predictions(
            scaler, y_train, y_pred_train_scaled, feature_columns
        )
        
        y_true_test_orig, y_pred_test_orig = inverse_transform_predictions(
            scaler, y_test, y_pred_test_scaled, feature_columns
        )
        
        # Calculate metrics for both training and test sets
        train_metrics = calculate_metrics(y_true_train_orig, y_pred_train_orig, "Training")
        test_metrics = calculate_metrics(y_true_test_orig, y_pred_test_orig, "Test")
        
        # Print results in table format
        print_metrics_table(stock_symbol, train_metrics, test_metrics)
        
        # Create output directory for plots if it doesn't exist
        import os
        os.makedirs('output', exist_ok=True)
        
        # Plot results
        plot_predictions(y_true_train_orig, y_pred_train_orig, f"{stock_symbol} Training Set", 
                        f"output/{stock_symbol}_training_predictions_plot.png")
        plot_predictions(y_true_test_orig, y_pred_test_orig, f"{stock_symbol} Test Set", 
                        f"output/{stock_symbol}_test_predictions_plot.png")
        
        # Save results
        save_results_to_csv(stock_symbol, train_metrics, test_metrics, 
                           y_true_train_orig, y_pred_train_orig,
                           y_true_test_orig, y_pred_test_orig)
        
        print(f"✅ {stock_symbol} analysis completed successfully!")
        
        return {
            'Stock': stock_symbol,
            'MAE ($)': test_metrics['MAE'],
            'RMSE ($)': test_metrics['RMSE'],
            'R2': test_metrics['R2'],
            'Correlation (r)': test_metrics['Correlation']
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {stock_symbol}: {e}")
        return None

def analyze_all_stocks(args):
    """Analyze all available stocks and generate summary CSV"""
    print("=== ANALYZING ALL STOCKS ===")
    
    # Get all available stocks
    available_stocks = get_available_stocks()
    
    if not available_stocks:
        print("No stocks found in the dataset.")
        return
    
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Store results for all stocks
    all_results = []
    successful_analyses = 0
    failed_analyses = 0
    
    print(f"\nStarting analysis of {len(available_stocks)} stocks...")
    print("This may take several minutes...")
    
    for i, stock in enumerate(available_stocks, 1):
        print(f"\n[{i}/{len(available_stocks)}] Processing {stock}...")
        
        result = analyze_single_stock(stock, args)
        
        if result:
            all_results.append(result)
            successful_analyses += 1
        else:
            failed_analyses += 1
    
    # Create summary CSV with all results
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Sort by R2 score (best performance first)
        summary_df = summary_df.sort_values('R2', ascending=False)
        
        # Round numerical values for better readability
        summary_df['MAE ($)'] = summary_df['MAE ($)'].round(2)
        summary_df['RMSE ($)'] = summary_df['RMSE ($)'].round(2)
        summary_df['R2'] = summary_df['R2'].round(4)
        summary_df['Correlation (r)'] = summary_df['Correlation (r)'].round(4)
        
        # Save summary CSV
        summary_df.to_csv('output/all_stocks_performance_summary.csv', index=False)
        
        # Print summary table
        print(f"\n{'='*100}")
        print("ALL STOCKS PERFORMANCE SUMMARY")
        print(f"{'='*100}")
        print(summary_df.to_string(index=False))
        print(f"{'='*100}")
        
        # Print statistics
        print(f"\n📊 ANALYSIS STATISTICS:")
        print(f"✅ Successful analyses: {successful_analyses}")
        print(f"❌ Failed analyses: {failed_analyses}")
        print(f"📈 Best performing stock (R²): {summary_df.iloc[0]['Stock']} ({summary_df.iloc[0]['R2']:.4f})")
        print(f"📉 Lowest MAE: {summary_df.loc[summary_df['MAE ($)'].idxmin()]['Stock']} (${summary_df['MAE ($)'].min():.2f})")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"- output/all_stocks_performance_summary.csv (main summary)")
        print(f"- output/{{STOCK}}_performance_metrics.csv (individual formatted metrics)")
        print(f"- output/{{STOCK}}_detailed_metrics.csv (individual detailed metrics)")
        print(f"- output/{{STOCK}}_predictions.csv (individual predictions)")
        print(f"- output/{{STOCK}}_*_plot.png (individual visualizations)")
    else:
        print("❌ No successful analyses completed.")

def main():
    """Main function to run the complete analysis"""
    # Parse arguments
    args = parse_arguments()
    
    if args.all_stocks:
        # Analyze all stocks
        analyze_all_stocks(args)
    else:
        # Analyze single stock
        stock_symbol = args.stock.upper()
        print("=== Stock Trend Prediction Model Performance Analysis ===")
        print(f"Analyzing stock: {stock_symbol}\n")
        
        result = analyze_single_stock(stock_symbol, args)
        
        if result:
            print(f"\n{'='*60}")
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print("All output files saved to 'output' folder:")
            print(f"- {stock_symbol}_performance_metrics.csv (formatted metrics)")
            print(f"- {stock_symbol}_detailed_metrics.csv (training & test metrics)")
            print(f"- {stock_symbol}_predictions.csv (detailed predictions)")
            print(f"- {stock_symbol}_training_predictions_plot.png (training plots)")
            print(f"- {stock_symbol}_test_predictions_plot.png (test plots)")
            print(f"{'='*60}")
        else:
            print(f"\n❌ Analysis failed for {stock_symbol}")

if __name__ == "__main__":
    main()
