import pandas as pd
import numpy as np
import os

# Paths (adjust if needed)
sentiment_path = os.path.join('results', 'sentiment_output.csv')
stock_path = os.path.join('data', 'stock_data.csv')

# Load data
sentiment_df = pd.read_csv(sentiment_path)
stock_df = pd.read_csv(stock_path)

# Parse dates
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date

# Aggregate sentiment by date (average sentiment per day)
daily_sentiment = sentiment_df.groupby('Date')['sentiment'].mean().reset_index()

# Merge on 'Date'
merged = pd.merge(daily_sentiment, stock_df, on='Date', how='inner')

# Example: correlate average sentiment with closing price
if 'sentiment' in merged.columns and 'Close' in merged.columns:
    corr = merged['sentiment'].corr(merged['Close'])
    print(f'Correlation between sentiment and closing price: {corr:.4f}')
    try:
        import matplotlib.pyplot as plt
        plt.scatter(merged['sentiment'], merged['Close'])
        plt.xlabel('Average Sentiment')
        plt.ylabel('Closing Price')
        plt.title('Sentiment vs Closing Price')
        plt.show()
    except ImportError:
        print('matplotlib not installed, skipping plot.')
else:
    print('Required columns not found in merged data.') 