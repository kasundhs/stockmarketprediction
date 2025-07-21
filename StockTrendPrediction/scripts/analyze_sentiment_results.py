import pandas as pd
import matplotlib.pyplot as plt

# Load the sentiment CSV
csv_path = 'results/sentiment_output.csv'
df = pd.read_csv(csv_path)

# 1. Sentiment distribution
positive = (df['sentiment'] > 0.05).sum()
negative = (df['sentiment'] < -0.05).sum()
neutral = ((df['sentiment'] >= -0.05) & (df['sentiment'] <= 0.05)).sum()

print(f'Positive tweets: {positive}')
print(f'Negative tweets: {negative}')
print(f'Neutral tweets: {neutral}')

# 2. Average sentiment
print('Average sentiment:', df['sentiment'].mean())

# 3. Most positive/negative tweets
print('\nMost positive tweets:')
print(df.sort_values('sentiment', ascending=False).head())

print('\nMost negative tweets:')
print(df.sort_values('sentiment').head())

# 4. Plot sentiment distribution
plt.figure(figsize=(6,4))
plt.bar(['Positive', 'Neutral', 'Negative'], [positive, neutral, negative], color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution')
plt.ylabel('Number of Tweets')
plt.show()

# 5. Plot average sentiment over time (if 'Date' column exists)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    daily_sentiment = df.groupby(df['Date'].dt.date)['sentiment'].mean()
    plt.figure(figsize=(10,4))
    daily_sentiment.plot()
    plt.title('Average Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment')
    plt.show()
else:
    print("\nNo 'Date' column found for time series plot.") 