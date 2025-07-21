import pandas as pd

# Path to the Parquet file
parquet_path = 'results/sentiment_output.parquet'
# Output paths
csv_path = 'results/sentiment_output.csv'
txt_path = 'results/sentiment_output.txt'

# Read the Parquet file
try:
    df = pd.read_parquet(parquet_path)
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f'Saved CSV to {csv_path}')
    # Save to TXT (tab-separated)
except Exception as e:
    print(f'Error reading or saving file: {e}') 