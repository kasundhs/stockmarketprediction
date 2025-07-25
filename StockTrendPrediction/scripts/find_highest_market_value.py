import pandas as pd
import os

# Path to integrated data
parquet_path = os.path.join('results', 'merged_dataset.parquet')

# Load data
try:
    df = pd.read_parquet(parquet_path)
except Exception as e:
    print(f'Error loading parquet file: {e}')
    exit(1)

# Use 'Close' as the market value
if 'Close' not in df.columns:
    print('No "Close" column found.')
    print('Available columns:', df.columns.tolist())
    exit(1)

# Find row with highest closing price
max_row = df.loc[df['Close'].idxmax()]
print('Row with highest closing price:')
print(max_row) 