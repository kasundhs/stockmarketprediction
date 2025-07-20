import argparse
import subprocess
import os

def run_script(script, args=None):
    cmd = ['python', script]
    if args:
        cmd += args
    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run the full stock trend prediction pipeline.')
    parser.add_argument('--stock_xlsx', default='../stock_yfinance_data.xlsx', help='Path to stock Excel file')
    parser.add_argument('--tweets_xlsx', default='../stock_tweets.xlsx', help='Path to tweets Excel file')
    parser.add_argument('--stock_csv', default='data/stock_data.csv', help='Output path for stock CSV')
    parser.add_argument('--tweets_json', default='data/tweets.json', help='Output path for tweets JSON')
    parser.add_argument('--cleaned_json', default='data/cleaned_tweets.json', help='Output path for cleaned tweets JSON')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 1. Ingest stock data
    run_script('ingest_stock.py')
    # 2. Ingest tweets data
    run_script('ingest_tweets.py')
    # 3. Clean tweets
    run_script('clean_tweets.py')
    # 4. Preprocess sentiment
    run_script('preprocess_sentiment.py')
    # 5. Integrate data
    run_script('integrate_data.py')
    # 6. Train LSTM
    run_script('train_lstm.py')

if __name__ == '__main__':
    main() 