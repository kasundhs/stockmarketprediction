import pandas as pd

df = pd.read_excel('../../stock_tweets.xlsx')
df.to_json('data/tweets.json', orient='records', lines=True)