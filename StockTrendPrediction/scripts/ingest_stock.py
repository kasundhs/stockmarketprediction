import pandas as pd

df = pd.read_excel('../../stock_yfinance_data.xlsx')
df.to_csv('data/stock_data.csv', index=False)