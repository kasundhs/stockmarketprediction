CREATE EXTERNAL TABLE IF NOT EXISTS stock_sentiment (
  stock_time TIMESTAMP,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume DOUBLE,
  avg_sentiment DOUBLE
)
STORED AS PARQUET
LOCATION '/user/yourusername/results/merged_dataset.parquet';