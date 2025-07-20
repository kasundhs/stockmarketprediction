from pyspark.sql import SparkSession
from pyspark.sql.functions import window, avg, to_date, col

spark = SparkSession.builder.appName("Integration").getOrCreate()
tweets = spark.read.parquet("results/sentiment_output.parquet")
stocks = spark.read.csv("data/stock_data.csv", header=True, inferSchema=True)
tweets = tweets.withColumn("timestamp", tweets["Date"].cast("timestamp"))
agg_sentiment = tweets.groupBy(window("timestamp", "1 day")).agg(avg("sentiment").alias("avg_sentiment"))
stocks = stocks.withColumn("stock_time", stocks["Date"].cast("timestamp"))

# Extract date only for join
stocks = stocks.withColumn("date_only", to_date(col("stock_time")))
agg_sentiment = agg_sentiment.withColumn("date_only", to_date(col("window").end))

print("Tweets DataFrame:")
tweets.select("timestamp").show(5)
print("Stocks DataFrame:")
stocks.select("stock_time", "date_only").show(5)
print("Aggregated Sentiment DataFrame:")
agg_sentiment.select("window", "avg_sentiment", "date_only").show(5)

joined = stocks.join(agg_sentiment, on="date_only", how="left")
print("Joined DataFrame:")
joined.show(5)
print("Joined DataFrame count:", joined.count())

joined.write.parquet("results/merged_dataset.parquet", mode="overwrite")