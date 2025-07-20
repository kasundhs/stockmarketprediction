from pyspark.sql import SparkSession
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("Sentiment").getOrCreate()
df = spark.read.json("data/tweets.json")
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    return analyzer.polarity_scores(text)["compound"]
sentiment_udf = udf(get_sentiment, DoubleType())
df = df.withColumn("sentiment", sentiment_udf(df['Tweet']))
df.write.parquet("results/sentiment_output.parquet", mode="overwrite")