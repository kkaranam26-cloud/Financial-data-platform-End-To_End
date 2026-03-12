from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, unix_timestamp, round
from pyspark.sql.window import Window

def create_spark():
    return SparkSession.builder \
        .appName("Transaction Recency Features") \
        .getOrCreate()


if __name__ == "__main__":

    spark = create_spark()

    fact_df = spark.read.parquet("data/gold/fact_transactions/")
    dim_date_df = spark.read.parquet("data/gold/dim_date/")

    # Join to get full_date
    df = fact_df.join(dim_date_df, "date_sk", "inner")

    # Convert to unix timestamp
    df = df.withColumn(
        "full_date_unix",
        unix_timestamp(col("full_date"))
    )

    # Define window per customer ordered by date
    window_spec = Window.partitionBy("customer_sk") \
        .orderBy("full_date_unix")

    # Get previous transaction timestamp
    df = df.withColumn(
        "previous_txn_unix",
        lag("full_date_unix").over(window_spec)
    )

    # Calculate difference in days
    df = df.withColumn(
        "days_since_last_txn",
        round(
            (col("full_date_unix") - col("previous_txn_unix")) / 86400,
            2
        )
    )

    feature_df = df.select(
        "transaction_sk",
        "customer_sk",
        "days_since_last_txn"
    )

    feature_df.write \
        .mode("overwrite") \
        .parquet("data/features/transaction_recency/")

    spark.stop()
