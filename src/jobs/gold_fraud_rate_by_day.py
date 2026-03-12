from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, round

def create_spark():
    return SparkSession.builder \
        .appName("Fraud Rate By Day Mart") \
        .getOrCreate()


if __name__ == "__main__":

    spark = create_spark()

    fact_df = spark.read.parquet("data/gold/fact_transactions/")
    dim_date_df = spark.read.parquet("data/gold/dim_date/")

    # Join on date_sk (clean join)
    mart_df = fact_df.join(
        dim_date_df,
        "date_sk",
        "inner"
    )

    # Aggregate fraud metrics
    mart_df = mart_df.groupBy(
        "date_sk",
        "full_date"
    ).agg(
        sum("fraud_flag").alias("fraud_count"),
        count("*").alias("total_transactions")
    )

    # Calculate fraud rate
    mart_df = mart_df.withColumn(
        "fraud_rate",
        round(col("fraud_count") / col("total_transactions"), 4)
    )

    mart_df.write \
        .mode("overwrite") \
        .parquet("data/gold/fraud_rate_by_day/")

    spark.stop()
