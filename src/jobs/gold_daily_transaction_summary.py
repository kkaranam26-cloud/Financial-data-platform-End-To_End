from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, round

def create_spark():
    return SparkSession.builder \
        .appName("Daily Transaction Summary Mart") \
        .getOrCreate()


if __name__ == "__main__":

    spark = create_spark()

    # Load fact table
    fact_df = spark.read.parquet("/opt/airflow/data/gold/fact_transactions/")

    # Load date dimension
    dim_date_df = spark.read.parquet("/opt/airflow/data/gold/dim_date/")

    # Join fact with date dimension
    mart_df = fact_df.join(
    dim_date_df,
    "date_sk",
    "inner"
    )

    # Aggregate
    mart_df = mart_df.groupBy(
    "date_sk",
    "full_date"
    ).agg(
        count("*").alias("total_transactions"),
        sum("amount").alias("total_amount"),
        sum("fraud_flag").alias("fraud_count")
    )

    # Calculate fraud rate
    mart_df = mart_df.withColumn(
        "fraud_rate",
        round(col("fraud_count") / col("total_transactions"), 4)
    )

    # Write mart
    mart_df.write \
        .mode("overwrite") \
        .parquet("/opt/airflow/data/gold/daily_transaction_summary/")

    spark.stop()
