from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, round

def create_spark():
    return SparkSession.builder \
        .appName("Monthly Revenue Summary Mart") \
        .getOrCreate()


if __name__ == "__main__":

    spark = create_spark()

    fact_df = spark.read.parquet("data/gold/fact_transactions/")
    dim_date_df = spark.read.parquet("data/gold/dim_date/")

    # Clean join (no ambiguity)
    mart_df = fact_df.join(
        dim_date_df,
        "date_sk",
        "inner"
    )

    # Aggregate by year + month
    mart_df = mart_df.groupBy(
        "year",
        "month"
    ).agg(
        count("*").alias("total_transactions"),
        sum("amount").alias("total_amount"),
        sum("fraud_flag").alias("fraud_count")
    )

    mart_df = mart_df.withColumn(
        "fraud_rate",
        round(col("fraud_count") / col("total_transactions"), 4)
    )

    mart_df.write \
        .mode("overwrite") \
        .parquet("data/gold/monthly_revenue_summary/")

    spark.stop()
