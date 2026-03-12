from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, round

def create_spark():
    return SparkSession.builder \
        .appName("Customer Activity Summary Mart") \
        .getOrCreate()


if __name__ == "__main__":

    spark = create_spark()

    fact_df = spark.read.parquet("data/gold/fact_transactions/")
    dim_customer_df = spark.read.parquet("data/gold/dim_customer/")

    # Join on surrogate key
    mart_df = fact_df.join(
        dim_customer_df,
        "customer_sk",
        "inner"
    )

    # Aggregate by customer
    mart_df = mart_df.groupBy(
        "customer_sk"
    ).agg(
        count("*").alias("transaction_count"),
        sum("amount").alias("total_spent"),
        avg("amount").alias("avg_transaction_amount"),
        sum("fraud_flag").alias("fraud_count")
    )

    mart_df = mart_df.withColumn(
        "fraud_rate",
        round(col("fraud_count") / col("transaction_count"), 4)
    )

    mart_df.write \
        .mode("overwrite") \
        .parquet("data/gold/customer_activity_summary/")

    spark.stop()
