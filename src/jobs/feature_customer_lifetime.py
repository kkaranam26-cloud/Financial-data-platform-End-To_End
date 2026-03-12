from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, round

def create_spark():
    return SparkSession.builder \
        .appName("Customer Lifetime Features") \
        .getOrCreate()


if __name__ == "__main__":

    spark = create_spark()

    # Load fact table
    fact_df = spark.read.parquet("data/gold/fact_transactions/")

    # Aggregate lifetime metrics per customer
    feature_df = fact_df.groupBy("customer_sk").agg(
        count("*").alias("total_transactions"),
        sum("amount").alias("total_spent"),
        avg("amount").alias("avg_transaction_amount"),
        sum("fraud_flag").alias("total_fraud_count")
    )

    # Fraud ratio
    feature_df = feature_df.withColumn(
        "fraud_ratio",
        round(col("total_fraud_count") / col("total_transactions"), 4)
    )

    # Write feature table
    feature_df.write \
        .mode("overwrite") \
        .parquet("data/features/customer_lifetime_features/")

    spark.stop()
