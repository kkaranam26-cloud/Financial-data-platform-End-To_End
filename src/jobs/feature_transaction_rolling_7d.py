from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, round, unix_timestamp
from pyspark.sql.window import Window
def create_spark():
    return SparkSession.builder \
        .appName("Rolling 7-Day Transaction Features") \
        .getOrCreate()


if __name__ == "__main__":

    spark = create_spark()

    fact_df = spark.read.parquet("/opt/airflow/data/gold/fact_transactions/")
    dim_date_df = spark.read.parquet("/opt/airflow/data/gold/dim_date/")

    # Join to get full_date
    df = fact_df.join(dim_date_df, "date_sk", "inner")

  
    

# Convert date to long (seconds)
df = df.withColumn(
    "full_date_unix",
    unix_timestamp(col("full_date"))
)

window_spec = Window.partitionBy("customer_sk") \
    .orderBy("full_date_unix") \
    .rangeBetween(-7 * 86400, 0)


    # Rolling features
df = df.withColumn(
        "transactions_last_7d",
        count("*").over(window_spec)
    ).withColumn(
        "total_spent_last_7d",
        sum("amount").over(window_spec)
    ).withColumn(
        "fraud_count_last_7d",
        sum("fraud_flag").over(window_spec)
    )

df = df.withColumn(
        "fraud_ratio_last_7d",
        round(col("fraud_count_last_7d") / col("transactions_last_7d"), 4)
    )

    # Select final feature columns
feature_df = df.select(
        "transaction_sk",
        "customer_sk",
        "transactions_last_7d",
        "total_spent_last_7d",
        "fraud_count_last_7d",
        "fraud_ratio_last_7d"
    )

feature_df.write \
        .mode("overwrite") \
        .parquet("/opt/airflow/data/features/transaction_rolling_7d/")

spark.stop()
