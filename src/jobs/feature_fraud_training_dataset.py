from pyspark.sql import SparkSession

def create_spark():
    return SparkSession.builder \
        .appName("Fraud Training Dataset Builder") \
        .getOrCreate()


if __name__ == "__main__":

    spark = create_spark()

    # Base fact table (contains fraud_flag = label)
    fact_df = spark.read.parquet("data/gold/fact_transactions/")

    # Load feature tables
    lifetime_df = spark.read.parquet("data/features/customer_lifetime_features/")
    rolling_df = spark.read.parquet("data/features/transaction_rolling_7d/")
    recency_df = spark.read.parquet("data/features/transaction_recency/")

    # Join rolling features
df = fact_df.join(
    rolling_df.drop("customer_sk"),
    "transaction_sk",
    "inner"
)

# Join recency features
df = df.join(
    recency_df.drop("customer_sk"),
    "transaction_sk",
    "inner"
)

# Join lifetime features
df = df.join(
    lifetime_df,
    "customer_sk",
    "inner"
)


    # Final dataset
training_df = df.select(
        "transaction_sk",
        "customer_sk",
        "amount",

        # Rolling features
        "transactions_last_7d",
        "total_spent_last_7d",
        "fraud_count_last_7d",
        "fraud_ratio_last_7d",

        # Recency
        "days_since_last_txn",

        # Lifetime
        "total_transactions",
        "total_spent",
        "avg_transaction_amount",
        "total_fraud_count",
        "fraud_ratio",

        # Target
        "fraud_flag"
    )

training_df.write \
        .mode("overwrite") \
        .parquet("data/features/fraud_training_dataset/")

spark.stop()
