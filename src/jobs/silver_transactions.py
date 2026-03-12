import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    monotonically_increasing_id,
    rand,
    floor,
    to_date,
    from_unixtime,
    to_timestamp,
    timestamp_seconds,
    expr
)

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

BRONZE_PATH = "/opt/airflow/data/bronze/transactions/"
SILVER_PATH = "/opt/airflow/data/silver/transactions/"
CUSTOMER_PATH = "/opt/airflow/data/bronze/customers/"

WATERMARK_FILE = "metadata/transactions_watermark.txt"


# ---------------------------------------------------
# Spark Session
# ---------------------------------------------------

def create_spark():
    return SparkSession.builder \
        .appName("Silver Transactions Incremental Pipeline") \
        .getOrCreate()


# ---------------------------------------------------
# Watermark Functions
# ---------------------------------------------------

def get_last_watermark():

    if os.path.exists(WATERMARK_FILE):
        with open(WATERMARK_FILE, "r") as f:
            return f.read().strip()

    return None


def update_watermark(new_date):

    os.makedirs("metadata", exist_ok=True)

    with open(WATERMARK_FILE, "w") as f:
        f.write(str(new_date))


# ---------------------------------------------------
# Transformation Logic
# ---------------------------------------------------

def transform_transactions(df, customer_count):

    # Rename columns
    df = df.withColumnRenamed("Class", "fraud_flag") \
           .withColumnRenamed("Amount", "amount")

    # Keep required columns
    df = df.select(
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "amount",
        "fraud_flag",
        "ingest_date"
    )

    # Convert Time → timestamp
    df = df.withColumn(
        "transaction_timestamp",
        timestamp_seconds(col("Time") + lit(1672531200))

    )

    df = df.withColumn(
        "transaction_date",
        to_date("transaction_timestamp")
    )

    df = df.drop("Time")

    # Deduplicate
    business_cols = [f"V{i}" for i in range(1, 29)] + ["amount", "fraud_flag"]

    df = df.dropDuplicates(business_cols)

    # Assign customer_id
    df = df.withColumn(
        "customer_id",
        floor(rand(seed=42) * customer_count)
    )

    # Generate transaction_id
    df = df.withColumn(
        "transaction_id",
        monotonically_increasing_id()
    )

    # Final ordering
    df = df.select(
        "transaction_id",
        "customer_id",
        "transaction_timestamp",
        "transaction_date",
        *[f"V{i}" for i in range(1, 29)],
        "amount",
        "fraud_flag",
        "ingest_date"
    )

    return df


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

if __name__ == "__main__":

    spark = create_spark()

    print("Reading Bronze Transactions...")

    bronze_df = spark.read.parquet(BRONZE_PATH)

    # Read watermark
    last_watermark = get_last_watermark()

    if last_watermark:
        print(f"Last processed ingest_date: {last_watermark}")

        bronze_df = bronze_df.filter(
            col("ingest_date") > last_watermark
        )

    if bronze_df.rdd.isEmpty():
        print("No new data to process. Exiting pipeline.")
        spark.stop()
        exit()

    print("Reading Customers...")

    customer_df = spark.read.parquet(CUSTOMER_PATH)

    customer_count = customer_df.count()

    print("Transforming Transactions...")

silver_df = transform_transactions(bronze_df, customer_count)


def run_data_quality_checks(df):

    print("Running data quality checks...")

    # Null check
    null_amounts = df.filter(col("amount").isNull()).count()

    if null_amounts > 0:
        raise Exception(f"Data quality failed: {null_amounts} rows have NULL amount")

    # Fraud flag validation
    invalid_fraud = df.filter(~col("fraud_flag").isin(0,1)).count()

    if invalid_fraud > 0:
        raise Exception(f"Data quality failed: {invalid_fraud} invalid fraud flags")

    # Duplicate transaction IDs
    dup_txn = df.groupBy("transaction_id").count().filter(col("count") > 1).count()

    if dup_txn > 0:
        raise Exception(f"Data quality failed: {dup_txn} duplicate transaction_ids")

    print("Data quality checks passed.")


run_data_quality_checks(silver_df)

print("Writing Silver Transactions...")

silver_df.write \
        .mode("append") \
        .partitionBy("transaction_date") \
        .parquet(SILVER_PATH)

    # Update watermark
max_ingest = bronze_df.selectExpr("max(ingest_date)").collect()[0][0]

if max_ingest:
        update_watermark(max_ingest)
        print(f"Updated watermark to: {max_ingest}")

spark.stop()

print("Pipeline completed successfully.")