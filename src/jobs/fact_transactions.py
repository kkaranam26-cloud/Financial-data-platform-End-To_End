from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, date_format

def create_spark():
    return SparkSession.builder \
        .appName("Fact Transactions Builder") \
        .getOrCreate()

if __name__ == "__main__":

    spark = create_spark()

    transactions_df = spark.read.parquet("/opt/airflow/data/silver/transactions/")
    dim_date_df = spark.read.parquet("/opt/airflow/data/gold/dim_date/")
    dim_customer_df = spark.read.parquet("/opt/airflow/data/gold/dim_customer/")

    fact_df = transactions_df.join(
        dim_customer_df,
        "customer_id",
        "inner"
    )

    fact_df = fact_df.withColumn(
        "date_sk",
        date_format(col("transaction_date"), "yyyyMMdd").cast("int")
    )

    fact_df = fact_df.join(
        dim_date_df.select("date_sk"),
        "date_sk",
        "left"
    )

    fact_df = fact_df.withColumn(
        "transaction_sk",
        monotonically_increasing_id()
    )

    fact_df = fact_df.select(
        "transaction_sk",
        "customer_sk",
        "date_sk",
        "amount",
        "fraud_flag"
    )

    fact_df.write.mode("overwrite").parquet("/opt/airflow/data/gold/fact_transactions/")

    spark.stop()