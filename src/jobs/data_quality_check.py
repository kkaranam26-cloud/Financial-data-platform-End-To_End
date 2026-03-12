from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Data Quality Checks").getOrCreate()

df = spark.read.parquet("/opt/airflow/data/gold/fact_transactions")

null_transactions = df.filter(col("transaction_sk").isNull()).count()

if null_transactions > 0:
    raise Exception("Data Quality Check Failed: transaction_sk contains NULLs")

print("Data Quality Checks Passed")