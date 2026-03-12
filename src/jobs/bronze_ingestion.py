from pyspark.sql import SparkSession
from pyspark.sql.functions import current_date

def create_spark_session():
    return SparkSession.builder \
        .appName("Bronze Ingestion Job") \
        .getOrCreate()


def ingest_dataset(spark, source_path, target_path, delimiter=","):
   
    df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("delimiter", delimiter) \
    .option("quote", '"') \
    .option("escape", '"') \
    .csv(source_path)



    df_with_ingest = df.withColumn("ingest_date", current_date())

    df_with_ingest.write \
        .mode("append") \
        .partitionBy("ingest_date") \
        .parquet(target_path)


if __name__ == "__main__":
    spark = create_spark_session()

    ingest_dataset(
        spark,
        "/opt/airflow/data/source/transactions/",
        "/opt/airflow/data/bronze/transactions/"
    )

    ingest_dataset(
        spark,
        "/opt/airflow/data/source/customers/",
        "/opt/airflow/data/bronze/customers/",
        delimiter=";"
    )

    spark.stop()
