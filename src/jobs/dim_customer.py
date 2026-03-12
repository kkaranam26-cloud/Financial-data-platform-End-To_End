from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, current_date, lit

def create_spark():
    return SparkSession.builder \
        .appName("Dim Customer Builder") \
        .getOrCreate()


def build_dim_customer(df):

    # Generate surrogate key
    df = df.withColumn("customer_sk", monotonically_increasing_id())

    # Add SCD columns
    df = df.withColumn("valid_from", current_date()) \
           .withColumn("valid_to", lit(None).cast("date")) \
           .withColumn("is_current", lit(1))

    # Reorder columns
    cols = ["customer_sk"] + [c for c in df.columns if c != "customer_sk"]
    df = df.select(*cols)

    return df


if __name__ == "__main__":

    spark = create_spark()

    silver_df = spark.read.parquet("data/silver/customers/")

    dim_df = build_dim_customer(silver_df)

    dim_df.write \
        .mode("overwrite") \
        .parquet("data/gold/dim_customer/")

    spark.stop()
