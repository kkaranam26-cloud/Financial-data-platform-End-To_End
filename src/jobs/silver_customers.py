from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, when

def create_spark():
    return SparkSession.builder \
        .appName("Silver Customers") \
        .getOrCreate()


def transform_customers(df):

    # Rename problematic columns (remove dots)
    df = df.withColumnRenamed("emp.var.rate", "emp_var_rate") \
           .withColumnRenamed("cons.price.idx", "cons_price_idx") \
           .withColumnRenamed("cons.conf.idx", "cons_conf_idx") \
           .withColumnRenamed("nr.employed", "nr_employed") \
           .withColumnRenamed("y", "response_flag")

    # Convert yes/no → 1/0
    df = df.withColumn(
        "response_flag",
        when(col("response_flag") == "yes", 1).otherwise(0)
    )
    df = df.drop("y")

    # Generate business key
    df = df.withColumn("customer_id", monotonically_increasing_id())

    # Reorder columns (customer_id first)
    final_cols = ["customer_id"] + [c for c in df.columns if c != "customer_id"]
    df = df.select(*final_cols)

    return df


if __name__ == "__main__":

    spark = create_spark()

    bronze_df = spark.read.parquet("data/bronze/customers/")

    silver_df = transform_customers(bronze_df)

    silver_df.write \
        .mode("overwrite") \
        .parquet("data/silver/customers/")

    spark.stop()
