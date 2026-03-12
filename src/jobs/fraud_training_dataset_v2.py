from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p, try_divide


def create_spark():
    return (
        SparkSession.builder
        .appName("Fraud Feature Engineering v2")
        .getOrCreate()
    )


if __name__ == "__main__":

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("Loading original feature dataset...")
    df = spark.read.parquet("data/features/fraud_training_dataset/")

    # ----------------------------
    # Relative Amount Deviation
    # ----------------------------
    df = df.withColumn(
        "amount_vs_avg",
        when(col("avg_transaction_amount") > 0,
             try_divide(col("amount"), col("avg_transaction_amount"))
        ).otherwise(0)
    )

    # ----------------------------
    # Short-Term Behavioral Deviation
    # ----------------------------
   
    df = df.withColumn(
       "amount_vs_7d_avg",
        try_divide(
           col("amount"),
           try_divide(col("total_spent_last_7d"), col("transactions_last_7d"))
      )
         )
    df = df.fillna(0, ["amount_vs_7d_avg"])
    # ----------------------------
    # Log Transforms
    # ----------------------------
    df = df.withColumn("log_amount", log1p(col("amount")))
    df = df.withColumn("log_total_spent", log1p(col("total_spent")))
    df = df.withColumn("log_total_transactions", log1p(col("total_transactions")))

    # ----------------------------
    # Interaction Features
    # ----------------------------
    df = df.withColumn(
        "amount_x_fraud_ratio",
        col("amount") * col("fraud_ratio")
    )

    df = df.withColumn(
        "velocity_x_fraud_ratio",
        col("transactions_last_7d") * col("fraud_ratio_last_7d")
    )

    print("New columns added successfully.")
    print("Updated Schema:")
    df.printSchema()

    # 🔎 DEBUG: Confirm data exists before saving
    print("Row count before saving:", df.count())
    
    # Save new dataset
    df.write.mode("overwrite").parquet(
        "/opt/airflow/data/features/fraud_training_dataset_v2/"
    )

    print("Feature dataset v2 saved successfully.")

    spark.stop()
    