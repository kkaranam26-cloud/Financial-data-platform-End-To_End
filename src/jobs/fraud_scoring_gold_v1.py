# ============================================================
# Fraud Detection - Gold Scoring Job (v1)
# Purpose:
#   1. Load full feature dataset
#   2. Load final trained model (v3)
#   3. Score all transactions
#   4. Write Gold fraud scoring table
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.functions import vector_to_array


# ------------------------------------------------------------
# Spark Session
# ------------------------------------------------------------
def create_spark():
    return (
        SparkSession.builder
        .appName("Fraud Detection - Gold Scoring v1")
        .getOrCreate()
    )


if __name__ == "__main__":

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------
    # 1️⃣ Load Feature Dataset (FULL DATA)
    # ------------------------------------------------------------
    df = spark.read.parquet(
        "/opt/airflow/data/features/fraud_training_dataset_v2/"
    )

    feature_cols = [
        "amount",
        "transactions_last_7d",
        "total_spent_last_7d",
        "fraud_count_last_7d",
        "fraud_ratio_last_7d",
        "days_since_last_txn",
        "total_transactions",
        "total_spent",
        "avg_transaction_amount",
        "total_fraud_count",
        "fraud_ratio",
        "amount_vs_avg",
        "amount_vs_7d_avg",
        "log_amount",
        "log_total_spent",
        "log_total_transactions",
        "amount_x_fraud_ratio",
        "velocity_x_fraud_ratio"
    ]

    df = df.fillna(0, subset=feature_cols)

    # ------------------------------------------------------------
    # 2️⃣ Assemble Feature Vector
    # ------------------------------------------------------------
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    df = assembler.transform(df)

    # ------------------------------------------------------------
    # 3️⃣ Load Final Model
    # ------------------------------------------------------------
    model_path = "/opt/airflow/data/models/v3_gbt_final/model"
    model = GBTClassificationModel.load(model_path)

    print("Model loaded successfully.")

    # ------------------------------------------------------------
    # 4️⃣ Score Full Dataset
    # ------------------------------------------------------------
    scored_df = model.transform(df)

    scored_df = scored_df.withColumn(
        "risk_score",
        vector_to_array(col("probability"))[1]
    )

    # ------------------------------------------------------------
    # 5️⃣ Create Risk Buckets
    # ------------------------------------------------------------
    scored_df = scored_df.withColumn(
        "risk_bucket",
        F.when(col("risk_score") > 0.7, "High")
         .when(col("risk_score") > 0.4, "Medium")
         .otherwise("Low")
    )

    # ------------------------------------------------------------
    # 6️⃣ Add Metadata Columns
    # ------------------------------------------------------------
    scored_df = scored_df.withColumn("model_version", lit("v3_gbt_final")) \
                         .withColumn("scoring_timestamp", current_timestamp())

    # ------------------------------------------------------------
    # 7️⃣ Select Final Gold Schema
    # ------------------------------------------------------------
    final_df = scored_df.select(
        "fraud_flag",
        "amount",
        "risk_score",
        "risk_bucket",
        "model_version",
        "scoring_timestamp"
    )

    # ------------------------------------------------------------
    # 8️⃣ Write Gold Table
    # ------------------------------------------------------------
    gold_path = "/opt/airflow/data/gold/fraud_scoring_v1/"

    final_df.write.mode("overwrite").parquet(gold_path)

    print("Gold fraud scoring table written successfully.")
    print("Location:", gold_path)

    spark.stop()