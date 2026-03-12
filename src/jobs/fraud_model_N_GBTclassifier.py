# ============================================================
# Fraud Detection - Gradient Boosted Trees (Structured)
# Purpose:
#   1. Train GBT model
#   2. Evaluate using fraud-focused metrics
#   3. Compare against Logistic baseline
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


# ------------------------------------------------------------
# Spark Session
# ------------------------------------------------------------
def create_spark():
    return (
        SparkSession.builder
        .appName("Fraud Detection - GBT Model")
        .getOrCreate()
    )


if __name__ == "__main__":

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------
    # 1️⃣ Load Feature Dataset (Enhanced v2)
    # ------------------------------------------------------------
    df = spark.read.parquet(
        "/mnt/c/users/karan/finP_M_Auto/data/features/fraud_training_dataset_v2/"
    )

    # ------------------------------------------------------------
    # 2️⃣ Feature Columns (Extended Feature Set)
    # ------------------------------------------------------------
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
    # 3️⃣ Assemble Feature Vector
    # ------------------------------------------------------------
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    df = assembler.transform(df)

    # ------------------------------------------------------------
    # 4️⃣ Train-Test Split
    # ------------------------------------------------------------
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # ------------------------------------------------------------
    # 5️⃣ Train GBT Model
    # ------------------------------------------------------------
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="fraud_flag",
        maxDepth=6,
        maxIter=100,
        stepSize=0.1,
        seed=42
    )

    model = gbt.fit(train_df)
    predictions = model.transform(test_df)

    # ------------------------------------------------------------
    # 6️⃣ Evaluation (Fraud-Focused Metrics)
    # ------------------------------------------------------------

    # --- PR-AUC
    evaluator = BinaryClassificationEvaluator(
        labelCol="fraud_flag",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    pr_auc = evaluator.evaluate(predictions)

    # --- Extract Fraud Probability
    scored_df = predictions.withColumn(
        "fraud_probability",
         vector_to_array(col("probability"))[1]
    )

    total_fraud = scored_df.filter(col("fraud_flag") == 1).count()
    total_rows = scored_df.count()

    # --- Top 1% Fraud Capture
    top_n = int(total_rows * 0.01)
    top_df = scored_df.orderBy(col("fraud_probability").desc()).limit(top_n)

    fraud_captured = top_df.filter(col("fraud_flag") == 1).count()
    top_1pct_capture = fraud_captured / total_fraud if total_fraud > 0 else 0

    # --- Precision & Recall @ 0.5 Threshold
    threshold_df = scored_df.withColumn(
        "prediction_label",
        F.when(col("fraud_probability") > 0.5, 1).otherwise(0)
    )

    tp = threshold_df.filter((col("prediction_label") == 1) & (col("fraud_flag") == 1)).count()
    fp = threshold_df.filter((col("prediction_label") == 1) & (col("fraud_flag") == 0)).count()
    fn = threshold_df.filter((col("prediction_label") == 0) & (col("fraud_flag") == 1)).count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\nGBT Metrics")
    print("PR-AUC:", pr_auc)
    print("Top 1% Capture:", top_1pct_capture)
    print("Precision:", precision)
    print("Recall:", recall)

    spark.stop()