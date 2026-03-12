# ============================================================
# Fraud Detection - Logistic Regression Baseline (Structured)
# Purpose:
#   1. Train weighted Logistic Regression
#   2. Evaluate using fraud-focused metrics (PR-AUC, Top 1%)
#   3. Serve as baseline for GBT comparison
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


# ------------------------------------------------------------
# Spark Session
# ------------------------------------------------------------
def create_spark():
    return (
        SparkSession.builder
        .appName("Fraud Detection - Logistic Baseline")
        .getOrCreate()
    )


if __name__ == "__main__":

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------
    # 1️⃣ Load Feature Dataset
    # ------------------------------------------------------------
    df = spark.read.parquet("data/features/fraud_training_dataset/")

    print("Class Distribution:")
    df.groupBy("fraud_flag").count().show()

    # ------------------------------------------------------------
    # 2️⃣ Feature Columns
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
        "fraud_ratio"
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
    # 4️⃣ Handle Class Imbalance Using Weights
    # ------------------------------------------------------------
    fraud_count = df.filter(col("fraud_flag") == 1).count()
    nonfraud_count = df.filter(col("fraud_flag") == 0).count()

    total = fraud_count + nonfraud_count

    fraud_weight = total / (2 * fraud_count)
    nonfraud_weight = total / (2 * nonfraud_count)

    df = df.withColumn(
        "classWeightCol",
        when(col("fraud_flag") == 1, fraud_weight).otherwise(nonfraud_weight)
    )

    # ------------------------------------------------------------
    # 5️⃣ Train-Test Split
    # ------------------------------------------------------------
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # ------------------------------------------------------------
    # 6️⃣ Train Logistic Regression Model
    # ------------------------------------------------------------
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="fraud_flag",
        weightCol="classWeightCol",
        maxIter=100
    )

    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    # ------------------------------------------------------------
    # 7️⃣ Evaluation (Fraud-Focused Metrics)
    # ------------------------------------------------------------

    # --- PR-AUC (Primary Metric for Imbalanced Fraud Data)
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

    print("\nLogistic Regression Metrics")
    print("PR-AUC:", pr_auc)
    print("Top 1% Capture:", top_1pct_capture)
    print("Precision:", precision)
    print("Recall:", recall)

    spark.stop()