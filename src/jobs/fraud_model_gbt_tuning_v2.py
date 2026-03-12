# ============================================================
# Fraud Detection - GBT Hyperparameter Tuning (Minimal)
# Purpose:
#   1. Test small grid of hyperparameters
#   2. Compare PR-AUC
#   3. Select best performing configuration
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
        .appName("Fraud Detection - GBT Tuning v2")
        .getOrCreate()
    )


if __name__ == "__main__":

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------
    # 1️⃣ Load Feature Dataset
    # ------------------------------------------------------------
    df = spark.read.parquet(
        "/mnt/c/users/karan/finP_M_Auto/data/features/fraud_training_dataset_v2/"
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

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    df = assembler.transform(df)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    evaluator = BinaryClassificationEvaluator(
        labelCol="fraud_flag",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    # ------------------------------------------------------------
    # 2️⃣ Hyperparameter Grid
    # ------------------------------------------------------------
    depths = [3, 5, 7]
    iters = [50, 100]

    best_pr_auc = 0
    best_config = None

    print("\nStarting Hyperparameter Tuning...\n")

    results = []

for depth in depths:
    for iteration in iters:

        print("=====================================")
        print(f"Training GBT: maxDepth={depth}, maxIter={iteration}")

        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="fraud_flag",
            maxDepth=depth,
            maxIter=iteration,
            stepSize=0.1,
            seed=42
        )

        model = gbt.fit(train_df)
        predictions = model.transform(test_df)

        pr_auc = evaluator.evaluate(predictions)

        print(f"PR-AUC: {pr_auc}")
        print("=====================================\n")

        results.append((depth, iteration, pr_auc))

# Print full summary table
print("\nALL RESULTS:")
for r in results:
    print(f"maxDepth={r[0]}, maxIter={r[1]} → PR-AUC={r[2]}")