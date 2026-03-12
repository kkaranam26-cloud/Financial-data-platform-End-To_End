from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def create_spark():
    return (
        SparkSession.builder
        .appName("Fraud Detection - Industry Grade v2")
        .getOrCreate()
    )


if __name__ == "__main__":

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("Loading v2 dataset...")
    df = spark.read.parquet("/mnt/c/users/karan/finP_M_Auto/data/features/fraud_training_dataset_v2/")

    print("Class Distribution:")
    df.groupBy("fraud_flag").count().show()

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

    # Stratified split
    train_df = df.sampleBy("fraud_flag", fractions={0: 0.8, 1: 0.8}, seed=42)
    test_df = df.subtract(train_df)

    # Downsample majority class
    fraud_train = train_df.filter(col("fraud_flag") == 1)
    nonfraud_train = train_df.filter(col("fraud_flag") == 0)

    fraud_count = fraud_train.count()
    nonfraud_count = nonfraud_train.count()

    ratio = fraud_count / nonfraud_count
    nonfraud_sampled = nonfraud_train.sample(False, min(1.0, ratio * 3), seed=42)

    balanced_train = fraud_train.union(nonfraud_sampled)

    print("Balanced Train Distribution:")
    balanced_train.groupBy("fraud_flag").count().show()

    # Train GBT
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="fraud_flag",
        maxDepth=6,
        maxIter=100,
        stepSize=0.1,
        seed=42
    )

    model = gbt.fit(balanced_train)

    predictions = model.transform(test_df)

    # Extract fraud probability
    extract_prob = udf(lambda v: float(v[1]), DoubleType())
    predictions = predictions.withColumn(
        "fraud_probability",
        extract_prob(col("probability"))
    )

    # PR-AUC
    evaluator = BinaryClassificationEvaluator(
        labelCol="fraud_flag",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    pr_auc = evaluator.evaluate(predictions)
    print("PR-AUC:", pr_auc)

    # Top-K capture
    total_fraud = predictions.filter(col("fraud_flag") == 1).count()
    total_rows = predictions.count()

    for pct in [0.01, 0.05]:
        top_n = int(total_rows * pct)
        top_k = predictions.orderBy(col("fraud_probability").desc()).limit(top_n)
        fraud_captured = top_k.filter(col("fraud_flag") == 1).count()
        capture_rate = fraud_captured / total_fraud if total_fraud > 0 else 0
        print(f"Top {int(pct*100)}% captures {capture_rate:.4f} of fraud")

    # Feature importance
    print("\nFeature Importance:")
    importances = model.featureImportances
    for i in range(len(feature_cols)):
        print(feature_cols[i], ":", importances[i])

    spark.stop()