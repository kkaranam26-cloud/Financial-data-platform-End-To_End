# ============================================================
# Fraud Detection - Final Production Model (GBT v3)
#
# Purpose:
#   1. Train final selected GBT model
#   2. Evaluate performance
#   3. Save model + metadata
#   4. Optimize fraud detection threshold
#   5. Export feature importance
#   6. Save model metadata
# ============================================================

import json
from datetime import datetime
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
        .appName("Fraud Detection - Final GBT v3")
        .getOrCreate()
    )


if __name__ == "__main__":

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------
    # 1️⃣ Load Feature Dataset
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

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    df = assembler.transform(df)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # ------------------------------------------------------------
    # 2️⃣ Train Final GBT Model
    # ------------------------------------------------------------
    final_depth = 3
    final_iter = 50

    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="fraud_flag",
        maxDepth=final_depth,
        maxIter=final_iter,
        stepSize=0.1,
        seed=42
    )

    model = gbt.fit(train_df)
    predictions = model.transform(test_df)

    predictions = predictions.withColumn(
        "fraud_probability",
        vector_to_array("probability")[1]
    )

    # ------------------------------------------------------------
    # 3️⃣ Evaluate Final Model
    # ------------------------------------------------------------
    evaluator = BinaryClassificationEvaluator(
        labelCol="fraud_flag",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    pr_auc = evaluator.evaluate(predictions)

    print("\nFinal Model Performance")
    print("PR-AUC:", pr_auc)

    # ------------------------------------------------------------
    # 4️⃣ Save Model Artifact
    # ------------------------------------------------------------
    model_path = "/opt/airflow/data/models/v3_gbt_final/model"
    model.write().overwrite().save(model_path)

    print("Model saved at:", model_path)

    # ------------------------------------------------------------
    # 5️⃣ Save Parameters
    # ------------------------------------------------------------
    parameters = {
        "model_type": "GBTClassifier",
        "maxDepth": final_depth,
        "maxIter": final_iter,
        "stepSize": 0.1,
        "seed": 42
    }

    with open("/opt/airflow/data/models/v3_gbt_final/parameters.json", "w") as f:
        json.dump(parameters, f, indent=4)

    # ------------------------------------------------------------
    # 6️⃣ Save Metrics
    # ------------------------------------------------------------
    metrics = {
        "PR_AUC": pr_auc
    }

    with open("/opt/airflow/data/models/v3_gbt_final/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # ------------------------------------------------------------
    # 7️⃣ Save Feature List
    # ------------------------------------------------------------
    with open("/opt/airflow/data/models/v3_gbt_final/feature_list.txt", "w") as f:
        for feature in feature_cols:
            f.write(feature + "\n")

    print("Metadata saved successfully.")

    # ------------------------------------------------------------
    # 8️⃣ Threshold Optimization (Fraud Detection)
    # ------------------------------------------------------------
    print("\nStarting Threshold Optimization...")

    import numpy as np
    from sklearn.metrics import precision_recall_curve

   # collect prediction results once
    pdf = predictions.select(
       "fraud_probability",
       "fraud_flag"
    ).toPandas()

    y_true = pdf["fraud_flag"]
    y_scores = pdf["fraud_probability"]

    # compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(
       y_true,
       y_scores
    )

    # compute F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    threshold_output = {
       "optimal_threshold": float(best_threshold),
       "best_f1_score": float(best_f1)
    }

    with open(
    "/opt/airflow/data/models/v3_gbt_final/optimal_threshold.json", "w"
    ) as f:
       json.dump(threshold_output, f, indent=4)

    print("Optimal Threshold:", best_threshold)
    print("Best F1 Score:", best_f1)


    # ------------------------------------------------------------
    # 9️⃣ Feature Importance Export
    # ------------------------------------------------------------
    print("\nExporting Feature Importance...")

    importances = model.featureImportances.toArray()

    importance_dict = dict(zip(feature_cols, importances.tolist()))

    with open("/opt/airflow/data/models/v3_gbt_final/feature_importance.json", "w") as f:
        json.dump(importance_dict, f, indent=4)

    print("Feature importance saved.")

    # ------------------------------------------------------------
    # 🔟 Model Metadata Logging
    # ------------------------------------------------------------
    print("\nSaving Model Metadata...")

    total_rows = df.count()
    fraud_rows = df.filter(col("fraud_flag") == 1).count()

    fraud_ratio = fraud_rows / total_rows if total_rows else 0

    metadata = {
        "model_type": "GBTClassifier",
        "training_rows": total_rows,
        "fraud_rows": fraud_rows,
        "fraud_ratio": fraud_ratio,
        "feature_count": len(feature_cols),
        "training_timestamp": datetime.utcnow().isoformat()
    }

    with open("/opt/airflow/data/models/v3_gbt_final/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Model metadata saved.")

    spark.stop()