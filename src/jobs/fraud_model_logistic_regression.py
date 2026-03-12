from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def create_spark():
    return SparkSession.builder \
        .appName("Fraud Logistic Regression Model") \
        .getOrCreate()


if __name__ == "__main__":

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # Load feature dataset
    df = spark.read.parquet("data/features/fraud_training_dataset/")

      # 🔎 Check class distribution FIRST
    print("Class Distribution:")
    df.groupBy("fraud_flag").count().show()



    # Feature columns
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

    # Assemble features into single vector
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    df = assembler.transform(df)

        # Compute Class Weights (Dynamic)
    # -----------------------------------
    fraud_count = df.filter(col("fraud_flag") == 1).count()
    nonfraud_count = df.filter(col("fraud_flag") == 0).count()
    
    total = fraud_count + nonfraud_count

    fraud_weight = total / (2 * fraud_count)
    nonfraud_weight = total / (2 * nonfraud_count)

    df = df.withColumn(
        "classWeightCol",
        when(col("fraud_flag") == 1, fraud_weight).otherwise(nonfraud_weight)
    )


    # Split data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

   


     # Train Weighted Logistic Regression
    # -----------------------------------
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="fraud_flag",
        weightCol="classWeightCol"
    )


    model = lr.fit(train_df)

  

    # Make predictions
    predictions = model.transform(test_df)

    predictions.select("fraud_flag", "probability", "prediction").show(20, False)

    # Evaluate model
    evaluator = BinaryClassificationEvaluator(
        labelCol="fraud_flag",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = evaluator.evaluate(predictions)
    print("AUC Score:", auc)


    precision_eval = MulticlassClassificationEvaluator(
        labelCol="fraud_flag",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    recall_eval = MulticlassClassificationEvaluator(
        labelCol="fraud_flag",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    print("Precision:", precision_eval.evaluate(predictions))
    print("Recall:", recall_eval.evaluate(predictions))

    # Confusion Matrix
# -----------------------------------
    print("Confusion Matrix:")
    predictions.groupBy("fraud_flag", "prediction").count().show()

  # Save trained model
    model.write().overwrite().save("models/fraud_logistic_regression")
    print("Model saved successfully!")

    spark.stop()