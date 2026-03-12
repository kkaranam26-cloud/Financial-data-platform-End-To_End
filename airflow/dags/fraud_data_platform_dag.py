from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="fraud_data_platform",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["data_platform"]
) as dag:

    # Test task
    test_task = BashOperator(
        task_id="test_task",
        bash_command="echo Airflow is working"
    )

    # Bronze ingestion
    bronze = BashOperator(
        task_id="bronze_ingestion",
        bash_command="python /opt/airflow/src/jobs/bronze_ingestion.py"
    )

    # Silver layer
    silver_transactions = BashOperator(
        task_id="silver_transactions",
        bash_command="python /opt/airflow/src/jobs/silver_transactions.py"
    )

    # Fact table
    fact_transactions = BashOperator(
        task_id="fact_transactions",
        bash_command="python /opt/airflow/src/jobs/fact_transactions.py"
    )

    # Gold aggregates
    gold_aggregates = BashOperator(
        task_id="gold_daily_summary",
        bash_command="python /opt/airflow/src/jobs/gold_daily_transaction_summary.py"
    )


    # data quality check
    data_quality_check = BashOperator(
       task_id="data_quality_check",
       bash_command="spark-submit /opt/airflow/src/jobs/data_quality_check.py"
    )

    # Feature engineering
    features = BashOperator(
        task_id="feature_engineering",
        bash_command="python /opt/airflow/src/jobs/feature_transaction_rolling_7d.py"
    )

    # Model training
    model = BashOperator(
        task_id="fraud_model_training",
        bash_command="python /opt/airflow/src/jobs/fraud_model_gbt_final_v3.py"
    )

    # Fraud scoring
    scoring = BashOperator(
        task_id="fraud_scoring",
        bash_command="python /opt/airflow/src/jobs/fraud_scoring_gold_v1.py"
    )

    # Pipeline order
    test_task >> bronze >> silver_transactions >> fact_transactions >> data_quality_check >> gold_aggregates >> features >> model >> scoring