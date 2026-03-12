# Financial Data Platform – End-to-End Data Engineering Architecture

This project implements a **financial data platform** designed to support analytics, machine learning workloads including fraud detection, and business intelligence use cases.

The platform ingests transactional data, processes it through a multi-layered data architecture, and produces curated datasets for analysts, data scientists, and downstream applications.

While the platform demonstrates a fraud detection ML workflow, the primary focus is building a **scalable data platform architecture**.

---

# Platform Architecture

The platform follows a **Bronze → Silver → Warehouse → Gold architecture**.

```
Data Sources
     ↓
Bronze Layer (Raw ingestion)
     ↓
Silver Layer (Cleaned and standardized)
     ↓
Dimensional Warehouse
     ↓
Gold Analytics Layer
     ↓
Feature Engineering
     ↓
Machine Learning Workloads
```

Consumers of the platform include:

- Business analysts
- Data scientists
- Machine learning models
- Analytics dashboards

---

# Technology Stack

Core technologies used in the platform:

- Python
- Apache Spark
- Apache Airflow
- Docker
- Parquet

The platform architecture is designed to be **cloud-deployable**.

Example cloud mapping:

| Platform Component | Cloud Equivalent |
|---|---|
Data Lake Storage | AWS S3 |
Processing Engine | Spark on EMR |
Pipeline Orchestration | AWS MWAA (Managed Airflow) |
Data Warehouse | Amazon Athena / Redshift |
Monitoring | CloudWatch |

---

# Data Platform Layers

## Bronze Layer

Raw ingestion of transactional data.

Purpose:

- Preserve raw data
- Immutable event storage
- Source of truth

Pipeline job:

```
bronze_ingestion.py
```

---

## Silver Layer

Data cleaning and normalization.

Responsibilities:

- schema standardization
- transformation
- enrichment

Jobs:

```
silver_transactions.py
silver_customers.py
```

---

## Dimensional Warehouse

The platform implements dimensional modeling for analytics.

Warehouse tables include:

```
fact_transactions
dim_customer
dim_date
```

These tables support analytical queries and business reporting.

---

## Gold Analytics Layer

Curated datasets for analytics and reporting.

Examples:

```
gold_daily_transaction_summary
gold_monthly_revenue_summary
gold_customer_activity_summary
gold_fraud_rate_by_day
```

These datasets represent the **business analytics layer**.

---

# Feature Engineering Layer

The platform generates features used for machine learning models.

Examples:

```
feature_customer_lifetime.py
feature_transaction_recency.py
feature_transaction_rolling_7d.py
```

Features include:

- customer transaction velocity
- rolling spending metrics
- fraud ratios
- recency features

---

# Machine Learning Workloads

The platform supports ML pipelines built on top of engineered features.

Example ML workload included in this project:

```
Fraud detection using Spark MLlib
```

Models evaluated:

- Logistic Regression
- Gradient Boosted Trees (GBT)

Primary production model:

```
fraud_model_gbt_final_v3.py
```

The ML pipeline performs:

- training dataset generation
- model training
- evaluation
- fraud scoring

---

# Fraud Scoring Pipeline

```
fraud_scoring_gold_v1.py
```

This job applies the trained model to transaction data and produces fraud predictions.

---

# Data Quality Validation

The platform includes automated data validation.

Job:

```
data_quality_check.py
```

These checks ensure:

- dataset integrity
- schema correctness
- availability of required tables

---

# Pipeline Orchestration

The platform is orchestrated using **Apache Airflow**.

Main DAG:

```
airflow/dags/fraud_data_platform_dag.py
```

Pipeline flow:

```
bronze_ingestion
     ↓
silver_transformations
     ↓
warehouse_tables
     ↓
gold_analytics
     ↓
data_quality_checks
     ↓
feature_engineering
     ↓
ml_training
     ↓
fraud_scoring
```

---

# Data Platform Storage Structure

The platform organizes data into logical layers:

```
BASE_PATH/
│
├── bronze/
├── silver/
├── warehouse/
├── gold/
├── features/
└── models/
```

The pipeline uses a configurable **base path**, allowing deployment across different environments such as local storage or cloud object storage.

---

# Example Analytics Queries

Example warehouse query:

```sql
SELECT
    transaction_date,
    SUM(total_amount) AS daily_revenue
FROM gold_daily_transaction_summary
GROUP BY transaction_date
ORDER BY transaction_date;
```

Example customer analytics:

```sql
SELECT
    customer_id,
    SUM(amount) AS total_spent
FROM fact_transactions
GROUP BY customer_id
ORDER BY total_spent DESC
LIMIT 10;
```

These queries demonstrate how analysts interact with the platform’s warehouse layer.

---

# Repository Structure

```
airflow/
    dags/

src/
    jobs/

docs/

README.md
```

---

# Author

Chaitanya Karanam
