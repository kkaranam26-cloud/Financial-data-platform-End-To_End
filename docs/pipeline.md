# Data Platform Pipeline

The Financial Data Platform is orchestrated using **Apache Airflow**, which manages the execution order of all data processing jobs.

The pipeline is designed to move data through multiple layers, gradually transforming raw transactional data into curated datasets used for analytics and machine learning.

---

# Pipeline Flow

The platform pipeline follows the sequence below:

```
bronze_ingestion
      ↓
silver_transformations
      ↓
warehouse_modeling
      ↓
gold_analytics_generation
      ↓
data_quality_validation
      ↓
feature_engineering
      ↓
machine_learning_training
      ↓
fraud_scoring
```

Each stage of the pipeline performs a specific responsibility within the data platform.

---

# Bronze Ingestion

The pipeline begins by ingesting raw transactional data into the **Bronze layer**.

Purpose of this stage:

- ingest raw data
- preserve source records
- maintain historical transaction data

Job:

```
bronze_ingestion.py
```

---

# Silver Transformations

The Silver layer cleans and standardizes the ingested data.

Responsibilities include:

- schema normalization
- data validation
- null handling
- enrichment

Jobs:

```
silver_transactions.py
silver_customers.py
```

This stage produces clean datasets used for downstream modeling.

---

# Warehouse Modeling

The warehouse stage builds dimensional tables used for analytics.

Tables created:

```
fact_transactions
dim_customer
dim_date
```

These tables support analytical queries and feature generation.

Jobs:

```
fact_transactions.py
dim_customer.py
dim_date.py
```

---

# Gold Analytics Generation

The Gold layer generates curated datasets designed for analytics and reporting.

Examples include:

```
gold_daily_transaction_summary
gold_monthly_revenue_summary
gold_customer_activity_summary
gold_fraud_rate_by_day
```

These datasets enable business analytics and reporting workloads.

---

# Data Quality Validation

Before downstream workloads execute, the platform performs data quality validation.

Validation checks include:

- dataset availability
- schema verification
- basic data integrity checks

Job:

```
data_quality_check.py
```

---

# Feature Engineering

Feature engineering pipelines generate machine learning features from transactional data.

Examples of engineered features include:

- transaction recency
- rolling spending metrics
- fraud ratios
- customer lifetime statistics

Jobs:

```
feature_transaction_recency.py
feature_transaction_rolling_7d.py
feature_customer_lifetime.py
```

---

# Machine Learning Training

Machine learning models are trained using engineered features.

Models evaluated include:

- Logistic Regression
- Gradient Boosted Trees

Primary model:

```
fraud_model_gbt_final_v3.py
```

---

# Fraud Scoring

The trained model is applied to transaction data to generate fraud predictions.

Job:

```
fraud_scoring_gold_v1.py
```

This stage produces fraud prediction outputs that can be used for monitoring and analytics.

---

# Pipeline Summary

The pipeline architecture ensures that raw data flows through multiple transformation stages, resulting in reliable datasets that support analytics and machine learning workloads.

Airflow orchestrates each stage to ensure correct execution order and pipeline reliability.
