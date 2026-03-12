from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, quarter, dayofweek, weekofyear
from pyspark.sql.types import IntegerType
from datetime import datetime, timedelta

def create_spark():
    return SparkSession.builder \
        .appName("Dim Date Builder") \
        .getOrCreate()

def generate_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    while start <= end:
        dates.append((start,))
        start += timedelta(days=1)

    return dates

if __name__ == "__main__":

    spark = create_spark()

    # Adjust range if needed
    date_data = generate_dates("2020-01-01", "2026-12-31")

    df = spark.createDataFrame(date_data, ["full_date"])

    df = df.withColumn("year", year(col("full_date"))) \
           .withColumn("month", month(col("full_date"))) \
           .withColumn("day", dayofmonth(col("full_date"))) \
           .withColumn("quarter", quarter(col("full_date"))) \
           .withColumn("day_of_week", dayofweek(col("full_date"))) \
           .withColumn("week_of_year", weekofyear(col("full_date"))) \
           .withColumn(
               "date_sk",
               (col("year") * 10000 + col("month") * 100 + col("day")).cast(IntegerType())
           )

    df = df.select(
        "date_sk",
        "full_date",
        "year",
        "month",
        "day",
        "quarter",
        "day_of_week",
        "week_of_year"
    )

    df.write \
        .mode("overwrite") \
        .parquet("data/gold/dim_date/")

    spark.stop()
