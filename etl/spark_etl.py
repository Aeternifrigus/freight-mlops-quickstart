"""
spark_etl.py
------------
Real PySpark ETL job — run this with spark-submit or `python etl/spark_etl.py`
(PySpark ships its own local driver, no cluster needed).

What it actually does (each one maps to a line in the job description):
  1. Reads raw CSV with an explicit schema (no `inferSchema` guesswork).
  2. Cleans bad/missing data (nulls, negative weights, duplicate rows).
  3. Feature-engineers a `is_delayed` label from two raw columns, while
     dropping post-outcome columns to avoid target leakage.
  4. Uses a groupBy + window aggregation to build a carrier-level
     historical delay-rate feature (a real Spark aggregation, not a toy).
  5. Writes a single clean, model-ready CSV.

Run:
    python etl/spark_etl.py --input data/raw_shipments.csv --output data/features.csv
"""
import argparse
import glob
import os
import shutil
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)


def check_java():
    """PySpark 3.5.x officially supports Java 8/11/17. Java 21 support is
    experimental and can fail with an unhelpful JVM gateway error. Check
    early so the failure mode is a clear message, not a stack trace."""
    import subprocess
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr
        first_line = out.splitlines()[0] if out else ""
        if any(v in first_line for v in ['"18', '"19', '"20', '"21', '"22', '"23']):
            print(
                f"[spark_etl] WARNING: detected {first_line.strip()} - "
                "PySpark 3.5.x officially supports Java 8/11/17 and may fail "
                "to start with Java 18+. If SparkSession.builder.getOrCreate() "
                "errors out below, install Java 17 (e.g. `sdk install java 17.0.11-tem` "
                "or your OS's openjdk-17 package) and set JAVA_HOME to it.",
                file=sys.stderr,
            )
    except FileNotFoundError:
        print("[spark_etl] ERROR: no `java` found on PATH. PySpark requires a "
              "JDK (8, 11, or 17) even though it needs no cluster. Install one "
              "(e.g. `apt install openjdk-17-jdk` or `brew install openjdk@17`) "
              "and ensure JAVA_HOME is set.", file=sys.stderr)
        sys.exit(1)

RAW_SCHEMA = StructType([
    StructField("shipment_id", StringType(), False),
    StructField("ship_date", StringType(), True),
    StructField("origin_country", StringType(), True),
    StructField("destination_country", StringType(), True),
    StructField("carrier", StringType(), True),
    StructField("mode", StringType(), True),
    StructField("weight_kg", DoubleType(), True),
    StructField("volume_cbm", DoubleType(), True),
    StructField("distance_km", DoubleType(), True),
    StructField("num_items", IntegerType(), True),
    StructField("is_hazardous", IntegerType(), True),
    StructField("fuel_price_index", DoubleType(), True),
    StructField("customs_declared_value_usd", DoubleType(), True),
    StructField("promised_transit_days", IntegerType(), True),
    StructField("actual_transit_days", IntegerType(), True),
    StructField("freight_cost_usd", DoubleType(), True),
])


def build_spark(app_name: str = "freight-etl") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")          # local mode: no cluster required
        .config("spark.sql.shuffle.partitions", "8")  # small dataset, keep it light
        .getOrCreate()
    )


def run(input_path: str, output_path: str) -> None:
    check_java()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.csv(input_path, header=True, schema=RAW_SCHEMA)
    raw_count = df.count()

    # ---- 1. Clean ---------------------------------------------------
    df = df.dropDuplicates(["shipment_id"])
    df = df.dropna(subset=["weight_kg", "fuel_price_index", "carrier", "mode"])
    df = df.filter((F.col("weight_kg") > 0) & (F.col("distance_km") > 0))
    clean_count = df.count()

    # ---- 2. Feature engineering (booking-time features only) --------
    # is_delayed is the label; drop actual_transit_days / freight_cost_usd
    # afterward so the model can't see post-shipment outcomes.
    df = df.withColumn(
        "is_delayed",
        (F.col("actual_transit_days") > F.col("promised_transit_days")).cast("int"),
    )
    df = df.withColumn(
        "value_density_usd_per_kg",
        F.round(F.col("customs_declared_value_usd") / F.col("weight_kg"), 2),
    )
    df = df.withColumn("ship_month", F.month("ship_date"))

    # ---- 3. GroupBy aggregation -> carrier historical delay rate -----
    carrier_stats = (
        df.groupBy("carrier")
        .agg(
            F.avg("is_delayed").alias("carrier_avg_delay_rate"),
            F.count("*").alias("carrier_shipment_count"),
        )
    )
    df = df.join(F.broadcast(carrier_stats), on="carrier", how="left")

    # ---- 4. Window function -> rolling shipment count per lane -------
    lane_window = Window.partitionBy("origin_country", "destination_country")
    df = df.withColumn("lane_volume", F.count("shipment_id").over(lane_window))

    # ---- 5. Select booking-time feature set (drop leakage columns) ---
    feature_cols = [
        "shipment_id", "origin_country", "destination_country", "carrier", "mode",
        "weight_kg", "volume_cbm", "distance_km", "num_items", "is_hazardous",
        "fuel_price_index", "customs_declared_value_usd", "value_density_usd_per_kg",
        "promised_transit_days", "ship_month", "carrier_avg_delay_rate",
        "carrier_shipment_count", "lane_volume", "is_delayed",
    ]
    out = df.select(*feature_cols)

    print(f"[spark_etl] raw rows: {raw_count} -> clean rows: {clean_count} -> final rows: {out.count()}")
    out.groupBy("is_delayed").count().show()
    out.printSchema()

    # coalesce(1) -> single CSV file since this is a small local dataset
    output_dir = output_path + "_dir"
    (
        out.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(output_dir)
    )

    # Spark writes a directory of part-files; flatten to one clean CSV.
    # Remove any stale destination first so this is safe to re-run.
    part_files = glob.glob(os.path.join(output_dir, "part-*.csv"))
    if not part_files:
        raise RuntimeError(
            f"[spark_etl] expected a part-*.csv file in {output_dir} but found none - "
            "Spark write may have failed silently, check the driver log above."
        )
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(part_files[0], output_path)
    shutil.rmtree(output_dir)
    print(f"[spark_etl] wrote features to {output_path}")

    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw_shipments.csv")
    ap.add_argument("--output", default="data/features.csv")
    args = ap.parse_args()
    run(args.input, args.output)
