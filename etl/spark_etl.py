"""
PySpark ETL: raw shipments CSV to a model-ready feature table.

Reads with an explicit schema, drops duplicates and bad rows, derives the
is_delayed label, adds carrier and lane aggregates, and drops columns that are
only known after delivery.

    python etl/spark_etl.py --input data/raw_shipments.csv --output data/features.csv

Runs in local mode, so plain `python` works as well as spark-submit.
"""
import argparse
import glob
import os
import shutil

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


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
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def run(input_path: str, output_path: str) -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.csv(input_path, header=True, schema=RAW_SCHEMA)
    raw_count = df.count()

    # ---- 1. Clean ---------------------------------------------------
    df = df.dropDuplicates(["shipment_id"])
    df = df.dropna(subset=["weight_kg", "fuel_price_index", "carrier", "mode"])
    df = df.filter((F.col("weight_kg") > 0) & (F.col("distance_km") > 0))
    clean_count = df.count()

    # ---- 2. Label and booking-time features --------------------------
    # actual_transit_days and freight_cost_usd are dropped in step 5
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

    # ---- 4. Window function -> shipment count per lane ---------------
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

    print(f"[spark_etl] raw rows: {raw_count} -> clean rows: {clean_count} "
          f"-> final rows: {out.count()}")
    out.groupBy("is_delayed").count().show()
    out.printSchema()

    # small dataset, so write a single part file
    output_dir = output_path + "_dir"
    (
        out.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(output_dir)
    )

    # move the part file to output_path and remove Spark's directory
    part_files = glob.glob(os.path.join(output_dir, "part-*.csv"))
    if not part_files:
        raise RuntimeError(
            f"[spark_etl] no part-*.csv in {output_dir}, check the driver log above"
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
