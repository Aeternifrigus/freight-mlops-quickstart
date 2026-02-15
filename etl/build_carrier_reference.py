"""
Write the carrier and lane aggregates from the feature table to a JSON file.

A request only has the carrier name, so the serving code looks these values up
here. A feature store would replace this file.

    python etl/build_carrier_reference.py \
        --input data/features.csv --out serve/carrier_reference.json
"""
import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/features.csv")
    ap.add_argument("--out", default="serve/carrier_reference.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    stats = (
        df.groupby("carrier")
        .agg(
            carrier_avg_delay_rate=("carrier_avg_delay_rate", "first"),
            carrier_shipment_count=("carrier_shipment_count", "first"),
        )
        .to_dict(orient="index")
    )
    stats["_default"] = {
        "carrier_avg_delay_rate": float(df["carrier_avg_delay_rate"].mean()),
        "carrier_shipment_count": int(df["carrier_shipment_count"].mean()),
    }
    stats["_lane_default"] = int(df["lane_volume"].mean())

    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[build_carrier_reference] wrote {args.out}")


if __name__ == "__main__":
    main()
