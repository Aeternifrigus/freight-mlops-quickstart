"""
generate_data.py
-----------------
Creates a synthetic freight-shipment dataset so the pipeline is fully
self-contained and runnable without any external data source.

The data is intentionally messy (some nulls, a few bad rows) so the
PySpark ETL step has real cleaning work to do, not just a pass-through.

Usage:
    python data/generate_data.py --rows 20000 --out data/raw_shipments.csv
"""
import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
random.seed(42)

ORIGINS = ["PL", "DE", "NL", "CN", "US", "IN", "VN", "TR", "GB", "FR"]
DESTS = ["PL", "DE", "NL", "CN", "US", "IN", "VN", "TR", "GB", "FR"]
CARRIERS = ["MaerskX", "DHL-Freight", "Kuehne+Nagel", "DB Schenker", "CMA-CGM", "Expeditors"]
MODES = ["Sea", "Air", "Road", "Rail"]

# Each carrier gets a hidden "reliability" factor that drives delay
# probability. The model has to discover this from the data.
CARRIER_RELIABILITY = {c: RNG.uniform(0.55, 0.97) for c in CARRIERS}
MODE_BASE_TRANSIT = {"Sea": 28, "Air": 4, "Road": 6, "Rail": 12}
MODE_VARIANCE = {"Sea": 1.5, "Air": 0.5, "Road": 0.8, "Rail": 1.0}


def gen_row(i: int, start_date: datetime) -> dict:
    origin = random.choice(ORIGINS)
    dest = random.choice([d for d in DESTS if d != origin])
    carrier = random.choice(CARRIERS)
    mode = random.choice(MODES)

    weight_kg = round(float(RNG.gamma(shape=2.0, scale=800)), 1)
    volume_cbm = round(weight_kg / RNG.uniform(150, 400), 2)  # rough density
    distance_km = round(float(RNG.uniform(300, 19000)), 1)
    num_items = int(RNG.integers(1, 500))
    is_hazardous = int(RNG.random() < 0.06)
    fuel_price_index = round(float(RNG.normal(100, 12)), 1)
    customs_declared_value_usd = round(weight_kg * RNG.uniform(4, 60), 2)

    base_transit = MODE_BASE_TRANSIT[mode] + distance_km / 4000
    promised_transit_days = max(1, round(base_transit + RNG.normal(0, 1)))

    reliability = CARRIER_RELIABILITY[carrier]
    # hazardous + long distance + bad-weather noise reduces reliability
    delay_prob = (1 - reliability) + (0.08 if is_hazardous else 0) + (
        0.10 if mode == "Sea" and distance_km > 12000 else 0
    )
    delayed = RNG.random() < min(delay_prob, 0.95)
    # non-delayed shipments arrive on or slightly before schedule;
    # delayed ones slip by a few days. Small mode-dependent noise on top.
    if delayed:
        actual_transit_days = max(1, round(
            promised_transit_days + RNG.uniform(2, 7) + RNG.normal(0, MODE_VARIANCE[mode])
        ))
    else:
        actual_transit_days = max(1, round(
            promised_transit_days - RNG.uniform(0, 1.5) + RNG.normal(0, MODE_VARIANCE[mode] * 0.4)
        ))

    freight_cost_usd = round(
        50
        + weight_kg * RNG.uniform(0.8, 2.2)
        + distance_km * RNG.uniform(0.05, 0.18)
        + (200 if is_hazardous else 0)
        + (fuel_price_index - 100) * 3
        + RNG.normal(0, 40),
        2,
    )

    ship_date = start_date + timedelta(days=int(RNG.integers(0, 540)))

    row = {
        "shipment_id": f"SHP{i:07d}",
        "ship_date": ship_date.strftime("%Y-%m-%d"),
        "origin_country": origin,
        "destination_country": dest,
        "carrier": carrier,
        "mode": mode,
        "weight_kg": weight_kg,
        "volume_cbm": volume_cbm,
        "distance_km": distance_km,
        "num_items": num_items,
        "is_hazardous": is_hazardous,
        "fuel_price_index": fuel_price_index,
        "customs_declared_value_usd": customs_declared_value_usd,
        "promised_transit_days": promised_transit_days,
        "actual_transit_days": actual_transit_days,
        "freight_cost_usd": freight_cost_usd,
    }
    return row


def dirty(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic messiness: a few nulls, a few bad values, dupes."""
    n = len(df)
    idx_null_weight = RNG.choice(n, size=max(1, n // 400), replace=False)
    df.loc[idx_null_weight, "weight_kg"] = np.nan

    idx_null_fuel = RNG.choice(n, size=max(1, n // 500), replace=False)
    df.loc[idx_null_fuel, "fuel_price_index"] = np.nan

    idx_bad_weight = RNG.choice(n, size=max(1, n // 800), replace=False)
    df.loc[idx_bad_weight, "weight_kg"] = -1.0  # sensor/data-entry error

    # duplicate a handful of rows (common in real ingestion)
    dupes = df.sample(n=max(1, n // 1000), random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--out", type=str, default="data/raw_shipments.csv")
    args = ap.parse_args()

    start_date = datetime(2024, 1, 1)
    rows = [gen_row(i, start_date) for i in range(args.rows)]
    df = pd.DataFrame(rows)
    df = dirty(df)
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)  # shuffle

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
