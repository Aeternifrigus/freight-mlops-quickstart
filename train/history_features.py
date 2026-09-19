"""
Features computed from shipment history: carrier delay rate, carrier volume
and lane volume.

These are fitted on the training split only and then looked up for every
other row, the same way serving looks them up. Computing them in the ETL over
the whole table would put test-set labels (and future volumes) into the
features the test set is scored on.
"""
import pandas as pd

LANE_SEP = "|"


def lane_key(origin: str, destination: str) -> str:
    return f"{origin}{LANE_SEP}{destination}"


def time_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Oldest rows train, newest rows test. Returns (train, test, cutoff)."""
    dates = pd.to_datetime(df["ship_date"])
    cutoff = dates.quantile(1 - test_fraction)
    train = df[dates <= cutoff]
    test = df[dates > cutoff]
    if train.empty or test.empty:
        raise ValueError("time split left an empty train or test set")
    return train, test, cutoff


def fit_reference(train: pd.DataFrame) -> dict:
    """History aggregates from training rows only, in the JSON shape serving reads."""
    carriers = (
        train.groupby("carrier")["is_delayed"]
        .agg(carrier_avg_delay_rate="mean", carrier_shipment_count="size")
    )
    lanes = train.groupby(["origin_country", "destination_country"]).size()
    return {
        "carriers": {
            name: {
                "carrier_avg_delay_rate": float(row.carrier_avg_delay_rate),
                "carrier_shipment_count": int(row.carrier_shipment_count),
            }
            for name, row in carriers.iterrows()
        },
        "lanes": {lane_key(o, d): int(n) for (o, d), n in lanes.items()},
        # An unseen carrier gets the overall rate and no volume; an unseen lane
        # has no history, so its volume is 0.
        "_default_carrier": {
            "carrier_avg_delay_rate": float(train["is_delayed"].mean()),
            "carrier_shipment_count": 0,
        },
        "_default_lane_volume": 0,
    }


def apply_reference(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Add the history features to any rows, using a fitted reference."""
    df = df.copy()
    default = ref["_default_carrier"]
    carrier_ref = [ref["carriers"].get(c, default) for c in df["carrier"]]
    df["carrier_avg_delay_rate"] = [r["carrier_avg_delay_rate"] for r in carrier_ref]
    df["carrier_shipment_count"] = [r["carrier_shipment_count"] for r in carrier_ref]
    df["lane_volume"] = [
        ref["lanes"].get(lane_key(o, d), ref["_default_lane_volume"])
        for o, d in zip(df["origin_country"], df["destination_country"], strict=True)
    ]
    return df
