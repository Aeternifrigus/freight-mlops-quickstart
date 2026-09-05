"""
lambda_handler.py
------------------
AWS Lambda entry point for real-time shipment-delay-risk scoring.

Works with two invocation styles:
  - API Gateway / Lambda Function URL proxy events (body is a JSON string)
  - Direct Lambda test-console invokes (event IS the payload dict)

Cold start loads the model bundle once at import time (module-global),
which is standard Lambda practice: subsequent warm invocations reuse it.

Expected input JSON:
{
  "origin_country": "PL", "destination_country": "US", "carrier": "MaerskX",
  "mode": "Sea", "weight_kg": 1200.0, "volume_cbm": 6.5, "distance_km": 9000,
  "num_items": 40, "is_hazardous": 0, "fuel_price_index": 102.5,
  "customs_declared_value_usd": 25000.0, "promised_transit_days": 25,
  "ship_month": 6
}
Note: carrier_avg_delay_rate / carrier_shipment_count / lane_volume are
historical aggregates computed offline in the ETL step. In production
these would be looked up from a Feature Store by carrier/lane at request
time; here they're looked up from a small bundled reference table so the
Lambda has no external dependency for this demo.
"""
import json
import os

import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_bundle.joblib")
REF_PATH = os.path.join(os.path.dirname(__file__), "carrier_reference.json")

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_encoder = _bundle["encoder"]
_categorical_cols = _bundle["categorical_cols"]
_numeric_cols = _bundle["numeric_cols"]

with open(REF_PATH) as f:
    _carrier_ref = json.load(f)

REQUIRED_FIELDS = [
    "origin_country", "destination_country", "carrier", "mode", "weight_kg",
    "volume_cbm", "distance_km", "num_items", "is_hazardous",
    "fuel_price_index", "customs_declared_value_usd", "promised_transit_days",
    "ship_month",
]


def _enrich(payload: dict) -> dict:
    """Attach the offline-computed carrier/lane aggregates and the
    derived value-density feature, mirroring the Spark ETL transform."""
    payload = dict(payload)
    ref = _carrier_ref.get(payload["carrier"], _carrier_ref["_default"])
    payload["carrier_avg_delay_rate"] = ref["carrier_avg_delay_rate"]
    payload["carrier_shipment_count"] = ref["carrier_shipment_count"]
    payload["lane_volume"] = _carrier_ref.get("_lane_default", 500)
    payload["value_density_usd_per_kg"] = round(
        payload["customs_declared_value_usd"] / max(payload["weight_kg"], 1e-6), 2
    )
    return payload


def predict(payload: dict) -> dict:
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    import pandas as pd

    row = _enrich(payload)
    cat_df = pd.DataFrame([[row[c] for c in _categorical_cols]], columns=_categorical_cols)
    cat_arr = _encoder.transform(cat_df)
    num_arr = np.array([[row[c] for c in _numeric_cols]])
    features = np.hstack([cat_arr, num_arr])

    proba = float(_model.predict_proba(features)[0, 1])
    label = int(proba >= 0.5)
    return {
        "is_delayed_prediction": label,
        "delay_probability": round(proba, 4),
        "risk_tier": "high" if proba >= 0.6 else "medium" if proba >= 0.35 else "low",
    }


def handler(event, context=None):
    try:
        if isinstance(event, dict) and "body" in event:
            body = event["body"]
            payload = json.loads(body) if isinstance(body, str) else body
        else:
            payload = event

        result = predict(payload)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }
    except ValueError as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
    except Exception as e:  # pragma: no cover - defensive
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "internal error", "detail": str(e)}),
        }
