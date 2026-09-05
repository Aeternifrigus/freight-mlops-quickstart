"""
inference.py
------------
Entry point for the SageMaker scikit-learn inference container.
Implements the four functions the container's model server calls:
model_fn, input_fn, predict_fn, output_fn. This file gets bundled
into model.tar.gz alongside model_bundle.joblib and carrier_reference.json.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    bundle = joblib.load(os.path.join(model_dir, "model_bundle.joblib"))
    with open(os.path.join(model_dir, "carrier_reference.json")) as f:
        ref = json.load(f)
    bundle["carrier_reference"] = ref
    return bundle


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(payload, bundle):
    model = bundle["model"]
    encoder = bundle["encoder"]
    cat_cols = bundle["categorical_cols"]
    num_cols = bundle["numeric_cols"]
    ref = bundle["carrier_reference"]

    carrier_ref = ref.get(payload["carrier"], ref["_default"])
    row = dict(payload)
    row["carrier_avg_delay_rate"] = carrier_ref["carrier_avg_delay_rate"]
    row["carrier_shipment_count"] = carrier_ref["carrier_shipment_count"]
    row["lane_volume"] = ref.get("_lane_default", 500)
    row["value_density_usd_per_kg"] = round(
        row["customs_declared_value_usd"] / max(row["weight_kg"], 1e-6), 2
    )

    cat_df = pd.DataFrame([[row[c] for c in cat_cols]], columns=cat_cols)
    cat_arr = encoder.transform(cat_df)
    num_arr = np.array([[row[c] for c in num_cols]])
    features = np.hstack([cat_arr, num_arr])

    proba = float(model.predict_proba(features)[0, 1])
    return {
        "is_delayed_prediction": int(proba >= 0.5),
        "delay_probability": round(proba, 4),
    }


def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
