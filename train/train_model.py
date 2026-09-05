"""
train_model.py
---------------
Reads the feature table produced by etl/spark_etl.py, trains a shipment
delay-risk classifier, evaluates it, and saves a deployable artifact
(model + encoder + feature schema bundled together) for the Lambda /
SageMaker serving step.

Usage:
    python train/train_model.py --input data/features.csv --out train/model_bundle.joblib
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["origin_country", "destination_country", "carrier", "mode"]
NUMERIC = [
    "weight_kg", "volume_cbm", "distance_km", "num_items", "is_hazardous",
    "fuel_price_index", "customs_declared_value_usd", "value_density_usd_per_kg",
    "promised_transit_days", "ship_month", "carrier_avg_delay_rate",
    "carrier_shipment_count", "lane_volume",
]
TARGET = "is_delayed"


def try_log_mlflow(params: dict, metrics: dict):
    """Log to MLflow if it's installed; otherwise skip silently.
    Keeps the script runnable even without an MLflow server configured,
    while still demonstrating experiment-tracking practice when it is."""
    try:
        import mlflow
        with mlflow.start_run(run_name="delay-risk-rf"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        print("[train] logged run to MLflow")
    except ImportError:
        print("[train] mlflow not installed - skipping experiment tracking "
              "(pip install mlflow to enable)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/features.csv")
    ap.add_argument("--out", default="train/model_bundle.joblib")
    ap.add_argument("--metrics-out", default="train/metrics.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[CATEGORICAL])
    X_test_cat = encoder.transform(X_test[CATEGORICAL])

    X_train_final = np.hstack([X_train_cat, X_train[NUMERIC].values])
    X_test_final = np.hstack([X_test_cat, X_test[NUMERIC].values])

    # class_weight="balanced" matters here: delays are the minority class
    # (~20%) and the business cost of missing a delay is asymmetric.
    params = dict(
        n_estimators=150, max_depth=10, min_samples_leaf=8,
        class_weight="balanced", random_state=42,
    )
    model = RandomForestClassifier(**params, n_jobs=-1)
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print("[train] metrics:", json.dumps(metrics, indent=2))

    importances = sorted(
        zip(
            list(encoder.get_feature_names_out(CATEGORICAL)) + NUMERIC,
            model.feature_importances_,
        ),
        key=lambda t: -t[1],
    )[:10]
    print("[train] top-10 feature importances:")
    for name, score in importances:
        print(f"    {name:35s} {score:.4f}")

    bundle = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL,
        "numeric_cols": NUMERIC,
        "metrics": metrics,
    }
    joblib.dump(bundle, args.out)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] saved model bundle to {args.out}")

    try_log_mlflow(params, metrics)


if __name__ == "__main__":
    main()
