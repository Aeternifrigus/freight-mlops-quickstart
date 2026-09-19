"""
Train the delay classifier on the ETL feature table and save the model, encoder
and column lists together as one bundle for serving.

The split is by ship date: the newest 20% of shipments are the test set. The
carrier and lane history features are fitted on the training rows only and
written to the carrier reference that serving reads, so training, evaluation
and serving all see the same values.

    python train/train_model.py --input data/features.csv --out train/model_bundle.joblib \
        --reference-out serve/carrier_reference.json
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

from history_features import apply_reference, fit_reference, time_split

CATEGORICAL = ["origin_country", "destination_country", "carrier", "mode"]
NUMERIC = [
    "weight_kg", "volume_cbm", "distance_km", "num_items", "is_hazardous",
    "fuel_price_index", "customs_declared_value_usd", "value_density_usd_per_kg",
    "promised_transit_days", "ship_month", "carrier_avg_delay_rate",
    "carrier_shipment_count", "lane_volume",
]
TARGET = "is_delayed"


def try_log_mlflow(params: dict, metrics: dict):
    """Log params and metrics to MLflow if it's installed."""
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
    ap.add_argument("--reference-out", default="serve/carrier_reference.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    train_df, test_df, cutoff = time_split(df)
    reference = fit_reference(train_df)
    train_df = apply_reference(train_df, reference)
    test_df = apply_reference(test_df, reference)
    print(f"[train] cutoff {cutoff.date()}: {len(train_df)} train, {len(test_df)} test")

    X_train, y_train = train_df[CATEGORICAL + NUMERIC], train_df[TARGET]
    X_test, y_test = test_df[CATEGORICAL + NUMERIC], test_df[TARGET]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[CATEGORICAL])
    X_test_cat = encoder.transform(X_test[CATEGORICAL])

    X_train_final = np.hstack([X_train_cat, X_train[NUMERIC].values])
    X_test_final = np.hstack([X_test_cat, X_test[NUMERIC].values])

    # delays are ~20% of rows and a missed delay costs more than a false alarm
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
        "split_cutoff": str(cutoff.date()),
    }
    print("[train] metrics:", json.dumps(metrics, indent=2))

    importances = sorted(
        zip(
            list(encoder.get_feature_names_out(CATEGORICAL)) + NUMERIC,
            model.feature_importances_,
            strict=True,
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
    with open(args.reference_out, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"[train] saved model bundle to {args.out}, reference to {args.reference_out}")

    try_log_mlflow(params, metrics)


if __name__ == "__main__":
    main()
