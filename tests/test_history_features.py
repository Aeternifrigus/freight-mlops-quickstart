import pandas as pd
import pytest

from history_features import apply_reference, fit_reference, time_split


def _shipments(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "ship_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "carrier": ["A", "B"] * (n // 2),
        "origin_country": ["PL"] * n,
        "destination_country": ["US", "DE"] * (n // 2),
        "is_delayed": [1, 0, 0, 0] * (n // 4),
    })


def test_time_split_puts_newest_rows_in_test():
    train, test, cutoff = time_split(_shipments())
    assert pd.to_datetime(train["ship_date"]).max() <= cutoff
    assert pd.to_datetime(test["ship_date"]).min() > cutoff
    assert len(train) + len(test) == 100


def test_test_labels_do_not_change_any_feature():
    """Regression: the carrier delay rate used to be computed over all rows,
    so the test set's own labels were part of its features."""
    df = _shipments()
    train, test, _ = time_split(df)

    flipped = test.copy()
    flipped["is_delayed"] = 1 - flipped["is_delayed"]

    ref = fit_reference(train)
    cols = ["carrier_avg_delay_rate", "carrier_shipment_count", "lane_volume"]
    pd.testing.assert_frame_equal(
        apply_reference(test, ref)[cols], apply_reference(flipped, ref)[cols]
    )


def test_reference_uses_training_rows_only():
    train, _, _ = time_split(_shipments())
    ref = fit_reference(train)
    a = train[train["carrier"] == "A"]
    assert ref["carriers"]["A"]["carrier_avg_delay_rate"] == pytest.approx(a["is_delayed"].mean())
    assert ref["carriers"]["A"]["carrier_shipment_count"] == len(a)
    assert ref["lanes"]["PL|US"] == int((train["destination_country"] == "US").sum())


def test_unseen_carrier_and_lane_get_defaults():
    train, _, _ = time_split(_shipments())
    ref = fit_reference(train)
    new = pd.DataFrame({"carrier": ["Z"], "origin_country": ["CN"], "destination_country": ["IN"]})
    row = apply_reference(new, ref).iloc[0]
    assert row["carrier_avg_delay_rate"] == pytest.approx(train["is_delayed"].mean())
    assert row["carrier_shipment_count"] == 0
    assert row["lane_volume"] == 0


def test_reference_survives_a_json_round_trip():
    """Serving reads the reference from JSON, so it must hold plain types."""
    import json

    train, test, _ = time_split(_shipments())
    ref = fit_reference(train)
    loaded = json.loads(json.dumps(ref))
    pd.testing.assert_frame_equal(apply_reference(test, ref), apply_reference(test, loaded))
