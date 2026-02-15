"""
Call the Lambda handler locally with a direct event, a proxy event and a bad
request.

    cd serve && python test_local.py
"""
import json

from lambda_handler import handler

console_event = {
    "origin_country": "PL", "destination_country": "US", "carrier": "CMA-CGM",
    "mode": "Sea", "weight_kg": 1200.0, "volume_cbm": 6.5, "distance_km": 9000,
    "num_items": 40, "is_hazardous": 0, "fuel_price_index": 102.5,
    "customs_declared_value_usd": 25000.0, "promised_transit_days": 25,
    "ship_month": 6,
}

api_gateway_event = {
    "httpMethod": "POST",
    "path": "/predict",
    "body": json.dumps({
        "origin_country": "DE", "destination_country": "PL", "carrier": "Expeditors",
        "mode": "Road", "weight_kg": 300.0, "volume_cbm": 1.5, "distance_km": 900,
        "num_items": 12, "is_hazardous": 0, "fuel_price_index": 95.0,
        "customs_declared_value_usd": 8000.0, "promised_transit_days": 6,
        "ship_month": 3,
    }),
}

bad_event = {"origin_country": "PL"}

if __name__ == "__main__":
    print("=== direct invoke (high-risk lane) ===")
    r = handler(console_event)
    print(json.dumps(r, indent=2))
    assert r["statusCode"] == 200

    print("\n=== proxy event (low-risk lane) ===")
    r = handler(api_gateway_event)
    print(json.dumps(r, indent=2))
    assert r["statusCode"] == 200

    print("\n=== missing fields ===")
    r = handler(bad_event)
    print(json.dumps(r, indent=2))
    assert r["statusCode"] == 400
