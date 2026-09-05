"""
test_local.py
--------------
Proves the Lambda handler works BEFORE you spend time deploying it.
Simulates both a Lambda-console test event and an API-Gateway-style
HTTP event.

Run:
    python serve/test_local.py
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

bad_event = {"origin_country": "PL"}  # missing required fields -> should 400

if __name__ == "__main__":
    print("=== Test 1: direct console-style invoke (high-risk lane) ===")
    print(json.dumps(handler(console_event), indent=2))

    print("\n=== Test 2: API Gateway proxy-style invoke (low-risk lane) ===")
    print(json.dumps(handler(api_gateway_event), indent=2))

    print("\n=== Test 3: malformed request (should return 400) ===")
    print(json.dumps(handler(bad_event), indent=2))
