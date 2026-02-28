"""
Delete the endpoint, endpoint config and model created by deploy_sagemaker.py.
The endpoint bills per hour while it exists, even idle.

    python infra/sagemaker/cleanup_sagemaker.py --endpoint-name freight-delay-predictor
"""
import argparse

import boto3

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-name", default="freight-delay-predictor")
    ap.add_argument("--region", default="eu-central-1")
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)

    try:
        desc = sm.describe_endpoint(EndpointName=args.endpoint_name)
        config_name = desc["EndpointConfigName"]
        sm.delete_endpoint(EndpointName=args.endpoint_name)
        print(f"Deleted endpoint: {args.endpoint_name}")
        sm.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"Deleted endpoint config: {config_name}")
    except sm.exceptions.ClientError as e:
        print(f"Endpoint cleanup skipped/failed: {e}")

    try:
        models = sm.list_models(NameContains=args.endpoint_name)["Models"]
        for m in models:
            sm.delete_model(ModelName=m["ModelName"])
            print(f"Deleted model: {m['ModelName']}")
    except Exception as e:  # noqa: BLE001
        print(f"Model cleanup skipped/failed: {e}")
