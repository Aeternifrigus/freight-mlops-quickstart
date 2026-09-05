"""
deploy_sagemaker.py
--------------------
Packages the trained model as a SageMaker-compatible model.tar.gz,
uploads it to S3, and deploys a real-time inference endpoint using the
prebuilt scikit-learn container (no custom Docker image needed here -
that's the Lambda path instead).

This is the "touch SageMaker" step for the job requirement. It uses a
real ml.t2.medium or ml.m5.large instance - check current AWS Free Tier
terms for your account before running, and ALWAYS run the cleanup step
below when you're done to avoid ongoing endpoint charges.

Prereqs:
    pip install boto3 sagemaker
    aws configure   (with a real AWS account)

Usage:
    python infra/sagemaker/deploy_sagemaker.py \
        --bucket my-freight-mlops-bucket \
        --role-arn arn:aws:iam::<account-id>:role/SageMakerExecutionRole
"""
import argparse
import shutil
import tarfile
from pathlib import Path


def build_model_tarball(out_path: str):
    """SageMaker's sklearn container expects:
    model.tar.gz
    ├── model_bundle.joblib
    ├── carrier_reference.json
    └── code/
        └── inference.py
    """
    staging = Path("infra/sagemaker/_staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "code").mkdir(parents=True)

    shutil.copy("train/model_bundle.joblib", staging / "model_bundle.joblib")
    shutil.copy("serve/carrier_reference.json", staging / "carrier_reference.json")
    shutil.copy("infra/sagemaker/inference.py", staging / "code" / "inference.py")

    with tarfile.open(out_path, "w:gz") as tar:
        for item in staging.iterdir():
            tar.add(item, arcname=item.name)
    shutil.rmtree(staging)
    print(f"[sagemaker] built {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="S3 bucket to upload the model artifact to")
    ap.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--endpoint-name", default="freight-delay-predictor")
    ap.add_argument("--instance-type", default="ml.t2.medium")
    ap.add_argument("--region", default="eu-central-1")
    args = ap.parse_args()

    import boto3
    import sagemaker
    from sagemaker.sklearn.model import SKLearnModel

    tarball_path = "infra/sagemaker/model.tar.gz"
    build_model_tarball(tarball_path)

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    s3_key = f"freight-mlops/{args.endpoint_name}/model.tar.gz"
    s3_uri = sm_session.upload_data(tarball_path, bucket=args.bucket, key_prefix=s3_key.rsplit("/", 1)[0])
    print(f"[sagemaker] uploaded model artifact to {s3_uri}")

    model = SKLearnModel(
        model_data=s3_uri,
        role=args.role_arn,
        entry_point="inference.py",
        source_dir=None,  # inference.py already bundled under code/ in the tarball
        framework_version="1.2-1",
        sagemaker_session=sm_session,
    )

    print(f"[sagemaker] deploying endpoint '{args.endpoint_name}' on {args.instance_type} ...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )
    print(f"[sagemaker] endpoint live: {args.endpoint_name}")
    print("[sagemaker] test with predictor.predict({...}) or the boto3 sagemaker-runtime client.")
    print("[sagemaker] IMPORTANT: run cleanup_sagemaker.py when done to avoid ongoing charges.")


if __name__ == "__main__":
    main()
