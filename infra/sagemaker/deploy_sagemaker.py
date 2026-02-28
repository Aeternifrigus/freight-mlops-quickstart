"""
Package the model as model.tar.gz, upload it to S3 and deploy a real-time
endpoint on the prebuilt scikit-learn container.

Needs boto3, sagemaker and configured AWS credentials. The instance bills per
hour, so run cleanup_sagemaker.py when done.

    python infra/sagemaker/deploy_sagemaker.py \
        --bucket my-freight-mlops-bucket \
        --role-arn arn:aws:iam::<account-id>:role/SageMakerExecutionRole
"""
import argparse
import shutil
import tarfile
from pathlib import Path


def build_model_tarball(out_path: str):
    """Layout the sklearn container expects:
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
    s3_uri = sm_session.upload_data(
        tarball_path, bucket=args.bucket, key_prefix=s3_key.rsplit("/", 1)[0]
    )
    print(f"[sagemaker] uploaded model artifact to {s3_uri}")

    model = SKLearnModel(
        model_data=s3_uri,
        role=args.role_arn,
        entry_point="inference.py",
        source_dir=None,  # inference.py is already under code/ in the tarball
        framework_version="1.2-1",
        sagemaker_session=sm_session,
    )

    print(f"[sagemaker] deploying endpoint '{args.endpoint_name}' on {args.instance_type} ...")
    model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )
    print(f"[sagemaker] endpoint live: {args.endpoint_name}")
    print("[sagemaker] call it with the sagemaker-runtime client")
    print("[sagemaker] run cleanup_sagemaker.py when done, the endpoint bills per hour")


if __name__ == "__main__":
    main()
