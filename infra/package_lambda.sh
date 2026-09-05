#!/usr/bin/env bash
# Builds the Lambda container image and pushes it to ECR.
# Run this from the project root. Requires: docker, aws cli configured
# with a free-tier-eligible AWS account (aws configure).
set -euo pipefail

REGION="${AWS_REGION:-eu-central-1}"
REPO_NAME="freight-delay-predictor"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"

echo "[1/4] Ensuring ECR repo exists..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" \
  >/dev/null 2>&1 || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

echo "[2/4] Logging in to ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "[3/4] Building image..."
docker build -t "${REPO_NAME}:latest" ./serve

echo "[4/4] Tagging and pushing to ${IMAGE_URI} ..."
docker tag "${REPO_NAME}:latest" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo ""
echo "Done. Image URI for Terraform:"
echo "  ${IMAGE_URI}"
echo "Set this as the lambda_image_uri variable, then: cd infra/terraform && terraform apply"
