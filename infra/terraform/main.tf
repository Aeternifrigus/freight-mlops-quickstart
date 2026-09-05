terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---- IAM: minimal execution role (CloudWatch Logs only) ----
resource "aws_iam_role" "lambda_exec" {
  name = "${var.function_name}-exec-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "basic_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ---- Lambda function (container image, built by infra/package_lambda.sh) ----
resource "aws_lambda_function" "delay_predictor" {
  function_name = var.function_name
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = var.lambda_image_uri
  timeout       = 15
  memory_size   = 512 # scikit-learn + pandas need headroom beyond the 128MB default

  # Free-tier note: 1M requests + 400,000 GB-seconds/month are free
  # perpetually (not just first 12 months), so this stays free for demo traffic.
}

# ---- Function URL: simplest public HTTPS endpoint, no API Gateway needed ----
resource "aws_lambda_function_url" "delay_predictor_url" {
  function_name      = aws_lambda_function.delay_predictor.function_name
  authorization_type = "NONE" # demo only - use "AWS_IAM" for anything real
}
