variable "aws_region" {
  type    = string
  default = "eu-central-1"
}

variable "function_name" {
  type    = string
  default = "freight-delay-predictor"
}

variable "lambda_image_uri" {
  description = "ECR image URI produced by infra/package_lambda.sh"
  type        = string
  # e.g. "123456789012.dkr.ecr.eu-central-1.amazonaws.com/freight-delay-predictor:latest"
}
