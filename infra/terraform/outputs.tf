output "function_url" {
  value       = aws_lambda_function_url.delay_predictor_url.function_url
  description = "Invoke with: curl -X POST <url> -d '{...payload...}'"
}

output "function_name" {
  value = aws_lambda_function.delay_predictor.function_name
}
