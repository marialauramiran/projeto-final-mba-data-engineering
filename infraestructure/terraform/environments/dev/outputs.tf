#Exibe informações úteis após a criação

output "data_lake_bucket_name" {
  description = "Nome do bucket S3"
  value       = aws_s3_bucket.data_lake.bucket
}

output "data_lake_bucket_arn" {
  description = "ARN do bucket S3"
  value       = aws_s3_bucket.data_lake.arn
}

output "aws_region" {
  description = "Regiao AWS do Data Lake"
  value       = var.aws_region
}

output "data_lake_access_policy_arn" {
  description = "ARN da politica IAM de acesso ao Data Lake"
  value       = aws_iam_policy.data_lake_access.arn
}
