#Centraliza região e nome do bucket.

variable "aws_region" {
  description = "Regiao AWS onde a infraestrutura sera criada"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Nome globalmente unico do bucket S3"
  type        = string
  default     = "projeto-final-mba-data-engineering"
}
