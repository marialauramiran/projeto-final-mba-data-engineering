#Cria o bucket S3 e suas proteções


resource "aws_s3_bucket" "data_lake" {
  bucket = var.bucket_name

  tags = {
    Name    = var.bucket_name
    Purpose = "Data Lake do TCC"
  }

  #Impedir que um terraform apply comum destrua o bucket acidentalmente  
  lifecycle {
    prevent_destroy = true
  }
}

#Evitar que os objetos sejam expostos publicamente
resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

#Ajuda a recuperar versões anteriores dos arquivos
resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  versioning_configuration {
    status = "Enabled"
  }
}

#Protege os objetos armazenados no S3
resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Pastas lógicas das camadas do Data Lake
resource "aws_s3_object" "data_lake_folders" {
  for_each = toset(["bronze/", "silver/", "gold/"])

  bucket  = aws_s3_bucket.data_lake.id
  key     = each.value
  content = ""
}
