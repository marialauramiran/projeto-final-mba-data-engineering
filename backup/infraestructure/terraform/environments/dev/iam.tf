#Define uma política IAM de acesso ao bucket

resource "aws_iam_policy" "data_lake_access" {
  name        = "projeto-final-mba-data-lake-access"
  description = "Acesso de leitura e escrita ao Data Lake do TCC"

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Sid    = "ListDataLakeBucket"
        Effect = "Allow"

        Action = [
          "s3:ListBucket"
        ]

        Resource = aws_s3_bucket.data_lake.arn
      },
      {
        Sid    = "ReadWriteDataLakeObjects"
        Effect = "Allow"

        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:AbortMultipartUpload"
        ]

        Resource = "${aws_s3_bucket.data_lake.arn}/*"
      }
    ]
  })

  tags = {
    Project = "projeto-final-mba"
  }
}
