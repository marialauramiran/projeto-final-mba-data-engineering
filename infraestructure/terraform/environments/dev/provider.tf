#Configuração da conexão com a AWS

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "projeto-final-mba"
      Environment = "dev"
      ManagedBy   = "Terraform"
    }
  }
}
