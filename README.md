# 🩺 Raio-X do Vendedor: diagnóstico de viabilidade para vendedores de marketplace

> Projeto Hands-On · MBA em Engenharia de Dados · Universidade Presbiteriana Mackenzie
> Professor: Gustavo Ferreira
> Base de dados: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)

![status](https://img.shields.io/badge/status-vers%C3%A3o%20final-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![arquitetura](https://img.shields.io/badge/arquitetura-Bronze%20%7C%20Silver%20%7C%20Gold-orange)

## Sumário

1. [Contexto e problema](#contexto-e-problema)
2. [Objetivos](#objetivos)
3. [O produto](#o-produto)
4. [Base de dados](#base-de-dados)
5. [Arquitetura de dados](#arquitetura-de-dados)
6. [Modelo preditivo](#modelo-preditivo)
7. [Principais resultados](#principais-resultados)
8. [Ferramentas e tecnologias](#ferramentas-e-tecnologias)
9. [Estrutura do repositório](#estrutura-do-repositório)
10. [Como executar](#como-executar)
11. [Governança e LGPD](#governança-e-lgpd)
12. [Limitações](#limitações)
13. [Próximos passos](#próximos-passos)
14. [Equipe](#equipe)
15. [Licença e créditos](#licença-e-créditos)

## Contexto e problema

Vendedores autônomos de marketplace decidem categoria, preço e logística com pouca informação. Quando o preço ignora o frete, quando a categoria já está saturada ou quando o prazo prometido não é cumprido, o resultado aparece na margem e na reputação. Nos dados da Olist (2016 a 2018):

- **8,1%** dos pedidos entregues chegam depois do prazo; esses pedidos recebem nota média **2,57**, contra **4,29** dos entregues no prazo;
- **36,6%** dos itens têm frete igual ou superior a 30% do preço;
- São Paulo concentra **42,0%** da demanda e **70,7%** dos vendedores.

## Objetivos

**Geral.** Desenvolver uma plataforma de dados em arquitetura lakehouse (Bronze, Silver e Gold) que combine análise exploratória, um modelo de risco de atraso e uma aplicação de consulta, para que o vendedor decida categoria, preço e logística com base em evidências.

**Específicos.**

1. Estruturar a ingestão e o tratamento das nove tabelas da Olist no Databricks, com qualidade verificável.
2. Quantificar concentração de mercado, peso do frete, prazo de entrega e reputação por categoria e região.
3. Treinar e validar, com separação temporal e sem vazamento, um modelo de risco de atraso.
4. Disponibilizar o resultado em uma aplicação Streamlit (semáforo de viabilidade) e em um painel executivo no Looker Studio.
5. Aplicar boas práticas de governança e adequação à LGPD.
6. Documentar o trabalho e relacionar as funções de uma equipe de dados às trilhas de carreira.

## O produto

| Módulo | O que entrega |
|--------|---------------|
| **Aplicação Raio-X do Vendedor** (Streamlit, Databricks App) | O vendedor informa categoria, UF de origem e faixas de preço e peso, e recebe um semáforo geral com cinco cartões (Mercado, Preço e frete, Risco de atraso, Reputação e Categoria) e recomendações |
| **Painel executivo** (Looker Studio) | Três páginas por perfil: Executivo, Negócio e Técnico |
| **Modelo de risco de atraso** | Probabilidade de atraso por pedido e faixas de risco (alto, médio e baixo) |

## Base de dados

Os CSVs vêm do Kaggle e **não são versionados** neste repositório (veja `data/README.md`).

| Tabela | Linhas (Silver) |
|--------|----------------:|
| `olist_orders_dataset` | 99.433 |
| `olist_customers_dataset` | 99.441 |
| `olist_order_items_dataset` | 112.650 |
| `olist_order_payments_dataset` | 103.886 |
| `olist_order_reviews_dataset` | 94.823 |
| `olist_products_dataset` | 32.341 |
| `olist_sellers_dataset` | 3.095 |
| `product_category_name_translation` | 71 |
| `olist_geolocation_dataset` | 1.000.163 |

A integração gera **98.658 pedidos** (uma linha por pedido) com GMV de **R$ 13.590.394,52**, reconciliado com a tabela de itens. A janela do modelo vai de 1º/01/2017 a 31/08/2018.

## Arquitetura de dados

![Arquitetura de dados](docs/img/arquitetura_dados.png)

O fluxo percorre cinco etapas: **fonte → preparação → armazenamento (lakehouse) → modelagem e análise → consumo**, com duas faixas transversais: governança e segurança, e engenharia de software.

| Camada | Conteúdo |
|--------|----------|
| **Bronze** | Cópia fiel dos 9 CSV em Parquet, sem regra de negócio |
| **Silver** | 9 tabelas limpas, tipadas e com chaves validadas |
| **Gold** | 5 tabelas de negócio (`gold_agg_mercado`, `gold_agg_frete`, `gold_ref_rotas`, `gold_agg_reputacao`, `gold_agg_categoria`) e tabelas de previsão (`predictions`, `aggregated_by_filter`, `filter_options`) |

- Ambiente: Databricks Free Edition (serverless, AWS). Os dados ficam em Parquet nos Volumes do Unity Catalog (`/Volumes/workspace/olist_{bronze,silver,gold}`), lidos com `spark.read.parquet` (sem `cache()`).
- O processamento é **em lote**. A base é histórica e fixa, sem atualização em tempo real, por isso o projeto não utiliza *streaming*. Essa camada só faria sentido se o projeto fosse conectado a uma fonte viva de pedidos.

### Regras de tratamento da Silver

| Regra | Impacto |
|-------|---------|
| Remover avaliações sem data, identificador ou pedido | 8.764 sem data; 1 sem `review_id`; 2.236 sem `order_id` |
| Remover pedidos entregues sem data de entrega | 8 pedidos |
| Remover produtos sem categoria | 610 produtos |
| Manter uma avaliação por pedido (a mais recente) | 94.823 avaliações |
| Padronizar tipos (datas, números, CEP com 5 dígitos) | Todas as tabelas |
| Verificar chaves primárias | Sem duplicatas |

## Modelo preditivo

- **Problema:** classificação binária (o pedido atrasa ou não), com 97.298 pedidos elegíveis e taxa de atraso de 7,84%.
- **Algoritmo:** regressão logística (C = 0,003) com 28 variáveis conhecidas no momento da compra (categóricas, produto, prazo e rota, vendedor, plataforma e calendário).
- **Validação temporal:** 3 dobras com embargo de 30 dias; treino até jan/2018, calibração em mar/2018 e teste de abr a ago/2018 (32.450 pedidos, 5,31% de atraso). Sem vazamento e com correção de prevalência.
- **Comparação:** regressão logística (PR-AUC 0,1296 com 25 variáveis) contra *Gradient Boosting* (0,1135); diferença por *bootstrap* de +0,0161 (IC 95%: +0,0099 a +0,0221).
- **Código:** `codes/machine-learning/olist_atraso_regressao_logistica_28vars.py`; histórico em `notebooks/histórico_modelo_de_ML/`.

## Principais resultados

| Métrica (teste, abr–ago/2018) | Valor |
|-------------------------------|------:|
| PR-AUC | 0,1307 (IC 95%: 0,1200 a 0,1428), 2,5 vezes o acaso (0,0531) |
| ROC-AUC | 0,7167 |
| Brier | 0,0489 |
| Precisão no top 10% de risco | 16,24% (lift 3,06) |
| Alerta sobre 40% dos pedidos | recall 69,6%, precisão 9,24%, acurácia 62,1% |

Na análise exploratória: atraso de 2,9% a 23,9% entre UFs (AL 23,9%, MA 19,6%, PI 15,9%), picos de atraso em mar/2018 (21,4%), fev/2018 (16,0%) e nov/2017 (14,3%), e correlação de −0,33 entre nota e tempo de entrega.

## Ferramentas e tecnologias

| Função | Tecnologia |
|--------|------------|
| Plataforma de dados | Databricks Free Edition (serverless, AWS) |
| Ingestão | kagglehub e PySpark |
| Processamento | PySpark |
| Armazenamento e catálogo | Parquet em Volumes do Unity Catalog |
| Modelagem | scikit-learn |
| Registro de modelos | MLflow (componente da arquitetura) |
| Aplicação | Streamlit em Databricks Apps |
| Painel | Looker Studio |
| Versionamento e CI | GitHub e GitHub Actions (compilação e `ruff`) |
| Infraestrutura como código | Terraform (data lake S3 com versionamento, criptografia e bloqueio de acesso público) |
| Gestão | Trello (Kanban e sprints) |

## Estrutura do repositório

```
.
├── codes/
│   ├── data-engineering/     # ingestão (Bronze), tratamento (Silver), agregação (Gold), modelo na Gold
│   ├── data-analysis/        # análise exploratória
│   └── machine-learning/     # regressão logística (28 variáveis)
├── notebooks/                # EDA (10 perguntas) e histórico do modelo
├── raio_x_app/               # aplicação Streamlit (Databricks App)
├── backup/infraestructure/   # Terraform do data lake (dev)
├── docs/                     # relatório técnico, escopo, briefing, dicionário de dados e imagens
├── .github/workflows/        # GitHub Actions (CI)
├── data/                     # dados locais (não versionados)
├── .gitignore
└── README.md
```

## Como executar

1. **Clonar o repositório**
   ```bash
   git clone https://github.com/marialauramiran/projeto-final-mba-data-engineering.git
   cd projeto-final-mba-data-engineering
   ```
2. **Pipeline de dados (Databricks).** Importe os notebooks de `codes/data-engineering/` e execute na ordem: `ingestao_bronze` → `tratamento_silver` → `agregacao_gold` → `modelo_ml_gold`. A ingestão baixa os CSV com `kagglehub` e grava Parquet nos Volumes.
3. **Aplicação.** Em `raio_x_app/`, instale as dependências e execute localmente (os dados da Gold são lidos dos Volumes quando publicada no Databricks Apps):
   ```bash
   cd raio_x_app
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
4. **Painel.** O painel do Looker Studio consome as tabelas da Gold e está descrito na seção 6 do relatório técnico.

## Governança e LGPD

- **Catálogo e metadados:** dicionário de dados (tabela, coluna, tipo, descrição, origem) e relatório de nulos da Silver.
- **Qualidade:** tipagem, tratamento de nulos, uma avaliação por pedido, chaves únicas, reconciliação do GMV e uma linha por pedido.
- **LGPD:** clientes anonimizados na origem, colunas classificadas por sensibilidade, aplicação com agregados e CEP com 5 dígitos. Uso acadêmico, dados pseudonimizados e reidentificação proibida.
- **Segurança:** dados em Volumes do Unity Catalog e aplicação com *service principal* e permissão de leitura.
- **Linhagem:** notebooks versionados; a tabela `predictions` guarda a versão do modelo e as 28 entradas.
- **Mapeamento:** Raw = Bronze, Trusted = Silver, Curated = Gold.

## Limitações

- Dados históricos de um único marketplace (2016 a 2018).
- Sinal preditivo moderado (PR-AUC 0,13 contra 0,053 do acaso); as probabilidades devem ser lidas como faixas de risco.
- Resultados indicam associação, não causalidade; faltam variáveis como transportadora, clima e promoções.
- Amostras pequenas em algumas rotas, categorias e estados do Nordeste.

## Próximos passos

O projeto é **em lote** por trabalhar com base histórica fixa. As evoluções previstas são:

- **Dados:** *star schema* e *wide table* persistida na Gold.
- **Modelo:** recalibração das probabilidades, outros algoritmos sob a mesma validação, registro no MLflow e, se necessário, *Model Serving*.
- **Análises:** agrupamento de vendedores (K-Means), simulador de preço e margem e análise de texto das avaliações.
- **Engenharia:** implantação automática no CI/CD, orquestração por *jobs*, testes de qualidade de dados e aplicação do Terraform.
- **Governança:** proprietário e curador dos dados, permissões por camada e linhagem no Unity Catalog.

## Equipe

| Integrante | Perfil |
|------------|--------|
| Maria Laura | Engenharia de Dados |
| Giulia Santiago | Análise de Dados e BI |
| Luana Rodrigues | Ciência de Dados |
| Vinicius Pontes | Engenharia de ML e governança |

## Licença e créditos

- Dados: *Brazilian E-Commerce Public Dataset by Olist* (Kaggle), licença **CC BY-NC-SA 4.0**. Uso exclusivamente acadêmico e não comercial.
- Código: projeto acadêmico do MBA em Engenharia de Dados, Mackenzie.
