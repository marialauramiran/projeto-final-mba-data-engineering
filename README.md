# 🩺 Raio-X do Vendedor: Diagnóstico 360° para Vendedores de E-commerce

> Projeto Hands-On · MBA Data Engineering · Universidade Presbiteriana Mackenzie
> Base de dados: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)

![status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![arquitetura](https://img.shields.io/badge/arquitetura-Bronze%20%7C%20Silver%20%7C%20Gold-orange)

---

## 📌 Sumário

1. [Contexto e problema](#-contexto-e-problema)
2. [Objetivo geral e objetivos específicos](#-objetivo-geral-e-objetivos-específicos)
3. [O produto](#-o-produto)
4. [Bases de dados](#-bases-de-dados)
5. [Arquitetura de dados](#-arquitetura-de-dados)
6. [Metodologia do Diagnóstico 360°](#-metodologia-do-diagnóstico-360)
7. [Ferramentas e tecnologias](#-ferramentas-e-tecnologias)
8. [Estrutura do repositório](#-estrutura-do-repositório)
9. [Como executar](#-como-executar-instruções-preliminares)
10. [Resultados preliminares da EDA](#-resultados-preliminares-da-eda)
11. [Escopo do MVP](#-escopo-do-mvp)
12. [Cronograma](#-cronograma)
13. [Equipe](#-equipe)
14. [Licença e créditos dos dados](#-licença-e-créditos-dos-dados)

---

## 🎯 Contexto e problema

Vendedores autônomos de marketplace costumam tomar decisões sem enxergar o próprio desempenho em relação ao mercado: precificam "no achômetro", não medem o peso do frete, convivem com atrasos de entrega e só descobrem o problema quando a nota de avaliação já caiu. Muitos negócios quebram por não terem planejamento e acompanhamento de indicadores.

Este projeto usa o ecossistema de dados da Olist (2016–2018) para mostrar, com dados reais, como uma **cultura data driven** melhora a saúde de um e-commerce.

## 🚀 Objetivo geral e objetivos específicos

### Objetivo geral

Desenvolver, a partir dos dados públicos da Olist, uma solução de dados ponta a ponta (pipeline em camadas, análises e Machine Learning) que gere automaticamente um **Diagnóstico 360°** para vendedores autônomos. O diagnóstico compara o desempenho do vendedor com o dos **melhores lojistas da mesma categoria**, apresentando a **Nota de Saúde do E-commerce** e as **3 ações prioritárias** para ampliar o faturamento no mês seguinte.

### Objetivos específicos

1. **Integrar as 9 tabelas** da Olist em um único pipeline de dados nas camadas Bronze, Silver e Gold, com tratamento de qualidade e versionamento em Git.
2. **Construir indicadores por vendedor** (receita, ticket, frete sobre preço, atraso, tempo de postagem, avaliações, cancelamentos, qualidade do anúncio) e o **benchmark dos melhores lojistas** de cada categoria.
3. **Calcular a Nota de Saúde (0 a 100)** e um motor de priorização que seleciona as 3 ações de maior impacto.
4. **Desenvolver o Radar de Oportunidades por Categoria e Geografia**: distância (Haversine) entre vendedor e comprador, mapa de calor logístico e análise de saturação (demanda × vendedores ativos).
5. **Treinar e validar o Motor Preditivo de Riscos Logísticos** (Random Forest ou XGBoost) para estimar a probabilidade de atraso por rota, exibida em um termômetro verde, amarelo ou vermelho.
6. **Entregar um relatório executivo automatizado** (PDF e/ou dashboard interativo) que reúne a nota, o radar, o termômetro e as ações prioritárias.
7. **Documentar e validar** a solução (testes, métricas e apresentação final).

## 📦 O produto

| Módulo | O que entrega | Tabelas principais |
|--------|---------------|--------------------|
| **Diagnóstico 360°** (principal) | Nota de Saúde, comparação com os melhores lojistas da categoria e 3 ações prioritárias, em PDF ou dashboard | Todas as tabelas |
| **Radar de Oportunidades** | Mapa de calor logístico, distância média vendedor → cliente e análise de saturação por categoria e região | `products`, `category_translation`, `customers`, `sellers`, `geolocation`, `order_items` |
| **Motor Preditivo de Riscos Logísticos** | Termômetro de Envio (verde, amarelo, vermelho) com probabilidade de atraso por rota | `orders`, `order_items`, `sellers`, `customers`, `products`, `geolocation` |

Os dois últimos módulos alimentam o Diagnóstico 360°: o Radar fornece o tamanho de mercado e a saturação, e o Motor Preditivo fornece o risco logístico da rota do vendedor.

## 🗂 Bases de dados

Os CSVs vêm do Kaggle e **não são versionados** neste repositório (veja `data/README.md`).

| Arquivo | Linhas | Uso no projeto |
|---------|-------:|----------------|
| `olist_orders_dataset.csv` | 99.441 | Status e datas do pedido (compra, aprovação, postagem, entrega, estimativa) |
| `olist_order_items_dataset.csv` | 112.650 | Vendedor, produto, preço e frete de cada item |
| `olist_order_payments_dataset.csv` | 103.886 | Forma de pagamento e parcelas |
| `olist_order_reviews_dataset.csv` | 99.224 | Nota de avaliação (1 a 5) |
| `olist_customers_dataset.csv` | 99.441 | Estado, cidade e CEP do comprador |
| `olist_sellers_dataset.csv` | 3.095 | Estado, cidade e CEP do vendedor |
| `olist_products_dataset.csv` | 32.951 | Categoria, peso, dimensões e dados do anúncio |
| `olist_geolocation_dataset.csv` | 1.000.163 | Latitude e longitude por prefixo de CEP |
| `product_category_name_translation.csv` | 71 | Tradução das categorias (PT → EN) |

Período dos pedidos: **setembro de 2016 a outubro de 2018**. Existem **1.237 vendedores com pelo menos 10 pedidos entregues**, volume mínimo usado como referência para o benchmark.

**Variáveis derivadas:** `delivery_time_days`, `delivery_delay_days` (negativo = entrega antecipada), `total_item_value`, `freight_ratio` (frete ÷ preço), `handling_time_days` (compra → postagem) e `distance_km` (Haversine).

## 🏗 Arquitetura de dados

```
 CSVs (Kaggle)
      │
      ▼
┌───────────┐   ┌─────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
│  BRONZE   │ → │   SILVER    │ → │        GOLD          │ → │      CONSUMO        │
│ CSVs      │   │ tipagem,    │   │ métricas por vendedor│   │ Relatório PDF       │
│ brutos    │   │ nulos,      │   │ benchmark por        │   │ Dashboard           │
│ (raw)     │   │ duplicados, │   │ categoria, rotas,    │   │ Termômetro de Envio │
│           │   │ joins       │   │ features de ML       │   │ Radar               │
└───────────┘   └─────────────┘   └──────────────────────┘   └─────────────────────┘
```

- **Bronze:** cópia fiel dos arquivos originais.
- **Silver:** conversão de datas, filtro de pedidos entregues, remoção de duplicados, tradução das categorias, geolocalização agregada por prefixo de CEP (mediana) e junção das tabelas.
- **Gold:** `gold_seller_metrics`, `gold_category_benchmark`, `gold_route_risk`, `gold_market_by_state` e `gold_seller_health_score`.

## 🧮 Metodologia do Diagnóstico 360°

1. **Categoria do vendedor:** categoria com maior faturamento do vendedor.
2. **Benchmark:** dentro de cada categoria, os melhores lojistas são os que têm volume mínimo de pedidos e estão no quartil superior de um índice composto (faturamento e satisfação). Categorias com poucos vendedores usam um agrupamento de fallback.
3. **Indicadores comparados:** faturamento e ticket médio, frete sobre preço, % de atraso, tempo de postagem, nota média e % de notas 1–2, % de cancelamento e qualidade do anúncio (fotos e descrição).
4. **Nota de Saúde (0 a 100):** média ponderada da posição do vendedor em cada indicador frente ao benchmark. Os pesos iniciais são hipóteses e serão calibrados e validados na fase de modelagem.
5. **3 ações prioritárias:** ranking das lacunas em relação ao benchmark, ponderadas pelo peso do indicador, convertido em recomendações textuais (por exemplo, reduzir o tempo de postagem, rever o frete, melhorar o anúncio).
6. **Motor preditivo de atraso:** classificação binária (atrasou ou não) com divisão temporal treino/teste, tratamento de desbalanceamento (cerca de 8% dos pedidos entregues chegam após o prazo estimado) e avaliação por PR-AUC, recall e precisão. Usa somente variáveis conhecidas antes da postagem, para evitar vazamento de dados.

> ⚠️ **Limitação:** o dataset é observacional. As ações recomendadas indicam associações com os melhores lojistas, não garantem causalidade nem a duplicação do faturamento. O relatório trará o **potencial estimado** com essa ressalva.

## 🛠 Ferramentas e tecnologias

- **Google Colab:** exploração e prototipagem
- **Databricks:** processamento e camadas Bronze, Silver e Gold
- **Git / GitHub Actions:** versionamento e CI/CD
- **Python:** pandas, scikit-learn, XGBoost, plotly/folium, Streamlit e geração de PDF
- **Trello:** gestão do Kanban e das sprints

## 📁 Estrutura do repositório

```
.
├── src/
│   ├── ingestion/        # CSVs → Bronze
│   ├── transformation/   # Bronze → Silver → Gold
│   ├── features/         # variáveis derivadas, distância Haversine
│   ├── models/           # Nota de Saúde e modelo de atraso
│   └── app/              # relatório PDF e dashboard
├── notebooks/            # EDA e experimentos
├── data/                 # dados locais (não versionados)
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── docs/                 # escopo, arquitetura, dicionário de dados, apresentações
├── infra/                # infraestrutura como código e CI/CD
├── .github/workflows/    # GitHub Actions
├── .gitignore
├── requirements.txt
└── README.md
```

## ▶️ Como executar (instruções preliminares)

> ⚠️ Em construção: será atualizado conforme o desenvolvimento avançar.

1. **Clonar o repositório**
   ```bash
   git clone https://github.com/<organizacao-ou-usuario>/<nome-do-repositorio>.git
   cd <nome-do-repositorio>
   ```
2. **Criar o ambiente e instalar dependências**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Baixar os dados** do [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) e colocar os 9 CSVs em `data/bronze/`.
4. **Executar os notebooks** em `notebooks/`, começando pela EDA.

## 📊 Resultados preliminares da EDA

- **Preço vs. frete:** preço médio de R$ 120,64 (mediana R$ 74,90) e frete médio de cerca de R$ 20,03. Em produtos de ticket baixo, o frete chega perto de 30% do valor pago.
- **Pagamentos:** cartão de crédito domina (quase 80 mil transações), seguido pelo boleto (~20 mil).
- **Logística:** tempo médio de entrega de 12,48 dias e atraso médio de -11,35 dias (as estimativas têm folga de segurança). Mesmo assim, cerca de 8% dos pedidos entregues chegam depois do prazo.
- **Satisfação:** correlação de -0,30 entre tempo de entrega e nota; acima de 15 dias de espera, as notas de 1 estrela aumentam bastante.
- **Geografia e demanda:** SP concentra os compradores, seguido por RJ e MG. As categorias líderes são Cama, Mesa e Banho e Saúde e Beleza.
- **Perfil dos vendedores:** a maioria é pequena (mediana de 7 pedidos por vendedor), o que reforça o público-alvo do projeto.

## 🎯 Escopo do MVP

**Entra:** pipeline Bronze, Silver e Gold; indicadores por vendedor e benchmark; Nota de Saúde e 3 ações prioritárias; Radar (mapa de calor e saturação); modelo de atraso com termômetro; relatório PDF e/ou dashboard.

**Evolução futura:** simulador de precificação e margem, análise de sentimento (NLP) nos comentários das avaliações e simulação da Reforma Tributária.

## 📅 Cronograma

| Sprint | Período | Foco |
|--------|---------|------|
| 1 | 06/10 – 12/10 | Definição do problema, bases, arquitetura inicial, ferramentas, Kanban e repositório |
| 2 | 13/10 – 19/10 | Modelagem das camadas, CI/CD e IaC, EDA inicial, protótipo de dashboard |
| 3 | 20/10 – 26/10 | MVP e requisitos, ambiente técnico, camada de dados, modelo preditivo (versão inicial) |
| 4 | 27/10 – 09/11 | Finalização do desenvolvimento, análise de resultados, testes, dashboard final |
| 5 | 10/11 – 16/11 | Revisão do escopo, documentação final e apresentação |

O escopo v2 (Raio-X do Vendedor) foi formalizado em `docs/Projeto_Final_Escopo_v2.docx`. Acompanhe o Kanban no [Trello do projeto](https://trello.com/invite/b/6ad7e8d5b2fbe988950e3811/ATTI46fc8d1c2565216a02b99b5c5f70bb0d53AE229C/projeto-rgm-simulador-de-precos-mba-data-engineering).

## 👥 Equipe

- Maria Laura
- Giulia Santiago
- Luana Rodrigues
- Vinicius

## 📄 Licença e créditos dos dados

- Dados: *Brazilian E-Commerce Public Dataset by Olist*, no Kaggle, sob a licença **CC BY-NC-SA 4.0**. Uso exclusivamente acadêmico e não comercial.
- Código: projeto acadêmico do MBA em Data Engineering, Mackenzie.
