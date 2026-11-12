# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC ## 0. Configuração

# COMMAND ----------

# MAGIC %pip install lightgbm shap seaborn --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd

#  fontes (Camada Silver)
SILVER_BASE = "/Volumes/workspace/olist_silver/parquet"
TABELAS = {
    "customers": "olist_customers_dataset",
    "orders": "olist_orders_dataset",
    "items": "olist_order_items_dataset",
    "payments": "olist_order_payments_dataset",
    "reviews": "olist_order_reviews_dataset",
    "products": "olist_products_dataset",
    "sellers": "olist_sellers_dataset",
    "translation": "product_category_name_translation",
}

TABELA_GEO_OPCIONAL = "olist_geolocation_dataset"


SAIDA_DIR = "/Volumes/workspace/olist_silver/parquet/_ml_saidas"  

# 
DATA_INICIO, DATA_FIM = "2017-01-01", "2018-08-31"  
FOLDS_VALIDACAO = [                                   
    ("2017-07-01", "2017-10-01"),
    ("2017-10-01", "2018-01-01"),
    ("2018-01-01", "2018-04-01"),
]
INICIO_TESTE = "2018-04-01"       # teste igual ao das versões anteriores (abr-ago/2018): comparação justa
MESES_CALIBRACAO = 3              # meses candidatos à calibração antes do teste (jan-mar/2018) ...
CALIBRAR_SEM_ATIPICOS = True      # ... dos quais saem os meses atípicos, pela mesma regra do treino
EMBARGO_DIAS = 30                 # pedidos comprados nos 30 dias antes de um corte não entram no treino
FIM_TREINO_FINAL = str((pd.Timestamp(INICIO_TESTE) - pd.DateOffset(months=MESES_CALIBRACAO)).date())
CALIBRAR = True                   # calibrador sigmoide (o Brier sem calibração é sempre reportado)

# ---- RECORTE: estratégia vencedora da comparação na validação
ESTRATEGIA_ATIVA = "equilibrado_sem_atipicos"   # sem meses atípicos + cada mês limitado à mediana de pedidos
FATOR_ATIPICO = 1.5               # mês atípico = taxa de atraso acima de 1,5x a mediana mensal do histórico
MIN_PEDIDOS_MES = 300             # meses com menos pedidos não são classificados como atípicos
VERSAO_RECORTE = "equilibrado_sem_atipicos_v3"  # entra no nome dos arquivos salvos (não reaproveita buscas antigas)

# Modelos testados nesta versão
MODELOS_TESTADOS = ["Regressão Logística", "Gradient Boosting"]

# features históricas
JANELA_RECENTE_DIAS = 30
MIN_EVENTOS_ROTA = 30             # rota UF>UF precisa de 30+ entregas anteriores; senão cai para região>região
MIN_EVENTOS_RECENTE = 20
MIN_EVENTOS_VENDEDOR = 3

# busca e reprodutibilidade
SEED = 42
N_ITER_BUSCA = 20                 # grades ampliadas (seção 6): mais combinações sorteadas                
MODO_RAPIDO = False               
if MODO_RAPIDO:
    N_ITER_BUSCA = 4

#  matriz de decisão (definida a priori)
PESOS_DECISAO = {
    "discriminacao_pr_auc": 0.35,     # classes desbalanceadas: PR-AUC e não acurácia
    "calibracao_brier": 0.20,         # a probabilidade vai para o app, então precisa ser confiável
    "alto_risco_precisao_top10": 0.15,  # utilidade da faixa vermelha (10% de maior risco)
    "estabilidade": 0.10,             # queda validação->teste e variação mês a mês
    "interpretabilidade": 0.10,       # 1 = coeficientes diretos; 0,5 = precisa de SHAP/permutação
    "viabilidade_operacional": 0.10,  # latência e tamanho do artefato
}
assert abs(sum(PESOS_DECISAO.values()) - 1) < 1e-9

# seleção de variáveis (matriz de correlação)
N_FEATURES = 30                   # variáveis que entram no modelo
LIMIAR_REDUNDANCIA = 0.85         # duas variáveis com correlação acima disso não entram juntas

# COMMAND ----------

import hashlib
import io
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
PCT = mtick.PercentFormatter(1)
np.random.seed(SEED)

try:
    os.makedirs(os.path.join(SAIDA_DIR, "figuras"), exist_ok=True)
except OSError:
    SAIDA_DIR = "/tmp/olist_ml_saidas"
    os.makedirs(os.path.join(SAIDA_DIR, "figuras"), exist_ok=True)
    print(f"AVISO: não foi possível escrever no volume; saídas em {SAIDA_DIR} (temporário)")
DIR_FIG = os.path.join(SAIDA_DIR, "figuras")


def mostrar(fig, nome):
    """Mostra o gráfico no notebook e salva uma cópia em SAIDA_DIR/figuras."""
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, f"{nome}.png"), dpi=110, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leitura e integração da Camada Silver

# COMMAND ----------

# [spark]
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def ler_silver(nome_tabela, obrigatoria=True):
    """Lê uma tabela Parquet da Silver. Aceita pasta (<nome>/) ou arquivo (<nome>.parquet)."""
    for caminho in (f"{SILVER_BASE}/{nome_tabela}", f"{SILVER_BASE}/{nome_tabela}.parquet"):
        try:
            return spark.read.parquet(caminho)
        except Exception:
            continue
    if obrigatoria:
        raise FileNotFoundError(f"Tabela {nome_tabela} não encontrada em {SILVER_BASE}")
    return None


silver = {chave: ler_silver(nome) for chave, nome in TABELAS.items()}
silver_geo = ler_silver(TABELA_GEO_OPCIONAL, obrigatoria=False)

for chave, df in silver.items():
    print(f"{TABELAS[chave]:40s} {df.count():>9,} linhas | {len(df.columns)} colunas")
print(f"{TABELA_GEO_OPCIONAL:40s} {'disponível' if silver_geo is not None else 'ausente (distância pela capital da UF)'}")

# COMMAND ----------

# [spark]
# Normalização defensiva: corrige nomes com erro de digitação da base original, tipos e CEP.
# Se a Silver já estiver padronizada, estas operações não alteram nada.
RENOMEAR = {"products": {"product_name_lenght": "product_name_length",
                         "product_description_lenght": "product_description_length"}}
DATAS = {"orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                    "order_delivered_customer_date", "order_estimated_delivery_date"],
         "items": ["shipping_limit_date"],
         "reviews": ["review_creation_date", "review_answer_timestamp"]}
NUMERICOS = {"items": ["price", "freight_value", "order_item_id"],
             "payments": ["payment_value", "payment_installments", "payment_sequential"],
             "reviews": ["review_score"],
             "products": ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
                          "product_photos_qty", "product_description_length", "product_name_length"]}
CEPS = {"customers": "customer_zip_code_prefix", "sellers": "seller_zip_code_prefix"}
OBRIGATORIAS = {
    "orders": ["order_id", "customer_id", "order_status"] + DATAS["orders"],
    "customers": ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
    "items": ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"],
    "payments": ["order_id", "payment_type", "payment_installments", "payment_value"],
    "reviews": ["order_id", "review_score"],
    "products": ["product_id", "product_category_name", "product_weight_g", "product_length_cm",
                 "product_height_cm", "product_width_cm"],
    "sellers": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    "translation": ["product_category_name", "product_category_name_english"],
}


def normalizar(chave, df):
    for antigo, novo in RENOMEAR.get(chave, {}).items():
        if antigo in df.columns and novo not in df.columns:
            df = df.withColumnRenamed(antigo, novo)
    faltando = [c for c in OBRIGATORIAS[chave] if c not in df.columns]
    assert not faltando, f"{TABELAS[chave]}: colunas ausentes na Silver {faltando}"
    for c in DATAS.get(chave, []):
        if c in df.columns:
            df = df.withColumn(c, F.expr(f"try_cast(`{c}` AS TIMESTAMP)"))
    for c in NUMERICOS.get(chave, []):
        if c in df.columns:
            df = df.withColumn(c, F.expr(f"try_cast(`{c}` AS DOUBLE)"))    # evita Decimal virar 'object' no pandas
    if chave in CEPS:
        c = CEPS[chave]
        df = df.withColumn(c, F.lpad(F.col(c).cast("int").cast("string"), 5, "0"))
    return df


silver = {chave: normalizar(chave, df) for chave, df in silver.items()}

# 1 avaliação por pedido (a mais recente). Na base original, review_id não é chave única.
col_ordem = "review_answer_timestamp" if "review_answer_timestamp" in silver["reviews"].columns else "review_creation_date"
w_rev = Window.partitionBy("order_id").orderBy(F.col(col_ordem).desc_nulls_last())
silver["reviews"] = (silver["reviews"].withColumn("_rn", F.row_number().over(w_rev))
                     .filter("_rn = 1").drop("_rn"))

# Chaves únicas esperadas: se falhar, a Silver tem duplicatas que distorceriam os joins
for chave, pk in [("orders", ["order_id"]), ("customers", ["customer_id"]), ("products", ["product_id"]),
                  ("sellers", ["seller_id"]), ("items", ["order_id", "order_item_id"]), ("reviews", ["order_id"])]:
    df = silver[chave]
    dup = df.count() - df.select(*pk).distinct().count()
    assert dup == 0, f"{TABELAS[chave]}: {dup} linhas com chave {pk} duplicada"
print("Checagem de chaves OK")

# COMMAND ----------

#  itens + produtos -> 1 linha por pedido
prod = (silver["products"]
        .join(silver["translation"], "product_category_name", "left")
        .withColumn("product_category_name", F.coalesce("product_category_name", F.lit("sem_categoria")))
        .withColumn("volume_cm3", F.col("product_length_cm") * F.col("product_height_cm") * F.col("product_width_cm"))
        .withColumn("maior_dim_cm", F.greatest("product_length_cm", "product_height_cm", "product_width_cm")))
for c in ["product_photos_qty", "product_description_length", "product_name_length"]:
    if c not in prod.columns:
        prod = prod.withColumn(c, F.lit(None).cast("double"))

itens = silver["items"].join(prod.select("product_id", "product_category_name", "product_category_name_english",
                                         "product_weight_g", "volume_cm3", "maior_dim_cm", "product_photos_qty",
                                         "product_description_length", "product_name_length"),
                             "product_id", "left")

agg_itens = itens.groupBy("order_id").agg(
    F.count("*").alias("n_itens"),
    F.countDistinct("product_id").alias("n_produtos_distintos"),
    F.countDistinct("seller_id").alias("n_vendedores"),
    F.sum("price").alias("valor_produtos"),
    F.sum("freight_value").alias("valor_frete"),
    F.avg("price").alias("preco_medio_item"),
    F.max("price").alias("preco_max_item"),
    F.sum("product_weight_g").alias("peso_total_g"),
    F.max("product_weight_g").alias("peso_max_g"),
    F.sum("volume_cm3").alias("volume_total_cm3"),
    F.max("maior_dim_cm").alias("maior_dimensao_cm"),
    F.avg("product_photos_qty").alias("fotos_media"),
    F.avg("product_description_length").alias("tam_descricao_media"),
    F.avg("product_name_length").alias("tam_nome_medio"),
    F.max("shipping_limit_date").alias("data_limite_postagem"),   # pós-compra (ver dicionário): só EDA
)

# Item principal (maior preço) define a categoria e o vendedor de referência do pedido
w_item = Window.partitionBy("order_id").orderBy(F.col("price").desc(), F.col("order_item_id").asc())
principal = (itens.withColumn("_rn", F.row_number().over(w_item)).filter("_rn = 1")
             .select("order_id", F.col("product_category_name").alias("categoria"),
                     F.col("product_category_name_english").alias("categoria_en"),
                     F.col("seller_id").alias("vendedor_principal")))

# ----------------------------------------------------------------- pagamentos -> 1 linha por pedido
pag = silver["payments"]
w_pag = Window.partitionBy("order_id").orderBy(F.col("payment_value").desc())
tipo_pag = (pag.withColumn("_rn", F.row_number().over(w_pag)).filter("_rn = 1")
            .select("order_id", F.col("payment_type").alias("tipo_pagamento")))
agg_pag = pag.groupBy("order_id").agg(
    F.sum("payment_value").alias("valor_pago"),
    F.countDistinct("payment_type").alias("n_formas_pagamento"),
    F.max(F.greatest(F.col("payment_installments"), F.lit(1.0))).alias("parcelas"),   # 0 parcelas = à vista
    F.sum(F.when(F.col("payment_type") == "voucher", F.col("payment_value")).otherwise(0.0)).alias("valor_voucher"),
).join(tipo_pag, "order_id", "left")

# ----------------------------------------------------------------- avaliação (pós-entrega: só EDA de satisfação)
rev = silver["reviews"]
msg = F.col("review_comment_message") if "review_comment_message" in rev.columns else F.lit(None).cast("string")
agg_rev = rev.select("order_id", "review_score",
                     F.when(F.length(F.trim(msg)) > 0, 1).otherwise(0).alias("tem_comentario"))

# ----------------------------------------------------------------- geolocalização opcional (1 ponto por CEP)
geo = None
if silver_geo is not None:
    geo = (silver_geo
           .withColumn("cep", F.lpad(F.col("geolocation_zip_code_prefix").cast("int").cast("string"), 5, "0"))
           .filter(F.col("geolocation_lat").between(-34, 5.5) & F.col("geolocation_lng").between(-74, -34))
           .groupBy("cep").agg(F.avg("geolocation_lat").alias("lat"), F.avg("geolocation_lng").alias("lng")))

# ----------------------------------------------------------------- base integrada (1 linha por pedido)
clientes = silver["customers"].select("customer_id", "customer_unique_id", "customer_zip_code_prefix",
                                      "customer_city", "customer_state")
vendedores = silver["sellers"].select(F.col("seller_id").alias("vendedor_principal"), "seller_zip_code_prefix",
                                      "seller_city", "seller_state")
integrada = (silver["orders"]
             .join(clientes, "customer_id", "left")
             .join(agg_itens, "order_id", "inner")          # pedidos sem item não têm produto nem frete
             .join(principal, "order_id", "left")
             .join(vendedores, "vendedor_principal", "left")
             .join(agg_pag, "order_id", "left")
             .join(agg_rev, "order_id", "left"))
if geo is not None:
    integrada = (integrada
                 .join(geo.select(F.col("cep").alias("customer_zip_code_prefix"), F.col("lat").alias("cliente_lat"),
                                  F.col("lng").alias("cliente_lng")), "customer_zip_code_prefix", "left")
                 .join(geo.select(F.col("cep").alias("seller_zip_code_prefix"), F.col("lat").alias("vendedor_lat"),
                                  F.col("lng").alias("vendedor_lng")), "seller_zip_code_prefix", "left"))
else:
    for c in ["cliente_lat", "cliente_lng", "vendedor_lat", "vendedor_lng"]:
        integrada = integrada.withColumn(c, F.lit(None).cast("double"))

# Testes de integridade da integração
n_ped, n_unicos = integrada.count(), integrada.select("order_id").distinct().count()
assert n_ped == n_unicos, "o join duplicou pedidos"
gmv_itens = silver["items"].join(integrada.select("order_id"), "order_id", "semi").agg(F.sum("price")).first()[0]
gmv_base = integrada.agg(F.sum("valor_produtos")).first()[0]
assert abs(gmv_itens - gmv_base) < 1, "GMV da base integrada não bate com order_items"
print(f"Base integrada: {n_ped:,} pedidos x {len(integrada.columns)} colunas | GMV R$ {gmv_base:,.2f} (reconciliado)")

# COMMAND ----------

# [spark]
# A base tem ~100 mil pedidos: cabe em memória no driver. As etapas seguintes usam pandas porque as
# features históricas precisam de junções "as-of" (o que se sabia em cada instante), que o pandas resolve
# com merge_asof de forma simples e auditável. Todo o histórico (inclusive 2016) é trazido: a janela de
# análise é aplicada só depois, para que o histórico de jan/2017 não comece vazio.
pedidos = integrada.toPandas()
print(pedidos.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Base analítica e definição do alvo

# COMMAND ----------

REGIAO = {**dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "Norte"),
          **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
          **dict.fromkeys(["DF", "GO", "MS", "MT"], "Centro-Oeste"),
          **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
          **dict.fromkeys(["PR", "RS", "SC"], "Sul")}
# Capitais (nome normalizado, lat, lng): fallback da distância e feature "cliente_em_capital"
CAPITAIS = {"AC": ("rio branco", -9.97, -67.81), "AL": ("maceio", -9.67, -35.74), "AP": ("macapa", 0.03, -51.07),
            "AM": ("manaus", -3.12, -60.02), "BA": ("salvador", -12.97, -38.51), "CE": ("fortaleza", -3.73, -38.52),
            "DF": ("brasilia", -15.79, -47.88), "ES": ("vitoria", -20.32, -40.34), "GO": ("goiania", -16.68, -49.25),
            "MA": ("sao luis", -2.53, -44.30), "MT": ("cuiaba", -15.60, -56.10), "MS": ("campo grande", -20.47, -54.62),
            "MG": ("belo horizonte", -19.92, -43.94), "PA": ("belem", -1.46, -48.50), "PB": ("joao pessoa", -7.12, -34.86),
            "PR": ("curitiba", -25.43, -49.27), "PE": ("recife", -8.05, -34.88), "PI": ("teresina", -5.09, -42.80),
            "RJ": ("rio de janeiro", -22.91, -43.17), "RN": ("natal", -5.79, -35.21), "RS": ("porto alegre", -30.03, -51.23),
            "RO": ("porto velho", -8.76, -63.90), "RR": ("boa vista", 2.82, -60.67), "SC": ("florianopolis", -27.60, -48.55),
            "SP": ("sao paulo", -23.55, -46.63), "SE": ("aracaju", -10.91, -37.07), "TO": ("palmas", -10.18, -48.33)}

COLS_DATA = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
             "order_delivered_customer_date", "order_estimated_delivery_date", "data_limite_postagem"]


def dias(delta):
    return delta.dt.total_seconds() / 86400


def normaliza_texto(s):
    s = s.astype("string").str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def preparar_base(ped):
    ped = ped.copy()
    for c in COLS_DATA:
        ped[c] = pd.to_datetime(ped[c], errors="coerce")
    ts = ped["order_purchase_timestamp"]
    entrega, prometido = ped["order_delivered_customer_date"], ped["order_estimated_delivery_date"]

    # ------------------------------------------------ variáveis PÓS-COMPRA (só EDA; nunca entram no modelo)
    ped["entregue"] = entrega.notna().astype(int)
    ped["horas_aprovacao"] = dias(ped["order_approved_at"] - ts) * 24
    ped["dias_aprovacao_postagem"] = dias(ped["order_delivered_carrier_date"] - ped["order_approved_at"])
    ped["dias_transporte"] = dias(entrega - ped["order_delivered_carrier_date"])
    ped["dias_entrega"] = dias(entrega - ts)
    ped["atraso_dias"] = (entrega.dt.normalize() - prometido.dt.normalize()).dt.days
    ped["postou_apos_limite"] = (ped["order_delivered_carrier_date"] > ped["data_limite_postagem"]).astype(float) \
        .where(ped["order_delivered_carrier_date"].notna())
    ped["review_negativa"] = (ped["review_score"] <= 2).astype(float).where(ped["review_score"].notna())

    # ------------------------------------------------ alvo
    fim_base = entrega.max().normalize()
    vencido = prometido.dt.normalize() < fim_base
    sem_entrega = ((ped["entregue"] == 0) & ped["order_delivered_carrier_date"].notna()
                   & (ped["order_status"] != "canceled") & vencido)
    ped["elegivel"] = (vencido & ((ped["entregue"] == 1) | sem_entrega)).astype(int)
    ped["atraso_sem_entrega"] = sem_entrega.astype(int)
    ped["atrasou"] = np.where(ped["entregue"] == 1, (ped["atraso_dias"] > 0).astype(float),
                              np.where(sem_entrega, 1.0, np.nan))
    dia_seguinte_prazo = prometido.dt.normalize() + pd.Timedelta(days=1)
    ped["data_conhecimento_alvo"] = entrega.where(ped["atrasou"] == 0, dia_seguinte_prazo)

    # ------------------------------------------------ atributos básicos, conhecidos na compra (usados na EDA)
    ped["prazo_prometido_dias"] = dias(prometido - ts)
    ped["ticket_total"] = ped["valor_produtos"] + ped["valor_frete"]
    ped["frete_pct_ticket"] = ped["valor_frete"] / ped["ticket_total"]
    for c in ["categoria", "tipo_pagamento", "customer_state", "seller_state"]:
        ped[c] = ped[c].fillna("desconhecido").astype(str)
    ped["regiao_cliente"] = ped["customer_state"].map(REGIAO).fillna("desconhecido")
    ped["regiao_vendedor"] = ped["seller_state"].map(REGIAO).fillna("desconhecido")
    ped["mesmo_estado"] = (ped["customer_state"] == ped["seller_state"]).astype(int)
    ped["mesma_regiao"] = (ped["regiao_cliente"] == ped["regiao_vendedor"]).astype(int)
    ped["cidade_cliente"] = normaliza_texto(ped["customer_city"])
    ped["cidade_vendedor"] = normaliza_texto(ped["seller_city"]).str.split(r"\s*[-/,\\]\s*", regex=True).str[0]

    # Distância: CEP quando há geolocalização; senão, capital da UF
    cap = pd.DataFrame(CAPITAIS, index=["nome", "lat", "lng"]).T
    for lado, uf in [("cliente", "customer_state"), ("vendedor", "seller_state")]:
        for eixo in ["lat", "lng"]:
            col = f"{lado}_{eixo}"
            ped[col] = pd.to_numeric(ped[col], errors="coerce").fillna(ped[uf].map(cap[eixo]).astype(float))
    ped["distancia_km"] = haversine_km(ped["vendedor_lat"], ped["vendedor_lng"], ped["cliente_lat"], ped["cliente_lng"])
    ped["ano_mes"] = ts.dt.to_period("M").astype(str)
    return ped


base = preparar_base(pedidos)
na_janela = base["order_purchase_timestamp"].between(DATA_INICIO, f"{DATA_FIM} 23:59:59")
base_modelo = base[na_janela & (base["elegivel"] == 1)].copy()
print(f"Pedidos na Silver integrada: {len(base):,} | na janela: {na_janela.sum():,} | elegíveis para o alvo: {len(base_modelo):,}")
print(f"Atrasos sem entrega recuperados: {base_modelo['atraso_sem_entrega'].sum():,}")
print(f"Taxa de atraso na base do modelo: {base_modelo['atrasou'].mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Análise exploratória e testes estatísticos

# COMMAND ----------

eda = base_modelo[base_modelo["order_purchase_timestamp"] < INICIO_TESTE].copy()
print(f"EDA: {len(eda):,} pedidos de {eda['order_purchase_timestamp'].min():%d/%m/%Y} a "
      f"{eda['order_purchase_timestamp'].max():%d/%m/%Y} | taxa de atraso {eda['atrasou'].mean():.2%}")


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    ordem = np.argsort(p)
    m = len(p)
    ajust = np.maximum.accumulate((m - np.arange(m)) * p[ordem])
    out = np.empty(m)
    out[ordem] = np.minimum(ajust, 1)
    return out


def rotulo_efeito(x):
    x = abs(x)
    return "desprezível" if x < 0.1 else "pequeno" if x < 0.3 else "médio" if x < 0.5 else "grande"


def cramers_v(tabela):
    chi2, p, _, _ = stats.chi2_contingency(tabela)
    n = tabela.to_numpy().sum()
    k = min(tabela.shape) - 1
    return chi2, p, np.sqrt(chi2 / (n * k)) if k > 0 else np.nan

# COMMAND ----------

# Visão geral: volume mensal (base inteira) e taxa de atraso (só desenvolvimento)
fig, axes = plt.subplots(1, 2, figsize=(16, 4.2))
vol = base[na_janela].groupby("ano_mes").size()
axes[0].bar(vol.index, vol.values, color=["#9DB4D8" if m < INICIO_TESTE[:7] else "#DDDDDD" for m in vol.index])
axes[0].set_title("Pedidos por mês (cinza = período de teste, fora da EDA)")
axes[0].tick_params(axis="x", rotation=70)
taxa_mes = eda.groupby("ano_mes")["atrasou"].mean()
axes[1].plot(taxa_mes.index, taxa_mes.values, marker="o", color="#C44E52")
axes[1].yaxis.set_major_formatter(PCT)
axes[1].set_title("Taxa de atraso por mês da compra (desenvolvimento)")
axes[1].tick_params(axis="x", rotation=70)
mostrar(fig, "eda_01_temporal")

# Atraso por faixas de distância, prazo prometido e frete relativo
faixas = {
    "distancia_km": ([0, 100, 500, 1000, 2000, 1e5], ["<100", "100-500", "500-1k", "1k-2k", "2k+"]),
    "prazo_prometido_dias": ([0, 10, 20, 30, 45, 1e4], ["<10", "10-20", "20-30", "30-45", "45+"]),
    "frete_pct_ticket": ([0, 0.1, 0.2, 0.3, 0.5, 1.01], ["<10%", "10-20%", "20-30%", "30-50%", "50%+"]),
}
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for ax, (var, (cortes, rot)) in zip(axes, faixas.items()):
    t = eda.groupby(pd.cut(eda[var], cortes, labels=rot, include_lowest=True), observed=True)["atrasou"].agg(["mean", "size"])
    ax.bar(t.index.astype(str), t["mean"], color="#C44E52")
    for i, (m, n) in enumerate(zip(t["mean"], t["size"])):
        ax.annotate(f"n={n:,}", (i, m), ha="center", va="bottom", fontsize=8)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title(f"Taxa de atraso por {var}")
mostrar(fig, "eda_02_faixas")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
rota = eda.pivot_table(index="regiao_vendedor", columns="regiao_cliente", values="atrasou", aggfunc="mean")
n_rota = eda.pivot_table(index="regiao_vendedor", columns="regiao_cliente", values="atrasou", aggfunc="count")
sns.heatmap(rota.where(n_rota >= 100), annot=True, fmt=".0%", cmap="Reds", ax=axes[0])
axes[0].set_title("Atraso por rota (vazio = menos de 100 pedidos)")
cat = eda.groupby("categoria")["atrasou"].agg(["mean", "size"]).query("size >= 300").sort_values("mean")
axes[1].barh(cat.index[-15:], cat["mean"].iloc[-15:], color="#8172B2")
axes[1].xaxis.set_major_formatter(PCT)
axes[1].set_title("15 categorias com maior atraso (300+ pedidos)")
mostrar(fig, "eda_03_rota_categoria")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Testes de hipótese: atraso

# COMMAND ----------

# Qui-quadrado + V de Cramér (categóricas). Categorias raras são agrupadas para evitar células com contagem baixa.
eda["rota_regiao"] = eda["regiao_vendedor"] + ">" + eda["regiao_cliente"]
eda["frete_gratis"] = (eda["valor_frete"] == 0).astype(int)
top_cat = eda["categoria"].value_counts().index[:25]
eda["categoria_top"] = eda["categoria"].where(eda["categoria"].isin(top_cat), "outras")
linhas = []
for var in ["customer_state", "seller_state", "rota_regiao", "categoria_top", "tipo_pagamento", "mesmo_estado", "frete_gratis"]:
    chi2, p, v = cramers_v(pd.crosstab(eda[var], eda["atrasou"]))
    linhas.append({"variável": var, "teste": "qui-quadrado", "estatística": chi2, "p": p,
                   "efeito": v, "medida": "V de Cramér"})

# Mann-Whitney U + correlação rank-biserial (numéricas): atrasados x no prazo
for var in ["valor_frete", "frete_pct_ticket", "valor_produtos", "distancia_km", "prazo_prometido_dias",
            "peso_total_g", "parcelas", "n_itens"]:
    a1 = eda.loc[eda["atrasou"] == 1, var].dropna()
    a0 = eda.loc[eda["atrasou"] == 0, var].dropna()
    u, p = stats.mannwhitneyu(a1, a0, alternative="two-sided")
    linhas.append({"variável": var, "teste": "Mann-Whitney U", "estatística": u, "p": p,
                   "efeito": 2 * u / (len(a1) * len(a0)) - 1, "medida": "rank-biserial (+ = maior nos atrasados)",
                   "mediana_atrasados": a1.median(), "mediana_no_prazo": a0.median()})

testes_atraso = pd.DataFrame(linhas)
testes_atraso["p_holm"] = holm(testes_atraso["p"])
testes_atraso["rejeita_H0 (5%)"] = testes_atraso["p_holm"] < 0.05
testes_atraso["magnitude"] = testes_atraso["efeito"].map(rotulo_efeito)
display(testes_atraso.round(4))

# COMMAND ----------

# Correlações de Pearson e Spearman com os dias de atraso (entregues). Spearman é a leitura principal:
# as variáveis são assimétricas (frete, preço, peso) e a relação não precisa ser linear.
ent = eda[eda["entregue"] == 1]
vars_corr = ["valor_frete", "frete_pct_ticket", "valor_produtos", "distancia_km", "prazo_prometido_dias", "peso_total_g"]
corr = pd.DataFrame({
    "pearson": [stats.pearsonr(ent[v].fillna(ent[v].median()), ent["atraso_dias"])[0] for v in vars_corr],
    "spearman": [stats.spearmanr(ent[v], ent["atraso_dias"], nan_policy="omit")[0] for v in vars_corr],
}, index=vars_corr)
corr["magnitude (spearman)"] = corr["spearman"].map(rotulo_efeito)
display(corr.round(3).reset_index().rename(columns={"index": "variável x atraso_dias"}))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(ent[vars_corr + ["atraso_dias"]].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Spearman: preço, frete, logística e atraso")
mostrar(fig, "eda_04_correlacoes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Testes de hipótese: satisfação

# COMMAND ----------

av = eda[eda["review_score"].notna() & (eda["entregue"] == 1)].copy()
av["faixa_atraso"] = pd.cut(av["atraso_dias"], [-200, -15, -7, 0, 3, 7, 15, 500],
                            labels=["15+ antes", "7-15 antes", "0-7 antes", "1-3 atraso", "4-7 atraso",
                                    "8-15 atraso", "15+ atraso"])

fig, axes = plt.subplots(1, 3, figsize=(18, 4.3))
av["review_score"].value_counts(normalize=True).sort_index().plot.bar(ax=axes[0], rot=0, color="#4C72B0")
axes[0].yaxis.set_major_formatter(PCT)
axes[0].set_title("Distribuição das notas")
av.groupby("faixa_atraso", observed=True)["review_negativa"].mean().plot.bar(ax=axes[1], color="#C44E52", rot=30)
axes[1].yaxis.set_major_formatter(PCT)
axes[1].set_title("% de notas 1-2 por dias de atraso")
av.groupby("atrasou")["review_score"].mean().rename({0: "no prazo", 1: "atrasado"}).plot.bar(ax=axes[2], rot=0,
                                                                                             color=["#55A868", "#C44E52"])
axes[2].set_title("Nota média")
mostrar(fig, "eda_05_satisfacao")

s1, s0 = av.loc[av["atrasou"] == 1, "review_score"], av.loc[av["atrasou"] == 0, "review_score"]
u, p_u = stats.mannwhitneyu(s1, s0, alternative="two-sided")
chi2, p_chi, v = cramers_v(pd.crosstab(av["atrasou"], av["review_negativa"]))
kw = stats.kruskal(*[g["review_score"].to_numpy() for _, g in av.groupby("faixa_atraso", observed=True)])
k = av["faixa_atraso"].nunique()
rho_atraso = stats.spearmanr(av["atraso_dias"], av["review_score"])
no_prazo = av[av["atrasou"] == 0]
rho_frete = stats.spearmanr(no_prazo["frete_pct_ticket"], no_prazo["review_score"], nan_policy="omit")
rho_preco = stats.spearmanr(no_prazo["valor_produtos"], no_prazo["review_score"], nan_policy="omit")

testes_satisf = pd.DataFrame([
    ["H5", "Nota: atrasado x no prazo", "Mann-Whitney U", p_u, 2 * u / (len(s1) * len(s0)) - 1, "rank-biserial"],
    ["H5", "Atraso x nota negativa (1-2)", "Qui-quadrado", p_chi, v, "V de Cramér"],
    ["H5", "Nota entre faixas de atraso", "Kruskal-Wallis", kw.pvalue, (kw.statistic - k + 1) / (len(av) - k), "épsilon²"],
    ["H5", "Dias de atraso x nota", "Spearman", rho_atraso.pvalue, rho_atraso.statistic, "rho"],
    ["H6", "Frete/ticket x nota (só no prazo)", "Spearman", rho_frete.pvalue, rho_frete.statistic, "rho"],
    ["H6", "Valor dos produtos x nota (só no prazo)", "Spearman", rho_preco.pvalue, rho_preco.statistic, "rho"],
], columns=["hipótese", "comparação", "teste", "p", "efeito", "medida"])
testes_satisf["p_holm"] = holm(testes_satisf["p"])
testes_satisf["magnitude"] = testes_satisf["efeito"].map(rotulo_efeito)
display(testes_satisf.round(4))
print(f"Nota média: {s0.mean():.2f} no prazo x {s1.mean():.2f} atrasado | "
      f"notas 1-2: {av.loc[av['atrasou'] == 0, 'review_negativa'].mean():.0%} x {av.loc[av['atrasou'] == 1, 'review_negativa'].mean():.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature engineering (variáveis candidatas)

# COMMAND ----------

def _acumulados(eventos, chave):
    ev = eventos.sort_values("_t", kind="mergesort")
    g = ev.groupby(chave, sort=False)
    out = ev[[chave, "_t"]].copy()
    out["_n"] = g.cumcount() + 1
    out["_s_atraso"] = g["atrasou"].cumsum()
    out["_s_dias"] = g["dias_entrega"].cumsum()
    return out


def historico_asof(ped, chave, janela_dias=None, minimo=1):
    """Estatísticas das entregas concluídas ESTRITAMENTE antes de cada compra, por `chave`.
    Com janela_dias, só as concluídas nos últimos N dias antes da compra."""
    ev = ped.loc[ped["entregue"] == 1, [chave, "order_delivered_customer_date", "atrasou", "dias_entrega"]].copy()
    ev = ev.rename(columns={"order_delivered_customer_date": "_t"})
    ev["dias_entrega"] = ev["dias_entrega"].clip(lower=0, upper=60)
    acc = _acumulados(ev, chave)
    cols = ["_n", "_s_atraso", "_s_dias"]

    def ate(deslocamento, incluir_igual):
        q = ped[[chave, "order_purchase_timestamp"]].copy()
        q["_t"] = q["order_purchase_timestamp"] - deslocamento
        q["_idx"] = np.arange(len(q))
        q = q.sort_values("_t", kind="mergesort")
        h = pd.merge_asof(q, acc, on="_t", by=chave, direction="backward", allow_exact_matches=incluir_igual)
        return h.sort_values("_idx")[cols].fillna(0).to_numpy()

    h = ate(pd.Timedelta(0), False)                       # tudo o que terminou antes da compra
    if janela_dias is not None:                           # menos o que terminou antes de (compra - janela)
        h = h - ate(pd.Timedelta(days=janela_dias), True)
    n = h[:, 0]
    ok = n >= minimo
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({"n": n, "taxa_atraso": np.where(ok, h[:, 1] / n, np.nan),
                             "dias_entrega": np.where(ok, h[:, 2] / n, np.nan)}, index=ped.index)


def contagem_anterior(ped, chave, janela_dias):
    """Nº de compras com a mesma chave nos últimos N dias, estritamente antes de cada compra."""
    out = np.zeros(len(ped))
    t_all = ped["order_purchase_timestamp"].to_numpy()
    for _, pos in ped.groupby(chave, sort=False).indices.items():
        t = t_all[pos]
        ordenado = np.sort(t)
        out[pos] = (np.searchsorted(ordenado, t, "left")
                    - np.searchsorted(ordenado, t - np.timedelta64(janela_dias, "D"), "left"))
    return out


def criar_features(ped):
    ped = ped.sort_values("order_purchase_timestamp", kind="mergesort").copy()
    ts = ped["order_purchase_timestamp"]

    # ---------------------------------------------------------- Preço e desconto
    ped["dif_pago_vs_ticket_pct"] = (ped["valor_pago"] - ped["ticket_total"]) / ped["ticket_total"]
    ped["usou_voucher"] = (ped["valor_voucher"].fillna(0) > 0).astype(int)
    ped["voucher_pct"] = ped["valor_voucher"] / ped["valor_pago"].where(ped["valor_pago"] > 0)
    ped["valor_parcela"] = ped["valor_pago"] / ped["parcelas"].clip(lower=1)

    # ---------------------------------------------------------- Frete e prazos
    ped["peso_cubado_kg"] = np.fmax(ped["peso_total_g"] / 1000, ped["volume_total_cm3"] / 6000)   # fator Correios
    ped["frete_por_kg"] = ped["valor_frete"] / ped["peso_cubado_kg"].clip(lower=0.1)
    ped["frete_gratis"] = (ped["valor_frete"] == 0).astype(int)

    # ---------------------------------------------------------- Logística e localidade
    ped["mesma_cidade"] = ((ped["cidade_cliente"] == ped["cidade_vendedor"]).fillna(False) & (ped["mesmo_estado"] == 1)).astype(int)
    capital = ped["customer_state"].map({uf: c[0] for uf, c in CAPITAIS.items()})
    ped["cliente_em_capital"] = (ped["cidade_cliente"] == capital).fillna(False).astype(int)

    ped["_rota_uf"] = ped["seller_state"] + ">" + ped["customer_state"]
    ped["_rota_reg"] = ped["regiao_vendedor"] + ">" + ped["regiao_cliente"]
    ped["_geral"] = "BR"
    ped["_vendedor"] = ped["vendedor_principal"].fillna("NA")
    ped["_cliente"] = ped["customer_unique_id"].fillna(ped["customer_id"])

    uf = historico_asof(ped, "_rota_uf", minimo=MIN_EVENTOS_ROTA)
    reg = historico_asof(ped, "_rota_reg", minimo=MIN_EVENTOS_ROTA)
    ger = historico_asof(ped, "_geral", minimo=30)
    ped["taxa_atraso_rota_hist"] = uf["taxa_atraso"].fillna(reg["taxa_atraso"]).fillna(ger["taxa_atraso"])
    ped["entrega_rota_hist_dias"] = uf["dias_entrega"].fillna(reg["dias_entrega"]).fillna(ger["dias_entrega"])
    uf_r = historico_asof(ped, "_rota_uf", JANELA_RECENTE_DIAS, MIN_EVENTOS_RECENTE)
    reg_r = historico_asof(ped, "_rota_reg", JANELA_RECENTE_DIAS, MIN_EVENTOS_RECENTE)
    ger_r = historico_asof(ped, "_geral", JANELA_RECENTE_DIAS, MIN_EVENTOS_RECENTE)
    ped["taxa_atraso_rota_recente"] = uf_r["taxa_atraso"].fillna(reg_r["taxa_atraso"]).fillna(ger_r["taxa_atraso"])
    ped["taxa_atraso_plataforma_recente"] = ger_r["taxa_atraso"]
    destino = historico_asof(ped, "customer_state", minimo=MIN_EVENTOS_ROTA)
    ped["taxa_atraso_destino_hist"] = destino["taxa_atraso"].fillna(ger["taxa_atraso"])
    ped["pedidos_plataforma_7d"] = contagem_anterior(ped, "_geral", 7)

    # Folgas: prazo prometido menos o tempo que a rota/vendedor costumava levar
    ped["folga_prometida"] = ped["prazo_prometido_dias"] - ped["entrega_rota_hist_dias"]

    # ---------------------------------------------------------- Produto e vendedor
    vend = historico_asof(ped, "_vendedor", minimo=MIN_EVENTOS_VENDEDOR)
    vend_r = historico_asof(ped, "_vendedor", JANELA_RECENTE_DIAS, MIN_EVENTOS_VENDEDOR)
    ped["vendedor_entregas_concluidas"] = vend["n"]
    ped["vendedor_taxa_atraso_hist"] = vend["taxa_atraso"]
    ped["vendedor_dias_entrega_hist"] = vend["dias_entrega"]
    ped["vendedor_taxa_atraso_recente"] = vend_r["taxa_atraso"]
    ped["vendedor_pedidos_anteriores"] = ped.groupby("_vendedor")["order_purchase_timestamp"].rank(method="min") - 1
    ped["vendedor_pedidos_30d"] = contagem_anterior(ped, "_vendedor", 30)
    primeira_venda = ped.groupby("_vendedor")["order_purchase_timestamp"].transform("min")
    ped["vendedor_idade_dias"] = dias(ts - primeira_venda)
    ped["folga_vs_vendedor"] = ped["prazo_prometido_dias"] - ped["vendedor_dias_entrega_hist"]

    # ---------------------------------------------------------- Calendário e cliente
    ped["mes"] = ts.dt.month
    ped["dia_semana"] = ts.dt.dayofweek
    ped["hora"] = ts.dt.hour
    ped["fim_de_semana"] = (ped["dia_semana"] >= 5).astype(int)
    ped["cliente_recorrente"] = (ped.groupby("_cliente")["order_purchase_timestamp"].rank(method="min") > 1).astype(int)

    return ped.drop(columns=["_rota_uf", "_rota_reg", "_geral", "_vendedor", "_cliente"])


t0 = time.time()
base_feat = criar_features(base)
print(f"Features criadas em {time.time() - t0:.0f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Razões entre variáveis

# COMMAND ----------

def razao_segura(num, den):
    """num / den; denominador <= 0 ou ausente vira nulo (sem infinito)."""
    num, den = num.astype(float), den.astype(float)
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


# nome: (componentes, descrição). Todas usam só variáveis conhecidas no momento da compra.
RAZOES_NOVAS = {
    "velocidade_exigida_km_dia": (("distancia_km", "prazo_prometido_dias"),
                                  "Km que o pedido precisa percorrer por dia de prazo prometido"),
    "razao_prazo_rota": (("prazo_prometido_dias", "entrega_rota_hist_dias"),
                         "Prazo prometido / tempo histórico da rota (< 1 = promessa apertada)"),
    "razao_prazo_vendedor": (("prazo_prometido_dias", "vendedor_dias_entrega_hist"),
                             "Prazo prometido / tempo histórico de entrega do vendedor"),
    "frete_por_km": (("valor_frete", "distancia_km"),
                     "Frete por km (frete barato para a distância sugere modal mais lento)"),
    "densidade_g_cm3": (("peso_total_g", "volume_total_cm3"),
                        "Peso / volume: itens leves e volumosos são cobrados e manuseados pela cubagem"),
    "pico_vendedor": (("vendedor_pedidos_30d", "vendedor_pedidos_anteriores", "vendedor_idade_dias"),
                      "Pedidos do vendedor nos últimos 30 dias / média mensal histórica dele (sobrecarga)"),
    "tendencia_atraso_rota": (("taxa_atraso_rota_recente", "taxa_atraso_rota_hist"),
                              "Taxa de atraso recente da rota / taxa histórica (> 1 = rota piorando)"),
    "tendencia_atraso_vendedor": (("vendedor_taxa_atraso_recente", "vendedor_taxa_atraso_hist"),
                                  "Taxa de atraso recente do vendedor / taxa histórica (> 1 = vendedor piorando)"),
}


def criar_razoes(df):
    df = df.copy()
    df["velocidade_exigida_km_dia"] = razao_segura(df["distancia_km"], df["prazo_prometido_dias"])
    df["razao_prazo_rota"] = razao_segura(df["prazo_prometido_dias"], df["entrega_rota_hist_dias"])
    df["razao_prazo_vendedor"] = razao_segura(df["prazo_prometido_dias"], df["vendedor_dias_entrega_hist"])
    df["frete_por_km"] = razao_segura(df["valor_frete"], df["distancia_km"].clip(lower=1))
    df["densidade_g_cm3"] = razao_segura(df["peso_total_g"], df["volume_total_cm3"])
    media_mensal = razao_segura(df["vendedor_pedidos_anteriores"], df["vendedor_idade_dias"].clip(lower=30)) * 30
    df["pico_vendedor"] = razao_segura(df["vendedor_pedidos_30d"], media_mensal)
    df["tendencia_atraso_rota"] = razao_segura(df["taxa_atraso_rota_recente"], df["taxa_atraso_rota_hist"])
    df["tendencia_atraso_vendedor"] = razao_segura(df["vendedor_taxa_atraso_recente"], df["vendedor_taxa_atraso_hist"])
    return df


base_feat = criar_razoes(base_feat)
desc = base_feat.loc[base_feat["elegivel"] == 1, list(RAZOES_NOVAS)].describe(percentiles=[0.05, 0.5, 0.95]).T
desc["pct_nulos"] = base_feat.loc[base_feat["elegivel"] == 1, list(RAZOES_NOVAS)].isna().mean()
display(desc.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dicionário de dados

# COMMAND ----------

D = []   # (feature, grupo, momento, tipo, descrição)
def _d(grupo, momento, tipo, itens):
    D.extend((f, grupo, momento, tipo, desc) for f, desc in itens)

_d("Preço e desconto", "compra", "numérica", [
    ("valor_produtos", "Soma dos preços dos itens"),
    ("preco_medio_item", "Preço médio por item"),
    ("n_itens", "Quantidade de itens no pedido"),
    ("ticket_total", "Produtos + frete"),
    ("dif_pago_vs_ticket_pct", "(valor pago - ticket) / ticket: negativo = desconto/voucher, positivo = juros do parcelamento"),
    ("usou_voucher", "Pagou parte com voucher (escolhido no checkout)"),
    ("voucher_pct", "Parcela do pagamento feita com voucher"),
    ("parcelas", "Número de parcelas escolhido"),
    ("valor_parcela", "Valor pago / parcelas"),
    ("preco_rel_categoria", "Valor dos produtos / mediana da categoria (calculada no pipeline, só com o treino)"),
])
_d("Preço e desconto", "compra", "categórica", [("tipo_pagamento", "Forma de pagamento de maior valor")])
_d("Frete e prazos estimados", "compra", "numérica", [
    ("valor_frete", "Frete cotado (soma dos itens)"),
    ("frete_pct_ticket", "Frete / ticket total"),
    ("frete_por_kg", "Frete / peso cubado"),
    ("frete_gratis", "Frete zerado"),
    ("frete_rel_rota", "Frete / mediana do frete da rota UF>UF (calculada no pipeline, só com o treino)"),
    ("prazo_prometido_dias", "Dias entre a compra e a data prometida ao cliente"),
    ("entrega_rota_hist_dias", "Tempo médio de entrega da rota nas entregas concluídas antes da compra"),
    ("folga_prometida", "Prazo prometido - tempo histórico da rota"),
    ("folga_vs_vendedor", "Prazo prometido - tempo histórico de entrega do vendedor"),
])
_d("Logística e localidade", "compra", "categórica", [
    ("customer_state", "UF do cliente"), ("seller_state", "UF do vendedor"),
    ("regiao_cliente", "Região do cliente"), ("regiao_vendedor", "Região do vendedor"),
])
_d("Logística e localidade", "compra", "numérica", [
    ("mesmo_estado", "Vendedor e cliente na mesma UF"),
    ("mesma_regiao", "Vendedor e cliente na mesma região"),
    ("mesma_cidade", "Vendedor e cliente na mesma cidade"),
    ("cliente_em_capital", "Cliente mora na capital da UF"),
    ("distancia_km", "Distância vendedor-cliente (CEP; capital da UF se faltar geolocalização)"),
    ("taxa_atraso_rota_hist", "Taxa de atraso da rota UF>UF antes da compra (região>região ou geral se poucos casos)"),
    ("taxa_atraso_rota_recente", f"Idem, só entregas concluídas nos últimos {JANELA_RECENTE_DIAS} dias"),
    ("taxa_atraso_destino_hist", "Taxa de atraso para a UF do cliente antes da compra"),
    ("taxa_atraso_plataforma_recente", f"Taxa de atraso de toda a plataforma nos últimos {JANELA_RECENTE_DIAS} dias (greves, picos)"),
    ("pedidos_plataforma_7d", "Volume de compras na plataforma nos 7 dias anteriores (pressão logística)"),
])
_d("Produto e vendedor", "compra", "categórica", [("categoria", "Categoria do item principal")])
_d("Produto e vendedor", "compra", "numérica", [
    ("peso_total_g", "Peso total dos itens"),
    ("volume_total_cm3", "Volume total dos itens"),
    ("peso_cubado_kg", "Maior entre peso real e volume/6000"),
    ("maior_dimensao_cm", "Maior dimensão entre os itens"),
    ("fotos_media", "Fotos por anúncio (média)"),
    ("tam_descricao_media", "Tamanho da descrição do anúncio (média)"),
    ("n_vendedores", "Vendedores distintos no pedido"),
    ("vendedor_pedidos_anteriores", "Pedidos do vendedor antes desta compra"),
    ("vendedor_pedidos_30d", "Pedidos do vendedor nos 30 dias anteriores"),
    ("vendedor_idade_dias", "Dias desde a primeira venda do vendedor"),
    ("vendedor_entregas_concluidas", "Entregas do vendedor concluídas antes da compra"),
    ("vendedor_taxa_atraso_hist", "Taxa de atraso do vendedor antes da compra (nulo se < 3 entregas)"),
    ("vendedor_taxa_atraso_recente", f"Idem, últimos {JANELA_RECENTE_DIAS} dias"),
    ("vendedor_dias_entrega_hist", "Tempo médio de entrega do vendedor antes da compra"),
])
_d("Calendário e cliente", "compra", "numérica", [
    ("mes", "Mês da compra"), ("dia_semana", "Dia da semana (0 = segunda)"), ("hora", "Hora da compra"),
    ("fim_de_semana", "Compra no sábado ou domingo"),
    ("cliente_recorrente", "Cliente já comprou antes"),
])
_d("Razões entre variáveis", "compra", "numérica", [(n, d) for n, (_, d) in RAZOES_NOVAS.items()])
# Pós-compra: documentadas para deixar explícito por que NÃO entram
_d("Pós-compra (proibida no modelo)", "pós-compra", "alvo", [
    ("atrasou", "ALVO: entrega após a data prometida"),
    ("atraso_dias", "Dias de atraso (define o alvo)"),
])
_d("Pós-compra (proibida no modelo)", "pós-compra", "numérica", [
    ("horas_aprovacao", "Horas entre compra e aprovação do pagamento"),
    ("dias_aprovacao_postagem", "Dias entre aprovação e postagem"),
    ("postou_apos_limite", "Vendedor postou depois do limite"),
    ("dias_transporte", "Dias entre postagem e entrega"),
    ("dias_entrega", "Dias entre compra e entrega"),
    ("entregue", "Pedido entregue"),
    ("review_score", "Nota da avaliação"),
    ("review_negativa", "Nota 1 ou 2"),
    ("tem_comentario", "Avaliação com comentário"),
    ("data_limite_postagem", "shipping_limit_date: pode ser redefinida após a aprovação"),
])
dicionario = pd.DataFrame(D, columns=["feature", "grupo", "momento", "tipo", "descricao"])
dicionario["calculada_no_pipeline"] = dicionario["feature"].isin(["preco_rel_categoria", "frete_rel_rota"])

FEATURES_COMPRA = dicionario.query("momento == 'compra'")
CATEGORICAS_CAND = FEATURES_COMPRA.query("tipo == 'categórica'")["feature"].tolist()
NUMERICAS_CAND = FEATURES_COMPRA.query("tipo == 'numérica'")["feature"].tolist()
NUMERICAS_X_CAND = [f for f in NUMERICAS_CAND if f not in ("preco_rel_categoria", "frete_rel_rota")]
GRUPO = dict(zip(dicionario["feature"], dicionario["grupo"]))

proibidas = set(dicionario.query("momento != 'compra'")["feature"])
assert not proibidas & set(CATEGORICAS_CAND + NUMERICAS_X_CAND), "vazamento no dicionário"
faltando = [f for f in CATEGORICAS_CAND + NUMERICAS_X_CAND if f not in base_feat.columns]
assert not faltando, f"features não construídas: {faltando}"

print(f"{len(FEATURES_COMPRA)} candidatas do momento da compra ({len(CATEGORICAS_CAND)} categóricas, "
      f"{len(NUMERICAS_CAND)} numéricas)")
display(FEATURES_COMPRA.groupby("grupo").size().rename("nº de candidatas").reset_index())
display(dicionario)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auditoria automática contra vazamento

# COMMAND ----------

# 1) força bruta
rng = np.random.default_rng(SEED)
amostra = base_feat[base_feat["elegivel"] == 1].sample(300, random_state=SEED)
entregues = base_feat[base_feat["entregue"] == 1]
erros = 0
for _, r in amostra.iterrows():
    ant = entregues[(entregues["vendedor_principal"] == r["vendedor_principal"])
                    & (entregues["order_delivered_customer_date"] < r["order_purchase_timestamp"])]
    erros += int(len(ant) != r["vendedor_entregas_concluidas"])
    if len(ant) >= MIN_EVENTOS_VENDEDOR:
        erros += int(not np.isclose(ant["atrasou"].mean(), r["vendedor_taxa_atraso_hist"]))
assert erros == 0, f"{erros} divergências no histórico do vendedor"
print("Força bruta OK: o histórico do vendedor usa só entregas concluídas antes de cada compra")

# 2) poder preditivo de cada variável sozinha (período de desenvolvimento)
dev = base_feat[(base_feat["elegivel"] == 1) & (base_feat["order_purchase_timestamp"] >= DATA_INICIO) & (base_feat["order_purchase_timestamp"] < INICIO_TESTE)]
y_dev = dev["atrasou"].to_numpy()
aud = []
for f in NUMERICAS_X_CAND + ["atraso_dias"]:
    x = dev[f].fillna(dev[f].median()).to_numpy()
    roc = roc_auc_score(y_dev, x)
    sinal = 1 if roc >= 0.5 else -1
    aud.append({"feature": f, "roc_auc": max(roc, 1 - roc), "pr_auc": average_precision_score(y_dev, sinal * x)})
aud = pd.DataFrame(aud).sort_values("roc_auc", ascending=False)
aud["suspeita"] = (aud["roc_auc"] > 0.85) | (aud["pr_auc"] > 0.5)
display(aud.round(3).head(12))
assert aud.set_index("feature").loc["atraso_dias", "suspeita"], "o controle positivo deveria ser detectado"
suspeitas = aud.query("suspeita and feature != 'atraso_dias'")["feature"].tolist()
assert not suspeitas, f"features suspeitas de vazamento: {suspeitas}"
print(f"Auditoria OK (taxa base {y_dev.mean():.3f}): só o controle pós-compra foi sinalizado")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Estratégia de validação temporal (recorte equilibrado sem meses atípicos)

# COMMAND ----------

bm = base_feat[(base_feat["elegivel"] == 1)
               & base_feat["order_purchase_timestamp"].between(DATA_INICIO, f"{DATA_FIM} 23:59:59")].copy()
bm["atrasou"] = bm["atrasou"].astype(int)
TS = bm["order_purchase_timestamp"]

# Recorte escolhido na comparação de estratégias (PR-AUC média na validação 0,193 contra 0,176 da expansiva):
# todo o histórico disponível no corte, SEM meses atípicos e com cada mês limitado à mediana de pedidos por mês.
# Tudo usa só pedidos comprados antes de (corte - embargo) e com resultado já conhecido no corte.


def base_conhecida(corte):
    """Tudo o que seria utilizável no corte: comprado antes do embargo e com resultado conhecido."""
    corte = pd.Timestamp(corte)
    return bm[(TS < corte - pd.Timedelta(days=EMBARGO_DIAS)) & (bm["data_conhecimento_alvo"] < corte)]


def taxa_mensal(df):
    t = df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["atrasou"].agg(["mean", "size"])
    return t[t["size"] >= MIN_PEDIDOS_MES]


def limite_atipico(referencia):
    """Taxa acima da qual um mês é atípico: FATOR_ATIPICO x a mediana das taxas mensais da referência."""
    return FATOR_ATIPICO * taxa_mensal(referencia)["mean"].median()


def meses_atipicos(df, referencia=None):
    """Meses de `df` com taxa de atraso acima do limite calculado na `referencia` (por padrão, o próprio df)."""
    t = taxa_mensal(df)
    return set(t.index[t["mean"] > limite_atipico(df if referencia is None else referencia)])


def limitar_volume(df):
    """Cada mês entra com no máximo a mediana de pedidos por mês (amostra aleatória dentro do mês, que preserva a
    taxa de atraso dele). Assim um mês de pico não domina o treino."""
    mes = df["order_purchase_timestamp"].dt.to_period("M")
    teto = int(mes.value_counts().median())
    ordem = pd.Series(np.random.default_rng(SEED).random(len(df)), index=df.index).groupby(mes).rank(method="first")
    return df[ordem <= teto]


def treino_ate(corte):
    """Estratégia 'equilibrado_sem_atipicos'."""
    df = base_conhecida(corte)
    df = df[~df["order_purchase_timestamp"].dt.to_period("M").isin(meses_atipicos(df))]
    return limitar_volume(df)


def Xy(df):
    return df[FEATURES_X], df["atrasou"].to_numpy()


def periodo(ini, fim):
    """Texto 'mm/aaaa a mm/aaaa' de um intervalo [ini, fim)."""
    return f"{pd.Timestamp(ini):%m/%Y} a {pd.Timestamp(fim) - pd.Timedelta(days=1):%m/%Y}"


folds = []
for ini, fim in FOLDS_VALIDACAO:
    tr = treino_ate(ini)
    assert tr["atrasou"].sum() >= 50, f"fold {ini}: só {tr['atrasou'].sum()} atrasos no treino"
    folds.append({"nome": f"valida {ini[:7]} a {pd.Timestamp(fim) - pd.Timedelta(days=1):%Y-%m}",
                  "tr": tr, "va": bm[(TS >= ini) & (TS < fim)], "aval": (pd.Timestamp(ini), pd.Timestamp(fim))})

treino_final = treino_ate(FIM_TREINO_FINAL)

# Calibração coerente com o recorte: meses candidatos entre FIM_TREINO_FINAL e o teste, SEM os meses atípicos.
# O limite de "atípico" vem do histórico do treino (nada do teste participa). Na versão anterior a calibração era
# só mar/2018, um mês de crise (13% de atraso na parte conhecida), e as probabilidades saíam infladas no teste.
calib_cand = bm[(TS >= FIM_TREINO_FINAL) & (TS < INICIO_TESTE) & (bm["data_conhecimento_alvo"] < INICIO_TESTE)]
CALIB_ATIPICOS = meses_atipicos(calib_cand, referencia=base_conhecida(FIM_TREINO_FINAL)) if CALIBRAR_SEM_ATIPICOS else set()
calib = calib_cand[~calib_cand["order_purchase_timestamp"].dt.to_period("M").isin(CALIB_ATIPICOS)]
if calib["atrasou"].sum() < 100:
    print("AVISO: calibração sem atípicos ficou pequena demais; usando todos os meses candidatos")
    calib, CALIB_ATIPICOS = calib_cand, set()
teste = bm[TS >= INICIO_TESTE]

# A seleção de variáveis (seção 7) continua usando o treino do fold 1 da versão ORIGINAL (todo o histórico):
# as 30 variáveis ficam iguais e a diferença nos resultados vem só do recorte e dos ajustes de modelagem.
TREINO_SELECAO = base_conhecida(FOLDS_VALIDACAO[0][0])

assert treino_final["order_purchase_timestamp"].max() < pd.Timestamp(FIM_TREINO_FINAL) - pd.Timedelta(days=EMBARGO_DIAS)
assert calib["order_purchase_timestamp"].min() >= pd.Timestamp(FIM_TREINO_FINAL)
assert teste["order_purchase_timestamp"].min() >= pd.Timestamp(INICIO_TESTE)
assert not set(teste["order_id"]) & (set(treino_final["order_id"]) | set(calib["order_id"]))
assert not set(treino_final["order_id"]) & set(calib["order_id"])


def desc(df):
    m = df["order_purchase_timestamp"].dt.to_period("M")
    return f"{periodo(m.min().start_time, m.max().end_time)} ({m.nunique()} meses)"


fora_treino = sorted(str(m) for m in meses_atipicos(base_conhecida(FIM_TREINO_FINAL)))
print(f"Recorte: equilibrado sem atípicos | limite de mês atípico: "
      f"{limite_atipico(base_conhecida(FIM_TREINO_FINAL)):.1%} de atraso | embargo {EMBARGO_DIAS} dias")
print(f"Meses atípicos fora do treino final: {fora_treino}")
print(f"Calibração: {sorted(calib['ano_mes'].unique())} | meses atípicos fora da calibração: "
      f"{sorted(str(m) for m in CALIB_ATIPICOS)}")
resumo = [{"conjunto": f["nome"], "compras_treino": desc(f["tr"]), "treino": len(f["tr"]),
           "atraso_treino": f["tr"]["atrasou"].mean(), "avaliação": len(f["va"]),
           "atraso_avaliação": f["va"]["atrasou"].mean()} for f in folds]
resumo += [{"conjunto": f"final (calibração {', '.join(sorted(calib['ano_mes'].unique()))})",
            "compras_treino": desc(treino_final), "treino": len(treino_final),
            "atraso_treino": treino_final["atrasou"].mean(), "avaliação": len(calib),
            "atraso_avaliação": calib["atrasou"].mean()},
           {"conjunto": f"TESTE {periodo(INICIO_TESTE, pd.Timestamp(DATA_FIM) + pd.Timedelta(days=1))}",
            "compras_treino": "-", "treino": np.nan, "atraso_treino": np.nan,
            "avaliação": len(teste), "atraso_avaliação": teste["atrasou"].mean()}]
display(pd.DataFrame(resumo).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Volume de vendas e taxa de atraso: do treino ao teste

# COMMAND ----------

# Volume de vendas e taxa de atraso do treino ao teste, para a estratégia escolhida.
# Os conjuntos da versão ORIGINAL são recalculados com as mesmas regras, para comparar.
CORTE_ORIGINAL = pd.Timestamp("2018-01-01")
originais = {
    "Treino original (jan-nov/17)": bm[(TS < CORTE_ORIGINAL - pd.Timedelta(days=EMBARGO_DIAS))
                                      & (bm["data_conhecimento_alvo"] < CORTE_ORIGINAL)],
    "Calibração original (jan-mar/18)": bm[(TS >= CORTE_ORIGINAL) & (TS < INICIO_TESTE)
                                           & (bm["data_conhecimento_alvo"] < INICIO_TESTE)],
}
novos = {f"Treino novo ({ESTRATEGIA_ATIVA})": treino_final,
         f"Calibração nova ({', '.join(sorted(calib['ano_mes'].unique()))})": calib,
         f"Teste ({periodo(INICIO_TESTE, pd.Timestamp(DATA_FIM) + pd.Timedelta(days=1))})": teste}

ORDEM_PAPEIS = ["Não usado pela estratégia", "Treino", "Excluído (embargo, resultado desconhecido ou mês atípico)",
                "Calibração", "Teste"]
CORES_PAPEL = dict(zip(ORDEM_PAPEIS, ["#E3E3E3", "#4C72B0", "#B0B0B0", "#DD8452", "#C44E52"]))
papel = pd.Series(ORDEM_PAPEIS[2], index=bm.index)
papel[bm["order_id"].isin(base_conhecida(FIM_TREINO_FINAL)["order_id"])] = ORDEM_PAPEIS[0]
papel[bm["order_id"].isin(treino_final["order_id"])] = "Treino"
papel[bm["order_id"].isin(calib["order_id"])] = "Calibração"
papel[bm["order_id"].isin(teste["order_id"])] = "Teste"

meses = sorted(bm["ano_mes"].unique())
vol = pd.crosstab(bm["ano_mes"], papel).reindex(index=meses, columns=ORDEM_PAPEIS, fill_value=0)
papel_mes = vol.idxmax(axis=1)            # papel predominante de cada mês (para sombrear o gráfico de taxa)

# ------------------------------------------------------------------ 1) Volume de vendas
gmv = bm.groupby("ano_mes")["ticket_total"].sum().reindex(meses) / 1e6
fig, axes = plt.subplots(1, 2, figsize=(20, 5.5), gridspec_kw={"width_ratios": [3, 1.2]})
ax = axes[0]
base_y = np.zeros(len(meses))
for p_ in ORDEM_PAPEIS:
    ax.bar(meses, vol[p_], bottom=base_y, color=CORES_PAPEL[p_], label=f"{p_} ({vol[p_].sum():,})")
    base_y += vol[p_].to_numpy()
ax.set_ylabel("Pedidos elegíveis")
ax.set_ylim(0, base_y.max() * 1.1)
ax.tick_params(axis="x", rotation=70)
ax2 = ax.twinx()
ax2.plot(meses, gmv.values, color="black", marker="o", lw=1.5, label="Faturamento (R$ mi)")
ax2.set_ylabel("Faturamento (R$ milhões)")
ax2.set_ylim(0, gmv.max() * 1.25)
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
ax.set_title(f"Volume de vendas por mês e papel no recorte (estratégia '{ESTRATEGIA_ATIVA}')")

ax = axes[1]
conj = {**originais, **novos}
nomes_c = list(conj)
n_ped = [len(conj[c]) for c in nomes_c]
n_atr = [int(conj[c]["atrasou"].sum()) for c in nomes_c]
ax.barh(nomes_c, n_ped, color=["#9DB4D8", "#F0C2A0", CORES_PAPEL["Treino"], CORES_PAPEL["Calibração"],
                               CORES_PAPEL["Teste"]])
for i, (n, a) in enumerate(zip(n_ped, n_atr)):
    ax.annotate(f" {n:,} pedidos | {a:,} atrasos", (n, i), va="center", fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, max(n_ped) * 1.6)
ax.set_title("Tamanho de cada conjunto (original x novo)")
mostrar(fig, f"recorte_01_volume_{VERSAO_RECORTE}")

# ------------------------------------------------------------------ 2) Taxa de atraso do treino ao teste
taxa_mes = bm.groupby("ano_mes")["atrasou"].mean().reindex(meses)
fig, ax = plt.subplots(figsize=(18, 5.5))
for i, m in enumerate(meses):
    ax.axvspan(i - 0.5, i + 0.5, color=CORES_PAPEL[papel_mes[m]], alpha=0.2, lw=0)
ax.plot(range(len(meses)), taxa_mes.values, color="black", marker="o", lw=1.6, label="Taxa de atraso do mês")
for i, v in enumerate(taxa_mes.values):
    ax.annotate(f"{v:.1%}", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
for nome_c, df_ in [("Treino", treino_final), ("Calibração", calib), ("Teste", teste)]:
    usados = [i for i, m in enumerate(meses) if m in set(df_["ano_mes"])]
    t = df_["atrasou"].mean()
    ax.hlines(t, min(usados) - 0.5, max(usados) + 0.5, color=CORES_PAPEL[nome_c], lw=3,
              label=f"Média {nome_c.lower()}: {t:.1%}")
t_orig = originais["Treino original (jan-nov/17)"]["atrasou"].mean()
ax.axhline(t_orig, color="#4C72B0", ls=":", lw=1.2, label=f"Média do treino ORIGINAL (jan-nov/17): {t_orig:.1%}")
ax.legend(handles=ax.get_legend_handles_labels()[0]
          + [plt.Rectangle((0, 0), 1, 1, color=CORES_PAPEL[k], alpha=0.3) for k in ORDEM_PAPEIS],
          labels=ax.get_legend_handles_labels()[1] + [f"Mês: {k.lower()}" for k in ORDEM_PAPEIS],
          fontsize=8, loc="upper left", ncol=2)
ax.set_xticks(range(len(meses)))
ax.set_xticklabels(meses, rotation=70)
ax.yaxis.set_major_formatter(PCT)
ax.set_ylim(0, taxa_mes.max() * 1.45)
ax.set_title(f"Taxa de atraso por mês da compra: do treino ao teste (estratégia '{ESTRATEGIA_ATIVA}')")
mostrar(fig, f"recorte_02_taxa_atraso_{VERSAO_RECORTE}")

# ------------------------------------------------------------------ 3) Meses usados por fold
grade = []
for f in folds:
    grade.append((f["nome"], f["tr"], f["aval"]))
grade.append(("Modelo final", treino_final, (pd.Timestamp(FIM_TREINO_FINAL), pd.Timestamp(INICIO_TESTE))))
n_mes = bm.groupby("ano_mes").size().reindex(meses)
fig, ax = plt.subplots(figsize=(18, 3.4))
for r, (nome_l, tr, (a_ini, a_fim)) in enumerate(grade):
    frac = (tr.groupby("ano_mes").size().reindex(meses, fill_value=0) / n_mes).fillna(0)
    for i, m in enumerate(meses):
        if frac[m] > 0:
            ax.barh(r, 1, left=i - 0.5, color=CORES_PAPEL["Treino"], alpha=0.25 + 0.75 * min(frac[m], 1),
                    edgecolor="white")
        if a_ini.strftime("%Y-%m") <= m < a_fim.strftime("%Y-%m"):
            ax.barh(r, 1, left=i - 0.5, color=CORES_PAPEL["Calibração"], edgecolor="white")
        if nome_l == "Modelo final" and m >= INICIO_TESTE[:7]:
            ax.barh(r, 1, left=i - 0.5, color=CORES_PAPEL["Teste"], edgecolor="white")
ax.set_yticks(range(len(grade)))
ax.set_yticklabels([g[0] for g in grade])
ax.invert_yaxis()
ax.set_xticks(range(len(meses)))
ax.set_xticklabels(meses, rotation=70)
ax.set_xlim(-0.5, len(meses) - 0.5)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=CORES_PAPEL[k]) for k in ["Treino", "Calibração", "Teste"]],
          ["Treino (mais claro = mês usado em parte)", "Validação / calibração", "Teste"],
          fontsize=8, loc="lower left", ncol=3)
ax.set_title(f"Meses usados no treino de cada fold (estratégia '{ESTRATEGIA_ATIVA}')")
mostrar(fig, f"recorte_03_linha_do_tempo_{VERSAO_RECORTE}")

equilibrio = pd.DataFrame([{"conjunto": c, "pedidos": len(df_), "atrasos": int(df_["atrasou"].sum()),
                            "taxa_atraso": df_["atrasou"].mean(),
                            "diferenca_para_teste_pp": (df_["atrasou"].mean() - teste["atrasou"].mean()) * 100}
                           for c, df_ in conj.items()])
display(equilibrio.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Modelagem: pré-processamento, balanceamento e os modelos (Regressão Logística e Gradient Boosting)

# COMMAND ----------

def log_com_sinal(x):
    return np.sign(x) * np.log1p(np.abs(x))


class RazoesPelaMediana(BaseEstimator, TransformerMixin):
    """Cria valor / mediana do grupo. As medianas vêm só do treino (fit); grupos com menos de
    `minimo` pedidos usam a mediana geral do treino."""
    REGRAS = (("preco_rel_categoria", "valor_produtos", ("categoria",)),
              ("frete_rel_rota", "valor_frete", ("seller_state", "customer_state")))

    def __init__(self, minimo=30, razoes=None):
        self.minimo = minimo
        self.razoes = razoes

    def _regras(self):
        return [r for r in self.REGRAS if self.razoes is None or r[0] in self.razoes]

    @staticmethod
    def _chave(X, grupo):
        chave = X[grupo[0]].astype(str)
        for c in grupo[1:]:
            chave = chave + ">" + X[c].astype(str)
        return chave

    def fit(self, X, y=None):
        self.medianas_ = {}
        for nome, valor, grupo in self._regras():
            chave = self._chave(X, grupo)
            g = X[valor].groupby(chave)
            med = g.median()[g.count() >= self.minimo]
            self.medianas_[nome] = (med, float(X[valor].median()))
        return self

    def transform(self, X):
        X = X.copy()
        for nome, valor, grupo in self._regras():
            med, geral = self.medianas_[nome]
            chave = self._chave(X, grupo)
            ref = chave.map(med).fillna(geral).replace(0, np.nan)
            X[nome] = X[valor] / ref
        return X


class SelecionaColunas(BaseEstimator, TransformerMixin):
    """Mantém só as variáveis escolhidas pela matriz de correlação."""
    def __init__(self, colunas):
        self.colunas = colunas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.colunas)]


class ParaCategoria(BaseEstimator, TransformerMixin):
    """Texto -> 'category' com as categorias vistas no treino (valor novo vira nulo)."""
    def __init__(self, colunas):
        self.colunas = colunas

    def fit(self, X, y=None):
        self.categorias_ = {c: sorted(X[c].dropna().unique()) for c in self.colunas}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.colunas:
            X[c] = pd.Categorical(X[c], categories=self.categorias_[c])
        return X


class ModeloBalanceado(BaseEstimator, ClassifierMixin):
    """Balanceamento aplicado DENTRO do fit, ou seja, só no treino do fold.
    estrategia:
      'nenhum'
      'pesos': erro em atraso custa mais; razao = proporção efetiva negativos:positivos (1 = classes iguais)
      'ensemble_subamostragem' (EasyEnsemble): n_membros modelos, cada um com todos os atrasos e uma amostra
          diferente de pedidos no prazo (razao x atrasos). A previsão é a média dos membros, então nenhum
          pedido no prazo é descartado de vez e a variância da subamostragem simples cai."""
    def __init__(self, estimador, estrategia="nenhum", razao=1.0, n_membros=5, random_state=SEED):
        self.estimador = estimador
        self.estrategia = estrategia
        self.razao = razao
        self.n_membros = n_membros
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        if self.estrategia == "ensemble_subamostragem":
            rng = np.random.default_rng(self.random_state)
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            n_neg = min(len(neg), int(self.razao * len(pos)))
            self.membros_ = []
            for _ in range(self.n_membros):
                idx = np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
                self.membros_.append(clone(self.estimador).fit(X.iloc[idx], y[idx]))
        elif self.estrategia == "pesos":
            w = np.where(y == 1, (y == 0).sum() / (self.razao * y.sum()), 1.0)
            self.membros_ = [clone(self.estimador).fit(X, y, modelo__sample_weight=w / w.mean())]
        else:
            self.membros_ = [clone(self.estimador).fit(X, y)]
        self.estimador_ = self.membros_[0]
        self.classes_ = self.estimador_.classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.membros_], axis=0)

    def decision_function(self, X):
        """Logit da probabilidade. O calibrador sigmoide usa este score; sem ele, algumas versões do
        scikit-learn pegam colunas diferentes de predict_proba no ajuste e na previsão e invertem o ranking."""
        p = np.clip(self.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def prep_linear():
    num = Pipeline([("imputa", SimpleImputer(strategy="median", add_indicator=True)),
                    ("log", FunctionTransformer(log_com_sinal, feature_names_out="one-to-one")),
                    ("spline", "passthrough"),   # a busca decide: linear ou curva (spline) por variável
                    ("padroniza", StandardScaler())])
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CATEGORICAS),
                              ("num", num, NUMERICAS)])


def prep_arvore():
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), CATEGORICAS),
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUMERICAS)])


def construir(nome):
    if nome == "Regressão Logística":
        etapas = [("prep", prep_linear()), ("modelo", LogisticRegression(max_iter=3000))]
    elif nome == "Random Forest":
        etapas = [("prep", prep_arvore()),
                  ("modelo", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED))]
    elif nome == "Gradient Boosting":
        etapas = [("prep", prep_arvore()), ("modelo", GradientBoostingClassifier(random_state=SEED))]
    elif nome == "LightGBM":
        etapas = [("cat", ParaCategoria(CATEGORICAS)),
                  ("modelo", LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1, subsample_freq=1))]
    else:
        raise ValueError(nome)
    return ModeloBalanceado(Pipeline([("razoes", RazoesPelaMediana(razoes=RAZOES_ATIVAS)),
                                  ("seleciona", SelecionaColunas(SELECIONADAS))] + etapas))


BALANCEAMENTO = {"estrategia": ["nenhum", "pesos", "ensemble_subamostragem"], "razao": [1.0, 2.0, 3.0, 5.0]}
ESPACOS = {
    "Regressão Logística": {"estimador__modelo__C": [0.0003, 0.001, 0.002, 0.003, 0.005, 0.01, 0.03],   # antes o melhor era o menor (0,003)
                            "estimador__prep__num__spline": ["passthrough",
                                                             SplineTransformer(n_knots=4, degree=3),
                                                             SplineTransformer(n_knots=6, degree=3)],
                            **BALANCEAMENTO},
    "Random Forest": {"estimador__modelo__max_depth": [8, 12, 16, None],
                      "estimador__modelo__min_samples_leaf": [20, 50, 100, 200],
                      "estimador__modelo__max_features": ["sqrt", 0.3, 0.5],
                      **BALANCEAMENTO},
    # antes os melhores ficaram na borda: 300 árvores (máx.), taxa 0,03 (mín.), profundidade 4 (máx.), folha 200 (máx.)
    "Gradient Boosting": {"estimador__modelo__n_estimators": [300, 500, 800],
                          "estimador__modelo__learning_rate": [0.01, 0.02, 0.03, 0.05],
                          "estimador__modelo__max_depth": [3, 4, 5],
                          "estimador__modelo__min_samples_leaf": [200, 400, 800],
                          "estimador__modelo__subsample": [0.7, 0.85],
                          "estimador__modelo__max_features": [0.5, "sqrt"],
                          **BALANCEAMENTO},
    "LightGBM": {"estimador__modelo__learning_rate": [0.02, 0.05, 0.1],
                 "estimador__modelo__n_estimators": [200, 400, 600],
                 "estimador__modelo__num_leaves": [15, 31, 63],
                 "estimador__modelo__min_child_samples": [50, 100, 200],
                 "estimador__modelo__reg_lambda": [0.0, 1.0, 5.0],
                 "estimador__modelo__colsample_bytree": [0.6, 0.8, 1.0],
                 "estimador__modelo__subsample": [0.7, 1.0],
                 **BALANCEAMENTO},
}
INTERPRETABILIDADE = {"Regressão Logística": 1.0, "Random Forest": 0.5, "Gradient Boosting": 0.5, "LightGBM": 0.5}
MODELOS = [m for m in ESPACOS if m in MODELOS_TESTADOS]
print(f"Modelos testados: {MODELOS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Seleção de variáveis por matriz de correlação (30 variáveis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Matriz de correlação das candidatas

# COMMAND ----------

def cramers_v_corrigido(a, b):
    """V de Cramér com correção de viés (Bergsma): não infla com muitas categorias."""
    tab = pd.crosstab(a, b)
    if min(tab.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    r, k = tab.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c, k_c = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    den = min(k_c - 1, r_c - 1)
    return float(np.sqrt(phi2 / den)) if den > 0 else 0.0


def razao_correlacao(categoria, x):
    """Razão de correlação (eta) entre uma categórica e uma numérica (0 = nenhuma, 1 = total)."""
    ok = categoria.notna() & x.notna()
    categoria, x = categoria[ok], x[ok]
    ss_total = ((x - x.mean()) ** 2).sum()
    g = x.groupby(categoria, observed=True)
    ss_entre = (g.size() * (g.mean() - x.mean()) ** 2).sum()
    return float(np.sqrt(ss_entre / ss_total)) if ss_total > 0 else 0.0


# Só o treino do 1º fold: nenhum período de validação, calibração ou teste participa da seleção
# (na versão anterior a seleção via os folds de validação e a validação cruzada ficou otimista)
dados_selecao = TREINO_SELECAO   # treino do fold 1 da versão original: mesmas 30 variáveis
RAZOES = [r[0] for r in RazoesPelaMediana.REGRAS]
CANDIDATAS = CATEGORICAS_CAND + NUMERICAS_CAND
sel = RazoesPelaMediana().fit_transform(dados_selecao[CATEGORICAS_CAND + NUMERICAS_X_CAND])
y_sel = pd.Series(dados_selecao["atrasou"].to_numpy(), index=sel.index)
cats = {c: sel[c].where(sel[c].map(sel[c].value_counts()) >= 50, "outras") for c in CATEGORICAS_CAND}
postos = sel[NUMERICAS_CAND].astype(float).rank()

# Matriz de correlação: |Spearman| (num x num), V de Cramér (cat x cat), razão de correlação (cat x num)
assoc = pd.DataFrame(np.eye(len(CANDIDATAS)), index=CANDIDATAS, columns=CANDIDATAS)
assoc.loc[NUMERICAS_CAND, NUMERICAS_CAND] = postos.corr().abs().fillna(0).to_numpy()
for i, a in enumerate(CATEGORICAS_CAND):
    for b in CATEGORICAS_CAND[i + 1:]:
        assoc.loc[a, b] = assoc.loc[b, a] = cramers_v_corrigido(cats[a], cats[b])
    for b in NUMERICAS_CAND:
        assoc.loc[a, b] = assoc.loc[b, a] = razao_correlacao(cats[a], postos[b])
for c in CANDIDATAS:
    assoc.loc[c, c] = 1.0

# Relevância: correlação de cada variável com o alvo (atrasou)
# Para numéricas usa o maior entre |Spearman| (relação monotônica) e o V de Cramér das faixas (decis), que
# também capta relações em U (ex.: mês, sazonalidade), que o Spearman sozinho não enxerga.
def relevancia_numerica(x, y):
    faixas = pd.qcut(x, 10, duplicates="drop").cat.codes  # nulo vira faixa própria (-1)
    return max(abs(x.rank().corr(y.rank())), cramers_v_corrigido(faixas, y))


relevancia = pd.Series({c: cramers_v_corrigido(cats[c], y_sel) for c in CATEGORICAS_CAND}
                       | {c: relevancia_numerica(sel[c].astype(float), y_sel) for c in NUMERICAS_CAND}).fillna(0)

ordem = sorted(CANDIDATAS, key=lambda f: (GRUPO[f], -relevancia[f]))
fig, ax = plt.subplots(figsize=(20, 17))
sns.heatmap(assoc.loc[ordem, ordem].astype(float), cmap="Reds", vmin=0, vmax=1, square=True, ax=ax,
            cbar_kws={"shrink": 0.6})
ax.set_title(f"Matriz de correlação das {len(CANDIDATAS)} candidatas (treino do 1º fold, {len(dados_selecao):,} pedidos)")
mostrar(fig, "selecao_01_matriz_candidatas")

triang = np.triu(np.ones(assoc.shape, dtype=bool), 1)
pares = (assoc.where(triang).stack().dropna().rename("correlacao")
         .loc[lambda s: s > LIMIAR_REDUNDANCIA].sort_values(ascending=False))
print(f"{len(pares)} pares com correlação acima de {LIMIAR_REDUNDANCIA}")
display(pares.reset_index().set_axis(["variável A", "variável B", "correlação"], axis=1).round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Novas razões: adicionar ou descartar

# COMMAND ----------

# Uma razão nova entra no modelo se a correlação dela com o atraso for pelo menos a mediana das originais e:
#   - não for cópia de um componente (correlação <= LIMIAR_REDUNDANCIA), ou
#   - for cópia, mas mais relevante que esse componente: nesse caso a razão SUBSTITUI o componente.
# Caso contrário, é descartada.
originais = [c for c in CANDIDATAS if c not in RAZOES_NOVAS]
corte_relevancia = relevancia[originais].median()
linhas = []
for nome, (comps, _) in RAZOES_NOVAS.items():
    corr_comp = assoc.loc[nome, list(comps)].astype(float)
    parecido = corr_comp.idxmax()
    linhas.append({"razão": nome, "componentes": " / ".join(comps),
                   "relevancia_alvo": relevancia[nome],
                   "relevancia_melhor_componente": relevancia[list(comps)].max(),
                   "maior_corr_componente": corr_comp.max(), "componente_mais_parecido": parecido,
                   "relevancia_parecido": relevancia[parecido]})
avaliacao_razoes = pd.DataFrame(linhas).set_index("razão").sort_values("relevancia_alvo", ascending=False)
a = avaliacao_razoes
interessante = a["relevancia_alvo"] >= corte_relevancia
copia = a["maior_corr_componente"] > LIMIAR_REDUNDANCIA
substitui = interessante & copia & (a["relevancia_alvo"] > a["relevancia_parecido"])
a["decisao"] = np.select([interessante & ~copia, substitui],
                         ["adicionada ao modelo", "adicionada (substitui " + a["componente_mais_parecido"] + ")"],
                         "descartada")
RAZOES_APROVADAS = a.index[a["decisao"] != "descartada"].tolist()
SUBSTITUIDAS = dict(zip(a.loc[substitui, "componente_mais_parecido"], a.index[substitui]))
print(f"Corte de relevância (mediana das {len(originais)} originais): {corte_relevancia:.4f}")
display(a.round(4).reset_index())

fig, axes = plt.subplots(1, 2, figsize=(20, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
r_ = a.sort_values("relevancia_alvo")
axes[0].barh(r_.index, r_["relevancia_alvo"],
             color=["#C44E52" if d == "descartada" else "#55A868" for d in r_["decisao"]])
axes[0].axvline(corte_relevancia, color="black", ls="--", lw=1, label="mediana das originais")
axes[0].legend(fontsize=8)
axes[0].set_title("Correlação das novas razões com o alvo (verde = adicionada)")
comps_todos = list(dict.fromkeys(c for comps, _ in RAZOES_NOVAS.values() for c in comps))
sns.heatmap(assoc.loc[list(RAZOES_NOVAS), comps_todos].astype(float), annot=True, fmt=".2f", annot_kws={"size": 7},
            cmap="Reds", vmin=0, vmax=1, ax=axes[1])
axes[1].set_title("Correlação das novas razões com as variáveis que as compõem")
mostrar(fig, "selecao_01b_novas_razoes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pesos e seleção das 30 variáveis

# COMMAND ----------

# Peso = relevância com o alvo x (1 - maior correlação com as já escolhidas). A cada passo entra a de maior peso;
# variáveis com correlação acima de LIMIAR_REDUNDANCIA com alguma escolhida ficam bloqueadas.
# As razões aprovadas entram primeiro; as descartadas saem da disputa.
selecionadas, peso_bruto = [], {}
for r in RAZOES_APROVADAS:
    red = assoc.loc[r, selecionadas].max() if selecionadas else 0.0
    if red <= LIMIAR_REDUNDANCIA:
        selecionadas.append(r)
        peso_bruto[r] = float(relevancia[r] * (1 - red))
fora = [r for r in RAZOES_NOVAS if r not in selecionadas] + list(SUBSTITUIDAS)
restantes = [f for f in relevancia.sort_values(ascending=False).index if f not in selecionadas and f not in fora]
while len(selecionadas) < N_FEATURES:
    if selecionadas:
        red = assoc.loc[restantes, selecionadas].max(axis=1)
    else:
        red = pd.Series(0.0, index=restantes)
    score = (relevancia[restantes] * (1 - red)).where(red <= LIMIAR_REDUNDANCIA, -1.0)
    melhor = score.idxmax()
    assert score[melhor] > 0, f"só {len(selecionadas)} variáveis passam no filtro de redundância"
    selecionadas.append(melhor)
    peso_bruto[melhor] = float(score[melhor])
    restantes.remove(melhor)

pesos = pd.Series(peso_bruto)
pesos = pesos / pesos.sum()

outras = assoc.loc[CANDIDATAS, selecionadas].astype(float).copy()
for f in selecionadas:
    outras.loc[f, f] = np.nan
ranking = pd.DataFrame({
    "grupo": [GRUPO[f] for f in CANDIDATAS],
    "relevancia_alvo": relevancia[CANDIDATAS],
    "maior_corr_selecionada": outras.max(axis=1),
    "mais_parecida_com": outras.idxmax(axis=1),
    "peso": pesos.reindex(CANDIDATAS),
    "ordem_entrada": pd.Series({f: i + 1 for i, f in enumerate(selecionadas)}).reindex(CANDIDATAS),
}, index=CANDIDATAS)
ranking["status"] = np.where(ranking.index.isin(selecionadas), "selecionada",
                             np.where(ranking["maior_corr_selecionada"] > LIMIAR_REDUNDANCIA,
                                      "removida: redundante", "removida: peso baixo"))
ranking.loc[ranking.index.isin([r for r in RAZOES_NOVAS if r not in selecionadas]), "status"] = "descartada: razão nova sem ganho"
for comp, razao in SUBSTITUIDAS.items():
    ranking.loc[comp, "status"] = f"substituída pela razão {razao}"
ranking = ranking.sort_values(["peso", "relevancia_alvo"], ascending=False)
display(ranking.round(3).reset_index().rename(columns={"index": "variável"}))
display(ranking.groupby(["grupo", "status"]).size().unstack(fill_value=0))

fig, axes = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={"width_ratios": [1, 1.6]})
p = pesos.sort_values()
cores_g = dict(zip(sorted({GRUPO[f] for f in CANDIDATAS}), sns.color_palette("deep")))
axes[0].barh(p.index, p.values, color=[cores_g[GRUPO[f]] for f in p.index])
axes[0].xaxis.set_major_formatter(PCT)
axes[0].legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in cores_g.values()], cores_g.keys(),
               fontsize=8, loc="lower right")
axes[0].set_title(f"Peso das {N_FEATURES} variáveis selecionadas")
sns.heatmap(assoc.loc[selecionadas, selecionadas].astype(float), annot=True, fmt=".2f", annot_kws={"size": 6},
            cmap="Reds", vmin=0, vmax=1, ax=axes[1])
axes[1].set_title(f"Matriz de correlação das {N_FEATURES} selecionadas (ordem de entrada)")
mostrar(fig, "selecao_02_pesos_matriz_selecionadas")

# Listas usadas pelo pipeline a partir daqui
SELECIONADAS = list(selecionadas)
CATEGORICAS = [f for f in CATEGORICAS_CAND if f in SELECIONADAS]
NUMERICAS = [f for f in NUMERICAS_CAND if f in SELECIONADAS]
RAZOES_ATIVAS = [r for r in RAZOES if r in SELECIONADAS]
entradas_razoes = [c for nome, valor, grupo in RazoesPelaMediana.REGRAS if nome in RAZOES_ATIVAS
                   for c in (valor, *grupo)]
FEATURES_X = list(dict.fromkeys([f for f in SELECIONADAS if f not in RAZOES] + entradas_razoes))
ASSINATURA = hashlib.md5(",".join(sorted(SELECIONADAS)).encode()).hexdigest()[:8]

assert len(SELECIONADAS) == N_FEATURES
assert not proibidas & set(FEATURES_X), "vazamento na seleção"
assert assoc.loc[SELECIONADAS, SELECIONADAS].where(~np.eye(N_FEATURES, dtype=bool)).max().max() <= LIMIAR_REDUNDANCIA
dicionario["selecionada"] = dicionario["feature"].isin(SELECIONADAS)
dicionario["peso"] = dicionario["feature"].map(pesos)
print(f"{len(CANDIDATAS)} candidatas -> {N_FEATURES} selecionadas ({len(CATEGORICAS)} categóricas, "
      f"{len(NUMERICAS)} numéricas) | colunas de entrada do pipeline: {len(FEATURES_X)} | assinatura {ASSINATURA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Busca de hiperparâmetros e balanceamento

# COMMAND ----------

import pickle
ARQ_BUSCA = os.path.join(SAIDA_DIR, f"buscas_{N_FEATURES}vars_{ASSINATURA}_bal_v30_{VERSAO_RECORTE}.pkl")
if os.path.exists(ARQ_BUSCA) and not MODO_RAPIDO:
    with open(ARQ_BUSCA, "rb") as f:
        buscas = pickle.load(f)
    print("Busca carregada do arquivo salvo; apague buscas.pkl para refazer.")
else:
    dados_folds = [(*Xy(f["tr"]), *Xy(f["va"])) for f in folds]
    buscas = {}
    for nome in MODELOS:
        combinacoes = list(ParameterSampler(ESPACOS[nome], n_iter=N_ITER_BUSCA, random_state=SEED))
        linhas_busca = []
        for i, params in enumerate(combinacoes, 1):
            scores = []
            for Xtr_, ytr_, Xva_, yva_ in dados_folds:
                m = construir(nome).set_params(**params).fit(Xtr_, ytr_)
                scores.append(average_precision_score(yva_, m.predict_proba(Xva_)[:, 1]))
            linhas_busca.append({"params": params, "pr_auc_cv": float(np.mean(scores))})
            print(f"  {nome} [{i}/{len(combinacoes)}] PR-AUC CV {np.mean(scores):.4f}")
        buscas[nome] = (pd.DataFrame(linhas_busca)
                        .sort_values("pr_auc_cv", ascending=False).reset_index(drop=True))
        print(f"{nome}: melhor PR-AUC CV {buscas[nome]['pr_auc_cv'].iloc[0]:.4f} entre {len(linhas_busca)} combinações")
    with open(ARQ_BUSCA, "wb") as f:
        pickle.dump(buscas, f)
    print(f"Busca concluída e salva em {ARQ_BUSCA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Treino final, calibração e avaliação no teste

# COMMAND ----------

import pickle
import cloudpickle


def calibrar(modelo, X, y):
    """Ajusta só o calibrador; o modelo fica congelado."""
    try:
        from sklearn.frozen import FrozenEstimator   # scikit-learn >= 1.6

        class Congelado(FrozenEstimator):
            """Mantém a tag de classificador (sklearn 1.7 a trata o wrapper como regressor)."""
            def __sklearn_tags__(self):
                tags = super().__sklearn_tags__()
                tags.estimator_type = "classifier"
                return tags

        return CalibratedClassifierCV(Congelado(modelo), method="sigmoid").fit(X, y)
    except ImportError:
        return CalibratedClassifierCV(modelo, method="sigmoid", cv="prefit").fit(X, y)


def modelos_base(modelo):
    """Pipelines originais (um por membro do ensemble), sem calibrador e sem wrapper de balanceamento."""
    if isinstance(modelo, CalibratedClassifierCV):
        est = modelo.calibrated_classifiers_[0].estimator
        modelo = getattr(est, "estimator", est)
    return modelo.membros_ if isinstance(modelo, ModeloBalanceado) else [modelo]


def limiar_f1(y, p):
    """Limiar que maximiza o F1, escolhido na calibração (nunca no teste)."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[np.nanargmax(f1)])


def avaliar(y, p, datas, n_boot=200):
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() > 0:
            boots.append(average_precision_score(y[i], p[i]))
    meses = pd.Series(datas).dt.to_period("M").to_numpy()
    pr_mes = []
    for mes in np.unique(meses):
        sel = meses == mes
        if y[sel].sum() > 0:
            pr_mes.append(average_precision_score(y[sel], p[sel]))
    top = p >= np.quantile(p, 0.9)
    return {
        "pr_auc": average_precision_score(y, p),
        "pr_auc_ic95": (np.percentile(boots, 2.5), np.percentile(boots, 97.5)),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "taxa_base": y.mean(),
        "precisao_top10": y[top].mean(),
        "lift_top10": y[top].mean() / y.mean(),
        "pr_auc_por_mes": pr_mes,
        "variacao_mensal": np.std(pr_mes) / np.mean(pr_mes),
    }


def feature_original(nome_transformado):
    """'cat__categoria_moveis' -> 'categoria'; 'num__missingindicator_voucher_pct' -> 'voucher_pct'."""
    n = nome_transformado.split("__", 1)[-1].replace("missingindicator_", "")
    candidatos = [f for f in CATEGORICAS + NUMERICAS if n == f or n.startswith(f + "_")]
    return max(candidatos, key=len) if candidatos else n


def importancia_pipeline(pipe):
    est = pipe.named_steps["modelo"]
    if "prep" in pipe.named_steps:
        nomes = pipe.named_steps["prep"].get_feature_names_out()
    else:
        nomes = getattr(est, "feature_name_", None) or list(getattr(est, "feature_names_in_"))
    if hasattr(est, "coef_"):
        valores = np.abs(est.coef_[0])
    elif hasattr(est, "booster_"):
        valores = est.booster_.feature_importance(importance_type="gain")
    else:
        valores = est.feature_importances_
    imp = pd.Series(valores, index=[feature_original(n) for n in nomes]).groupby(level=0).sum()
    return imp / imp.sum()


def importancia_nativa(modelo):
    """Importância do próprio modelo (|coeficiente|, impureza ou ganho), média entre os membros do ensemble."""
    return pd.concat([importancia_pipeline(p) for p in modelos_base(modelo)], axis=1).mean(axis=1)


Xtr, ytr = Xy(treino_final)
Xcal, ycal = Xy(calib)
Xte, yte = Xy(teste)
datas_te = teste["order_purchase_timestamp"].to_numpy()
DIR_CHECKPOINT = os.path.join(SAIDA_DIR, f"checkpoint_finais_{N_FEATURES}vars_{ASSINATURA}_bal_v30_{VERSAO_RECORTE}")
os.makedirs(DIR_CHECKPOINT, exist_ok=True)

finais, res, prev, imp_nativa = {}, {}, {}, {}
for nome in MODELOS:
    arq = os.path.join(DIR_CHECKPOINT, f"{nome.replace(' ', '_')}.pkl")
    if os.path.exists(arq):
        with open(arq, "rb") as f:
            res[nome], prev[nome], imp_nativa[nome] = pickle.load(f)
        print(f"{nome:20s} carregado do checkpoint")
        continue
    params = buscas[nome]["params"].iloc[0]
    print(f"{nome:20s} treinando... {params}", flush=True)
    t0 = time.time()
    base_m = construir(nome).set_params(**params).fit(Xtr, ytr)
    t_treino = time.time() - t0
    modelo = calibrar(base_m, Xcal, ycal) if CALIBRAR else base_m
    t0 = time.time()
    p = modelo.predict_proba(Xte)[:, 1]
    ms_1000 = (time.time() - t0) / len(yte) * 1e6
    p_sem = base_m.predict_proba(Xte)[:, 1]
    assert stats.spearmanr(p, p_sem).statistic > 0.99, f"{nome}: a calibração alterou a ordem dos scores"
    m = avaliar(yte, p, datas_te)
    m.update({
        "params": params,
        "pr_auc_cv": buscas[nome]["pr_auc_cv"].iloc[0],
        "brier_sem_calibracao": brier_score_loss(yte, p_sem),
        "limiar": limiar_f1(ycal, modelo.predict_proba(Xcal)[:, 1]),
        "segundos_treino": t_treino,
        "ms_por_1000_previsoes": ms_1000,
        "interpretabilidade": INTERPRETABILIDADE[nome],
    })
    try:
        imp_nome = importancia_nativa(modelo)
    except Exception as e:                         # a importância nunca deve derrubar o treino
        print(f"  importância de {nome} não calculada ({type(e).__name__}: {e})")
        imp_nome = pd.Series(dtype=float)
    finais[nome], res[nome], prev[nome], imp_nativa[nome] = modelo, m, p, imp_nome
    with open(arq, "wb") as f:                     # só métricas e previsões: arquivo pequeno e rápido
        pickle.dump((res[nome], prev[nome], imp_nativa[nome]), f)
    print(f"{nome:20s} pronto em {t_treino:.0f}s")

print()
for nome, m in res.items():
    print(f"{nome:20s} TESTE PR-AUC {m['pr_auc']:.4f} (IC95 {m['pr_auc_ic95'][0]:.4f}-{m['pr_auc_ic95'][1]:.4f}) | "
          f"ROC-AUC {m['roc_auc']:.4f} | Brier {m['brier_sem_calibracao']:.4f} -> {m['brier']:.4f} | "
          f"top10% {m['precisao_top10']:.1%}")

# Referência sem machine learning: taxa histórica de atraso da rota
p_ref = teste["taxa_atraso_rota_hist"].fillna(ytr.mean()).to_numpy()
ref = avaliar(yte, p_ref, datas_te)
print(f"\nReferência (taxa da rota) PR-AUC {ref['pr_auc']:.4f} | acaso = taxa base {ref['taxa_base']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Matriz de decisão e escolha do modelo

# COMMAND ----------

r = pd.DataFrame(res).T
for c in ["pr_auc", "pr_auc_cv", "brier", "precisao_top10", "variacao_mensal", "interpretabilidade",
          "ms_por_1000_previsoes", "segundos_treino"]:
    r[c] = r[c].astype(float)


def faixa(v, limites):
    return 1.0 if v <= limites[0] else 0.75 if v <= limites[1] else 0.5


queda = ((r["pr_auc_cv"] - r["pr_auc"]).abs() / r["pr_auc_cv"]).clip(0, 1)
matriz = pd.DataFrame({
    "discriminacao_pr_auc": r["pr_auc"] / r["pr_auc"].max(),
    "calibracao_brier": r["brier"].min() / r["brier"],
    "alto_risco_precisao_top10": r["precisao_top10"] / r["precisao_top10"].max(),
    "estabilidade": 1 - (queda + r["variacao_mensal"].clip(0, 1)) / 2,
    "interpretabilidade": r["interpretabilidade"],
    # latência por 1.000 previsões (<= 50 ms ótimo) e tempo de retreino (<= 1 min ótimo, <= 10 min aceitável)
    "viabilidade_operacional": [(faixa(a, (50, 500)) + faixa(b, (60, 600))) / 2
                                for a, b in zip(r["ms_por_1000_previsoes"], r["segundos_treino"])],
})
matriz["nota_final"] = sum(matriz[c] * w for c, w in PESOS_DECISAO.items())
matriz = matriz.sort_values("nota_final", ascending=False)

# Desempate: se a diferença de PR-AUC entre 1º e 2º pode ser ruído, vence o mais interpretável
vencedor, segundo = matriz.index[0], matriz.index[1]
rng = np.random.default_rng(SEED)
dif = []
for _ in range(300):
    i = rng.integers(0, len(yte), len(yte))
    if yte[i].sum() > 0:
        dif.append(average_precision_score(yte[i], prev[vencedor][i]) - average_precision_score(yte[i], prev[segundo][i]))
ic_dif = np.percentile(dif, [2.5, 97.5])
if ic_dif[0] <= 0 <= ic_dif[1] and INTERPRETABILIDADE[segundo] > INTERPRETABILIDADE[vencedor]:
    print(f"Desempate: {vencedor} e {segundo} empatam em PR-AUC (IC95 da diferença {ic_dif[0]:+.4f} a "
          f"{ic_dif[1]:+.4f}); vence o mais interpretável.")
    vencedor = segundo
else:
    print(f"Diferença de PR-AUC {vencedor} - {segundo}: IC95 {ic_dif[0]:+.4f} a {ic_dif[1]:+.4f}")

tabela = r[["pr_auc", "pr_auc_cv", "roc_auc", "brier_sem_calibracao", "brier", "precisao_top10", "lift_top10",
            "variacao_mensal", "limiar", "segundos_treino", "ms_por_1000_previsoes"]].astype(float)
tabela.loc["Referência (taxa da rota)", ["pr_auc", "roc_auc", "brier", "precisao_top10", "lift_top10"]] = \
    [ref["pr_auc"], ref["roc_auc"], ref["brier"], ref["precisao_top10"], ref["lift_top10"]]
print(f"Taxa base de atraso no teste (PR-AUC do acaso): {yte.mean():.4f}")
display(tabela.round(4).reset_index().rename(columns={"index": "modelo"}))
display(matriz.round(3).reset_index().rename(columns={"index": "modelo"}))
print(f"\n>>> MODELO ESCOLHIDO: {vencedor}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparação entre as versões (original, v2 e v3)

# COMMAND ----------

# Comparação entre as três versões, no MESMO teste (abr-ago/2018):
#   original  : treino expansivo jan-nov/2017, calibração jan-mar/2018
#   v2        : recorte equilibrado sem atípicos (jan/17-jan/18), calibração só mar/2018, busca original
#   v3 (esta) : mesmo recorte, calibração sem meses atípicos e busca com grades ampliadas
# Valores das versões anteriores copiados das saídas da seção 9 de cada execução.
ANTERIORES = {
    "original": {"Regressão Logística": {"pr_auc": 0.1300, "roc_auc": 0.7077, "brier": 0.0586, "precisao_top10": 0.1633},
                 "Gradient Boosting": {"pr_auc": 0.0997, "roc_auc": 0.6776, "brier": 0.0669, "precisao_top10": 0.1177}},
    "v2 (equilibrado, calib. mar/18)": {
        "Regressão Logística": {"pr_auc": 0.1307, "roc_auc": 0.7120, "brier": 0.0653, "precisao_top10": 0.1606},
        "Gradient Boosting": {"pr_auc": 0.1182, "roc_auc": 0.6915, "brier": 0.0632, "precisao_top10": 0.1371}},
}
VERSOES = {**ANTERIORES, f"v3 ({VERSAO_RECORTE})": {m: {k: float(res[m][k]) for k in
                                                     ["pr_auc", "roc_auc", "brier", "precisao_top10"]} for m in MODELOS}}
METRICAS_COMP = {"pr_auc": "PR-AUC (maior = melhor)", "roc_auc": "ROC-AUC (maior = melhor)",
                 "brier": "Brier calibrado (menor = melhor)", "precisao_top10": "Precisão top 10% (maior = melhor)"}
comp_versoes = pd.DataFrame([{"modelo": m, "métrica": rot, **{v: VERSOES[v][m][met] for v in VERSOES}}
                             for m in MODELOS for met, rot in METRICAS_COMP.items()])
brier_constante = yte.mean() * (1 - yte.mean())
print(f"Referência de Brier: prever sempre a taxa base do teste ({yte.mean():.2%}) dá Brier {brier_constante:.4f}. "
      f"Um modelo bem calibrado deveria ficar ABAIXO disso.")
display(comp_versoes.round(4))

cores_v = ["#C8C8C8", "#9DB4D8", "#4C72B0"]
fig, axes = plt.subplots(1, len(METRICAS_COMP), figsize=(20, 4.2))
for ax, (met, rotulo) in zip(axes, METRICAS_COMP.items()):
    x = np.arange(len(MODELOS))
    larg = 0.8 / len(VERSOES)
    for k, (v, cor) in enumerate(zip(VERSOES, cores_v)):
        vals = [VERSOES[v][m][met] for m in MODELOS]
        ax.bar(x + (k - (len(VERSOES) - 1) / 2) * larg, vals, larg, color=cor, label=v)
    if met == "brier":
        ax.axhline(brier_constante, color="red", ls=":", lw=1.2, label="Prever sempre a taxa base")
    if met == "pr_auc":
        ax.axhline(yte.mean(), color="red", ls=":", lw=1.2, label="Acaso")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELOS, fontsize=8)
    ax.set_title(rotulo, fontsize=10)
axes[0].legend(fontsize=7)
axes[2].legend(fontsize=7)
mostrar(fig, f"comparacao_versoes_{VERSAO_RECORTE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Peso pela correlação x importância no modelo escolhido

# COMMAND ----------

imp = imp_nativa.get(vencedor, pd.Series(dtype=float))
comp = (pd.DataFrame({"peso_correlacao": pesos, f"importancia_{vencedor}": imp})
        .reindex(SELECIONADAS).fillna(0).sort_values("peso_correlacao"))
fig, ax = plt.subplots(figsize=(10, 9))
comp.plot.barh(ax=ax, width=0.8)
ax.xaxis.set_major_formatter(PCT)
ax.set_title(f"Peso pela matriz de correlação x importância no modelo ({vencedor})")
mostrar(fig, "selecao_03_peso_vs_importancia")
print(f"Spearman entre peso e importância: {stats.spearmanr(comp.iloc[:, 0], comp.iloc[:, 1]).statistic:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Avaliação e diagnóstico

# COMMAND ----------

# MAGIC %md
# MAGIC ### Matrizes de confusão

# COMMAND ----------

# Matrizes de confusão: os modelos testados no TESTE (abr-ago/2018), em dois pontos de corte
#  1) Limiar F1: escolhido na calibração (período de calibração), nunca no teste
#  2) Top 10% de maior risco: a "faixa vermelha" usada na matriz de decisão
# Usa prev, res, yte e MODELOS das células anteriores (rode a seção de treino final antes).

def metricas_corte(y, pred):
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"VN": vn, "FP": fp, "FN": fn, "VP": vp,
            "acuracia": (vp + vn) / len(y),
            "precisao": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "especificidade": vn / (vn + fp) if (vn + fp) else np.nan,
            "pct_sinalizados": pred.mean()}


CORTES = {
    "Limiar F1 (calibração)": lambda nome, p: (p >= res[nome]["limiar"]).astype(int),
    "Top 10% de risco": lambda nome, p: (p >= np.quantile(p, 0.9)).astype(int),
}

linhas = []
fig, axes = plt.subplots(len(CORTES), len(MODELOS), figsize=(4.3 * len(MODELOS), 4.1 * len(CORTES)))
for i, (corte, regra) in enumerate(CORTES.items()):
    for j, nome in enumerate(MODELOS):
        pred = regra(nome, prev[nome])
        m = metricas_corte(yte, pred)
        linhas.append({"corte": corte, "modelo": nome, **m, "pr_auc": float(res[nome]["pr_auc"])})
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        anot = np.array([[f"{cm[a, b]:,}\n({cm[a, b] / cm[a].sum():.1%})" for b in range(2)] for a in range(2)])
        ax = axes[i, j]
        sns.heatmap(cm, annot=anot, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No prazo", "Atrasou"], yticklabels=["No prazo", "Atrasou"])
        ax.set_title(f"{nome}\nprec {m['precisao']:.1%} | recall {m['recall']:.1%} | acur {m['acuracia']:.1%}",
                     fontsize=10)
        ax.set_xlabel("Previsto")
        ax.set_ylabel(f"{corte}\n\nReal" if j == 0 else "Real")
fig.suptitle("Matrizes de confusão no TESTE (abr-ago/2018) | % = parcela da classe real", y=1.02)
mostrar(fig, "matrizes_confusao")

# Referência ingênua: dizer que NENHUM pedido atrasa (mostra por que acurácia engana aqui)
linhas.append({"corte": "-", "modelo": "Ingênuo: 'nunca atrasa'",
               **metricas_corte(yte, np.zeros_like(yte)), "pr_auc": yte.mean()})

comparacao = pd.DataFrame(linhas)
cols_pct = ["acuracia", "precisao", "recall", "f1", "especificidade", "pct_sinalizados"]
display(comparacao.style.format({c: "{:.1%}" for c in cols_pct} | {"pr_auc": "{:.4f}"}))

# Diferença de PR-AUC: modelo escolhido contra cada um dos outros (bootstrap pareado no teste)
escolhido = vencedor if "vencedor" in globals() else "Regressão Logística"
rng = np.random.default_rng(SEED)
idx_boot = [i for i in (rng.integers(0, len(yte), len(yte)) for _ in range(300)) if yte[i].sum() > 0]
print(f"\nDiferença de PR-AUC ({escolhido} menos o outro), IC95 por bootstrap pareado:")
for nome in MODELOS:
    if nome == escolhido:
        continue
    dif = [average_precision_score(yte[i], prev[escolhido][i]) - average_precision_score(yte[i], prev[nome][i])
           for i in idx_boot]
    lo, hi = np.percentile(dif, [2.5, 97.5])
    veredito = "diferença real" if lo > 0 or hi < 0 else "empate estatístico"
    print(f"  vs {nome:18s} {res[escolhido]['pr_auc'] - res[nome]['pr_auc']:+.4f} "
          f"(IC95 {lo:+.4f} a {hi:+.4f}) -> {veredito}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Curva Precision-Recall

# COMMAND ----------

# Curva Precision-Recall no TESTE: modelos testados + acaso + referência da rota
# Os marcadores mostram os dois cortes da matriz de confusão (limiar F1 e top 10%).
base_te = yte.mean()
cores = dict(zip(MODELOS, sns.color_palette("deep", len(MODELOS))))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
for nome in MODELOS:
    p = prev[nome]
    pr, rc, _ = precision_recall_curve(yte, p)
    ax.plot(rc, pr, color=cores[nome], lw=1.8, label=f"{nome} (PR-AUC {res[nome]['pr_auc']:.3f})")
    for regra, marcador in [(p >= res[nome]["limiar"], "o"), (p >= np.quantile(p, 0.9), "s")]:
        ax.scatter(recall_score(yte, regra), precision_score(yte, regra, zero_division=0),
                   color=cores[nome], marker=marcador, s=60, edgecolor="black", zorder=5)
pr_r, rc_r, _ = precision_recall_curve(yte, p_ref)
ax.plot(rc_r, pr_r, color="gray", ls="--", lw=1.2, label=f"Referência: taxa da rota ({ref['pr_auc']:.3f})")
ax.axhline(base_te, color="red", ls=":", lw=1.5, label=f"Acaso = taxa base ({base_te:.3f})")
ax.scatter([], [], marker="o", color="white", edgecolor="black", label="Limiar F1 (calibração)")
ax.scatter([], [], marker="s", color="white", edgecolor="black", label="Top 10% de risco")
ax.set(xlabel="Recall (atrasos capturados)", ylabel="Precisão (alertas corretos)",
       title="Curva Precision-Recall | teste abr-ago/2018", xlim=(0, 1), ylim=(0, 0.5))
ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
ax.legend(fontsize=8, loc="upper right")

# Precisão por faixa de pedidos sinalizados (visão operacional: "se eu checar os X% mais arriscados...")
ax = axes[1]
fracoes = np.linspace(0.01, 0.5, 50)
for nome in MODELOS:
    ordem = np.argsort(-prev[nome])
    ax.plot(fracoes, [yte[ordem[:max(1, int(f * len(yte)))]].mean() for f in fracoes],
            color=cores[nome], lw=1.8, label=nome)
ax.axhline(base_te, color="red", ls=":", lw=1.5, label="Acaso")
ax.axvline(0.10, color="black", ls="--", lw=0.8)
ax.set(xlabel="% de pedidos sinalizados (maior risco primeiro)", ylabel="Precisão na faixa",
       title="Precisão por tamanho da faixa de risco")
ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
ax.legend(fontsize=8)
mostrar(fig, "curva_precision_recall")

# Precisão em níveis fixos de recall (melhor precisão alcançável com pelo menos aquele recall)
linhas_pr = []
for nome in MODELOS:
    pr, rc, _ = precision_recall_curve(yte, prev[nome])
    linha = {"modelo": nome, "pr_auc": res[nome]["pr_auc"], "lift_pr_auc": res[nome]["pr_auc"] / base_te}
    for alvo in (0.1, 0.2, 0.3, 0.5, 0.7):
        linha[f"precisao@recall{int(alvo * 100)}"] = pr[rc >= alvo].max()
    linhas_pr.append(linha)
tab_pr = pd.DataFrame(linhas_pr).set_index("modelo")
display(tab_pr.style.format({c: "{:.1%}" for c in tab_pr.columns if c.startswith("precisao")}
                            | {"pr_auc": "{:.4f}", "lift_pr_auc": "{:.2f}x"}))