# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Atraso de entrega Olist: Regressão Logística (28 variáveis, versão para o aplicativo)
# MAGIC
# MAGIC Modelo final da v6 (25 variáveis) **mais** os campos que o aplicativo recebe e que não estavam nele:
# MAGIC UF de origem (`seller_state`), preço (`valor_produtos`) e peso (`peso_total_g`). Todo o resto é igual à v6.
# MAGIC
# MAGIC | Escolha fixada | Valor |
# MAGIC |---|---|
# MAGIC | Modelo | Regressão Logística, C = 0,003, variáveis lineares (sem spline), sem balanceamento |
# MAGIC | Variáveis | 25 da v6 (assinatura `ad9b03e4`) + UF de origem, preço e peso = 28 |
# MAGIC | Recorte do treino | `equilibrado_sem_atipicos`, embargo de 30 dias |
# MAGIC | Períodos | treino até jan/2018 · calibração mar/2018 · teste abr–ago/2018 |
# MAGIC | Regra de alerta | X% de maior risco, com X escolhido pelo menor custo na validação (razão 10×) |
# MAGIC
# MAGIC **Ao final, a próxima etapa tem em memória:**
# MAGIC
# MAGIC | Objeto | O que é |
# MAGIC |---|---|
# MAGIC | `modelo` | modelo treinado; `modelo.prever(df)` devolve a probabilidade de atraso |
# MAGIC | `LIMIAR_ALERTA` | probabilidade a partir da qual o pedido é sinalizado (40% de maior risco) |
# MAGIC | `previsoes` | uma linha por pedido: probabilidade, alerta, nível de risco e resultado real |
# MAGIC | `res` | métricas do teste |
# MAGIC | `coeficientes` | coeficientes e odds ratio da regressão |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuração

# COMMAND ----------

import time
import hashlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.special import expit, logit
import uuid
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---- fontes (Camada Silver)
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

# ---- períodos (iguais à v6)
DATA_INICIO, DATA_FIM = "2017-01-01", "2018-08-31"
FOLDS_VALIDACAO = [("2017-07-01", "2017-10-01"), ("2017-10-01", "2018-01-01"), ("2018-01-01", "2018-04-01")]
INICIO_TESTE = "2018-04-01"
MESES_CALIBRACAO = 1
EMBARGO_DIAS = 30
FIM_TREINO_FINAL = str((pd.Timestamp(INICIO_TESTE) - pd.DateOffset(months=MESES_CALIBRACAO)).date())

# ---- recorte do treino (escolhido na validação da v6)
ESTRATEGIA_RECORTE = "equilibrado_sem_atipicos"
FATOR_ATIPICO = 1.5
MIN_PEDIDOS_MES = 300

# ---- features históricas (iguais à v6)
JANELA_RECENTE_DIAS = 30
MIN_EVENTOS_ROTA = 30
MIN_EVENTOS_RECENTE = 20
MIN_EVENTOS_VENDEDOR = 3
SEED = 42

# ---- modelo e regra de alerta (escolhidos na v6)
PARAMS_RL = {"C": 0.003, "max_iter": 3000}
FAIXAS_ALERTA = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
RAZAO_CUSTO = 10
FAIXA_V6 = 0.40                                    # faixa escolhida pela v6 (só para comparação)
FAIXA_ALTO_RISCO = 0.10                            # "alto" = 10% de maior risco; "médio" = até a faixa de alerta

# ---- as 25 variáveis do modelo final (seleção da v6)
CATEGORICAS = ["customer_state", "categoria"]
NUMERICAS = [
    "razao_prazo_rota", "atrasos_abertos_plataforma", "velocidade_exigida_km_dia", "taxa_atraso_rota_hist",
    "hora", "taxa_atraso_plataforma_recente", "tendencia_atraso_vendedor", "tendencia_atraso_rota",
    "pico_vendedor", "razao_prazo_vendedor", "vendedor_idade_dias", "dia_semana", "cliente_em_capital",
    "entrega_rota_hist_dias", "frete_pct_ticket", "n_vendedores", "dif_pago_vs_ticket_pct",
    "vendedor_dias_entrega_hist", "valor_frete", "prazo_prometido_dias", "taxa_atraso_destino_hist",
    "maior_dimensao_cm", "mesma_regiao",
]
ASSINATURA = hashlib.md5(",".join(sorted(CATEGORICAS + NUMERICAS)).encode()).hexdigest()[:8]
assert ASSINATURA == "ad9b03e4", f"lista de variáveis diferente da v6 ({ASSINATURA})"

# ---- campos do aplicativo que não estavam no modelo v6
CATEGORICAS = CATEGORICAS + ["seller_state"]                  # UF de origem
NUMERICAS = NUMERICAS + ["valor_produtos", "peso_total_g"]     # preço e peso
FEATURES = CATEGORICAS + NUMERICAS
assert len(FEATURES) == 28

# ---- resultados do modelo v6 (25 variáveis) no mesmo teste, para comparação
REFERENCIA_V6 = {"pr_auc": 0.1296, "roc_auc": 0.7149, "brier": 0.0489, "precisao_top10": 0.1615,
                 "recall_alerta": 0.6937}

print(f"{len(FEATURES)} variáveis (assinatura {ASSINATURA})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leitura e integração da Camada Silver
# MAGIC Mesmo código da v6: cada tabela é reduzida a uma linha por pedido antes dos joins, com testes de chave única,
# MAGIC de uma linha por pedido e de reconciliação do GMV.

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
# ~100 mil pedidos cabem no driver; as variáveis históricas usam merge_asof do pandas.
pedidos = integrada.toPandas()
print(pedidos.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Base analítica e definição do alvo
# MAGIC Atrasou = entrega depois da data prometida, ou prazo vencido sem entrega. Só entram no treino pedidos com resultado conhecido.
# MAGIC

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
# MAGIC ## 3. Variáveis (mesmo código da v6)
# MAGIC Todas calculadas só com o que se sabia no instante da compra: os históricos usam apenas entregas concluídas antes de cada pedido.
# MAGIC

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


def contar_antes(ped, chave, col_t, janela_dias=None):
    """Nº de eventos (col_t) da mesma chave ocorridos ESTRITAMENTE antes de cada compra.
    Com janela_dias, só os ocorridos nos últimos N dias antes da compra."""
    ev = ped.loc[ped[col_t].notna(), [chave, col_t]].rename(columns={col_t: "_t"}).sort_values("_t", kind="mergesort")
    ev["_t"] = ev["_t"].astype("datetime64[ns]")
    ev["_n"] = ev.groupby(chave, sort=False).cumcount() + 1

    def ate(deslocamento, incluir_igual):
        q = ped[[chave, "order_purchase_timestamp"]].copy()
        q["_t"] = (q["order_purchase_timestamp"] - deslocamento).astype("datetime64[ns]")
        q["_idx"] = np.arange(len(q))
        q = q.sort_values("_t", kind="mergesort")
        h = pd.merge_asof(q, ev[[chave, "_t", "_n"]], on="_t", by=chave, direction="backward",
                          allow_exact_matches=incluir_igual)
        return h.sort_values("_idx")["_n"].fillna(0).to_numpy()

    n = ate(pd.Timedelta(0), False)
    if janela_dias is not None:
        n = n - ate(pd.Timedelta(days=janela_dias), True)
    return n


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

    # ---------------------------------------------------------- v6: atrasos em aberto (tempo real, sem vazamento)
    # No instante da compra: pedidos com data prometida JÁ VENCIDA e ainda NÃO entregues. Um pedido deixa de
    # contar como "em aberto" quando é entregue ou, no máximo, 90 dias após o prometido (regra fixa, para que
    # pedidos extraviados não se acumulem para sempre).
    valido = ~ped["order_status"].isin(["canceled", "unavailable"]) & ped["order_estimated_delivery_date"].notna()
    prometido = ped["order_estimated_delivery_date"]
    limite = prometido + pd.Timedelta(days=90)
    entrega = ped["order_delivered_customer_date"]
    resolvido = entrega.where(entrega > prometido, prometido).where(entrega.notna())   # NaT se não entregue
    ped["_vencido_em"] = prometido.where(valido)
    ped["_resolvido_em"] = resolvido.where(resolvido < limite, limite).where(valido)
    for chave, nome in [("_geral", "plataforma"), ("customer_state", "destino")]:
        abertos = contar_antes(ped, chave, "_vencido_em") - contar_antes(ped, chave, "_resolvido_em")
        vencidos_janela = contar_antes(ped, chave, "_vencido_em", JANELA_RECENTE_DIAS)
        if nome == "plataforma":
            taxa_plat = np.where(vencidos_janela > 0, abertos / np.maximum(vencidos_janela, 1), 0.0)
            ped["atrasos_abertos_plataforma"] = taxa_plat
        else:
            ped["atrasos_abertos_destino"] = ((abertos + MIN_EVENTOS_RECENTE * taxa_plat)
                                              / (vencidos_janela + MIN_EVENTOS_RECENTE))

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

    return ped.drop(columns=["_rota_uf", "_rota_reg", "_geral", "_vendedor", "_cliente", "_vencido_em", "_resolvido_em"])


t0 = time.time()
base_feat = criar_features(base)
print(f"Features criadas em {time.time() - t0:.0f}s")

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

faltando = [f for f in FEATURES if f not in base_feat.columns]
assert not faltando, f"variáveis não construídas: {faltando}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Recorte temporal (treino, validação, calibração e teste)
# MAGIC Estratégia fixa `equilibrado_sem_atipicos`: tira os meses com atraso acima de 1,5× a mediana mensal do
# MAGIC próprio treino e limita cada mês à mediana de pedidos. O teste (abr–ago/2018) fica intacto.

# COMMAND ----------

bm = base_feat[(base_feat["elegivel"] == 1)
               & base_feat["order_purchase_timestamp"].between(DATA_INICIO, f"{DATA_FIM} 23:59:59")].copy()
bm["atrasou"] = bm["atrasou"].astype(int)
TS = bm["order_purchase_timestamp"]


def base_conhecida(corte):
    """Comprado antes de (corte - embargo) e com o resultado já conhecido no corte."""
    corte = pd.Timestamp(corte)
    return bm[(TS < corte - pd.Timedelta(days=EMBARGO_DIAS)) & (bm["data_conhecimento_alvo"] < corte)]


def meses_atipicos(df):
    t = df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["atrasou"].agg(["mean", "size"])
    t = t[t["size"] >= MIN_PEDIDOS_MES]
    return set(t.index[t["mean"] > FATOR_ATIPICO * t["mean"].median()])


def limitar_volume(df):
    mes = df["order_purchase_timestamp"].dt.to_period("M")
    teto = int(mes.value_counts().median())
    ordem = pd.Series(np.random.default_rng(SEED).random(len(df)), index=df.index).groupby(mes).rank(method="first")
    return df[ordem <= teto]


def treino_ate(corte):
    df = base_conhecida(corte)
    df = df[~df["order_purchase_timestamp"].dt.to_period("M").isin(meses_atipicos(df))]
    return limitar_volume(df)


folds = [{"nome": f"valida {ini[:7]} a {pd.Timestamp(fim) - pd.Timedelta(days=1):%Y-%m}",
          "tr": treino_ate(ini), "va": bm[(TS >= ini) & (TS < fim)]} for ini, fim in FOLDS_VALIDACAO]
treino_final = treino_ate(FIM_TREINO_FINAL)
calib = bm[(TS >= FIM_TREINO_FINAL) & (TS < INICIO_TESTE) & (bm["data_conhecimento_alvo"] < INICIO_TESTE)]
teste = bm[TS >= INICIO_TESTE]

assert not set(teste["order_id"]) & (set(treino_final["order_id"]) | set(calib["order_id"])), "teste vazou para o treino"
assert treino_final["order_purchase_timestamp"].max() < pd.Timestamp(FIM_TREINO_FINAL) - pd.Timedelta(days=EMBARGO_DIAS)

resumo = pd.DataFrame(
    [{"conjunto": f["nome"], "treino": len(f["tr"]), "atraso_treino": f["tr"]["atrasou"].mean(),
      "avaliacao": len(f["va"]), "atraso_avaliacao": f["va"]["atrasou"].mean()} for f in folds]
    + [{"conjunto": "final (calibração mar/2018)", "treino": len(treino_final),
        "atraso_treino": treino_final["atrasou"].mean(), "avaliacao": len(calib),
        "atraso_avaliacao": calib["atrasou"].mean()},
       {"conjunto": "TESTE abr-ago/2018", "treino": np.nan, "atraso_treino": np.nan,
        "avaliacao": len(teste), "atraso_avaliacao": teste["atrasou"].mean()}])
display(resumo.round(4))
# Esperado (v6): treinos 6.704 / 17.759 / 25.420 / 30.633 e teste 32.450 pedidos com 5,31% de atraso

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Pipeline da Regressão Logística
# MAGIC Todo o preparo é ajustado só com o treino: mediana para faltantes (+ indicador de faltante), log com sinal,
# MAGIC padronização e one-hot das categorias (categorias com menos de 50 pedidos viram "infrequentes").

# COMMAND ----------

def log_com_sinal(x):
    return np.sign(x) * np.log1p(np.abs(x))


def construir_pipeline():
    num = Pipeline([("imputa", SimpleImputer(strategy="median", add_indicator=True)),
                    ("log", FunctionTransformer(log_com_sinal, feature_names_out="one-to-one")),
                    ("padroniza", StandardScaler())])
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CATEGORICAS),
                              ("num", num, NUMERICAS)])
    return Pipeline([("prep", prep), ("modelo", LogisticRegression(**PARAMS_RL))])


class ModeloAtraso:
    """Pipeline treinado + correção de prevalência (só desloca o intercepto para que a probabilidade média
    prevista no treino seja a taxa real de atraso do treino; não muda a ordem dos pedidos)."""

    def __init__(self, pipe, X, y):
        self.pipe = pipe
        z = logit(np.clip(pipe.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7))
        self.deslocamento = brentq(lambda b: expit(z + b).mean() - float(np.mean(y)), -30, 30)
        self.features = list(FEATURES)

    def prever(self, df):
        """Probabilidade de atraso para um DataFrame com as 25 colunas de FEATURES."""
        p = self.pipe.predict_proba(df[self.features])[:, 1]
        return expit(logit(np.clip(p, 1e-7, 1 - 1e-7)) + self.deslocamento)


def Xy(df):
    return df[FEATURES], df["atrasou"].to_numpy()


def metricas_faixa(y, p, faixa):
    """Sinaliza os `faixa` x 100% pedidos de maior risco e mede o resultado."""
    alerta = p >= np.quantile(p, 1 - faixa)
    vp = (alerta & (y == 1)).sum()
    prec = vp / max(alerta.sum(), 1)
    return {"faixa": faixa, "recall": vp / max(y.sum(), 1), "precisao": prec,
            "acuracia": (alerta == (y == 1)).mean(), "lift": prec / y.mean()}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validação temporal e escolha da faixa de alerta
# MAGIC Em cada fold: treina com o passado, prevê o trimestre seguinte. A faixa é a de menor custo na curva média
# MAGIC (perder um atraso = `RAZAO_CUSTO` alarmes falsos). Nada aqui olha o teste.

# COMMAND ----------

curvas, linhas_fold = [], []
for f in folds:
    Xf, yf = Xy(f["tr"])
    Xv, yv = Xy(f["va"])
    pipe_f = construir_pipeline().fit(Xf, yf)
    pv = pipe_f.predict_proba(Xv)[:, 1]
    curvas.append(pd.DataFrame([metricas_faixa(yv, pv, fx) for fx in FAIXAS_ALERTA]))
    linhas_fold.append({"fold": f["nome"], "pr_auc": average_precision_score(yv, pv), "taxa_base": yv.mean(),
                        "lift": average_precision_score(yv, pv) / yv.mean(), "roc_auc": roc_auc_score(yv, pv)})
validacao = pd.DataFrame(linhas_fold)
display(validacao.round(4))
print(f"PR-AUC média na validação: {validacao['pr_auc'].mean():.4f} (v6 com 25 variáveis: 0.2034) | "
      f"lift médio {validacao['lift'].mean():.2f} (v6: 2.20)")

curva_val = pd.concat(curvas).groupby("faixa", as_index=False).mean()
taxa = curva_val["precisao"] / curva_val["lift"]
curva_val["custo"] = RAZAO_CUSTO * taxa * (1 - curva_val["recall"]) + curva_val["faixa"] * (1 - curva_val["precisao"])
FAIXA_ALERTA = float(curva_val.loc[curva_val["custo"].idxmin(), "faixa"])
print(f"Faixa de alerta escolhida na validação: {FAIXA_ALERTA:.0%}")
print(f"(a v6 com 25 variáveis escolheu {FAIXA_V6:.0%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Treino final e avaliação no teste (abr–ago/2018)

# COMMAND ----------

Xtr, ytr = Xy(treino_final)
Xte, yte = Xy(teste)

t0 = time.time()
pipe_final = construir_pipeline().fit(Xtr, ytr)
segundos_treino = time.time() - t0
modelo = ModeloAtraso(pipe_final, Xtr, ytr)

p_te = modelo.prever(teste)
assert stats.spearmanr(p_te, pipe_final.predict_proba(Xte)[:, 1]).statistic > 0.99

# Cortes de risco: definidos pela distribuição dos scores do teste, sem usar os rótulos (igual à v6).
# Em produção, recalcular periodicamente com os scores do período mais recente.
LIMIAR_ALERTA = float(np.quantile(p_te, 1 - FAIXA_ALERTA))
LIMIAR_ALTO = float(np.quantile(p_te, 1 - FAIXA_ALTO_RISCO))

rng = np.random.default_rng(SEED)
boots = [average_precision_score(yte[i], p_te[i]) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(200))]
top10 = p_te >= LIMIAR_ALTO
alerta_te = metricas_faixa(yte, p_te, FAIXA_ALERTA)
res = {
    "pr_auc": average_precision_score(yte, p_te),
    "pr_auc_ic95_inf": np.percentile(boots, 2.5),
    "pr_auc_ic95_sup": np.percentile(boots, 97.5),
    "roc_auc": roc_auc_score(yte, p_te),
    "brier": brier_score_loss(yte, p_te),
    "brier_taxa_base": brier_score_loss(yte, np.full(len(yte), yte.mean())),
    "taxa_base": yte.mean(),
    "lift_pr_auc": average_precision_score(yte, p_te) / yte.mean(),
    "precisao_top10": yte[top10].mean(),
    "lift_top10": yte[top10].mean() / yte.mean(),
    "faixa_alerta": FAIXA_ALERTA,
    "limiar_alerta": LIMIAR_ALERTA,
    "limiar_alto_risco": LIMIAR_ALTO,
    "recall_alerta": alerta_te["recall"],
    "precisao_alerta": alerta_te["precisao"],
    "acuracia_alerta": alerta_te["acuracia"],
    "lift_treino": average_precision_score(ytr, pipe_final.predict_proba(Xtr)[:, 1]) / ytr.mean(),
    "lift_validacao": validacao["lift"].mean(),
    "pr_auc_validacao": validacao["pr_auc"].mean(),
    "segundos_treino": segundos_treino,
    "pedidos_treino": len(treino_final),
    "pedidos_teste": len(teste),
}
for k, v in res.items():
    print(f"  {k:22s} {v:.4f}")

# Comparação com o modelo v6 (25 variáveis) no mesmo teste
comparacao = pd.DataFrame({"28 variáveis": {k: res[k] for k in REFERENCIA_V6}, "v6 (25 variáveis)": REFERENCIA_V6})
comparacao["diferenca"] = comparacao["28 variáveis"] - comparacao["v6 (25 variáveis)"]
display(comparacao.round(4))

curva_teste = pd.DataFrame([metricas_faixa(yte, p_te, fx) for fx in FAIXAS_ALERTA])
curva_alerta = (curva_val[["faixa", "recall", "precisao", "lift", "custo"]]
                .rename(columns={"recall": "recall_validacao", "precisao": "precisao_validacao",
                                 "lift": "lift_validacao", "custo": "custo_validacao"})
                .merge(curva_teste.rename(columns={"recall": "recall_teste", "precisao": "precisao_teste",
                                                   "acuracia": "acuracia_teste", "lift": "lift_teste"}), on="faixa"))
curva_alerta["escolhida"] = np.isclose(curva_alerta["faixa"], FAIXA_ALERTA)
display(curva_alerta.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Coeficientes (explicação do modelo)

# COMMAND ----------

nomes_cols = pipe_final.named_steps["prep"].get_feature_names_out()


def variavel_original(nome):
    """'cat__customer_state_RJ' -> 'customer_state'; 'num__missingindicator_x' -> 'x'."""
    n = nome.split("__", 1)[-1].replace("missingindicator_", "")
    cand = [f for f in FEATURES if n == f or n.startswith(f + "_")]
    return max(cand, key=len) if cand else n


coef = pipe_final.named_steps["modelo"].coef_[0]
coeficientes = pd.DataFrame({
    "coluna_modelo": nomes_cols,
    "variavel": [variavel_original(n) for n in nomes_cols],
    "tipo": ["categoria" if n.startswith("cat__") else ("indicador_faltante" if "missingindicator_" in n else "numerica")
             for n in nomes_cols],
    "coeficiente": coef,
    "odds_ratio": np.exp(coef),
})
coeficientes["efeito"] = np.where(coeficientes["coeficiente"] > 0, "aumenta o risco", "reduz o risco")
coeficientes = pd.concat([coeficientes, pd.DataFrame([{
    "coluna_modelo": "intercepto", "variavel": "intercepto", "tipo": "intercepto",
    "coeficiente": pipe_final.named_steps["modelo"].intercept_[0] + modelo.deslocamento,
    "odds_ratio": np.exp(pipe_final.named_steps["modelo"].intercept_[0] + modelo.deslocamento),
    "efeito": "-"}])], ignore_index=True)
coeficientes["abs_coef"] = coeficientes["coeficiente"].abs()
coeficientes = coeficientes.sort_values("abs_coef", ascending=False).drop(columns="abs_coef").reset_index(drop=True)
display(coeficientes.head(20).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Previsões de todos os pedidos
# MAGIC Uma linha por pedido da janela (jan/2017–ago/2018). `conjunto` diz o papel do pedido no modelo: para
# MAGIC avaliar desempenho, use só `teste` (nos de `treino` a previsão é dentro da amostra).

# COMMAND ----------

janela = base_feat[base_feat["order_purchase_timestamp"].between(DATA_INICIO, f"{DATA_FIM} 23:59:59")]
p_all = modelo.prever(janela)

conjunto = np.select(
    [janela["order_id"].isin(treino_final["order_id"]), janela["order_id"].isin(calib["order_id"]),
     janela["order_id"].isin(teste["order_id"]), janela["elegivel"] == 1],
    ["treino", "calibracao", "teste", "fora_do_treino"], default="sem_resultado")

previsoes = pd.DataFrame({
    "order_id": janela["order_id"].to_numpy(),
    "data_compra": janela["order_purchase_timestamp"].to_numpy(),
    "categoria": janela["categoria"].to_numpy(),
    "uf_origem": janela["seller_state"].to_numpy(),
    "uf_destino": janela["customer_state"].to_numpy(),
    "valor_produtos": janela["valor_produtos"].to_numpy(),
    "peso_total_g": janela["peso_total_g"].to_numpy(),
    "prob_atraso": p_all,
    "alerta": (p_all >= LIMIAR_ALERTA).astype(int),
    "nivel_risco": np.where(p_all >= LIMIAR_ALTO, "alto", np.where(p_all >= LIMIAR_ALERTA, "medio", "baixo")),
    "conjunto": conjunto,
    "atrasou": janela["atrasou"].to_numpy(),          # NaN = resultado ainda desconhecido
})
assert previsoes["order_id"].is_unique

display(previsoes.groupby("conjunto").agg(pedidos=("order_id", "size"), prob_media=("prob_atraso", "mean"),
                                          pct_alerta=("alerta", "mean"), taxa_real=("atrasou", "mean")).round(4))
t = previsoes[previsoes["conjunto"] == "teste"]
print(f"Teste: recall do alerta {t.loc[t['atrasou'] == 1, 'alerta'].mean():.1%} | limiar de alerta {LIMIAR_ALERTA:.4f}")

# Visão útil para o aplicativo: risco médio previsto e taxa real por rota (UF origem -> UF destino), no teste
rotas = (t.assign(rota=t["uf_origem"] + " -> " + t["uf_destino"])
          .groupby("rota").agg(pedidos=("order_id", "size"), risco_medio=("prob_atraso", "mean"),
                               taxa_real=("atrasou", "mean"))
          .query("pedidos >= 100").sort_values("risco_medio", ascending=False))
display(rotas.head(15).round(4))

# COMMAND ----------


# Tabela principal de previsões do modelo de Machine Learning

# 1. Definir as 28 features do novo modelo
features_modelo = FEATURES

# 2. Recuperar resultados do novo modelo
modelo_escolhido = "regressao_logistica"
probabilidades = np.asarray(p_te)
limiar = float(LIMIAR_ALERTA)
limiar_alto = float(LIMIAR_ALTO)
resultados_reais = np.asarray(yte)

# 3. Validar alinhamento
assert len(teste) == len(probabilidades)
assert len(teste) == len(resultados_reais)
assert all(col in teste.columns for col in features_modelo)

# 4. Criar cópia dos dados
base = teste.reset_index(drop=True).copy()

# 5. Classificar nível de risco
nivel_risco = np.where(
    probabilidades >= limiar_alto,
    "alto",
    np.where(probabilidades >= limiar, "medio", "baixo")
)

# 6. Criar tabela principal
df_predictions = pd.DataFrame({
    "prediction_id": [str(uuid.uuid4()) for _ in range(len(base))],
    "order_id": base["order_id"].astype(str),

    # Campos para filtros da aplicação
    "categoria": base["categoria"],
    "seller_state": base["seller_state"],
    "customer_state": base["customer_state"],
    "valor_produtos": base["valor_produtos"],
    "peso_total_g": base["peso_total_g"],

    # Resultados do modelo
    "probability": probabilidades,
    "predicted_label": (probabilidades >= limiar).astype(int),
    "actual_label": resultados_reais.astype(int),
    "threshold": limiar,
    "nivel_risco": nivel_risco,

    # Metadados
    "prediction_type": "test",
    "model_name": modelo_escolhido,
    "model_version": "2.0",
    "prediction_timestamp": datetime.now()
})

# 7. INPUT: armazenar as 28 features do modelo
features_json = base[features_modelo].copy()

df_predictions["inputs_json"] = features_json.apply(
    lambda linha: linha.to_json(force_ascii=False),
    axis=1
)

# 8. OUTPUT: armazenar os 4 resultados do modelo
df_predictions["outputs_json"] = df_predictions.apply(
    lambda linha: pd.Series({
        "probability": float(linha["probability"]),
        "predicted_label": int(linha["predicted_label"]),
        "threshold": float(linha["threshold"]),
        "nivel_risco": str(linha["nivel_risco"])
    }).to_json(),
    axis=1
)

# 9. Conferir estrutura
print("Dimensões:", df_predictions.shape)
print("Quantidade de features:", len(features_modelo))

display(df_predictions.head())

# 10. Converter Pandas para Spark
df_predictions_spark = spark.createDataFrame(df_predictions)

# 11. Salvar no Volume
PATH_PREDICTIONS = "/Volumes/workspace/olist_gold/ml_predictions/predictions"

(
    df_predictions_spark.write
    .format("parquet")
    .mode("overwrite")
    .save(PATH_PREDICTIONS)
)

print("Previsões salvas no Volume com sucesso!")

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Calibração: gráfico de confiabilidade
# MAGIC Agrupa os pedidos em 10 faixas de probabilidade prevista (mesmo número de pedidos em cada uma) e compara,
# MAGIC em cada faixa, a **probabilidade média prevista** com a **taxa real de atraso**. Modelo perfeitamente calibrado
# MAGIC = pontos sobre a diagonal. Abaixo da diagonal o modelo superestima o risco; acima, subestima.
# MAGIC
# MAGIC Três períodos: validação (os 3 folds, previsões fora da amostra), calibração (mar/2018) e teste (abr–ago/2018).

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Previsões fora da amostra na validação: cada fold treinado só com o seu passado (igual à seção 6)
y_val, p_val = [], []
for f in folds:
    Xf, yf = Xy(f["tr"])
    m_f = ModeloAtraso(construir_pipeline().fit(Xf, yf), Xf, yf)   # com a mesma correção de prevalência do modelo final
    y_val.append(f["va"]["atrasou"].to_numpy())
    p_val.append(m_f.prever(f["va"]))

periodos = {
    "Validação (3 folds)": (np.concatenate(y_val), np.concatenate(p_val)),
    "Calibração (mar/2018)": (calib["atrasou"].to_numpy(), modelo.prever(calib)),
    "Teste (abr–ago/2018)": (teste["atrasou"].to_numpy(), modelo.prever(teste)),
}


def resumo_calibracao(y, p, n_bins=10):
    """Média prevista x real, Brier, ECE (erro médio de calibração) e intercepto/inclinação de calibração."""
    real, prev = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    pesos = np.histogram(p, bins=np.quantile(p, np.linspace(0, 1, n_bins + 1)))[0]
    pesos = pesos[: len(real)] / pesos[: len(real)].sum()
    z = logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    rl = LogisticRegression(penalty=None, max_iter=1000).fit(z, y)
    return real, prev, {
        "pedidos": len(y), "prob_media_prevista": p.mean(), "taxa_real": y.mean(),
        "brier": brier_score_loss(y, p), "brier_taxa_base": brier_score_loss(y, np.full(len(y), y.mean())),
        "ece": float(np.sum(pesos * np.abs(real - prev))),
        "inclinacao_calibracao": float(rl.coef_[0][0]),   # 1 = ideal; < 1 = previsões extremas demais
    }


curvas_cal, linhas_cal = {}, []
for nome, (y, p) in periodos.items():
    real, prev, r = resumo_calibracao(y, p)
    curvas_cal[nome] = (prev, real)
    linhas_cal.append({"periodo": nome, **r})
tab_cal = pd.DataFrame(linhas_cal).set_index("periodo")
display(tab_cal.round(4).reset_index())

# Gráfico: curva de confiabilidade + distribuição das probabilidades
cores = {"Validação (3 folds)": "#7F7F7F", "Calibração (mar/2018)": "#D9822B", "Teste (abr–ago/2018)": "#3B6FA8"}
lim = max(max(np.max(prev), np.max(real)) for prev, real in curvas_cal.values()) * 1.1

fig, (ax, axh) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
ax.plot([0, lim], [0, lim], ls="--", color="gray", lw=1, label="Calibração perfeita")
for nome, (prev, real) in curvas_cal.items():
    ax.plot(prev, real, marker="o", lw=2, ms=6, color=cores[nome],
            label=f"{nome}: previsto {tab_cal.loc[nome, 'prob_media_prevista']:.1%} × real {tab_cal.loc[nome, 'taxa_real']:.1%}")
ax.axvline(LIMIAR_ALERTA, color="#3B6FA8", ls=":", lw=1)
ax.text(LIMIAR_ALERTA, lim * 0.97, " limiar de alerta", color="#3B6FA8", fontsize=8, va="top")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_ylabel("Taxa real de atraso na faixa")
ax.set_title("Gráfico de confiabilidade: Regressão Logística (10 faixas com o mesmo nº de pedidos)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

for nome, (y, p) in periodos.items():
    axh.hist(p, bins=np.linspace(0, lim, 40), histtype="step", lw=1.5, color=cores[nome], density=True, label=nome)
axh.set_xlabel("Probabilidade de atraso prevista")
axh.set_ylabel("Densidade")
axh.grid(alpha=0.3)
plt.tight_layout()
plt.show()

t = tab_cal.loc["Teste (abr–ago/2018)"]
print(f"Teste: média prevista {t['prob_media_prevista']:.2%} × real {t['taxa_real']:.2%} | Brier {t['brier']:.4f} "
      f"(taxa base {t['brier_taxa_base']:.4f}) | ECE {t['ece']:.4f} | inclinação {t['inclinacao_calibracao']:.2f}")