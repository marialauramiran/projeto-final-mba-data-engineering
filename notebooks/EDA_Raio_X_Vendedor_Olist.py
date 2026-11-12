# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # EDA – Raio-X do Vendedor (Olist) | Hands-On Mackenzie
# MAGIC **Objetivo:** responder, com dados, quais fatores de mercado, preço, frete e logística estão associados ao **atraso na entrega** e à **satisfação do cliente**.
# MAGIC
# MAGIC **Como usar:** 1) tenha as 9 tabelas Parquet na Bronze; 2) ajuste `BRONZE_PATH` na célula 0 (se necessário); 3) `Run all`.
# MAGIC
# MAGIC **Estrutura:** 0 Setup → 1 Qualidade dos dados → 2 Tabela analítica (1 linha por pedido) → 3 a 12 Perguntas de negócio → 13 Fatores do atraso + testes estatísticos → 14 Resumo/exportação.
# MAGIC
# MAGIC Cada seção traz a **pergunta**, o **código** e a **leitura do resultado**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup e carga (camada Bronze)
# MAGIC **Pergunta:** os dados estão acessíveis e com a estrutura esperada?

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

W = Window  # alias usado nas janelas do notebook

# Caminho da Camada Bronze nos Volumes do Databricks
BRONZE_PATH = "/Volumes/workspace/olist_bronze/parquet"

# Carregando as 9 tabelas brutas diretamente da Bronze
df_customers = spark.read.parquet(f"{BRONZE_PATH}/olist_customers_dataset")
df_orders = spark.read.parquet(f"{BRONZE_PATH}/olist_orders_dataset")
df_items = spark.read.parquet(f"{BRONZE_PATH}/olist_order_items_dataset")
df_payments = spark.read.parquet(f"{BRONZE_PATH}/olist_order_payments_dataset")
df_reviews = spark.read.parquet(f"{BRONZE_PATH}/olist_order_reviews_dataset")
df_products = spark.read.parquet(f"{BRONZE_PATH}/olist_products_dataset")
df_sellers = spark.read.parquet(f"{BRONZE_PATH}/olist_sellers_dataset")
df_geolocation = spark.read.parquet(f"{BRONZE_PATH}/olist_geolocation_dataset")
df_translation = spark.read.parquet(f"{BRONZE_PATH}/product_category_name_translation")

# Aliases: o restante do notebook usa df_geo e df_cat_tr
df_geo = df_geolocation
df_cat_tr = df_translation

# Garante os tipos corretos (inofensivo se a Bronze já estiver tipada; necessário se tudo veio como string)
def cast_cols(df, tipo, cols):
    # try_cast: valor malformado vira NULL (em vez de derrubar o notebook com CAST_INVALID_INPUT)
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.expr(f"try_cast(`{c}` as {tipo})"))
    return df

def diagnostico_cast(df_antes, df_depois, cols, nome):
    # conta valores que existiam e viraram NULL na conversão (sinal de dado corrompido na Bronze)
    for c in cols:
        if c in df_antes.columns:
            n = df_antes.filter(F.col(c).isNotNull()).count() - df_depois.filter(F.col(c).isNotNull()).count()
            if n > 0:
                print(f"ATENÇÃO {nome}.{c}: {n:,} valores inválidos viraram NULL")

df_orders = cast_cols(df_orders, "timestamp", ["order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"])
df_items = cast_cols(df_items, "double", ["price", "freight_value"])
df_items = cast_cols(df_items, "int", ["order_item_id"])
_rev_raw = df_reviews
df_reviews = cast_cols(df_reviews, "double", ["review_score"])
diagnostico_cast(_rev_raw, df_reviews, ["review_score"], "reviews")
# nota válida é de 1 a 5; linhas fora disso vêm de quebra de linha mal tratada na ingestão (comentários multilinha)
df_reviews = df_reviews.filter(F.col("review_score").between(1, 5))
df_payments = cast_cols(df_payments, "double", ["payment_value"])
df_payments = cast_cols(df_payments, "int", ["payment_installments"])
df_products = cast_cols(df_products, "double", ["product_weight_g", "product_photos_qty"])
df_geo = cast_cols(df_geo, "double", ["geolocation_lat", "geolocation_lng"])
# CEP-prefixo como texto em TODAS as tabelas, para o join não falhar por tipo diferente
for nome in ["customer_zip_code_prefix"]:
    df_customers = cast_cols(df_customers, "string", [nome])
df_sellers = cast_cols(df_sellers, "string", ["seller_zip_code_prefix"])
df_geo = cast_cols(df_geo, "string", ["geolocation_zip_code_prefix"])

tabelas = {"customers": df_customers, "geolocation": df_geo, "order_items": df_items,
           "payments": df_payments, "reviews": df_reviews, "orders": df_orders,
           "products": df_products, "sellers": df_sellers, "category_translation": df_cat_tr}
display(pd.DataFrame([(k, v.count(), len(v.columns)) for k, v in tabelas.items()],
                     columns=["tabela", "linhas", "colunas"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Qualidade dos dados
# MAGIC **Pergunta:** quais problemas (nulos, duplicatas, inconsistências) precisamos tratar antes de modelar?
# MAGIC
# MAGIC **Como ler:** nulos em datas de entrega são esperados em pedidos não entregues; duplicatas em `geolocation` e `reviews` exigem agregação antes do join (senão o join multiplica linhas).

# COMMAND ----------

# 1.1 Status dos pedidos
display(df_orders.groupBy("order_status").count()
        .withColumn("pct", F.round(F.col("count") / df_orders.count() * 100, 2)).orderBy(F.desc("count")))

# COMMAND ----------

# 1.2 Nulos por coluna (orders, products)
def nulos(df):
    n = df.count()
    return df.select([F.round(F.sum(F.col(c).isNull().cast("int")) / n * 100, 2).alias(c) for c in df.columns])

print("% nulos – orders");   display(nulos(df_orders))
print("% nulos – products"); display(nulos(df_products))

# COMMAND ----------

# 1.3 Duplicatas e integridade de chaves
print("orders: order_id duplicado        :", df_orders.count() - df_orders.select("order_id").distinct().count())
print("reviews: pedidos com >1 review    :", df_reviews.count() - df_reviews.select("order_id").distinct().count())
print("geolocation: linhas 100% duplicadas:", df_geo.count() - df_geo.dropDuplicates().count())
print("items sem produto cadastrado      :", df_items.join(df_products, "product_id", "left_anti").count())
print("items sem vendedor cadastrado     :", df_items.join(df_sellers, "seller_id", "left_anti").count())
print("orders sem cliente cadastrado     :", df_orders.join(df_customers, "customer_id", "left_anti").count())
print("orders sem nenhum item            :", df_orders.join(df_items.select("order_id").distinct(), "order_id", "left_anti").count())
print("produtos sem categoria            :", df_products.filter(F.col("product_category_name").isNull()).count())
print("produtos com peso = 0             :", df_products.filter(F.col("product_weight_g") == 0).count())
print("items com frete = 0               :", df_items.filter(F.col("freight_value") == 0).count())

# COMMAND ----------

# 1.4 Consistência temporal (entrega não pode vir antes da compra; entrega ao cliente após postagem)
print("entrega antes da compra   :", df_orders.filter(F.col("order_delivered_customer_date") < F.col("order_purchase_timestamp")).count())
print("entrega antes da postagem :", df_orders.filter(F.col("order_delivered_customer_date") < F.col("order_delivered_carrier_date")).count())
print("Período dos pedidos       :", df_orders.agg(F.min("order_purchase_timestamp"), F.max("order_purchase_timestamp")).collect()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tabela analítica – 1 linha por pedido (camada Silver → Gold)
# MAGIC **Pergunta:** conseguimos consolidar as 9 tabelas em uma base única, sem duplicar pedidos?
# MAGIC
# MAGIC **Regras:** só pedidos `delivered` com data de entrega (é onde existe "atrasou ou não"); `atrasou = entrega real > data estimada`; reviews, pagamentos e itens são agregados **antes** do join; categoria do pedido = categoria do item mais caro; distância = Haversine entre CEP-prefixo do vendedor e do cliente.

# COMMAND ----------

# 2.1 Agregações por pedido
items_enr = (df_items.join(df_products, "product_id", "left")
             .join(df_cat_tr, "product_category_name", "left")
             .join(df_sellers, "seller_id", "left")
             .withColumn("categoria", F.coalesce("product_category_name_english", F.lit("sem_categoria"))))

w = W.partitionBy("order_id").orderBy(F.desc("price"), "order_item_id")
item_principal = (items_enr.withColumn("rn", F.row_number().over(w)).filter("rn = 1")
                  .select("order_id", "categoria", "seller_id", "seller_state", "seller_zip_code_prefix"))

agg_items = (items_enr.groupBy("order_id").agg(
    F.sum("price").alias("preco_total"),
    F.sum("freight_value").alias("frete_total"),
    F.count("*").alias("qtd_itens"),
    F.countDistinct("seller_id").alias("qtd_vendedores"),
    F.sum("product_weight_g").alias("peso_total_g"),
    F.avg("product_photos_qty").alias("fotos_media")))

agg_reviews  = df_reviews.groupBy("order_id").agg(F.avg("review_score").alias("nota"))
agg_payments = df_payments.groupBy("order_id").agg(
    F.sum("payment_value").alias("valor_pago"),
    F.max("payment_installments").alias("parcelas"),
    F.first("payment_type").alias("tipo_pagamento"))

geo_zip = (df_geo.groupBy("geolocation_zip_code_prefix")
           .agg(F.avg("geolocation_lat").alias("lat"), F.avg("geolocation_lng").alias("lng")))

# COMMAND ----------

# 2.2 Haversine e montagem da tabela
def haversine_km(lat1, lng1, lat2, lng2):
    r = lambda c: F.radians(c)
    a = (F.sin((r(lat2) - r(lat1)) / 2) ** 2
         + F.cos(r(lat1)) * F.cos(r(lat2)) * F.sin((r(lng2) - r(lng1)) / 2) ** 2)
    return F.round(2 * 6371 * F.asin(F.sqrt(a)), 1)

ped = (df_orders.filter((F.col("order_status") == "delivered") & F.col("order_delivered_customer_date").isNotNull())
       .join(df_customers, "customer_id", "left")
       .join(agg_items, "order_id", "left")
       .join(item_principal, "order_id", "left")
       .join(agg_reviews, "order_id", "left")
       .join(agg_payments, "order_id", "left")
       .join(geo_zip.withColumnRenamed("lat", "c_lat").withColumnRenamed("lng", "c_lng"),
             F.col("customer_zip_code_prefix") == F.col("geolocation_zip_code_prefix"), "left").drop("geolocation_zip_code_prefix")
       .join(geo_zip.withColumnRenamed("lat", "s_lat").withColumnRenamed("lng", "s_lng"),
             F.col("seller_zip_code_prefix") == F.col("geolocation_zip_code_prefix"), "left").drop("geolocation_zip_code_prefix"))

df_mart = (ped
    .withColumn("atrasou", (F.col("order_delivered_customer_date") > F.col("order_estimated_delivery_date")).cast("int"))
    .withColumn("dias_atraso", F.datediff("order_delivered_customer_date", "order_estimated_delivery_date"))
    .withColumn("prazo_prometido_dias", F.datediff("order_estimated_delivery_date", "order_purchase_timestamp"))
    .withColumn("prazo_real_dias", F.datediff("order_delivered_customer_date", "order_purchase_timestamp"))
    .withColumn("frete_pct_valor", F.col("frete_total") / (F.col("preco_total") + F.col("frete_total")))
    .withColumn("distancia_km", haversine_km("s_lat", "s_lng", "c_lat", "c_lng"))
    .withColumn("mesmo_estado", (F.col("seller_state") == F.col("customer_state")).cast("int"))
    .withColumn("ano_mes", F.date_format("order_purchase_timestamp", "yyyy-MM"))
    .withColumn("dia_semana", F.dayofweek("order_purchase_timestamp"))
    .withColumn("nota_baixa", F.when(F.col("nota").isNull(), None).otherwise((F.col("nota") <= 2).cast("int")))
    .withColumn("faixa_preco", F.when(F.col("preco_total") <= 50, "1) até R$50")
                                .when(F.col("preco_total") <= 200, "2) R$50–200")
                                .when(F.col("preco_total") <= 500, "3) R$200–500").otherwise("4) acima de R$500"))
    .withColumn("faixa_dist", F.when(F.col("distancia_km").isNull(), "sem geoloc")
                                .when(F.col("distancia_km") <= 100, "1) até 100 km")
                                .when(F.col("distancia_km") <= 500, "2) 100–500 km")
                                .when(F.col("distancia_km") <= 1000, "3) 500–1.000 km")
                                .when(F.col("distancia_km") <= 2000, "4) 1.000–2.000 km").otherwise("5) acima de 2.000 km"))
    .select("order_id", "customer_unique_id", "customer_state", "customer_city", "seller_id", "seller_state", "categoria",
            "order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date",
            "ano_mes", "dia_semana", "preco_total", "frete_total", "frete_pct_valor", "qtd_itens", "qtd_vendedores",
            "peso_total_g", "fotos_media", "valor_pago", "parcelas", "tipo_pagamento", "distancia_km", "mesmo_estado",
            "prazo_prometido_dias", "prazo_real_dias", "dias_atraso", "atrasou", "nota", "nota_baixa",
            "faixa_preco", "faixa_dist"))

df_mart.cache()
print("Linhas na tabela analítica:", df_mart.count())

# COMMAND ----------

# 2.3 TESTE DE RECONCILIAÇÃO (exigido no objetivo específico 1)
n_mart  = df_mart.count()
n_dist  = df_mart.select("order_id").distinct().count()
n_orig  = df_orders.filter((F.col("order_status") == "delivered") & F.col("order_delivered_customer_date").isNotNull()).count()
soma_mart = df_mart.agg(F.sum("preco_total")).collect()[0][0]
ids = df_mart.select("order_id")
soma_items = df_items.join(ids, "order_id").agg(F.sum("price")).collect()[0][0]

assert n_mart == n_dist == n_orig, "Duplicação/perda de pedidos no join!"
assert abs(soma_mart - soma_items) < 0.01, "Receita da mart difere da tabela de itens!"
print(f"OK – {n_mart:,} pedidos, 1 linha por pedido | receita reconciliada: R$ {soma_mart:,.2f}")

# COMMAND ----------

# 2.4 Nulos da tabela analítica
display(nulos(df_mart))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Qual é a taxa geral de atraso e qual o tamanho do problema?
# MAGIC **Como ler:** compara o prazo prometido com o real. Se o prazo real médio é bem menor que o prometido, o atraso é concentrado (cauda), não generalizado.

# COMMAND ----------

display(df_mart.agg(F.count("*").alias("pedidos"),
                    F.round(F.avg("atrasou") * 100, 2).alias("pct_atrasados"),
                    F.round(F.avg("prazo_prometido_dias"), 1).alias("prazo_prometido_medio"),
                    F.round(F.avg("prazo_real_dias"), 1).alias("prazo_real_medio")))

display(df_mart.groupBy("atrasou").agg(F.count("*").alias("pedidos"),
        F.round(F.avg("prazo_prometido_dias"), 1).alias("prometido_medio"),
        F.round(F.avg("prazo_real_dias"), 1).alias("real_medio")))

# gravidade: quantos dias de atraso entre os atrasados
display(df_mart.filter("atrasou = 1").select(
    F.expr("percentile_approx(dias_atraso, array(0.25,0.5,0.75,0.9,0.99))").alias("p25_p50_p75_p90_p99_dias_atraso"),
    F.round(F.avg("dias_atraso"), 1).alias("media_dias_atraso")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. O atraso é estável ao longo do tempo?
# MAGIC **Como ler:** meses com picos (ex.: Black Friday/nov-2017, fev–mar/2018) indicam choques de capacidade logística. Isso **impacta a validação temporal do modelo**

# COMMAND ----------

pdf_mes = (df_mart.groupBy("ano_mes").agg(F.count("*").alias("pedidos"),
           F.round(F.avg("atrasou") * 100, 2).alias("pct_atraso"),
           F.round(F.avg("nota"), 2).alias("nota_media")).orderBy("ano_mes").toPandas())
display(pdf_mes)

pm = pdf_mes[pdf_mes.pedidos >= 200]   # ignora meses quase vazios (2016)
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(pm.ano_mes, pm.pedidos, color="#cfd8dc"); ax.set_ylabel("Pedidos")
ax2 = ax.twinx(); ax2.plot(pm.ano_mes, pm.pct_atraso, color="#c62828", marker="o"); ax2.set_ylabel("% atraso")
plt.xticks(rotation=70); plt.title("Volume e taxa de atraso por mês"); plt.tight_layout(); plt.show()

# COMMAND ----------

# Sazonalidade por dia da semana (1 = domingo)
display(df_mart.groupBy("dia_semana").agg(F.count("*").alias("pedidos"),
        F.round(F.avg("atrasou") * 100, 2).alias("pct_atraso")).orderBy("dia_semana"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. O atraso derruba a nota? (impacto na reputação)
# MAGIC **Como ler:** compare a nota média e a % de notas 1–2. O teste de Mann-Whitney (nota é ordinal) e o qui-quadrado (nota baixa × atraso) respondem se a diferença é estatisticamente significativa (p < 0,05). A curva por faixa de dias de atraso mostra se há um "ponto de dor".

# COMMAND ----------

display(df_mart.filter("nota is not null").groupBy("atrasou").agg(
    F.count("*").alias("pedidos"), F.round(F.avg("nota"), 2).alias("nota_media"),
    F.round(F.avg("nota_baixa") * 100, 1).alias("pct_nota_1_2")))

pdf_n = df_mart.filter("nota is not null").select("atrasou", "nota", "nota_baixa").toPandas()
u = stats.mannwhitneyu(pdf_n[pdf_n.atrasou == 1].nota, pdf_n[pdf_n.atrasou == 0].nota)
chi = stats.chi2_contingency(pd.crosstab(pdf_n.atrasou, pdf_n.nota_baixa))
print(f"Mann-Whitney p-valor: {u.pvalue:.3e} | Qui-quadrado p-valor: {chi[1]:.3e}")

# COMMAND ----------

# Nota por gravidade do atraso (dias em relação à data prometida)
display(df_mart.filter("nota is not null").withColumn("grav",
    F.when(F.col("dias_atraso") < 0, "0) antes do prazo").when(F.col("dias_atraso") == 0, "1) no dia")
     .when(F.col("dias_atraso") <= 3, "2) 1–3 dias").when(F.col("dias_atraso") <= 7, "3) 4–7 dias")
     .otherwise("4) mais de 7 dias"))
    .groupBy("grav").agg(F.count("*").alias("pedidos"), F.round(F.avg("nota"), 2).alias("nota_media"),
                         F.round(F.avg("nota_baixa") * 100, 1).alias("pct_nota_1_2")).orderBy("grav"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Qual o tamanho do mercado por categoria? (Radar de Oportunidades)
# MAGIC **Como ler:** receita e nº de pedidos mostram onde está a demanda; ticket mediano mostra o perfil de preço; o Pareto mostra quão concentrado é o mercado (concentração alta = poucas categorias dominam, mais competição).

# COMMAND ----------

tot_rev = df_mart.agg(F.sum("preco_total")).collect()[0][0]
pdf_cat = (df_mart.groupBy("categoria").agg(
    F.count("*").alias("pedidos"), F.round(F.sum("preco_total"), 0).alias("receita"),
    F.round(F.expr("percentile_approx(preco_total, 0.5)"), 1).alias("ticket_mediano"),
    F.countDistinct("seller_id").alias("vendedores"),
    F.round(F.avg("atrasou") * 100, 1).alias("pct_atraso"),
    F.round(F.avg("nota"), 2).alias("nota_media"),
    F.round(F.avg("nota_baixa") * 100, 1).alias("pct_nota_1_2"),
    F.round(F.expr("percentile_approx(frete_pct_valor, 0.5)") * 100, 1).alias("frete_pct_mediano"))
    .withColumn("share_receita", F.round(F.col("receita") / tot_rev * 100, 2))
    .withColumn("pedidos_por_vendedor", F.round(F.col("pedidos") / F.col("vendedores"), 1))
    .orderBy(F.desc("receita")).toPandas())
display(pdf_cat)

pdf_cat["acum"] = pdf_cat.share_receita.cumsum()
print("Share de receita das 10 maiores categorias: %.1f%%" % pdf_cat.share_receita.head(10).sum())
print("Nº de categorias para chegar a 80%% da receita: %d de %d" % ((pdf_cat.acum < 80).sum() + 1, len(pdf_cat)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Quanto o frete pesa no preço?
# MAGIC **Como ler:** o frete é relativamente fixo; em produtos baratos ele vira grande parte do valor total (desestimula a compra e reduz competitividade). Esta é a base do "peso do frete" do Raio-X.

# COMMAND ----------

display(df_mart.groupBy("faixa_preco").agg(
    F.count("*").alias("pedidos"),
    F.round(F.avg("frete_pct_valor") * 100, 1).alias("frete_pct_medio"),
    F.round(F.avg("frete_total"), 1).alias("frete_medio_rs"),
    F.round(F.avg("atrasou") * 100, 2).alias("pct_atraso"),
    F.round(F.avg("nota"), 2).alias("nota_media")).orderBy("faixa_preco"))

# categorias onde o frete mais pesa (mín. 300 pedidos)
display(pdf_cat[pdf_cat.pedidos >= 300].sort_values("frete_pct_mediano", ascending=False)
        [["categoria", "pedidos", "ticket_mediano", "frete_pct_mediano"]].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. De onde vendem e para onde entregam? (geografia)
# MAGIC **Como ler:** concentração de oferta (vendedores) e demanda (clientes) por estado; e o atraso por **estado de destino** – base do "risco por região de destino".

# COMMAND ----------

display(df_mart.groupBy("seller_state").agg(F.count("*").alias("pedidos"),
        F.countDistinct("seller_id").alias("vendedores"),
        F.round(F.avg("atrasou") * 100, 2).alias("pct_atraso")).orderBy(F.desc("pedidos")).limit(10))

pdf_uf = (df_mart.groupBy("customer_state").agg(F.count("*").alias("pedidos"),
          F.round(F.avg("atrasou") * 100, 2).alias("pct_atraso"),
          F.round(F.avg("frete_total"), 1).alias("frete_medio"),
          F.round(F.avg("nota"), 2).alias("nota_media"),
          F.round(F.avg("distancia_km"), 0).alias("dist_media_km")).orderBy("pct_atraso").toPandas())
display(pdf_uf)

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(pdf_uf.customer_state, pdf_uf.pct_atraso, color="#ef6c00")
ax.axhline(df_mart.agg(F.avg("atrasou")).collect()[0][0] * 100, color="k", ls="--", label="média geral")
plt.title("% de atraso por estado do cliente"); plt.legend(); plt.tight_layout(); plt.show()

# COMMAND ----------

# Vender dentro do próprio estado reduz o risco?
display(df_mart.groupBy("mesmo_estado").agg(F.count("*").alias("pedidos"),
        F.round(F.avg("atrasou") * 100, 2).alias("pct_atraso"),
        F.round(F.avg("frete_total"), 1).alias("frete_medio"),
        F.round(F.avg("nota"), 2).alias("nota_media")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. A distância aumenta o risco de atraso?
# MAGIC **Como ler:** se a taxa de atraso cresce de forma monótona com a distância, distância é feature central do modelo (e está disponível no momento da compra: CEP vendedor + CEP cliente). Teste: qui-quadrado de tendência via correlação de Spearman.

# COMMAND ----------

display(df_mart.groupBy("faixa_dist").agg(F.count("*").alias("pedidos"),
        F.round(F.avg("atrasou") * 100, 2).alias("pct_atraso"),
        F.round(F.avg("frete_pct_valor") * 100, 1).alias("frete_pct_medio"),
        F.round(F.avg("nota"), 2).alias("nota_media")).orderBy("faixa_dist"))

pdf_d = df_mart.filter("distancia_km is not null").select("distancia_km", "atrasou").toPandas()
rho, p = stats.spearmanr(pdf_d.distancia_km, pdf_d.atrasou)
print(f"Spearman distância × atraso: rho={rho:.3f}, p={p:.2e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Quais são as rotas (origem → destino) mais arriscadas?
# MAGIC **Como ler:** só rotas com volume mínimo (≥ 200 pedidos) para evitar conclusões em amostra pequena. Rotas acima da média geral devem disparar alerta amarelo/vermelho no semáforo.

# COMMAND ----------

media_geral = df_mart.agg(F.avg("atrasou")).collect()[0][0]
rotas = (df_mart.groupBy("seller_state", "customer_state").agg(F.count("*").alias("pedidos"),
         F.round(F.avg("atrasou") * 100, 2).alias("pct_atraso"), F.round(F.avg("nota"), 2).alias("nota_media"))
         .filter("pedidos >= 200"))
print("Top 10 rotas mais arriscadas"); display(rotas.orderBy(F.desc("pct_atraso")).limit(10))
print("Top 10 rotas mais seguras");    display(rotas.orderBy("pct_atraso").limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Vendedores: mercado concentrado? Quem atrasa?
# MAGIC **Como ler:** concentração de vendas em poucos vendedores = barreira de entrada para o iniciante. A dispersão da taxa de atraso entre vendedores (com volume mínimo) mostra o quanto o atraso é "do vendedor" e não só da rota.

# COMMAND ----------

pdf_v = (df_mart.groupBy("seller_id").agg(F.count("*").alias("pedidos"),
         F.avg("atrasou").alias("taxa_atraso"), F.avg("nota").alias("nota_media")).toPandas()
         .sort_values("pedidos", ascending=False))
pdf_v["acum"] = pdf_v.pedidos.cumsum() / pdf_v.pedidos.sum()
print("Vendedores ativos:", len(pdf_v))
print("Mediana de pedidos por vendedor:", pdf_v.pedidos.median())
print("Share dos 10%% maiores vendedores: %.1f%%" % (pdf_v.pedidos.head(int(len(pdf_v) * .1)).sum() / pdf_v.pedidos.sum() * 100))
print("Vendedores com < 10 pedidos: %.1f%%" % ((pdf_v.pedidos < 10).mean() * 100))

v10 = pdf_v[pdf_v.pedidos >= 30]
print(f"Entre vendedores com ≥30 pedidos (n={len(v10)}): atraso p25/p50/p75/p90 =",
      np.round(v10.taxa_atraso.quantile([.25, .5, .75, .9]).values * 100, 1))

# COMMAND ----------

# Pedidos com mais de um vendedor (carrinho multi-vendedor)
display(df_mart.groupBy("qtd_vendedores").agg(F.count("*").alias("pedidos"), F.round(F.avg("atrasou") * 100, 2).alias("pct_atraso")).orderBy("qtd_vendedores"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Reputação por categoria e perfil de pagamento/recompra
# MAGIC **Como ler:** categorias com nota baixa mesmo com atraso baixo indicam problema de produto/expectativa (não logístico) – o semáforo de reputação deve separar os dois motivos. Recompra baixa = cada venda precisa "acertar de primeira".

# COMMAND ----------

print("Piores notas (≥300 pedidos)")
display(pdf_cat[pdf_cat.pedidos >= 300].sort_values("nota_media")[["categoria", "pedidos", "nota_media", "pct_nota_1_2", "pct_atraso"]].head(8))
print("Maiores taxas de atraso (≥300 pedidos)")
display(pdf_cat[pdf_cat.pedidos >= 300].sort_values("pct_atraso", ascending=False)[["categoria", "pedidos", "pct_atraso", "nota_media"]].head(8))

# COMMAND ----------

# Pagamento
display(df_payments.groupBy("payment_type").agg(F.count("*").alias("pagamentos"),
        F.round(F.avg("payment_installments"), 1).alias("parcelas_media")).orderBy(F.desc("pagamentos")))

# Recompra: clientes únicos com mais de 1 pedido
cli = df_customers.join(df_orders.select("customer_id", "order_id"), "customer_id").groupBy("customer_unique_id").agg(F.countDistinct("order_id").alias("pedidos"))
print(f"Clientes que compraram mais de 1 vez: {cli.filter('pedidos > 1').count() / cli.count() * 100:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Quais fatores estão mais associados ao atraso? (testes estatísticos)
# MAGIC **Como ler:**
# MAGIC * **Spearman** (variável numérica × atraso): sinal e magnitude da associação monotônica;
# MAGIC * **Mann-Whitney** (atrasados × no prazo) + **tamanho de efeito** (r = Z/√N): diz se a distribuição muda e o quanto;
# MAGIC * **Qui-quadrado + V de Cramér** (categóricas): força da associação.
# MAGIC
# MAGIC ⚠️ **Vazamento de dados:** a coluna `Disponível na compra?` separa o que o vendedor/cliente já sabe ao fechar o pedido (pode entrar no modelo) do que só se sabe depois (`prazo_real_dias`, `dias_atraso`, `nota` – NÃO podem entrar).

# COMMAND ----------

num_cols = {"distancia_km": True, "preco_total": True, "frete_total": True, "frete_pct_valor": True,
            "peso_total_g": True, "qtd_itens": True, "qtd_vendedores": True, "prazo_prometido_dias": True,
            "parcelas": True, "fotos_media": True, "mesmo_estado": True}
pdf = df_mart.select("atrasou", "customer_state", "seller_state", "categoria", "faixa_preco", *num_cols.keys()).toPandas()

linhas = []
for c, disp in num_cols.items():
    s = pdf[[c, "atrasou"]].dropna()
    rho, p_s = stats.spearmanr(s[c], s["atrasou"])
    a, b = s[s.atrasou == 1][c], s[s.atrasou == 0][c]
    mw = stats.mannwhitneyu(a, b)
    z = (mw.statistic - len(a) * len(b) / 2) / np.sqrt(len(a) * len(b) * (len(a) + len(b) + 1) / 12)
    linhas.append((c, round(rho, 3), p_s, round(abs(z) / np.sqrt(len(s)), 3), round(a.median(), 2), round(b.median(), 2), "Sim" if disp else "Não"))
res_num = pd.DataFrame(linhas, columns=["variavel", "spearman_rho", "p_valor", "efeito_r", "mediana_atrasados", "mediana_no_prazo", "Disponível na compra?"])
display(res_num.sort_values("efeito_r", ascending=False))

# COMMAND ----------

def cramers_v(x, y):
    ct = pd.crosstab(x, y); chi2, p, _, _ = stats.chi2_contingency(ct)
    return round(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))), 3), p

linhas = []
for c in ["customer_state", "seller_state", "categoria", "faixa_preco"]:
    v, p = cramers_v(pdf[c], pdf["atrasou"]); linhas.append((c, v, p))
display(pd.DataFrame(linhas, columns=["variavel_categorica", "cramers_v", "p_valor"]).sort_values("cramers_v", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Exportar tabela Gold e resumo
# MAGIC A tabela analítica fica disponível para as próximas etapas (feature engineering, modelagem, Streamlit).

# COMMAND ----------

# Opção A: tabela gerenciada no Unity Catalog (ajuste catálogo.schema)
# df_mart.write.mode("overwrite").saveAsTable("workspace.default.gold_pedidos_olist")

# Opção B: Delta em Volume
# df_mart.write.mode("overwrite").format("delta").save(BASE_PATH + "gold_pedidos_olist")

print("EDA concluída. Próximo passo: engenharia de variáveis (≈50) separando disponíveis na compra × pós-compra.")