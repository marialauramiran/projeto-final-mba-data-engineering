# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 07 · Exportar a Gold para o dashboard (Power BI ou Looker Studio)
# MAGIC Gera 6 CSVs pequenos (separador `;`, UTF-8) em `/Volumes/workspace/olist_gold/export_dashboard/`.
# MAGIC Depois: Catalog > workspace > olist_gold > export_dashboard > baixar cada arquivo.
# MAGIC Não usa `cache()` (serverless) e lê só Parquet.

# COMMAND ----------

import inspect
print(inspect.getsource(salvar))

# COMMAND ----------

from pyspark.sql import functions as F

SILVER = "/Volumes/workspace/olist_silver/parquet"
GOLD = "/Volumes/workspace/olist_gold/parquet"
PRED = "/Volumes/workspace/olist_gold/ml_predictions/predictions"
OUT = "/Volumes/workspace/olist_gold/export_dashboard"
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.olist_gold.export_dashboard")


def salvar(df, nome):
    (df.coalesce(1).write.mode("overwrite")
       .option("header", True).option("sep", ";").csv(f"{OUT}/{nome}_csv"))
    print(f"{nome}: {df.count():,} linhas -> {nome}_csv/part-*.csv | {df.columns}")

# COMMAND ----------

# 1. KPIs por mês: pedidos entregues, taxa de atraso, tempo de entrega e nota (tendência do painel executivo)
orders = spark.read.parquet("/Volumes/workspace/olist_silver/parquet/olist_orders_dataset")
reviews = spark.read.parquet("/Volumes/workspace/olist_silver/parquet/olist_order_reviews_dataset")
kpi_mes = (orders.where("order_status = 'delivered'")
    .join(reviews.select("order_id", "review_score"), "order_id", "left")
    .withColumn("mes", F.date_trunc("month", "order_purchase_timestamp").cast("date"))
    .groupBy("mes")
    .agg(F.countDistinct("order_id").alias("pedidos_entregues"),
         F.sum((F.col("order_delivered_customer_date") > F.col("order_estimated_delivery_date")).cast("int")).alias("pedidos_atrasados"),
         F.avg(F.datediff("order_delivered_customer_date", "order_purchase_timestamp")).alias("dias_entrega_medio"),
         F.avg("review_score").alias("nota_media"))
    .withColumn("taxa_atraso", F.col("pedidos_atrasados") / F.col("pedidos_entregues"))
    .orderBy("mes"))
display(kpi_mes)

# COMMAND ----------

# 2 a 5. Tabelas Gold de negócio (sem gravar arquivo)
for t in ["gold_agg_mercado", "gold_agg_categoria", "gold_agg_frete", "gold_agg_reputacao", "gold_ref_rotas"]:
    print(t)
    display(spark.read.parquet(f"{GOLD}/{t}"))

# COMMAND ----------

# 6. Risco do modelo por rota (UF origem x UF destino) e categoria, sem as colunas JSON
pred = spark.read.parquet("/Volumes/workspace/olist_gold/ml_predictions/predictions")
risco_rota = (pred.groupBy("seller_state", "customer_state", "categoria")
    .agg(F.count("*").alias("pedidos"),
         F.avg("probability").alias("risco_medio"),
         F.avg((F.col("nivel_risco") == "alto").cast("int")).alias("pct_alto_risco"),
         F.avg("predicted_label").alias("pct_alerta"),
         F.avg("actual_label").alias("taxa_real_atraso")))
display(risco_rota)

# COMMAND ----------

