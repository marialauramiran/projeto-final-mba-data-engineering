# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Exportar a Gold para o app Streamlit
# MAGIC Gera `gold.agg_mercado` e `gold.ref_rotas` no contrato que o app espera
# MAGIC e exporta em Parquet. Ajuste os nomes de colunas de `gold.pedidos` ao seu schema.

# COMMAND ----------
from pyspark.sql import functions as F, Window

CAT = "raio_x"  # catálogo do Unity Catalog
SAIDA = "/Volumes/raio_x/gold/export_app"

p = (spark.table(f"{CAT}.gold.pedidos")
     .where("order_status = 'delivered'")
     .where(F.col("order_purchase_timestamp").between("2017-01-01", "2018-08-31 23:59:59")))

# COMMAND ----------
# gold.agg_mercado : categoria x UF de destino
cat_w = Window.partitionBy("categoria")
queixa = (p.where("review_score <= 2").groupBy("categoria", "queixa_tema").count()
          .withColumn("r", F.row_number().over(Window.partitionBy("categoria").orderBy(F.desc("count"))))
          .where("r = 1").select("categoria", F.col("queixa_tema").alias("queixa_top")))

agg = (p.groupBy(F.col("product_category_name_english").alias("categoria"),
                 F.col("customer_state").alias("uf_destino"))
       .agg(F.countDistinct("order_id").alias("pedidos"),
            F.sum(F.when(F.col("order_purchase_timestamp").between("2017-01-01", "2017-08-31 23:59:59"), 1)).alias("pedidos_jan_ago_2017"),
            F.sum(F.when(F.col("order_purchase_timestamp").between("2018-01-01", "2018-08-31 23:59:59"), 1)).alias("pedidos_jan_ago_2018"),
            F.avg("review_score").alias("nota_media"),
            F.avg((F.col("review_score") <= 2).cast("int")).alias("pct_notas_1_2"),
            F.avg(F.when(F.col("atrasou") == 1, F.col("review_score"))).alias("nota_se_atrasar"),
            F.mode("payment_installments").alias("parcelas_moda"))
       .withColumn("meses", F.lit(20))
       .join(p.groupBy(F.col("product_category_name_english").alias("categoria"))
              .agg(F.countDistinct("seller_id").alias("vendedores_ativos"),
                   F.percentile_approx("price", [0.25, 0.5, 0.75]).alias("q")),
             "categoria")
       .select("*", F.col("q")[0].alias("preco_p25"), F.col("q")[1].alias("preco_p50"), F.col("q")[2].alias("preco_p75"))
       .drop("q")
       .join(queixa, "categoria", "left")
       .fillna({"queixa_top": "Não recebi", "pedidos_jan_ago_2017": 0, "pedidos_jan_ago_2018": 0}))

# COMMAND ----------
# gold.ref_rotas : UF origem x UF destino (frete ~ base + R$/kg, via regressão simples por rota)
rot = (p.groupBy(F.col("seller_state").alias("uf_origem"), F.col("customer_state").alias("uf_destino"))
       .agg(F.countDistinct("order_id").alias("pedidos"),
            F.expr("percentile_approx(distancia_km, 0.5)").alias("distancia_km"),
            F.avg("atrasou").alias("taxa_atraso_hist"),
            F.regr_intercept("freight_value", F.col("product_weight_g") / 1000).alias("frete_base"),
            F.regr_slope("freight_value", F.col("product_weight_g") / 1000).alias("frete_por_kg"))
       .where("pedidos >= 30"))  # rotas com pouco histórico ficam de fora

# COMMAND ----------
for nome, df in {"agg_mercado": agg, "ref_rotas": rot}.items():
    df.write.mode("overwrite").saveAsTable(f"{CAT}.gold.{nome}")
    df.toPandas().to_parquet(f"{SAIDA}/{nome}.parquet", index=False)
    print(nome, df.count())
