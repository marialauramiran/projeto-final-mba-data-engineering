# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Validar contrato Gold × App
# MAGIC Substitui o antigo `06_export_gold_app.py` (que usava o catálogo `raio_x` e a tabela
# MAGIC `gold.pedidos`, inexistentes). O pipeline atual já grava a Gold em
# MAGIC `/Volumes/workspace/olist_gold/parquet`; este notebook só confere se as colunas
# MAGIC batem com o que o app espera. Rode depois do Silver → Gold e antes do deploy do app.

# COMMAND ----------
BASE = "/Volumes/workspace/olist_gold/parquet"

# Campos obrigatórios por tabela (aceita qualquer um dos nomes da lista).
CONTRATO = {
    "gold_ref_rotas": {
        "uf_origem": ["uf_origem"], "uf_destino": ["uf_destino"],
        "pedidos": ["total_pedidos", "pedidos"],
        "atraso": ["pedidos_atrasados", "risco_atraso_pct", "taxa_atraso"],
        "distancia": ["distancia_media_km", "distancia_km"],
    },
    "gold_agg_categoria": {
        "categoria": ["category"], "preco_medio": ["preco_medio"], "preco_mediano": ["preco_mediano"],
        "preco_min": ["preco_minimo"], "preco_max": ["preco_maximo"], "volume": ["total_pedidos"],
    },
    "gold_agg_mercado": {
        "categoria": ["category"], "pedidos": ["total_pedidos", "pedidos"],
        "vendedores": ["vendedores_ativos", "vendedores"],
    },
    "gold_agg_frete": {
        "categoria": ["category"], "frete_medio": ["frete_medio"],
    },
    "gold_agg_reputacao": {
        "categoria": ["category"], "nota_media": ["nota_media", "review_score_medio"],
    },
}

# COMMAND ----------
erros, linhas = [], []
for tabela, campos in CONTRATO.items():
    caminho = f"{BASE}/{tabela}"
    try:
        df = spark.read.parquet(caminho)
    except Exception as e:  # noqa: BLE001
        erros.append(f"{tabela}: não encontrada ({e.__class__.__name__})")
        continue
    cols = {c.lower() for c in df.columns}
    for campo, opcoes in campos.items():
        achou = next((o for o in opcoes if o.lower() in cols), None)
        linhas.append((tabela, campo, achou or "—", "OK" if achou else "FALTA"))
        if not achou:
            erros.append(f"{tabela}.{campo}: esperado um de {opcoes}")
    if "uf_origem" in campos and df.filter("uf_origem IS NULL").count() > 0:
        erros.append(f"{tabela}: uf_origem com nulos")

display(spark.createDataFrame(linhas, "tabela string, campo_app string, coluna_gold string, status string"))

# COMMAND ----------
# Colunas reais de cada tabela (para atualizar src/config.py quando algo mudar)
for tabela in CONTRATO:
    try:
        print(tabela, "→", spark.read.parquet(f"{BASE}/{tabela}").columns)
    except Exception:  # noqa: BLE001
        print(tabela, "→ não encontrada")

# COMMAND ----------
if erros:
    raise AssertionError("Contrato Gold × App quebrado:\n- " + "\n- ".join(erros))
print("Contrato OK: o app consegue ler todas as tabelas Gold.")
