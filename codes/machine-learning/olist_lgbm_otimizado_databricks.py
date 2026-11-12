# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "seaborn",
#   "lightgbm",
# ]
# ///
# MAGIC %md
# MAGIC # Otimização de Modelo Preditivo de Atrasos Olist (LightGBM + Ajuste Dinâmico de Limiar)
# MAGIC
# MAGIC Este notebook implementa a pipeline completa para previsão de atrasos nas entregas da Olist, combinando:
# MAGIC 1. Engenharia de features temporais e logísticas.
# MAGIC 2. Tratamento nativo de desbalanceamento via `scale_pos_weight` (evitando vazamento de dados de oversampling nos folds).
# MAGIC 3. Validação Cruzada Estratificada com otimização dinâmica do limiar de decisão (*threshold*) na curva Precision-Recall para atingir a meta de recall mínimo (>= 50%) maximizando a precisão.

# COMMAND ----------

# DBTITLE 1,Instalar dependências (seaborn, lightgbm)
# MAGIC %pip install seaborn lightgbm

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    brier_score_loss
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# Configurações de exibição
pd.set_option('display.max_columns', None)
RANDOM_STATE = 42
RECALL_MINIMO = 0.50

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregamento e Preparação dos Dados
# MAGIC *Nota: Ajuste os caminhos abaixo caso suas tabelas estejam no Unity Catalog / Hive Metastore ou em volumes/DBFS.*

# COMMAND ----------

def carregar_dados_olist():
    """
    Carrega as tabelas da Olist a partir do Spark Catalog ou arquivos CSV no DBFS/Volumes.
    Adapte o caminho conforme o seu ambiente no Databricks.
    """
    try:
        # Tentativa 1: Leitura via Spark SQL / Tabelas registradas no Databricks
        orders = spark.table("olist_orders_dataset").toPandas()
        items = spark.table("olist_order_items_dataset").toPandas()
        customers = spark.table("olist_customers_dataset").toPandas()
        sellers = spark.table("olist_sellers_dataset").toPandas()
        products = spark.table("olist_products_dataset").toPandas()
        print("Dados carregados com sucesso via Spark Catalog.")
    except Exception as e:
        print(f"Tabelas não encontradas no catálogo padrão ({e}). Tentando leitura via DBFS...")
        base_path = "/dbfs/FileStore/tables/olist/"
        orders = pd.read_csv(f"{base_path}olist_orders_dataset.csv")
        items = pd.read_csv(f"{base_path}olist_order_items_dataset.csv")
        customers = pd.read_csv(f"{base_path}olist_customers_dataset.csv")
        sellers = pd.read_csv(f"{base_path}olist_sellers_dataset.csv")
        products = pd.read_csv(f"{base_path}olist_products_dataset.csv")
        print("Dados carregados com sucesso via CSV no DBFS.")
        
    return orders, items, customers, sellers, products

orders, items, customers, sellers, products = carregar_dados_olist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Engenharia de Features e Definição do Target (`atraso`)

# COMMAND ----------

def construir_dataset_modelagem(orders, items, customers, sellers, products):
    # Filtrar apenas pedidos entregues e com datas válidas
    df = orders[orders['order_status'] == 'delivered'].copy()
    
    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    df = df.dropna(subset=[
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]).copy()
    
    # Definição da variável alvo (Target): 1 se entregue após a data estimada, 0 caso contrário
    df['atraso'] = (
        df['order_delivered_customer_date'].dt.date > df['order_estimated_delivery_date'].dt.date
    ).astype(int)
    
    # Agregação dos itens por pedido
    items_products = items.merge(products, on='product_id', how='left')
    
    agg_items = items_products.groupby('order_id').agg(
        qtd_itens=('order_item_id', 'count'),
        preco_total=('price', 'sum'),
        frete_total=('freight_value', 'sum'),
        peso_total_g=('product_weight_g', 'sum'),
        volume_total_cm3=('product_length_cm', lambda x: np.nan), # calculado abaixo
        seller_id=('seller_id', 'first'),
        categoria_principal=('product_category_name', 'first')
    ).reset_index()
    
    # Cálculo de volume cúbico dos itens
    items_products['volume_cm3'] = (
        items_products['product_length_cm'].fillna(0) *
        items_products['product_height_cm'].fillna(0) *
        items_products['product_width_cm'].fillna(0)
    )
    vol_agg = items_products.groupby('order_id')['volume_cm3'].sum().reset_index().rename(
        columns={'volume_cm3': 'volume_total_cm3'}
    )
    agg_items = agg_items.drop(columns=['volume_total_cm3']).merge(vol_agg, on='order_id', how='left')
    
    # Join com clientes e vendedores
    df = df.merge(agg_items, on='order_id', how='inner')
    df = df.merge(customers[['customer_id', 'customer_state', 'customer_zip_code_prefix']], on='customer_id', how='left')
    df = df.merge(sellers[['seller_id', 'seller_state', 'seller_zip_code_prefix']], on='seller_id', how='left')
    
    # Features Temporais e de Prazo (disponíveis no momento da aprovação do pedido)
    df['prazo_estimado_dias'] = (
        df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
    ).dt.total_seconds() / 86400.0
    
    df['tempo_aprovacao_horas'] = (
        df['order_approved_at'] - df['order_purchase_timestamp']
    ).dt.total_seconds() / 3600.0
    
    df['dia_semana_compra'] = df['order_purchase_timestamp'].dt.dayofweek
    df['mes_compra'] = df['order_purchase_timestamp'].dt.month
    df['hora_compra'] = df['order_purchase_timestamp'].dt.hour
    
    # Features Logísticas e de Rota
    df['mesmo_estado'] = (df['customer_state'] == df['seller_state']).astype(int)
    df['rota_uf'] = df['seller_state'].fillna('NA') + '_' + df['customer_state'].fillna('NA')
    df['ratio_frete_preco'] = df['frete_total'] / (df['preco_total'] + 1e-5)
    df['diff_prefixo_cep'] = np.abs(
        df['customer_zip_code_prefix'].fillna(0) - df['seller_zip_code_prefix'].fillna(0)
    )
    
    # Ordenar cronologicamente para split temporal realista
    df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
    return df

df_model = construir_dataset_modelagem(orders, items, customers, sellers, products)
print(f"Total de pedidos válidos: {len(df_model):,}")
print(f"Taxa global de atrasos: {df_model['atraso'].mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Separação Treino / Teste (Split Temporal) e Pré-processamento

# COMMAND ----------

features_numericas = [
    'qtd_itens',
    'preco_total',
    'frete_total',
    'peso_total_g',
    'volume_total_cm3',
    'prazo_estimado_dias',
    'tempo_aprovacao_horas',
    'dia_semana_compra',
    'mes_compra',
    'hora_compra',
    'mesmo_estado',
    'ratio_frete_preco',
    'diff_prefixo_cep'
]

features_categoricas = [
    'customer_state',
    'seller_state',
    'rota_uf',
    'categoria_principal'
]

X = df_model[features_numericas + features_categoricas].copy()
y = df_model['atraso'].copy()

# Split temporal (80% mais antigos para treino/CV, 20% mais recentes para teste holdout)
split_idx = int(len(df_model) * 0.80)
X_train_full, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train_full, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

print(f"Treino: {len(X_train_full):,} registros (Atrasos: {y_train_full.mean():.2%})")
print(f"Teste (Holdout): {len(X_test):,} registros (Atrasos: {y_test.mean():.2%})")

# Função de pré-processamento compatível com LightGBM
def criar_preprocessador():
    num_transformer = SimpleImputer(strategy='median')
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='DESCONHECIDO')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', num_transformer, features_numericas),
            ('cat', cat_transformer, features_categoricas)
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Função de Otimização Dinâmica do Limiar (*Threshold*)

# COMMAND ----------

def encontrar_limiar_otimo(y_true, y_scores, recall_minimo=0.50):
    """
    Encontra o limiar de probabilidade que garante recall >= recall_minimo
    enquanto maximiza a precisão (minimizando falsos positivos).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # precision_recall_curve retorna len(thresholds) == len(recalls) - 1
    recalls_validos = recalls[:-1]
    precisions_validas = precisions[:-1]
    
    indices_validos = np.where(recalls_validos >= recall_minimo)[0]
    
    if len(indices_validos) > 0:
        melhor_sub_idx = np.argmax(precisions_validas[indices_validos])
        melhor_idx = indices_validos[melhor_sub_idx]
        limiar_otimo = float(thresholds[melhor_idx])
    else:
        limiar_otimo = 0.50
        
    return limiar_otimo

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validação Cruzada Estratificada (5-Folds) com `scale_pos_weight`
# MAGIC Calculamos o `scale_pos_weight` exclusivamente dentro de cada fold de treino para evitar qualquer vazamento (*data leakage*) e calibramos o limiar ótimo na validação out-of-fold.

# COMMAND ----------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_probs = np.zeros(len(X_train_full))
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
    X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
    X_va, y_va = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
    
    # Pré-processamento ajustado apenas no fold de treino
    prep = criar_preprocessador()
    X_tr_proc = prep.fit_transform(X_tr)
    X_va_proc = prep.transform(X_va)
    
    # Cálculo dinâmico do peso da classe positiva dentro do fold
    neg_count = (y_tr == 0).sum()
    pos_count = (y_tr == 1).sum()
    scale_weight = neg_count / pos_count
    
    # Modelo LightGBM regularizado para evitar overfitting em base desbalanceada
    modelo = LGBMClassifier(
        scale_pos_weight=scale_weight,
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbose=-1
    )
    
    modelo.fit(X_tr_proc, y_tr)
    
    # Probabilidades na validação
    probs_va = modelo.predict_proba(X_va_proc)[:, 1]
    oof_probs[val_idx] = probs_va
    
    # Limiar ótimo do fold
    limiar_fold = encontrar_limiar_otimo(y_va, probs_va, recall_minimo=RECALL_MINIMO)
    preds_va = (probs_va >= limiar_fold).astype(int)
    
    fold_metrics.append({
        'fold': fold,
        'scale_pos_weight': round(scale_weight, 2),
        'limiar_otimo': round(limiar_fold, 4),
        'pr_auc': average_precision_score(y_va, probs_va),
        'roc_auc': roc_auc_score(y_va, probs_va),
        'recall': recall_score(y_va, preds_va),
        'precision': precision_score(y_va, preds_va, zero_division=0),
        'f1': f1_score(y_va, preds_va),
        'accuracy': accuracy_score(y_va, preds_va)
    })

df_cv_results = pd.DataFrame(fold_metrics)
print("=== Resultados por Fold (Validação Cruzada) ===")
display(df_cv_results)

print("\n=== Média dos Folds ===")
print(df_cv_results.mean(numeric_only=True).round(4))

# Limiar global consolidado via predições Out-Of-Fold (OOF)
limiar_global_oof = encontrar_limiar_otimo(y_train_full, oof_probs, recall_minimo=RECALL_MINIMO)
print(f"\nLimiar Ótimo Consolidado (OOF) para Recall >= {RECALL_MINIMO:.0%}: {limiar_global_oof:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Treinamento Final e Avaliação no Conjunto de Teste (Holdout)

# COMMAND ----------

# Ajuste do pré-processador em todo o conjunto de treino
prep_final = criar_preprocessador()
X_train_proc = prep_final.fit_transform(X_train_full)
X_test_proc = prep_final.transform(X_test)

# Peso final das classes
scale_weight_final = (y_train_full == 0).sum() / (y_train_full == 1).sum()

modelo_final = LGBMClassifier(
    scale_pos_weight=scale_weight_final,
    n_estimators=400,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

modelo_final.fit(X_train_proc, y_train_full)

# Predições no conjunto de teste
probs_test = modelo_final.predict_proba(X_test_proc)[:, 1]
preds_test_otimizado = (probs_test >= limiar_global_oof).astype(int)
preds_test_padrao = (probs_test >= 0.50).astype(int)

# Comparativo de Métricas: Limiar Padrão (0.50) vs Limiar Otimizado
resumo_comparativo = pd.DataFrame([
    {
        'Estratégia': 'Limiar Padrão (0.50)',
        'Threshold': 0.5000,
        'PR-AUC': average_precision_score(y_test, probs_test),
        'ROC-AUC': roc_auc_score(y_test, probs_test),
        'Recall': recall_score(y_test, preds_test_padrao),
        'Precision': precision_score(y_test, preds_test_padrao, zero_division=0),
        'F1-Score': f1_score(y_test, preds_test_padrao),
        'Acurácia': accuracy_score(y_test, preds_test_padrao),
        'Brier Score': brier_score_loss(y_test, probs_test)
    },
    {
        'Estratégia': f'Limiar Otimizado (Recall >= {RECALL_MINIMO:.0%})',
        'Threshold': round(limiar_global_oof, 4),
        'PR-AUC': average_precision_score(y_test, probs_test),
        'ROC-AUC': roc_auc_score(y_test, probs_test),
        'Recall': recall_score(y_test, preds_test_otimizado),
        'Precision': precision_score(y_test, preds_test_otimizado, zero_division=0),
        'F1-Score': f1_score(y_test, preds_test_otimizado),
        'Acurácia': accuracy_score(y_test, preds_test_otimizado),
        'Brier Score': brier_score_loss(y_test, probs_test)
    }
])

print("=== Comparativo no Conjunto de Teste (Holdout) ===")
display(resumo_comparativo.round(4))

print("\n=== Relatório de Classificação (Limiar Otimizado) ===")
print(classification_report(y_test, preds_test_otimizado, target_names=['No Prazo (0)', 'Atrasado (1)']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualização: Matriz de Confusão e Curva Precision-Recall

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Matriz de Confusão com Limiar Otimizado
cm = confusion_matrix(y_test, preds_test_otimizado)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
    xticklabels=['Previsto: No Prazo', 'Previsto: Atrasado'],
    yticklabels=['Real: No Prazo', 'Real: Atrasado']
)
axes[0].set_title(f'Matriz de Confusão (Limiar = {limiar_global_oof:.4f})')

# 2. Curva Precision-Recall com ponto de operação destacado
precisions, recalls, thresholds = precision_recall_curve(y_test, probs_test)
pr_auc_test = average_precision_score(y_test, probs_test)

axes[1].plot(recalls, precisions, label=f'LightGBM (PR-AUC = {pr_auc_test:.4f})', color='#1f77b4', lw=2)
axes[1].scatter(
    recall_score(y_test, preds_test_otimizado),
    precision_score(y_test, preds_test_otimizado),
    color='red', s=80, zorder=5,
    label=f'Ponto Otimizado (Recall={recall_score(y_test, preds_test_otimizado):.2%})'
)
axes[1].axvline(RECALL_MINIMO, color='gray', linestyle='--', label=f'Meta Recall ({RECALL_MINIMO:.0%})')
axes[1].set_xlabel('Recall (Sensibilidade)')
axes[1].set_ylabel('Precision (Precisão)')
axes[1].set_title('Curva Precision-Recall (Teste Holdout)')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()