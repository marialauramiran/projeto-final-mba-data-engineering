"""Consulta a tabela `predictions` (Volume ml_predictions) e calcula os indicadores do cenário."""
import io
import os

import pandas as pd
import streamlit as st

BASE_PRED = os.getenv("RAIOX_PRED_PATH", "/Volumes/workspace/olist_gold/ml_predictions")
T_PRED = "predictions"
LIMIAR_ALTO = 0.5  # usado só se a tabela não tiver coluna de classe prevista

# Nomes possíveis das colunas (a primeira que existir é usada).
COLUNAS = {
    "categoria": ["categoria", "product_category_name_english", "product_category_name"],
    "uf": ["seller_state", "uf", "uf_origem", "estado"],
    "preco": ["valor_produtos", "preco", "price", "valor"],
    "peso": ["peso_total_g", "peso_g", "peso", "product_weight_g", "weight_g"],
    "risco": ["probability", "prob_atraso", "risco_atraso", "probabilidade_atraso", "risk_score", "score"],
    "alto": ["predicted_label", "risco_alto", "alto_risco", "is_high_risk", "flag_alto_risco"],
    "pedido": ["order_id", "pedido_id"],
}


def _col(df: pd.DataFrame, chave: str):
    return next((c for c in COLUNAS[chave] if c in df.columns), None)


# ---------------- Leitura (Files API do Databricks SDK, usa o service principal do app) ----------------
@st.cache_resource
def _cliente():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _listar(path: str):
    return list(_cliente().files.list_directory_contents(path))


def _parquets(path: str) -> list:
    achados = []
    for e in _listar(path):
        if e.is_directory:
            if not e.name.startswith(("_", ".")):
                achados += _parquets(e.path)
        elif e.name.endswith(".parquet"):
            achados.append(e.path)
    return achados


def _ler_arquivo(path: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_cliente().files.download(path).contents.read()))


@st.cache_data(ttl=600, show_spinner="Consultando predictions…")
def carregar_predictions() -> pd.DataFrame:
    erros = []
    for p in (f"{BASE_PRED}/{T_PRED}", f"{BASE_PRED}/{T_PRED}.parquet"):
        try:
            if p.endswith(".parquet"):
                return _ler_arquivo(p)
            arquivos = _parquets(p)
            if arquivos:
                return pd.concat([_ler_arquivo(a) for a in arquivos], ignore_index=True)
            erros.append(f"{p}: nenhum .parquet encontrado")
        except Exception as e:  # noqa: BLE001
            erros.append(f"{p}: {type(e).__name__}: {e}")
    raise FileNotFoundError(" | ".join(erros))


# ---------------- Indicadores ----------------
def _risco(serie: pd.Series) -> pd.Series:
    r = serie.astype(float)
    return r / 100 if r.max() > 1 else r  # aceita escala 0–1 ou 0–100


def calcular_cenario(df, categoria, uf, preco_min, preco_max, peso_min, peso_max) -> dict:
    c = {k: _col(df, k) for k in COLUNAS}
    faltando = [k for k in ("categoria", "uf", "preco", "peso", "risco") if not c[k]]
    if faltando:
        raise KeyError(f"colunas ausentes em predictions: {faltando}. Colunas existentes: {list(df.columns)}")

    base_cat = df[df[c["categoria"]] == categoria]
    f = base_cat[(base_cat[c["uf"]] == uf)
                 & base_cat[c["preco"]].between(preco_min, preco_max)
                 & base_cat[c["peso"]].between(peso_min, peso_max)]
    if f.empty:
        return {"pedidos": 0}

    risco = _risco(f[c["risco"]])
    alto = f[c["alto"]].astype(bool) if c["alto"] else risco >= LIMIAR_ALTO
    medio, pct = float(risco.mean()), float(alto.mean())
    nivel, cor = (("Alto", "vermelho") if medio >= 0.4 or pct >= 0.35 else
                  ("Moderado", "amarelo") if medio >= 0.2 or pct >= 0.15 else
                  ("Baixo", "verde"))
    return {
        "pedidos": int(f[c["pedido"]].nunique()) if c["pedido"] else len(f),
        "risco_medio": medio, "pct_alto": pct, "nivel": nivel, "cor": cor,
        "risco_categoria": float(_risco(base_cat[c["risco"]]).mean()),
    }


def recomendacoes_cenario(ind: dict) -> list:
    if not ind or ind["pedidos"] == 0:
        return ["Sem pedidos nas predições para esse cenário: amplie as faixas de preço e peso."]
    recs = []
    if ind["nivel"] == "Alto":
        recs.append("Risco de atraso elevado neste cenário: revise o prazo prometido e a transportadora.")
    if ind["pct_alto"] >= 0.15:
        recs.append(f"{fmt_pct(ind['pct_alto'])} dos pedidos são de alto risco: acrescente margem no prazo de entrega.")
    dif = ind["risco_medio"] - ind["risco_categoria"]
    if abs(dif) >= 0.02:
        recs.append(f"Risco médio {'acima' if dif > 0 else 'abaixo'} da média da categoria "
                    f"({fmt_pct(ind['risco_categoria'])}) em {abs(dif) * 100:.1f} p.p.".replace(".", ",", 1))
    if ind["nivel"] == "Baixo":
        recs.append("Cenário saudável nas predições: mantenha o prazo atual e monitore o risco.")
    if ind["pedidos"] < 30:
        recs.append("Poucos pedidos neste cenário: interprete os indicadores com cautela.")
    return recs


def fmt_pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",")


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")
