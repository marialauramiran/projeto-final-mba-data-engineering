"""Risco de atraso por UF de destino.

Usa o modelo registrado no MLflow (model/modelo.joblib + config.json) quando existir.
Sem modelo, usa a referência sem ML do briefing: taxa histórica de atraso da rota.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

PASTA = Path(__file__).resolve().parent.parent / "model"


@st.cache_resource
def carregar_modelo():
    arq, cfg = PASTA / "modelo.joblib", PASTA / "config.json"
    if arq.exists() and cfg.exists():
        import joblib

        return joblib.load(arq), json.loads(cfg.read_text(encoding="utf-8"))
    return None, None


def risco_por_destino(rotas: pd.DataFrame, categoria: str, preco: float, peso_g: float) -> pd.DataFrame:
    """Recebe as rotas do UF de origem e devolve a coluna prob_atraso por UF de destino."""
    rotas = rotas.copy()
    modelo, config = carregar_modelo()
    if modelo is not None:
        X = rotas.assign(product_category_name=categoria, price=preco, product_weight_g=peso_g)
        X = X.reindex(columns=config["features"])
        rotas["prob_atraso"] = modelo.predict_proba(X)[:, 1]
        rotas["fonte_risco"] = "modelo"
    else:
        rotas["prob_atraso"] = rotas["taxa_atraso_hist"]
        rotas["fonte_risco"] = "taxa histórica da rota"
    return rotas


def fonte_atual() -> str:
    modelo, _ = carregar_modelo()
    return "Modelo preditivo (MLflow)" if modelo is not None else "Taxa histórica da rota (sem ML)"
