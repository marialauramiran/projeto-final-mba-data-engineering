"""Analisa a base `predictions` (Volume ml_predictions) e roda o modelo de ML para prever o risco do cenário.

Fluxo: lê a base -> filtra o cenário (categoria, UF, preço, peso) -> monta as features dos pedidos
-> chama o modelo (Model Serving) -> calcula risco médio, % de alto risco e nº de pedidos.
Se o endpoint não estiver configurado ou falhar, usa as probabilidades já gravadas na base.
"""
import io
import json
import os

import pandas as pd
import streamlit as st

BASE_PRED = os.getenv("RAIOX_PRED_PATH", "/Volumes/workspace/olist_gold/ml_predictions")
T_PRED = "predictions"
ENDPOINT = os.getenv("RAIOX_MODEL_ENDPOINT", "").strip()   # nome do endpoint de Model Serving
LIMIAR_ALTO = 0.5     # usado só se a base não tiver a coluna `threshold`
MAX_CENARIO = 1000    # máx. de pedidos do cenário enviados ao modelo (amostra fixa, reprodutível)
MAX_CATEGORIA = 500   # amostra da categoria inteira, usada como referência de comparação
LOTE = 250            # registros por chamada ao endpoint

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
# Colunas que NÃO são features do modelo (só usadas se a base não tiver inputs_json)
META = {"prediction_id", "order_id", "pedido_id", "probability", "predicted_label", "actual_label", "threshold",
        "prediction_type", "model_name", "model_version", "prediction_timestamp", "inputs_json", "outputs_json"}


def _col(df: pd.DataFrame, chave: str):
    return next((c for c in COLUNAS[chave] if c in df.columns), None)


# ---------------- Leitura da base (Files API do Databricks SDK, service principal do app) ----------------
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


@st.cache_data(ttl=600, show_spinner="Analisando a base de predictions…")
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


# ---------------- Modelo de ML (Model Serving) ----------------
def _features(frame: pd.DataFrame) -> pd.DataFrame:
    """Features exatamente como o modelo recebeu na predição gravada (inputs_json); senão, colunas não-meta."""
    if "inputs_json" in frame.columns:
        try:
            linhas = []
            for s in frame["inputs_json"]:
                d = json.loads(s) if isinstance(s, str) else s
                if isinstance(d, dict) and len(d) == 1 and isinstance(next(iter(d.values())), (dict, list)):
                    d = next(iter(d.values()))
                if isinstance(d, list):
                    d = d[0]
                linhas.append(d)
            return pd.DataFrame(linhas)
        except Exception:  # noqa: BLE001
            pass
    return frame[[c for c in frame.columns if c not in META]].reset_index(drop=True)


def _para_prob(p) -> float:
    if isinstance(p, dict):
        p = p.get("probability", p.get("prob", next(iter(p.values()))))
    if isinstance(p, (list, tuple)):
        p = p[-1]  # [P(classe 0), P(classe 1)] -> P(atraso)
    return float(p)


def prever_modelo(frame: pd.DataFrame) -> pd.Series:
    """Envia os pedidos ao endpoint do modelo e devolve a probabilidade de atraso de cada um."""
    registros = json.loads(_features(frame).to_json(orient="records"))
    saida = []
    for i in range(0, len(registros), LOTE):
        resp = _cliente().serving_endpoints.query(name=ENDPOINT, dataframe_records=registros[i:i + LOTE])
        saida += list(resp.predictions or [])
    if len(saida) != len(registros):
        raise ValueError(f"o modelo devolveu {len(saida)} predições para {len(registros)} pedidos")
    return pd.Series([_para_prob(p) for p in saida], dtype=float)


# ---------------- Indicadores ----------------
def _risco(serie: pd.Series) -> pd.Series:
    r = serie.astype(float)
    return r / 100 if r.max() > 1 else r  # aceita escala 0–1 ou 0–100


def _amostra(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    return frame if len(frame) <= n else frame.sample(n, random_state=0)


@st.cache_data(ttl=600, show_spinner="Rodando o modelo no cenário…")
def calcular_cenario(_df, categoria, uf, preco_min, preco_max, peso_min, peso_max) -> dict:
    c = {k: _col(_df, k) for k in COLUNAS}
    faltando = [k for k in ("categoria", "uf", "preco", "peso") if not c[k]]
    if faltando or (not c["risco"] and not ENDPOINT):
        raise KeyError(f"colunas ausentes em predictions: {faltando or ['risco']}. Colunas existentes: {list(_df.columns)}")

    base_cat = _df[_df[c["categoria"]] == categoria]
    f = base_cat[(base_cat[c["uf"]] == uf)
                 & base_cat[c["preco"]].between(preco_min, preco_max)
                 & base_cat[c["peso"]].between(peso_min, peso_max)]
    if f.empty:
        return {"pedidos": 0, "base": len(_df)}

    thr = float(f["threshold"].median()) if "threshold" in f.columns and f["threshold"].notna().any() else LIMIAR_ALTO

    # 1) Plano B: probabilidades já gravadas na base
    fonte, aviso, amostra = "probabilidades gravadas na base", None, len(f)
    if c["risco"]:
        risco = _risco(f[c["risco"]])
        alto = f[c["alto"]].astype(bool) if c["alto"] else risco >= thr
        risco_cat = float(_risco(base_cat[c["risco"]]).mean())
    else:
        risco, alto, risco_cat = None, None, None

    # 2) Modelo de ML ao vivo
    if ENDPOINT:
        try:
            am = _amostra(f, MAX_CENARIO)
            r_live = _risco(prever_modelo(am))
            cat_live = float(_risco(prever_modelo(_amostra(base_cat, MAX_CATEGORIA))).mean())
            risco, alto, risco_cat = r_live, r_live >= thr, cat_live
            fonte, amostra = f"modelo de ML ao vivo ({ENDPOINT})", len(am)
        except Exception as e:  # noqa: BLE001
            aviso = f"{type(e).__name__}: {e}"
    else:
        aviso = "endpoint do modelo não configurado (RAIOX_MODEL_ENDPOINT)"
    if risco is None:
        raise RuntimeError(f"não foi possível prever o risco: {aviso}")

    medio, pct = float(risco.mean()), float(alto.mean())
    nivel, cor = (("Alto", "vermelho") if medio >= 0.4 or pct >= 0.35 else
                  ("Moderado", "amarelo") if medio >= 0.2 or pct >= 0.15 else
                  ("Baixo", "verde"))
    nome = f["model_name"].dropna() if "model_name" in f.columns else []
    return {
        "pedidos": int(f[c["pedido"]].nunique()) if c["pedido"] else len(f),
        "risco_medio": medio, "pct_alto": pct, "nivel": nivel, "cor": cor, "risco_categoria": risco_cat,
        "fonte": fonte, "aviso": aviso, "amostra": amostra, "base": len(_df), "limiar": thr,
        "modelo_base": str(nome.mode().iloc[0]) if len(nome) else None,
    }


def recomendacoes_cenario(ind: dict) -> list:
    if not ind or ind["pedidos"] == 0:
        return ["Sem pedidos na base para esse cenário: amplie as faixas de preço e peso."]
    recs = []
    if ind["nivel"] == "Alto":
        recs.append("Risco de atraso elevado neste cenário: revise o prazo prometido e a transportadora.")
    if ind["pct_alto"] >= 0.15:
        recs.append(f"{fmt_pct(ind['pct_alto'])} dos pedidos são de alto risco: acrescente margem no prazo de entrega.")
    if ind.get("risco_categoria") is not None:
        dif = ind["risco_medio"] - ind["risco_categoria"]
        if abs(dif) >= 0.02:
            recs.append(f"Risco médio {'acima' if dif > 0 else 'abaixo'} da média da categoria "
                        f"({fmt_pct(ind['risco_categoria'])}) em {abs(dif) * 100:.1f} p.p.".replace(".", ",", 1))
    if ind["nivel"] == "Baixo":
        recs.append("Cenário saudável: mantenha o prazo atual e monitore o risco.")
    if ind["pedidos"] < 30:
        recs.append("Poucos pedidos neste cenário: interprete os indicadores com cautela.")
    return recs


def rodape_cenario(ind: dict) -> str:
    if not ind or ind["pedidos"] == 0:
        return f"Base analisada: {fmt_int((ind or {}).get('base', 0))} registros"
    partes = [f"Predição: {ind['fonte']}", f"base analisada: {fmt_int(ind['base'])} registros"]
    if ind["amostra"] < ind["pedidos"]:
        partes.append(f"modelo aplicado a amostra de {fmt_int(ind['amostra'])} de {fmt_int(ind['pedidos'])} pedidos")
    return " · ".join(partes)


def fmt_pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",")


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")
