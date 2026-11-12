"""Leitura das tabelas Gold em Parquet a partir do Volume do Unity Catalog.

- No Databricks Apps: usa o databricks-sdk (Files API) com o service principal do app.
- Em cluster/notebook ou teste local: se o caminho existir no disco, lê direto com pandas.

Aceita os dois formatos que o Spark costuma gerar:
  <BASE>/gold_agg_mercado/part-0000*.parquet   (pasta)
  <BASE>/gold_agg_mercado.parquet              (arquivo único)
"""
import io
import os

import pandas as pd
import streamlit as st

from . import config as C


class TabelaNaoEncontrada(Exception):
    pass


# ---------- leitura local (disco) ----------
def _ler_local(base: str, tabela: str) -> pd.DataFrame:
    for alvo in (os.path.join(base, tabela), os.path.join(base, f"{tabela}.parquet")):
        if os.path.isdir(alvo):
            arqs = sorted(os.path.join(r, f) for r, _, fs in os.walk(alvo) for f in fs
                          if not f.startswith(("_", ".")) and f.endswith((".parquet", ".csv")))
            if not arqs:
                break
            return pd.concat([pd.read_csv(a) if a.endswith(".csv") else pd.read_parquet(a) for a in arqs],
                             ignore_index=True)
        if os.path.isfile(alvo):
            return pd.read_parquet(alvo)
    raise TabelaNaoEncontrada(f"{tabela} não encontrada em {base}")


# ---------- leitura pelo SDK (Databricks Apps) ----------
@st.cache_resource
def _cliente():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _listar_parquet(w, pasta: str) -> list[str]:
    arqs = []
    for e in w.files.list_directory_contents(pasta):
        nome = e.name.rstrip("/")
        if nome.startswith(("_", ".")):
            continue
        if e.is_directory:
            arqs += _listar_parquet(w, e.path)
        elif nome.endswith(".parquet"):
            arqs.append(e.path)
    return arqs


def _ler_sdk(base: str, tabela: str) -> pd.DataFrame:
    w = _cliente()
    entradas = {e.name.rstrip("/"): e for e in w.files.list_directory_contents(base)}
    for nome in (tabela, f"{tabela}.parquet"):
        e = entradas.get(nome)
        if e is None:
            continue
        arqs = _listar_parquet(w, e.path) if e.is_directory else [e.path]
        if arqs:
            partes = [pd.read_parquet(io.BytesIO(w.files.download(a).contents.read())) for a in arqs]
            return pd.concat(partes, ignore_index=True)
    raise TabelaNaoEncontrada(f"{tabela} não encontrada em {base}")


@st.cache_data(ttl=3600, show_spinner="Carregando dados da camada Gold...")
def ler_bruto(tabela: str) -> pd.DataFrame:
    base = C.BASE_PATH.rstrip("/")
    if os.path.isdir(base):
        return _ler_local(base, tabela)
    return _ler_sdk(base, tabela)


def carregar(tabela: str, obrigatoria: bool = True) -> pd.DataFrame | None:
    """Lê a tabela e renomeia as colunas reais para os nomes usados pelo app."""
    try:
        df = ler_bruto(tabela)
    except TabelaNaoEncontrada:
        if obrigatoria:
            raise
        return None
    mapa = {real: logico for logico, real in C.COLUNAS.get(tabela, {}).items() if real in df.columns}
    df = df.rename(columns=mapa)
    for col in ("categoria", "uf_origem", "uf_destino"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def diagnostico() -> list[dict]:
    """Para o expander de diagnóstico: o que foi encontrado em cada tabela."""
    linhas = []
    for tabela, mapa in C.COLUNAS.items():
        try:
            df = ler_bruto(tabela)
            faltando = [f"{l} → {r}" for l, r in mapa.items() if r not in df.columns]
            linhas.append({"tabela": tabela, "status": "OK", "linhas": len(df),
                           "colunas na tabela": ", ".join(map(str, df.columns)),
                           "mapeadas e não encontradas": ", ".join(faltando) or "—"})
        except Exception as e:  # noqa: BLE001
            linhas.append({"tabela": tabela, "status": "não encontrada", "linhas": 0,
                           "colunas na tabela": "—", "mapeadas e não encontradas": str(e)[:150]})
    return linhas
