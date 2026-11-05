"""Leitura das tabelas Gold exportadas pelo Databricks (Parquet) ou via Amazon Athena."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

PASTA = Path(__file__).resolve().parent.parent / "data"
FONTE = os.getenv("RAIOX_FONTE", "arquivo")  # "arquivo" (Parquet/CSV) ou "athena"


def _ler_arquivo(tabela: str) -> pd.DataFrame:
    pq, csv = PASTA / f"{tabela}.parquet", PASTA / f"{tabela}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"Tabela '{tabela}' não encontrada em {PASTA}. "
        "Exporte a Gold do Databricks ou rode: python gerar_dados_demo.py"
    )


def _ler_athena(tabela: str) -> pd.DataFrame:
    from pyathena import connect

    conn = connect(
        s3_staging_dir=st.secrets["ATHENA_S3_STAGING"],
        region_name=st.secrets["AWS_REGION"],
        schema_name=st.secrets.get("ATHENA_SCHEMA", "gold"),
    )
    return pd.read_sql(f"SELECT * FROM {tabela}", conn)


@st.cache_data(ttl=3600, show_spinner="Carregando dados da camada Gold...")
def carregar(tabela: str) -> pd.DataFrame:
    return _ler_athena(tabela) if FONTE == "athena" else _ler_arquivo(tabela)
