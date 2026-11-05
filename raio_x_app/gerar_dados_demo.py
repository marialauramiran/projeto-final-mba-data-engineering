"""Gera tabelas Gold SINTÉTICAS (mesmo schema do contrato) para testar o app sem o Databricks.

Os valores são calibrados nos KPIs do briefing (atraso ~8%, 6% até 100 km e 14% acima
de 2.000 km, 67% da demanda em SP/RJ/MG, +137% de crescimento), mas NÃO são os dados reais.
Substitua pelos Parquet exportados da Gold assim que estiverem prontos.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import UF_REGIAO

rng = np.random.default_rng(42)
PASTA = Path(__file__).parent / "data"
PASTA.mkdir(exist_ok=True)

# Centroides aproximados (lat, lon) e peso de demanda por UF
UFS = {
    "SP": (-23.5, -46.6, 42), "RJ": (-22.9, -43.2, 13), "MG": (-19.9, -43.9, 12), "RS": (-30.0, -51.2, 5.5),
    "PR": (-25.4, -49.3, 5), "SC": (-27.6, -48.5, 3.7), "BA": (-12.97, -38.5, 3.4), "DF": (-15.8, -47.9, 2.1),
    "ES": (-20.3, -40.3, 2), "GO": (-16.7, -49.3, 2), "PE": (-8.05, -34.9, 1.7), "CE": (-3.7, -38.5, 1.3),
    "PA": (-1.45, -48.5, 1), "MT": (-15.6, -56.1, 0.9), "MA": (-2.5, -44.3, 0.75), "MS": (-20.4, -54.6, 0.7),
    "PB": (-7.1, -34.9, 0.55), "PI": (-5.1, -42.8, 0.5), "RN": (-5.8, -35.2, 0.5), "AL": (-9.65, -35.7, 0.4),
    "SE": (-10.9, -37.1, 0.35), "TO": (-10.2, -48.3, 0.28), "RO": (-8.76, -63.9, 0.25), "AM": (-3.1, -60.0, 0.15),
    "AC": (-9.97, -67.8, 0.08), "AP": (0.03, -51.05, 0.07), "RR": (2.8, -60.7, 0.05),
}
# categoria: (pedidos/mês Brasil, preço mediano, nota média, queixa mais comum)
CATEGORIAS = {
    "Cama, mesa e banho": (480, 80, 3.9, "Produto diferente do anúncio"),
    "Saúde e beleza": (440, 90, 4.1, "Não recebi"),
    "Esporte e lazer": (370, 80, 4.1, "Não recebi"),
    "Móveis e decoração": (350, 70, 3.9, "Produto com defeito"),
    "Informática e acessórios": (330, 80, 3.9, "Produto com defeito"),
    "Utilidades domésticas": (300, 50, 4.0, "Não recebi"),
    "Relógios e presentes": (250, 120, 4.0, "Não recebi"),
    "Telefonia": (190, 30, 3.9, "Produto diferente do anúncio"),
    "Brinquedos": (175, 75, 4.1, "Não recebi"),
    "Automotivo": (175, 75, 4.0, "Peça incompatível"),
    "Bebês": (130, 90, 4.0, "Não recebi"),
    "Papelaria": (100, 55, 4.2, "Não recebi"),
    "Pet shop": (75, 65, 4.2, "Não recebi"),
    "Instrumentos musicais": (28, 140, 4.1, "Produto com defeito"),
    "Artes e artesanato": (12, 60, 4.1, "Não recebi"),
}
ORIGENS = ["SP", "MG", "PR", "RJ", "SC", "RS", "DF", "BA", "GO", "ES", "PE", "CE"]
MESES = 20  # jan/2017 a ago/2018


def km(a, b):
    (la1, lo1, _), (la2, lo2, _) = UFS[a], UFS[b]
    p = math.pi / 180
    h = (math.sin((la2 - la1) * p / 2) ** 2
         + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2)
    return max(2 * 6371 * math.asin(math.sqrt(h)), 60.0)


# ---- gold.agg_mercado (categoria x UF de destino) ----
tot_peso = sum(v[2] for v in UFS.values())
linhas = []
for cat, (pm, preco, nota, queixa) in CATEGORIAS.items():
    vend = int(pm * 1.1 * rng.uniform(0.9, 1.1))
    cresc = rng.uniform(0.9, 1.6)
    for uf, (_, _, w) in UFS.items():
        ped = max(int(round(pm * MESES * w / tot_peso * rng.uniform(0.85, 1.15))), 1)
        p17 = ped * 8 / MESES / (1 + cresc) * 2
        n = float(np.clip(nota + rng.normal(0, 0.08) - (0.15 if UF_REGIAO[uf] in ("Nordeste", "Norte") else 0), 1, 5))
        linhas.append(dict(
            categoria=cat, uf_destino=uf, pedidos=ped, meses=MESES,
            pedidos_jan_ago_2017=int(p17), pedidos_jan_ago_2018=int(p17 * (1 + cresc)),
            vendedores_ativos=vend, nota_media=round(n, 2),
            pct_notas_1_2=round(float(np.clip(0.09 + (4.3 - n) * 0.25, 0.05, 0.4)), 3),
            nota_se_atrasar=round(float(np.clip(2.57 + rng.normal(0, 0.1), 1.5, 3.5)), 2),
            preco_p25=round(preco * 0.55, 2), preco_p50=float(preco), preco_p75=round(preco * 1.7, 2),
            parcelas_moda=1 if preco < 40 else 2 if preco < 150 else 4, queixa_top=queixa,
        ))
agg = pd.DataFrame(linhas)

# ---- gold.ref_rotas (UF origem x UF destino) ----
rot = []
for o in ORIGENS:
    for d, (_, _, w) in UFS.items():
        dist = km(o, d)
        taxa = 0.055 + 0.04 * math.log1p(dist / 400) / math.log1p(5)
        if UF_REGIAO[d] == "Nordeste":
            taxa += 0.03
        if UF_REGIAO[d] == "Norte":
            taxa += 0.01
        rot.append(dict(
            uf_origem=o, uf_destino=d, distancia_km=round(dist),
            pedidos=max(int(w * 300 * UFS[o][2] / 42 * rng.uniform(0.8, 1.2)), 3),
            taxa_atraso_hist=round(float(np.clip(taxa + rng.normal(0, 0.005), 0.03, 0.3)), 4),
            frete_base=round(10 + dist / 300, 2), frete_por_kg=round(3.5 + dist / 450, 2),
        ))
rotas = pd.DataFrame(rot)

for nome, df in {"agg_mercado": agg, "ref_rotas": rotas}.items():
    try:
        df.to_parquet(PASTA / f"{nome}.parquet", index=False)
        print(f"data/{nome}.parquet  ({len(df)} linhas)")
    except Exception:  # sem pyarrow: grava CSV (o app lê os dois formatos)
        df.to_csv(PASTA / f"{nome}.csv", index=False)
        print(f"data/{nome}.csv  ({len(df)} linhas)")
