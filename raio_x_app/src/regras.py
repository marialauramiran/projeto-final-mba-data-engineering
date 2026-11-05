"""Cálculo dos quatro sinais, do semáforo geral e das recomendações."""
from dataclasses import dataclass, field

import pandas as pd

from . import config as C


@dataclass
class Sinal:
    cor: str
    metricas: dict
    linhas: list = field(default_factory=list)


def _cor(valor, amarelo, vermelho, maior_e_pior=True):
    if maior_e_pior:
        return "vermelho" if valor >= vermelho else "amarelo" if valor >= amarelo else "verde"
    return "vermelho" if valor < vermelho else "amarelo" if valor < amarelo else "verde"


def sinal_mercado(m: pd.DataFrame) -> Sinal:
    pedidos_mes = m["pedidos"].sum() / m["meses"].max()
    base17, base18 = m["pedidos_jan_ago_2017"].sum(), m["pedidos_jan_ago_2018"].sum()
    cresc = (base18 / base17 - 1) if base17 else 0.0
    cor = _cor(pedidos_mes, C.MERCADO_AMARELO, C.MERCADO_VERMELHO, maior_e_pior=False)
    if cor == "verde" and cresc < 0:
        cor = "amarelo"
    top = m.groupby("uf_destino")["pedidos"].sum().nlargest(3).index.tolist()
    return Sinal(cor, {"Pedidos/mês": f"{pedidos_mes:,.0f}".replace(",", "."),
                       "Crescimento": f"{cresc:+.0%}"},
                 [("Maiores compradores", ", ".join(top)),
                  ("Vendedores ativos", f"{int(m['vendedores_ativos'].max()):,}".replace(",", "."))])


def sinal_frete(rotas: pd.DataFrame, m: pd.DataFrame, preco: float, peso_g: float) -> Sinal:
    w = rotas["pedidos"]
    frete_rota = rotas["frete_base"] + rotas["frete_por_kg"] * peso_g / 1000
    frete = float((frete_rota * w).sum() / w.sum())
    pct = frete / (preco + frete)
    p25, p50, p75 = m["preco_p25"].median(), m["preco_p50"].median(), m["preco_p75"].median()
    quartil = 1 if preco <= p25 else 2 if preco <= p50 else 3 if preco <= p75 else 4
    return Sinal(_cor(pct, C.FRETE_AMARELO, C.FRETE_VERMELHO),
                 {"Frete estimado": f"R$ {frete:,.0f}".replace(",", "."), "Do ticket": f"{pct:.0%}"},
                 [("Preço na categoria", f"{quartil}º quartil"),
                  ("Parcelamento comum", f"{int(m['parcelas_moda'].mode().iloc[0])}x")])


def sinal_atraso(rotas: pd.DataFrame) -> Sinal:
    rotas = rotas.assign(regiao=rotas["uf_destino"].map(C.UF_REGIAO))
    por_regiao = (rotas.groupby("regiao")
                  .apply(lambda g: (g["prob_atraso"] * g["pedidos"]).sum() / g["pedidos"].sum(),
                         include_groups=False)
                  .reindex(C.ORDEM_REGIOES).dropna())
    linhas = [(r, v, _cor(v, C.ATRASO_AMARELO, C.ATRASO_VERMELHO)) for r, v in por_regiao.items()]
    return Sinal(_cor(por_regiao.max(), C.ATRASO_AMARELO, C.ATRASO_VERMELHO), {}, linhas)


def sinal_reputacao(m: pd.DataFrame) -> Sinal:
    w = m["pedidos"]
    nota = (m["nota_media"] * w).sum() / w.sum()
    neg = (m["pct_notas_1_2"] * w).sum() / w.sum()
    nota_atraso = (m["nota_se_atrasar"] * w).sum() / w.sum()
    return Sinal(_cor(nota, C.NOTA_AMARELO, C.NOTA_VERMELHO, maior_e_pior=False),
                 {"Nota média": f"{nota:.1f}".replace(".", ","), "Notas 1-2": f"{neg:.0%}"},
                 [("Queixa mais comum", m["queixa_top"].mode().iloc[0]),
                  ("Nota se atrasar", f"{nota_atraso:.1f}".replace(".", ","))])


def semaforo_geral(sinais: dict) -> tuple[str, int, str]:
    pontos = sum(C.PONTOS[s.cor] for s in sinais.values())
    alertas = [k for k, s in sinais.items() if s.cor != "verde"]
    if pontos >= C.GERAL_VERMELHO:
        cor, titulo = "vermelho", "Inviável no cenário atual"
    elif pontos >= C.GERAL_AMARELO or any(s.cor == "vermelho" for s in sinais.values()):
        cor, titulo = "amarelo", f"Atenção: viável, com {len(alertas)} alerta{'s' if len(alertas) > 1 else ''}"
    else:
        cor, titulo = "verde", "Viável: combinação saudável"
    return cor, pontos, titulo


def recomendacoes(sinais: dict, atraso: Sinal) -> list[str]:
    recs = []
    criticas = [r for r, _, c in atraso.linhas if c == "vermelho"]
    if criticas:
        boas = [r for r, _, c in atraso.linhas if c == "verde"]
        recs.append(f"Anuncie prazo maior para {', '.join(criticas)}"
                    + (f" ou priorize {' e '.join(boas)} na divulgação" if boas else ""))
    if sinais["frete"].cor != "verde":
        recs.append("Venda em kit de 2 unidades para diluir o frete no ticket")
    if sinais["reputacao"].cor != "verde":
        recs.append("Capriche na descrição e na embalagem: a categoria tem reputação abaixo da média")
    if sinais["mercado"].cor != "verde":
        recs.append("Mercado pequeno ou em queda: teste com estoque reduzido antes de escalar")
    recs.append("Poste dentro do prazo limite: é a etapa que você controla")
    return recs


def resumo_alertas(sinais: dict, atraso: Sinal) -> str:
    partes = []
    if sinais["frete"].cor != "verde":
        partes.append("Frete alto para o ticket")
    criticas = [r for r, _, c in atraso.linhas if c == "vermelho"]
    if criticas:
        partes.append(f"risco de atraso para o {', '.join(criticas)}")
    elif sinais["atraso"].cor != "verde":
        partes.append("risco de atraso moderado em algumas regiões")
    if sinais["reputacao"].cor != "verde":
        partes.append("reputação da categoria abaixo da média")
    if sinais["mercado"].cor != "verde":
        partes.append("mercado com baixa demanda")
    txt = " e ".join(partes) if len(partes) <= 2 else ", ".join(partes[:-1]) + " e " + partes[-1]
    return (txt[0].upper() + txt[1:]) if txt else "Nenhum alerta para esta combinação"
