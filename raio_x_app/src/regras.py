"""Cálculo dos cards, do semáforo geral e das recomendações a partir das tabelas Gold reais."""
from dataclasses import dataclass, field

import pandas as pd

from . import config as C


@dataclass
class Card:
    cor: str                      # verde | amarelo | vermelho | neutro
    metricas: dict
    linhas: list = field(default_factory=list)   # (rótulo, valor) ou (rótulo, valor, cor)
    pill: str | None = None
    nota: str | None = None       # observação exibida no rodapé


# ---------------- utilitários ----------------
def filtra(df, **kw):
    if df is None or df.empty:
        return pd.DataFrame(columns=getattr(df, "columns", []))
    m = pd.Series(True, index=df.index)
    for col, v in kw.items():
        if v is not None and col in df.columns:
            m &= df[col].astype("string").str.upper() == str(v).upper()
    return df[m]


def janela(df, ini=C.JANELA_INICIO, fim=C.JANELA_FIM):
    if "mes" not in df.columns:
        return df
    return df[df["mes"].between(pd.Timestamp(ini), pd.Timestamp(fim))]


def soma(df, col):
    return float(df[col].sum()) if col in df.columns and not df.empty and df[col].notna().any() else None


def media(df, col, peso=None):
    if col not in df.columns or df.empty or df[col].notna().sum() == 0:
        return None
    s = df[col]
    if peso and peso in df.columns:
        w = df[peso].where(s.notna()).fillna(0)
        if w.sum() > 0:
            return float((s.fillna(0) * w).sum() / w.sum())
    return float(s.mean())


def brl(v, casas=0):
    return "—" if v is None else f"R$ {v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def inteiro(v):
    return "—" if v is None else f"{v:,.0f}".replace(",", ".")


def pct(v, casas=0):
    return "—" if v is None else f"{v:.{casas}%}".replace(".", ",")


def dec(v, casas=1):
    return "—" if v is None else f"{v:.{casas}f}".replace(".", ",")


def _cor(v, amarelo, vermelho, maior_e_pior=True):
    if v is None:
        return "neutro"
    if maior_e_pior:
        return "vermelho" if v >= vermelho else "amarelo" if v >= amarelo else "verde"
    return "vermelho" if v < vermelho else "amarelo" if v < amarelo else "verde"


# ---------------- 🏪 Mercado (gold_agg_mercado + gold_ref_rotas) ----------------
def card_mercado(mercado, rotas, categoria):
    base = janela(mercado)
    m = filtra(base, categoria=categoria)
    pedidos, total = soma(m, "pedidos"), soma(base, "pedidos")
    meses = m["mes"].nunique() if "mes" in m.columns and not m.empty else None
    part = pedidos / total if pedidos is not None and total else None

    atual = soma(janela(m, *C.CRESC_ATUAL), "pedidos")
    anterior = soma(janela(m, *C.CRESC_ANTERIOR), "pedidos")
    cresc = (atual / anterior - 1) if atual is not None and anterior else None

    cor = _cor(part, C.MERCADO_PART_AMARELO, C.MERCADO_PART_VERMELHO, maior_e_pior=False)
    if cor == "verde" and cresc is not None and cresc < 0:
        cor = "amarelo"

    # vendedores_ativos é distinto por mês: somar meses contaria o mesmo vendedor várias vezes
    vend_mes = media(m, "vendedores")
    linhas = [("Pedidos por mês (média)", inteiro(pedidos / meses) if pedidos and meses else "—"),
              ("Compradores únicos (soma mensal)", inteiro(soma(m, "compradores"))),
              ("Vendedores ativos por mês (média)", inteiro(vend_mes))]
    r = filtra(rotas, categoria=categoria)
    if not r.empty and {"uf_destino", "pedidos"} <= set(r.columns):
        top = r.groupby("uf_destino")["pedidos"].sum().nlargest(3).index
        linhas.append(("Maiores estados compradores", ", ".join(top)))
    linhas.append(("Participação no marketplace", pct(part, 1)))
    return Card(cor, {"Pedidos": inteiro(pedidos), "Crescimento": "—" if cresc is None else f"{cresc:+.0%}"},
                linhas, nota="Crescimento: jan–ago/2018 × jan–ago/2017")


# ---------------- 🚚 Preço e frete (gold_agg_frete) ----------------
def card_frete(frete, categoria, uf, preco):
    f_cat = filtra(frete, categoria=categoria)
    f = filtra(f_cat, uf_origem=uf)
    nota = f"Médias da categoria vendida a partir de {uf}"
    if f.empty:
        f, nota = f_cat, f"Sem vendas da categoria a partir de {uf}: média Brasil"
    frete_medio = media(f, "frete_medio", "pedidos")
    pct_ticket = frete_medio / (preco + frete_medio) if frete_medio is not None else None
    peso = media(f, "peso_medio_g", "pedidos")
    parc = media(f, "parcelas", "pedidos")
    return Card(_cor(pct_ticket, C.FRETE_AMARELO, C.FRETE_VERMELHO),
                {"Frete médio por item": brl(frete_medio), "Do ticket": pct(pct_ticket)},
                [("Ticket médio por pedido (sem frete)", brl(media(f, "ticket_medio", "pedidos"))),
                 ("Preço médio do item", brl(media(f, "preco_medio", "pedidos"))),
                 ("Frete ÷ preço na categoria", pct(media(f, "frete_pct_preco", "pedidos"))),
                 ("Peso médio", "—" if peso is None else f"{inteiro(peso)} g"),
                 ("Parcelamento médio", "—" if parc is None else f"{dec(parc)}x"),
                 ("Pedidos considerados", inteiro(soma(f, "pedidos")))],
                nota=nota)


# ---------------- ⏰ Risco de atraso (gold_ref_rotas) ----------------
def tabela_rotas(rotas, categoria, uf):
    """Rotas da UF de origem para a categoria. Com pouco histórico, usa todas as categorias da UF."""
    r_uf = filtra(rotas, uf_origem=uf)
    r = filtra(r_uf, categoria=categoria)
    escopo = "categoria escolhida"
    if (soma(r, "pedidos") or 0) < C.MIN_PEDIDOS_ROTA:
        r, escopo = r_uf, "todas as categorias (pouco histórico na categoria)"
    if r.empty:
        return r, escopo
    r = r.groupby("uf_destino", as_index=False).agg(
        pedidos=("pedidos", "sum"), atrasos=("atrasos", "sum"),
        distancia_km=("distancia_km", lambda s: s.mean()))
    r["taxa"] = r["atrasos"] / r["pedidos"].where(r["pedidos"] > 0)
    r["regiao"] = r["uf_destino"].str.upper().map(C.UF_REGIAO)
    return r, escopo


def card_atraso(r, escopo):
    if r.empty or r["taxa"].isna().all():
        return Card("neutro", {}, [("Sem histórico de entregas para essa UF de origem", "—")])
    g = r.dropna(subset=["taxa", "regiao"]).groupby("regiao")[["atrasos", "pedidos"]].sum()
    por_regiao = (g["atrasos"] / g["pedidos"]).reindex([x for x in C.ORDEM_REGIOES if x in g.index])
    linhas = [(reg, pct(v), _cor(v, C.ATRASO_AMARELO, C.ATRASO_VERMELHO)) for reg, v in por_regiao.items()]
    taxa_geral = soma(r, "atrasos") / soma(r, "pedidos") if soma(r, "pedidos") else None
    return Card(_cor(por_regiao.max(), C.ATRASO_AMARELO, C.ATRASO_VERMELHO),
                {"Taxa de atraso": pct(taxa_geral, 1), "Distância média": f"{inteiro(media(r, 'distancia_km', 'pedidos'))} km"},
                linhas, nota=f"Taxa histórica de atraso · {escopo} · {inteiro(soma(r, 'pedidos'))} pedidos entregues")


# ---------------- ⭐ Reputação (gold_agg_reputacao) ----------------
def card_reputacao(reput, categoria):
    r = filtra(reput, categoria=categoria)
    nota = media(r, "nota_media", "avaliacoes")
    aval, neg = soma(r, "avaliacoes"), soma(r, "avaliacoes_negativas")
    pneg = neg / aval if aval and neg is not None else media(r, "pct_negativas")
    queixa = r["reclamacao_top"].dropna().iloc[0] if "reclamacao_top" in r.columns and r["reclamacao_top"].notna().any() else "—"
    if str(queixa).lower().startswith("outros"):
        queixa = "Não identificada no texto"
    return Card(_cor(nota, C.NOTA_AMARELO, C.NOTA_VERMELHO, maior_e_pior=False),
                {"Nota média": dec(nota), "Notas 1-2": pct(pneg)},
                [("Total de avaliações", inteiro(aval)), ("Avaliações com nota 1 ou 2", inteiro(neg)),
                 ("Queixa mais comum", queixa)])


# ---------------- 🏷️ Categoria (gold_agg_categoria) ----------------
def card_categoria(categ, categoria, preco):
    c = filtra(janela(categ), categoria=categoria)
    medio = media(c, "preco_medio", "itens")
    mediano = media(c, "preco_mediano", "itens")     # média ponderada das medianas mensais
    pmin = float(c["preco_min"].min()) if "preco_min" in c.columns and not c.empty else None
    pmax = float(c["preco_max"].max()) if "preco_max" in c.columns and not c.empty else None
    if pmin is not None and preco < pmin:
        pos = "abaixo do mínimo"
    elif pmax is not None and preco > pmax:
        pos = "acima do máximo"
    elif mediano is not None:
        pos = "abaixo da mediana" if preco < mediano else "acima da mediana"
    else:
        pos = None
    return Card("neutro", {"Preço médio": brl(medio, 2), "Preço mediano (aprox.)": brl(mediano, 2)},
                [("Preço mínimo", brl(pmin, 2)), ("Preço máximo", brl(pmax, 2)),
                 ("Pedidos", inteiro(soma(c, "pedidos"))), ("Itens vendidos", inteiro(soma(c, "itens")))],
                pill=f"Seu preço: {pos}" if pos else None,
                nota="Mediana aproximada: média ponderada das medianas mensais")


# ---------------- semáforo e textos ----------------
SINAIS = ("mercado", "frete", "atraso", "reputacao")


def semaforo_geral(cards):
    av = {k: cards[k] for k in SINAIS if cards[k].cor != "neutro"}
    pontos = sum(C.PONTOS[c.cor] for c in av.values())
    alertas = [k for k, c in av.items() if c.cor != "verde"]
    if pontos >= C.GERAL_VERMELHO:
        return "vermelho", pontos, "Inviável no cenário atual"
    if pontos >= C.GERAL_AMARELO or any(c.cor == "vermelho" for c in av.values()):
        return "amarelo", pontos, f"Atenção: viável, com {len(alertas)} alerta{'s' if len(alertas) != 1 else ''}"
    return "verde", pontos, "Viável: combinação saudável"


def _regioes(card, cor):
    return [l[0] for l in card.linhas if len(l) == 3 and l[2] == cor]


def resumo(cards):
    partes = []
    if cards["frete"].cor in ("amarelo", "vermelho"):
        partes.append("frete alto para o ticket")
    if _regioes(cards["atraso"], "vermelho"):
        partes.append(f"risco de atraso para {', '.join(_regioes(cards['atraso'], 'vermelho'))}")
    elif cards["atraso"].cor == "amarelo":
        partes.append("risco de atraso moderado em algumas regiões")
    if cards["reputacao"].cor in ("amarelo", "vermelho"):
        partes.append("reputação da categoria abaixo da média")
    if cards["mercado"].cor in ("amarelo", "vermelho"):
        partes.append("mercado pequeno ou em queda")
    if not partes:
        return "Nenhum alerta para esta combinação"
    txt = " e ".join(partes) if len(partes) <= 2 else ", ".join(partes[:-1]) + " e " + partes[-1]
    return txt[0].upper() + txt[1:]


def recomendacoes(cards):
    recs = []
    criticas, boas = _regioes(cards["atraso"], "vermelho"), _regioes(cards["atraso"], "verde")
    if criticas:
        recs.append(f"Anuncie prazo maior para {', '.join(criticas)}"
                    + (f" ou priorize {' e '.join(boas)} na divulgação" if boas else ""))
    if cards["frete"].cor in ("amarelo", "vermelho"):
        recs.append("Venda em kit para diluir o frete no ticket")
    if cards["reputacao"].cor in ("amarelo", "vermelho"):
        recs.append("Capriche na descrição e na embalagem: a categoria tem reputação abaixo da média")
    if cards["mercado"].cor in ("amarelo", "vermelho"):
        recs.append("Mercado pequeno ou em queda: teste com estoque reduzido antes de escalar")
    pill = cards["categoria"].pill or ""
    if "acima do máximo" in pill:
        recs.append("Seu preço está acima do máximo praticado na categoria: revise o posicionamento")
    elif "abaixo do mínimo" in pill:
        recs.append("Seu preço está abaixo do mínimo da categoria: confira se a margem cobre o frete")
    recs.append("Poste dentro do prazo limite: é a etapa que você controla")
    return recs
