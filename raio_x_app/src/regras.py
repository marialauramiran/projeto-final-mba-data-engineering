"""Cálculo dos cards, do semáforo geral e das recomendações. Tolerante a colunas ausentes."""
from dataclasses import dataclass, field

import pandas as pd

from . import config as C


@dataclass
class Card:
    cor: str                      # verde | amarelo | vermelho | neutro
    metricas: dict                # destaque (2 números grandes)
    linhas: list = field(default_factory=list)   # (rótulo, valor) ou (rótulo, valor, cor)
    pill: str | None = None


# ---------------- utilitários ----------------
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def filtra(df, **kw):
    if df is None:
        return pd.DataFrame()
    for col, v in kw.items():
        if v is not None and col in df.columns:
            df = df[df[col].str.lower() == str(v).lower()]
    return df


def soma(df, col):
    return float(_num(df[col]).sum()) if col in df.columns and not df.empty else None


def media(df, col, peso="pedidos"):
    if col not in df.columns or df.empty:
        return None
    s = _num(df[col])
    if s.notna().sum() == 0:
        return None
    if peso in df.columns:
        w = _num(df[peso]).where(s.notna()).fillna(0)
        if w.sum() > 0:
            return float((s.fillna(0) * w).sum() / w.sum())
    return float(s.mean())


def primeiro(df, col):
    if col not in df.columns or df.empty:
        return None
    s = df[col].dropna()
    return None if s.empty else s.iloc[0]


def brl(v, casas=0):
    if v is None:
        return "—"
    return f"R$ {v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


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


# ---------------- cards ----------------
def card_mercado(mercado, categoria):
    m = filtra(mercado, categoria=categoria)
    pedidos = soma(m, "pedidos")
    total = soma(mercado, "pedidos")
    part = pedidos / total if pedidos is not None and total else None
    ant, atual = soma(m, "pedidos_ano_anterior"), soma(m, "pedidos_ano_atual")
    cresc = (atual / ant - 1) if ant and atual is not None else None
    compradores = soma(m, "compradores")
    # Se a tabela tiver uma linha por UF de destino, o mesmo vendedor se repete entre UFs: usa o máximo.
    vendedores = (float(_num(m["vendedores"]).max()) if "uf_destino" in m.columns else soma(m, "vendedores")) \
        if "vendedores" in m.columns and not m.empty else None

    cor = _cor(part, C.MERCADO_PART_AMARELO, C.MERCADO_PART_VERMELHO, maior_e_pior=False)
    if cor == "verde" and cresc is not None and cresc < 0:
        cor = "amarelo"
    linhas = [("Compradores", inteiro(compradores)), ("Vendedores", inteiro(vendedores))]
    if "uf_destino" in m.columns and "pedidos" in m.columns and not m.empty:
        top = m.assign(p=_num(m["pedidos"])).groupby("uf_destino")["p"].sum().nlargest(3).index
        linhas.append(("Maiores estados compradores", ", ".join(top)))
    if part is not None:
        linhas.append(("Participação no marketplace", pct(part, 1)))
    return Card(cor, {"Pedidos": inteiro(pedidos), "Crescimento": "—" if cresc is None else f"{cresc:+.0%}"}, linhas)


def card_frete(frete, categoria, uf, preco):
    f = filtra(frete, categoria=categoria)
    f_uf = filtra(f, uf_origem=uf)
    if not f_uf.empty:
        f = f_uf
    frete_medio = media(f, "frete_medio")
    pct_ticket = frete_medio / (preco + frete_medio) if frete_medio is not None else None
    linhas = [("Ticket médio da categoria", brl(media(f, "ticket_medio"))),
              ("Preço médio da categoria", brl(media(f, "preco_medio"))),
              ("Peso médio", "—" if media(f, "peso_medio_g") is None else f"{inteiro(media(f, 'peso_medio_g'))} g"),
              ("Parcelamento médio", "—" if media(f, "parcelas") is None else f"{dec(media(f, 'parcelas'))}x")]
    return Card(_cor(pct_ticket, C.FRETE_AMARELO, C.FRETE_VERMELHO),
                {"Frete médio": brl(frete_medio), "Do ticket": pct(pct_ticket)}, linhas)


def tabela_rotas(rotas, modelo, categoria, uf):
    """UF de destino com taxa de atraso: modelo de ML se disponível, senão histórico da rota."""
    r = filtra(rotas, uf_origem=uf).copy()
    if r.empty:
        return r, "sem dados"
    if "atrasos" in r.columns and "pedidos" in r.columns:
        r["taxa"] = _num(r["atrasos"]) / _num(r["pedidos"]).replace(0, pd.NA)
    elif "taxa_atraso" in r.columns:
        r["taxa"] = _num(r["taxa_atraso"])
    else:
        r["taxa"] = pd.NA
    fonte = "taxa histórica da rota"
    if modelo is not None and {"uf_destino", "prob_atraso"} <= set(modelo.columns):
        mm = filtra(filtra(modelo, uf_origem=uf), categoria=categoria)
        if not mm.empty:
            prob = mm.assign(p=_num(mm["prob_atraso"])).groupby("uf_destino")["p"].mean()
            r["taxa"] = r["uf_destino"].map(prob).fillna(r["taxa"])
            fonte = "modelo de ML"
    if "regiao" not in r.columns:
        r["regiao"] = r["uf_destino"].str.upper().map(C.UF_REGIAO)
    return r, fonte


def card_atraso(r, fonte):
    if r.empty or r["taxa"].isna().all():
        return Card("neutro", {}, [("Sem histórico para essa UF de origem", "—")])
    w = _num(r["pedidos"]).fillna(1) if "pedidos" in r.columns else pd.Series(1, index=r.index)
    r = r.assign(w=w, tw=_num(r["taxa"]) * w)
    g = r.dropna(subset=["taxa"]).groupby("regiao")[["tw", "w"]].sum()
    por_regiao = (g["tw"] / g["w"]).reindex([x for x in C.ORDEM_REGIOES if x in g.index] +
                                            [x for x in g.index if x not in C.ORDEM_REGIOES])
    linhas = [(reg, pct(v), _cor(v, C.ATRASO_AMARELO, C.ATRASO_VERMELHO)) for reg, v in por_regiao.items()]
    dist = media(r, "distancia_km")
    metr = {"Pedidos na origem": inteiro(soma(r, "pedidos")), "Distância média": "—" if dist is None else f"{inteiro(dist)} km"}
    return Card(_cor(por_regiao.max(), C.ATRASO_AMARELO, C.ATRASO_VERMELHO), metr, linhas,
                pill=f"Fonte: {fonte}")


def card_reputacao(reput, categoria):
    r = filtra(reput, categoria=categoria)
    nota = media(r, "nota_media", peso="avaliacoes" if "avaliacoes" in r.columns else "pedidos")
    aval, neg = soma(r, "avaliacoes"), soma(r, "avaliacoes_negativas")
    pneg = neg / aval if aval and neg is not None else media(r, "pct_negativas")
    if pneg is not None and pneg > 1:   # veio em 0-100
        pneg = pneg / 100
    return Card(_cor(nota, C.NOTA_AMARELO, C.NOTA_VERMELHO, maior_e_pior=False),
                {"Nota média": dec(nota), "Avaliações negativas": pct(pneg)},
                [("Total de avaliações", inteiro(aval)),
                 ("Avaliações 1-2", inteiro(neg)),
                 ("Reclamação mais comum", primeiro(r, "reclamacao_top") or "—")])


def card_categoria(categ, categoria, preco):
    c = filtra(categ, categoria=categoria)
    medio, mediano = media(c, "preco_medio", peso="volume"), media(c, "preco_mediano", peso="volume")
    pmin = float(_num(c["preco_min"]).min()) if "preco_min" in c.columns and not c.empty else None
    pmax = float(_num(c["preco_max"]).max()) if "preco_max" in c.columns and not c.empty else None
    if pmin is not None and preco < pmin:
        pos = "Abaixo do mínimo"
    elif pmax is not None and preco > pmax:
        pos = "Acima do máximo"
    elif mediano is not None:
        pos = "Abaixo da mediana" if preco < mediano else "Acima da mediana"
    else:
        pos = None
    return Card("neutro", {"Preço médio": brl(medio, 2), "Preço mediano": brl(mediano, 2)},
                [("Preço mínimo", brl(pmin, 2)), ("Preço máximo", brl(pmax, 2)),
                 ("Volume", inteiro(soma(c, "volume")))],
                pill=f"Seu preço: {pos}" if pos else None)


# ---------------- semáforo e textos ----------------
def semaforo_geral(cards: dict):
    avaliados = {k: v for k, v in cards.items() if v.cor != "neutro"}
    pontos = sum(C.PONTOS[c.cor] for c in avaliados.values())
    alertas = [k for k, c in avaliados.items() if c.cor != "verde"]
    if pontos >= C.GERAL_VERMELHO:
        return "vermelho", pontos, "Inviável no cenário atual"
    if pontos >= C.GERAL_AMARELO or any(c.cor == "vermelho" for c in avaliados.values()):
        return "amarelo", pontos, f"Atenção: viável, com {len(alertas)} alerta{'s' if len(alertas) > 1 else ''}"
    return "verde", pontos, "Viável: combinação saudável"


def resumo(cards: dict) -> str:
    partes = []
    if cards["frete"].cor in ("amarelo", "vermelho"):
        partes.append("frete alto para o ticket")
    criticas = [l[0] for l in cards["atraso"].linhas if len(l) == 3 and l[2] == "vermelho"]
    if criticas:
        partes.append(f"risco de atraso para {', '.join(criticas)}")
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


def recomendacoes(cards: dict) -> list[str]:
    recs = []
    lin = [l for l in cards["atraso"].linhas if len(l) == 3]
    criticas = [l[0] for l in lin if l[2] == "vermelho"]
    if criticas:
        boas = [l[0] for l in lin if l[2] == "verde"]
        recs.append(f"Anuncie prazo maior para {', '.join(criticas)}"
                    + (f" ou priorize {' e '.join(boas)} na divulgação" if boas else ""))
    if cards["frete"].cor in ("amarelo", "vermelho"):
        recs.append("Venda em kit para diluir o frete no ticket")
    if cards["reputacao"].cor in ("amarelo", "vermelho"):
        recs.append("Capriche na descrição e na embalagem: a categoria tem reputação abaixo da média")
    if cards["mercado"].cor in ("amarelo", "vermelho"):
        recs.append("Mercado pequeno ou em queda: teste com estoque reduzido antes de escalar")
    pill = cards["categoria"].pill or ""
    if "Acima do máximo" in pill:
        recs.append("Seu preço está acima do máximo praticado na categoria: revise o posicionamento")
    elif "Abaixo do mínimo" in pill:
        recs.append("Seu preço está abaixo do mínimo da categoria: confira se a margem cobre o frete")
    recs.append("Poste dentro do prazo limite: é a etapa que você controla")
    return recs
