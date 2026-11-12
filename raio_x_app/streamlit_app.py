"""Raio-X do Vendedor — app Streamlit lendo a camada Gold (Parquet) do Volume do Databricks."""
import html

import pandas as pd
import streamlit as st

from predicoes import (BASE_PRED, calcular_cenario, carregar_predictions, fmt_int, fmt_pct,
                       recomendacoes_cenario)
from src import config as C
from src.dados import TabelaNaoEncontrada, carregar, diagnostico
from src.regras import (card_atraso, card_categoria, card_frete, card_mercado, card_reputacao,
                        recomendacoes, resumo, semaforo_geral, tabela_rotas)

st.set_page_config(page_title="Raio-X do Vendedor", page_icon="🔎", layout="centered")

# ---------------- Tema (claro / escuro) ----------------
def _tema_escuro() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # Streamlit antigo: segue o tema claro
        return False


ESCURO = _tema_escuro()
# cor: (fundo da pílula, texto da pílula, destaque/borda)
CORES = ({"verde": ("#1E3A1A", "#9FD88A", "#5FAE45"), "amarelo": ("#3D3216", "#F0C66B", "#D6A13A"),
          "vermelho": ("#45201E", "#F4A19C", "#E0625C"), "neutro": ("#1C2C45", "#9CC3F5", "#5B9DF0")}
         if ESCURO else
         {"verde": ("#E9F1DC", "#265B19", "#437426"), "amarelo": ("#F6EEDF", "#5A4815", "#805C1F"),
          "vermelho": ("#F7ECEC", "#7F2C28", "#A73D39"), "neutro": ("#EEF2F8", "#2B4C7E", "#2A78D6")})
PALETA = ({"card": "#1A1D23", "borda": "rgba(255,255,255,.12)", "linha": "rgba(255,255,255,.08)",
           "texto": "#E8E6E1", "suave": "#B4B2AC", "fraco": "#8A8882", "apagado": "rgba(255,255,255,.14)",
           "sombra": "0 1px 2px rgba(0,0,0,.4)"}
          if ESCURO else
          {"card": "#FFFFFF", "borda": "rgba(31,30,29,.15)", "linha": "rgba(31,30,29,.12)",
           "texto": "#141413", "suave": "#3D3D3A", "fraco": "#73726C", "apagado": "rgba(31,30,29,.12)",
           "sombra": "0 1px 2px rgba(0,0,0,.04)"})
ROTULO = {"verde": "Verde", "amarelo": "Amarelo", "vermelho": "Vermelho", "neutro": "Informativo"}

st.markdown(f"""<style>
:root {{
    --rx-card: {PALETA['card']};
    --rx-borda: {PALETA['borda']};
    --rx-linha: {PALETA['linha']};
    --rx-texto: {PALETA['texto']};
    --rx-suave: {PALETA['suave']};
    --rx-fraco: {PALETA['fraco']};
    --rx-apagado: {PALETA['apagado']};
    --rx-sombra: {PALETA['sombra']};
}}

/* 1. Esconde a barra superior nativa do Streamlit para não brigar com a barra do Databricks */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"] {{
    display: none !important;
    height: 0 !important;
}}

/* 2. Força o respiro no topo em todas as versões do Streamlit (antigas e novas) */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
.block-container {{
    padding-top: 2.5rem !important;
    margin-top: 0 !important;
}}

.block-container,
[data-testid="stMainBlockContainer"] {{
    max-width: 880px !important;
    padding-bottom: 2.5rem !important;
}}

/* 3. Garante que o header do app tenha espaço extra e nunca corte */
.rx-head {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 1rem 0 1.2rem 0 !important;
    padding-top: 0.5rem !important;
    line-height: 1.4 !important;
}}
.rx-head .t {{ font-size: 22px; font-weight: 600; color: var(--rx-texto); }}
.rx-head .s {{ margin-left: auto; font-size: 12px; color: var(--rx-fraco); }}

/* 4. Estilos dos cards e indicadores */
.rx-card {{
    background: var(--rx-card); border: 1px solid var(--rx-borda); border-radius: 12px;
    padding: 16px 18px; margin-bottom: 14px; box-shadow: var(--rx-sombra); color: var(--rx-texto);
}}
.rx-top {{ display: flex; align-items: center; justify-content: space-between; font-weight: 600; font-size: 15px; margin-bottom: 12px; }}
.rx-pill {{ padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
.rx-mets {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 10px; }}
.rx-lbl {{ margin: 0; font-size: 12px; color: var(--rx-fraco); }}
.rx-met {{ margin: 2px 0 0; font-size: 22px; font-weight: 700; color: var(--rx-texto); }}
.rx-sub {{ margin: 4px 0 0; font-size: 13px; color: var(--rx-suave); }}
.rx-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
.rx-row {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-top: 1px solid var(--rx-linha); font-size: 14px; }}
.rx-foot {{ margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--rx-linha); font-size: 11px; color: var(--rx-fraco); }}
</style>""", unsafe_allow_html=True)


def pill(cor, texto=None):
    bg, fg, _ = CORES[cor]
    return f'<span class="rx-pill" style="background:{bg};color:{fg}">{html.escape(texto or ROTULO[cor])}</span>'


def render(icone, titulo, card, rodape=None):
    mets = "".join(f'<div><p class="rx-lbl">{html.escape(k)}</p><p class="rx-met">{html.escape(str(v))}</p></div>'
                   for k, v in card.metricas.items())
    linhas = ""
    for linha in card.linhas:
        valor = pill(linha[2], str(linha[1])) if len(linha) == 3 else html.escape(str(linha[1]))
        linhas += f'<div class="rx-row"><span>{html.escape(str(linha[0]))}</span><span>{valor}</span></div>'
    topo_pill = pill("neutro", card.pill) if card.cor == "neutro" and card.pill else pill(card.cor)
    foot = f'<div class="rx-foot">{html.escape(rodape)}</div>' if rodape else ""
    st.markdown(f'<div class="rx-card"><div class="rx-top"><span>{icone} {titulo}</span>{topo_pill}</div>'
                + (f'<div class="rx-mets">{mets}</div>' if mets else "") + linhas + foot + "</div>",
                unsafe_allow_html=True)


# ---------------- Dados (Volume do Databricks) ----------------
try:
    mercado = carregar(C.T_MERCADO)
    frete = carregar(C.T_FRETE)
    rotas = carregar(C.T_ROTAS)
    reputacao = carregar(C.T_REPUTACAO)
    categorias_df = carregar(C.T_CATEGORIA)
    modelo = carregar(C.T_MODELO, obrigatoria=False)   # em desenvolvimento
except TabelaNaoEncontrada as e:
    st.error(f"{e}. Confira o caminho {C.BASE_PATH} e as permissões do app no Volume.")
    st.dataframe(pd.DataFrame(diagnostico()), hide_index=True)
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Não consegui acessar {C.BASE_PATH}: {e}")
    st.info("No Databricks Apps, o service principal do app precisa de USE CATALOG, USE SCHEMA e READ VOLUME.")
    st.stop()

st.markdown('<div class="rx-head"><span style="font-size:22px">🔎</span><span class="t">Raio-X do vendedor</span>'
            '<span class="s">Dados Olist · camada Gold</span></div>', unsafe_allow_html=True)

# ---------------- Entradas ----------------
cats = sorted(set().union(*[set(d["categoria"].dropna()) for d in (mercado, categorias_df, reputacao, frete)
                            if "categoria" in d.columns]))
ufs = sorted(rotas["uf_origem"].dropna().unique()) if "uf_origem" in rotas.columns else []
if not cats or not ufs:
    st.error("Não encontrei a coluna de categoria ou de UF de origem. Ajuste o mapeamento em src/config.py.")
    st.dataframe(pd.DataFrame(diagnostico()), hide_index=True)
    st.stop()

with st.form("entradas", border=False):
    c1, c2 = st.columns([2.2, 1])
    categoria = c1.selectbox("Categoria", cats)
    uf = c2.selectbox("UF de origem", ufs, index=ufs.index("SP") if "SP" in ufs else 0)
    c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1.2], vertical_alignment="bottom")
    preco = c3.number_input("Preço mín. (R$)", min_value=1.0, max_value=50000.0, value=60.0, step=5.0, format="%.2f")
    preco_max = c4.number_input("Preço máx. (R$)", min_value=1.0, max_value=50000.0, value=50000.0, step=5.0, format="%.2f")
    peso = c5.number_input("Peso mín. (g)", min_value=1, max_value=40000, value=400, step=50)
    peso_max = c6.number_input("Peso máx. (g)", min_value=1, max_value=40000, value=40000, step=50)
    c7.form_submit_button("Gerar raio-x", type="primary", use_container_width=True)

# ---------------- Cards ----------------
rotas_uf, fonte_risco = tabela_rotas(rotas, modelo, categoria, uf)
cards = {
    "mercado": card_mercado(mercado, categoria),
    "frete": card_frete(frete, categoria, uf, preco),
    "atraso": card_atraso(rotas_uf, fonte_risco),
    "reputacao": card_reputacao(reputacao, categoria),
    "categoria": card_categoria(categorias_df, categoria, preco),
}
cor, pontos, titulo = semaforo_geral(cards)
bg, fg, borda = CORES[cor]
luzes = "".join(f'<span class="rx-dot" style="background:{CORES[c][2] if c == cor else "var(--rx-apagado)"}"></span>'
                for c in ("verde", "amarelo", "vermelho"))
st.markdown(f'<div class="rx-card" style="display:flex;align-items:center;gap:14px;border-color:{borda}">'
            f'<div aria-hidden="true">{luzes}</div><div><p style="margin:0;font-weight:500;color:{fg}">{titulo}</p>'
            f'<p class="rx-sub">{html.escape(resumo(cards))}</p></div>'
            f'<span style="margin-left:auto;font-size:13px;color:var(--rx-suave)">{pontos} ponto{"s" if pontos != 1 else ""}</span></div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    render("🏪", "Mercado", cards["mercado"], f"Fonte: {C.T_MERCADO}")
    render("⏰", "Risco de atraso", cards["atraso"],
           f"Fonte: {C.T_ROTAS}" + (f" + {C.T_MODELO}" if fonte_risco == "modelo de ML" else " · modelo de ML em desenvolvimento"))
with col2:
    render("🚚", "Preço e frete", cards["frete"], f"Fonte: {C.T_FRETE} · peso informado: {peso} g")
    render("⭐", "Reputação", cards["reputacao"], f"Fonte: {C.T_REPUTACAO}")
render("🏷️", f"Categoria · {categoria}", cards["categoria"], f"Fonte: {C.T_CATEGORIA}")

# ---------------- Predições do cenário (Volume ml_predictions) ----------------
ind, recs_pred = None, []
if preco_max < preco or peso_max < peso:
    st.warning("O valor máximo não pode ser menor que o mínimo (preço ou peso).")
else:
    try:
        ind = calcular_cenario(carregar_predictions(), categoria, uf, preco, preco_max, peso, peso_max)
        recs_pred = recomendacoes_cenario(ind)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Não consegui consultar predictions em {BASE_PRED}: {e}")

if ind is not None:
    if ind["pedidos"] == 0:
        topo = pill("neutro", "Sem dados")
        corpo = '<p class="rx-sub">Nenhum pedido encontrado nas predições com esses filtros.</p>'
    else:
        topo = pill(ind["cor"], f"Risco {ind['nivel']}")
        mets = (("Risco médio de atraso", fmt_pct(ind["risco_medio"])),
                ("Pedidos de alto risco", fmt_pct(ind["pct_alto"])),
                ("Pedidos no cenário", fmt_int(ind["pedidos"])))
        corpo = '<div class="rx-mets">' + "".join(
            f'<div><p class="rx-lbl">{k}</p><p class="rx-met">{v}</p></div>' for k, v in mets) + "</div>"
    st.markdown(f'<div class="rx-card"><div class="rx-top"><span>📈 Predições do cenário</span>{topo}</div>{corpo}'
                f'<div class="rx-foot">Fonte: {html.escape(BASE_PRED)}/predictions · {html.escape(categoria)} · {html.escape(str(uf))} · '
                f'R$ {preco:,.2f}–{preco_max:,.2f} · {peso}–{peso_max} g</div></div>', unsafe_allow_html=True)

recs = "".join(f'<div class="rx-row"><span>{html.escape(r)}</span></div>'
               for r in list(recomendacoes(cards)) + recs_pred)
st.markdown(f'<div class="rx-card"><p style="font-weight:500;margin:0 0 8px">💡 Recomendações</p>{recs}</div>',
            unsafe_allow_html=True)

with st.expander("Diagnóstico dos dados (para ajustar o mapeamento de colunas)"):
    st.caption(f"Caminho: {C.BASE_PATH}")
    st.table(pd.DataFrame(diagnostico()))
    st.caption(f"Predições: {BASE_PRED}")
