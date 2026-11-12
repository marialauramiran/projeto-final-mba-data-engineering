"""Raio-X do Vendedor — app Streamlit lendo a camada Gold (Parquet) do Volume do Databricks."""
import html

import pandas as pd
import streamlit as st

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
:root{{--rx-card:{PALETA['card']};--rx-borda:{PALETA['borda']};--rx-linha:{PALETA['linha']};
--rx-texto:{PALETA['texto']};--rx-suave:{PALETA['suave']};--rx-fraco:{PALETA['fraco']};
--rx-apagado:{PALETA['apagado']};--rx-sombra:{PALETA['sombra']}}}
.block-container{{padding-top:2rem;max-width:880px}}
.rx-card{{background:var(--rx-card);color:var(--rx-texto);border:1px solid var(--rx-borda);border-radius:12px;
padding:1rem 1.25rem;margin-bottom:12px;box-shadow:var(--rx-sombra)}}
.rx-top{{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;font-weight:500}}
.rx-mets{{display:flex;gap:24px;margin-bottom:8px;flex-wrap:wrap}}
.rx-lbl{{font-size:12px;color:var(--rx-suave);margin:0 0 2px}}
.rx-met{{font-size:21px;font-weight:500;margin:0;color:var(--rx-texto)}}
.rx-row{{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:13px;padding:6px 0;
border-top:1px solid var(--rx-linha);color:var(--rx-texto)}}
.rx-pill{{font-size:12px;padding:2px 10px;border-radius:8px;white-space:nowrap}}
.rx-foot{{font-size:11px;color:var(--rx-fraco);margin-top:6px}}
.rx-sub{{margin:0;font-size:13px;color:var(--rx-suave)}}
.rx-dot{{width:14px;height:14px;border-radius:50%;display:inline-block;margin-right:6px}}
</style>""", unsafe_allow_html=True)


def pill(cor, texto=None):
    bg, fg, _ = CORES[cor]
    return f'<span class="rx-pill" style="background:{bg};color:{fg}">{html.escape(texto or ROTULO[cor])}</span>'


def render(icone, titulo, card, rodape=None):
    mets = "".join(f'<div><p class="rx-lbl">{html.escape(k)}</p><p class="rx-met">{html.escape(str(v))}</p></div>'
                   for k, v in card.metricas.items())
    linhas = ""
    for l in card.linhas:
        valor = pill(l[2], str(l[1])) if len(l) == 3 else html.escape(str(l[1]))
        linhas += f'<div class="rx-row"><span>{html.escape(str(l[0]))}</span><span>{valor}</span></div>'
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

st.markdown('<div style="display:flex;align-items:center;gap:10px;margin-bottom:.5rem">'
            '<span style="font-size:22px">🔎</span><span style="font-size:22px;font-weight:600;color:var(--rx-texto)">Raio-X do vendedor</span>'
            '<span style="margin-left:auto;font-size:12px;color:var(--rx-fraco)">Dados Olist · camada Gold</span></div>',
            unsafe_allow_html=True)

# ---------------- Entradas ----------------
cats = sorted(set().union(*[set(d["categoria"].dropna()) for d in (mercado, categorias_df, reputacao, frete)
                            if "categoria" in d.columns]))
ufs = sorted(rotas["uf_origem"].dropna().unique()) if "uf_origem" in rotas.columns else []
if not cats or not ufs:
    st.error("Não encontrei a coluna de categoria ou de UF de origem. Ajuste o mapeamento em src/config.py.")
    st.dataframe(pd.DataFrame(diagnostico()), hide_index=True)
    st.stop()

with st.form("entradas", border=False):
    c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1.1, 1.1, 1.3], vertical_alignment="bottom")
    categoria = c1.selectbox("Categoria", cats)
    uf = c2.selectbox("UF de origem", ufs, index=ufs.index("SP") if "SP" in ufs else 0)
    preco = c3.number_input("Preço (R$)", min_value=1.0, max_value=50000.0, value=60.0, step=5.0, format="%.2f")
    peso = c4.number_input("Peso (g)", min_value=1, max_value=40000, value=400, step=50)
    c5.form_submit_button("Gerar raio-x", type="primary", use_container_width=True)

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

recs = "".join(f'<div class="rx-row"><span>{html.escape(r)}</span></div>' for r in recomendacoes(cards))
st.markdown(f'<div class="rx-card"><p style="font-weight:500;margin:0 0 8px">💡 Recomendações</p>{recs}</div>',
            unsafe_allow_html=True)

# ---------------- Diagnóstico (sempre visível na barra lateral) ----------------
with st.sidebar:
    st.subheader("🔧 Diagnóstico dos dados")
    st.caption(f"Caminho: {C.BASE_PATH}")
    if st.toggle("Mostrar colunas das tabelas", value=False):
        try:
            for item in diagnostico():
                with st.expander(f"{item['tabela']} · {item['status']}"):
                    st.write(f"**Linhas:** {item['linhas']}")
                    st.write(f"**Colunas na tabela:** {item['colunas na tabela']}")
                    st.write(f"**Mapeadas e não encontradas:** {item['mapeadas e não encontradas']}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro ao ler o Volume: {e}")