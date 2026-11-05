"""Raio-X do Vendedor — app Streamlit (camada de consumo da arquitetura medalhão)."""
import html

import streamlit as st

from src import config as C
from src.dados import carregar
from src.modelo import fonte_atual, risco_por_destino
from src.regras import (recomendacoes, resumo_alertas, semaforo_geral, sinal_atraso,
                        sinal_frete, sinal_mercado, sinal_reputacao)

st.set_page_config(page_title="Raio-X do Vendedor", page_icon="🔎", layout="centered")

CORES = {  # fundo, texto, borda
    "verde": ("#E9F1DC", "#265B19", "#437426"),
    "amarelo": ("#F6EEDF", "#5A4815", "#805C1F"),
    "vermelho": ("#F7ECEC", "#7F2C28", "#A73D39"),
}
ROTULO = {"verde": "Verde", "amarelo": "Amarelo", "vermelho": "Vermelho"}

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:860px}
.rx-head{display:flex;align-items:center;gap:10px;margin-bottom:.5rem}
.rx-head .tt{font-size:22px;font-weight:600}
.rx-head .sub{margin-left:auto;font-size:12px;color:#73726C}
.rx-card{background:#fff;border:1px solid rgba(31,30,29,.15);border-radius:12px;padding:1rem 1.25rem;margin-bottom:12px}
.rx-lbl{font-size:12px;color:#3D3D3A;margin:0 0 2px}
.rx-met{font-size:22px;font-weight:500;margin:0}
.rx-row{display:flex;justify-content:space-between;align-items:center;font-size:13px;padding:6px 0;border-top:1px solid rgba(31,30,29,.12)}
.rx-pill{font-size:12px;padding:2px 10px;border-radius:8px}
.rx-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;font-weight:500}
.rx-mets{display:flex;gap:24px;margin-bottom:8px}
.rx-sem{display:flex;align-items:center;gap:14px}
.rx-dot{width:14px;height:14px;border-radius:50%;background:rgba(31,30,29,.12);display:inline-block;margin-right:6px}
</style>""", unsafe_allow_html=True)


def pill(cor, texto=None):
    bg, fg, _ = CORES[cor]
    return f'<span class="rx-pill" style="background:{bg};color:{fg}">{html.escape(texto or ROTULO[cor])}</span>'


def card(icone, titulo, sinal, linhas_html=None):
    mets = "".join(f'<div><p class="rx-lbl">{k}</p><p class="rx-met"'
                   + (f' style="color:{CORES[sinal.cor][1]}"' if k == "Do ticket" and sinal.cor != "verde" else "")
                   + f'>{html.escape(v)}</p></div>' for k, v in sinal.metricas.items())
    linhas = linhas_html or "".join(
        f'<div class="rx-row"><span>{html.escape(a)}</span><span>{html.escape(str(b))}</span></div>'
        for a, b in sinal.linhas)
    return (f'<div class="rx-card"><div class="rx-top"><span>{icone} {titulo}</span>{pill(sinal.cor)}</div>'
            + (f'<div class="rx-mets">{mets}</div>' if mets else "") + linhas + "</div>")


# ---------- Dados da Gold ----------
try:
    mercado = carregar("agg_mercado")
    rotas_ref = carregar("ref_rotas")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.markdown('<div class="rx-head"><span style="font-size:22px">🔎</span><span class="tt">Raio-X do vendedor</span>'
            '<span class="sub">Dados Olist 2017–2018</span></div>', unsafe_allow_html=True)

# ---------- Entradas ----------
with st.form("entradas", border=False):
    c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1.1, 1.1, 1.3], vertical_alignment="bottom")
    categorias = sorted(mercado["categoria"].unique())
    categoria = c1.selectbox("Categoria", categorias,
                             index=categorias.index("Saúde e beleza") if "Saúde e beleza" in categorias else 0)
    ufs = sorted(rotas_ref["uf_origem"].unique())
    uf = c2.selectbox("UF de origem", ufs, index=ufs.index("SP") if "SP" in ufs else 0)
    preco = c3.number_input("Preço (R$)", min_value=1.0, max_value=20000.0, value=60.0, step=5.0, format="%.2f")
    peso = c4.number_input("Peso (g)", min_value=1, max_value=40000, value=400, step=50)
    gerar = c5.form_submit_button("Gerar raio-x", type="primary", use_container_width=True)

m = mercado[mercado["categoria"] == categoria]
rotas = risco_por_destino(rotas_ref[rotas_ref["uf_origem"] == uf], categoria, preco, peso)

if m.empty or rotas.empty:
    st.warning("Sem histórico suficiente para essa combinação de categoria e UF de origem.")
    st.stop()

sinais = {
    "mercado": sinal_mercado(m),
    "frete": sinal_frete(rotas, m, preco, peso),
    "atraso": sinal_atraso(rotas),
    "reputacao": sinal_reputacao(m),
}
cor_geral, pontos, titulo = semaforo_geral(sinais)

# ---------- Semáforo geral ----------
bg, fg, borda = CORES[cor_geral]
luzes = "".join(f'<span class="rx-dot" style="background:{CORES[c][2] if c == cor_geral else "rgba(31,30,29,.12)"}"></span>'
                for c in ["verde", "amarelo", "vermelho"])
st.markdown(
    f'<div class="rx-card rx-sem" style="border-color:{borda}">'
    f'<div aria-hidden="true">{luzes}</div>'
    f'<div><p style="margin:0;font-weight:500;color:{fg}">{titulo}</p>'
    f'<p style="margin:0;font-size:13px;color:#3D3D3A">{html.escape(resumo_alertas(sinais, sinais["atraso"]))}</p></div>'
    f'<span style="margin-left:auto;font-size:13px;color:#3D3D3A">{pontos} ponto{"s" if pontos != 1 else ""}</span></div>',
    unsafe_allow_html=True)

# ---------- Quatro blocos ----------
linhas_atraso = "".join(f'<div class="rx-row"><span>{r}</span>{pill(c, f"{v:.0%}")}</div>'
                        for r, v, c in sinais["atraso"].linhas)
col1, col2 = st.columns(2)
with col1:
    st.markdown(card("🏪", "Mercado", sinais["mercado"]), unsafe_allow_html=True)
    st.markdown(card("⏰", "Risco de atraso", sinais["atraso"], linhas_atraso), unsafe_allow_html=True)
with col2:
    st.markdown(card("🚚", "Preço e frete", sinais["frete"]), unsafe_allow_html=True)
    st.markdown(card("⭐", "Reputação", sinais["reputacao"]), unsafe_allow_html=True)

# ---------- Recomendações ----------
recs = "".join(f'<div class="rx-row"><span>{html.escape(r)}</span></div>'
               for r in recomendacoes(sinais, sinais["atraso"]))
st.markdown(f'<div class="rx-card"><p style="font-weight:500;margin:0 0 8px">💡 Recomendações</p>{recs}</div>',
            unsafe_allow_html=True)

with st.expander("Detalhe por UF de destino"):
    st.dataframe(
        rotas.assign(regiao=rotas["uf_destino"].map(C.UF_REGIAO))
             [["uf_destino", "regiao", "distancia_km", "pedidos", "prob_atraso"]]
             .sort_values("prob_atraso", ascending=False),
        hide_index=True, use_container_width=True,
        column_config={"prob_atraso": st.column_config.ProgressColumn("Risco de atraso", format="percent",
                                                                     min_value=0, max_value=0.3),
                       "distancia_km": st.column_config.NumberColumn("Distância (km)", format="%d")})

st.caption(f"Risco de atraso calculado por: {fonte_atual()}. "
           "Fonte: camada Gold (Databricks) — gold.agg_mercado e gold.ref_rotas.")
