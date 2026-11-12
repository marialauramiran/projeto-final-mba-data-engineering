"""Configuração do app: caminho do Volume, tabelas, mapeamento de colunas e limites do semáforo."""
import os

# ---------------- Origem dos dados ----------------
BASE_PATH = os.getenv("RAIOX_PATH", "/Volumes/workspace/olist_gold/parquet")

T_MERCADO = "gold_agg_mercado"
T_FRETE = "gold_agg_frete"
T_ROTAS = "gold_ref_rotas"
T_REPUTACAO = "gold_agg_reputacao"
T_CATEGORIA = "gold_agg_categoria"
# Tabela do modelo de ML (em desenvolvimento). Enquanto não existir, o app usa a taxa histórica da rota.
T_MODELO = os.getenv("RAIOX_TABELA_MODELO", "gold_risco_modelo")

# ---------------- Mapeamento de colunas ----------------
COLUNAS = {
    T_MERCADO: {
        "categoria": "category",
        "uf_destino": "uf_destino",
        "pedidos": "total_pedidos",
        "pedidos_ano_anterior": "pedidos_2017",
        "pedidos_ano_atual": "pedidos_2018",
        "compradores": "compradores",
        "vendedores": "vendedores_ativos",
    },
    T_FRETE: {
        "categoria": "category",
        "uf_origem": "uf_origem",
        "pedidos": "total_pedidos",
        "frete_medio": "frete_medio",
        "ticket_medio": "ticket_medio",
        "preco_medio": "preco_medio",
        "peso_medio_g": "peso_medio_g",
        "parcelas": "parcelas_media",
    },
    T_ROTAS: {
        "uf_origem": "uf_origem",
        "uf_destino": "uf_destino",
        "regiao": "regiao_destino",
        "distancia_km": "distancia_km",
        "pedidos": "pedidos",
        "atrasos": "atrasos",
        "taxa_atraso": "taxa_atraso",
    },
    T_MODELO: {
        "categoria": "category",
        "uf_origem": "uf_origem",
        "uf_destino": "uf_destino",
        "prob_atraso": "prob_atraso",
    },
    T_REPUTACAO: {
        "categoria": "category",
        "pedidos": "total_pedidos",
        "nota_media": "nota_media",
        "avaliacoes": "avaliacoes",
        "avaliacoes_negativas": "avaliacoes_negativas",
        "pct_negativas": "pct_negativas",
        "reclamacao_top": "reclamacao_top",
    },
    T_CATEGORIA: {
        "categoria": "category",
        "preco_medio": "preco_medio",
        "preco_mediano": "preco_mediano",
        "preco_min": "preco_minimo",
        "preco_max": "preco_maximo",
        "volume": "total_pedidos",
    },
}

# ---------------- Regiões ----------------
REGIOES = {
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}
UF_REGIAO = {uf: r for r, ufs in REGIOES.items() for uf in ufs}
ORDEM_REGIOES = ["Sudeste", "Sul", "Centro-Oeste", "Norte", "Nordeste"]

# ---------------- Limites do semáforo (validar com o grupo) ----------------
MERCADO_PART_AMARELO, MERCADO_PART_VERMELHO = 0.01, 0.003   # participação da categoria no total de pedidos
FRETE_AMARELO, FRETE_VERMELHO = 0.20, 0.30                   # frete / (preço + frete)
ATRASO_AMARELO, ATRASO_VERMELHO = 0.08, 0.12                 # pior região
NOTA_AMARELO, NOTA_VERMELHO = 4.0, 3.5

PONTOS = {"verde": 0, "amarelo": 1, "vermelho": 2, "neutro": 0}
GERAL_AMARELO, GERAL_VERMELHO = 2, 5
