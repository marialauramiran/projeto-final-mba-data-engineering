"""Configuração do app: caminhos, tabelas, mapeamento de colunas (Gold real) e limites do semáforo."""
import os

# ---------------- Origem dos dados ----------------
BASE_PATH = os.getenv("RAIOX_PATH", "/Volumes/workspace/olist_gold/parquet")

T_MERCADO = "gold_agg_mercado"       # mês × categoria
T_FRETE = "gold_agg_frete"           # categoria × UF de origem
T_ROTAS = "gold_ref_rotas"           # categoria × UF de origem × UF de destino
T_REPUTACAO = "gold_agg_reputacao"   # categoria
T_CATEGORIA = "gold_agg_categoria"   # mês × categoria

# Janela de análise do briefing (jan/2017 a ago/2018) e base do crescimento anual.
JANELA_INICIO, JANELA_FIM = "2017-01-01", "2018-08-01"
CRESC_ATUAL = ("2018-01-01", "2018-08-01")      # jan–ago/2018
CRESC_ANTERIOR = ("2017-01-01", "2017-08-01")   # jan–ago/2017

# ---------------- Mapeamento de colunas ----------------
# Esquerda: nome usado pelo código (NÃO mudar). Direita: nome REAL na Gold (agregacao_gold.py).
COLUNAS = {
    T_MERCADO: {
        "mes": "mes_pedido",
        "categoria": "category",
        "pedidos": "total_pedidos",
        "compradores": "compradores_unicos",
        "vendedores": "vendedores_ativos",
        "crescimento_mm": "crescimento_pct",       # mês contra mês, em %
    },
    T_FRETE: {
        "categoria": "category",
        "uf_origem": "uf_origem",
        "pedidos": "total_pedidos",
        "frete_medio": "frete_medio",
        "preco_medio": "preco_medio",
        "peso_medio_g": "peso_medio_g",
        "parcelas": "parcelamento_medio",
        "ticket_medio": "ticket_medio",             # valor de produtos por pedido, sem frete
        "frete_pct_preco": "frete_pct_preco",       # em %
    },
    T_ROTAS: {
        "categoria": "category",
        "uf_origem": "uf_origem",
        "uf_destino": "uf_destino",
        "pedidos": "total_pedidos",
        "atrasos": "pedidos_atrasados",
        "taxa_atraso": "risco_atraso_pct",          # em %
        "distancia_km": "distancia_media_km",
    },
    T_REPUTACAO: {
        "categoria": "category",
        "nota_media": "nota_media",
        "avaliacoes": "total_avaliacoes",
        "avaliacoes_negativas": "avaliacoes_baixas",
        "pct_negativas": "notas_1_2_pct",           # em %
        "reclamacao_top": "queixa_mais_comum",
    },
    T_CATEGORIA: {
        "mes": "mes_pedido",
        "categoria": "category",
        "pedidos": "total_pedidos",
        "itens": "total_itens",
        "preco_medio": "preco_medio",
        "preco_mediano": "preco_mediano",
        "preco_min": "preco_minimo",
        "preco_max": "preco_maximo",
    },
}

# Colunas que a Gold grava em 0–100: o app converte para 0–1.
PERCENTUAIS = {"taxa_atraso", "pct_negativas", "crescimento_mm", "frete_pct_preco"}

# Linhas sem categoria ficam fora da lista de categorias.
CATEGORIAS_IGNORADAS = {"", "nan", "none", "<na>", "categoria_nao_identificada"}

# ---------------- Nome amigável da categoria (só exibição) ----------------
# agro_industria_e_comercio -> Agro Industria e Comercio
PALAVRAS_MINUSCULAS = {"e", "de", "da", "do", "das", "dos", "em", "para", "a", "o"}
# Opcional: acentos e siglas, palavra por palavra (vale para todas as categorias).
CORRECOES_NOME = {
    # "Industria": "Indústria",
    # "Comercio": "Comércio",
    # "Saude": "Saúde",
    # "Eletronicos": "Eletrônicos",
    # "Pcs": "PCs",
    # "Dvds": "DVDs",
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
MERCADO_PART_AMARELO, MERCADO_PART_VERMELHO = 0.01, 0.003   # participação da categoria nos pedidos
FRETE_AMARELO, FRETE_VERMELHO = 0.20, 0.30                   # frete ÷ (preço + frete)
ATRASO_AMARELO, ATRASO_VERMELHO = 0.08, 0.12                 # pior região de destino
NOTA_AMARELO, NOTA_VERMELHO = 4.0, 3.5
MIN_PEDIDOS_ROTA = 30        # abaixo disso, a taxa por região usa todas as categorias da UF

PONTOS = {"verde": 0, "amarelo": 1, "vermelho": 2, "neutro": 0}
GERAL_AMARELO, GERAL_VERMELHO = 2, 5
