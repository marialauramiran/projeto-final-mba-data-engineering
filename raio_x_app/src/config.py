"""Parâmetros do app: limites do semáforo (definidos a partir dos KPIs do briefing)."""

REGIOES = {
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}
UF_REGIAO = {uf: r for r, ufs in REGIOES.items() for uf in ufs}
ORDEM_REGIOES = ["Sudeste", "Sul", "Centro-Oeste", "Norte", "Nordeste"]

# Frete / ticket: alerta do briefing em 30%
FRETE_AMARELO = 0.20
FRETE_VERMELHO = 0.30

# Risco de atraso por região (taxa média 8,1%; alerta de rota acima de 10%)
ATRASO_AMARELO = 0.08
ATRASO_VERMELHO = 0.12

# Reputação (nota média da categoria)
NOTA_AMARELO = 4.0
NOTA_VERMELHO = 3.5

# Mercado (pedidos/mês da categoria no Brasil)
MERCADO_AMARELO = 100
MERCADO_VERMELHO = 20

# Pontos por sinal e corte do semáforo geral
PONTOS = {"verde": 0, "amarelo": 1, "vermelho": 2}
GERAL_AMARELO = 2   # 2 a 4 pontos
GERAL_VERMELHO = 5  # 5 ou mais
