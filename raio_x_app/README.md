# Raio-X do Vendedor · App Streamlit (Databricks Apps)

Lê as tabelas Gold em Parquet de `/Volumes/workspace/olist_gold/parquet` e as predições de
`/Volumes/workspace/olist_gold/ml_predictions`.

| Card | Tabela(s) | Grão |
|---|---|---|
| Mercado | gold_agg_mercado + gold_ref_rotas (estados compradores) | mês × categoria |
| Preço e frete | gold_agg_frete | categoria × UF de origem |
| Risco de atraso | gold_ref_rotas | categoria × UF origem × UF destino |
| Reputação | gold_agg_reputacao | categoria |
| Categoria | gold_agg_categoria | mês × categoria |
| Predições do cenário | ml_predictions/predictions | pedido |

## Estrutura
```
streamlit_app.py      tela (formulário, cards, semáforo, recomendações)
predicoes.py          card Predições do cenário (base predictions / Model Serving)
src/config.py         caminhos, mapeamento de colunas, limites do semáforo, nomes de categoria
src/dados.py          leitura do Volume e padronização das colunas
src/regras.py         cálculo dos cards e do semáforo
src/formatacao.py     nome amigável da categoria (só exibição)
app.yaml              comando e variáveis do Databricks Apps
```

## Nome das categorias
A tela mostra `agro_industria_e_comercio` como **Agro Industria e Comercio**. O valor interno não muda,
então os filtros continuam batendo com a Gold e com as predições. Para acentos e siglas, preencha
`CORRECOES_NOME` em `src/config.py` (troca palavra por palavra).

## Regras do semáforo (src/config.py)
| Sinal | Amarelo | Vermelho |
|---|---|---|
| Mercado: participação da categoria nos pedidos | < 1% (ou crescimento anual negativo) | < 0,3% |
| Frete ÷ (preço + frete) | ≥ 20% | ≥ 30% |
| Atraso da pior região de destino | ≥ 8% | ≥ 12% |
| Nota média da categoria | < 4,0 | < 3,5 |

Pontos: verde 0, amarelo 1, vermelho 2. Geral: 0–1 verde, 2–4 amarelo, ≥ 5 vermelho
(qualquer card vermelho deixa o geral no mínimo amarelo). Categoria e Predições não pontuam.

## Definições
- Janela: jan/2017 a ago/2018. Crescimento: pedidos jan–ago/2018 ÷ jan–ago/2017 − 1.
- Colunas `*_pct` da Gold (0–100) são convertidas para 0–1 no app.
- Vendedores ativos: média mensal (a Gold conta vendedores distintos por mês).
- Risco de atraso: pedidos_atrasados ÷ total_pedidos, filtrado pela categoria; com menos de 30 pedidos
  entregues, usa todas as categorias da UF de origem.
- Preço mín. do formulário é o preço de referência nos cards Preço e frete e Categoria.

## Diagnóstico
Barra lateral → "Mostrar colunas das tabelas": mostra colunas reais e campos não encontrados.

## Permissões (uma vez, no SQL Editor)
GRANT USE CATALOG ON CATALOG workspace TO `<service-principal-do-app>`;
GRANT USE SCHEMA ON SCHEMA workspace.olist_gold TO `<service-principal-do-app>`;
GRANT READ VOLUME ON SCHEMA workspace.olist_gold TO `<service-principal-do-app>`;
