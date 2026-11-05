# Raio-X do Vendedor · App Streamlit

Camada de consumo da arquitetura medalhão: lê as tabelas Gold exportadas do Databricks
(Parquet) ou consulta via Amazon Athena, e mostra o semáforo de viabilidade com 4 sinais
(Mercado, Preço e frete, Risco de atraso, Reputação) e recomendações.

## Rodar localmente
```bash
pip install -r requirements.txt
python gerar_dados_demo.py        # dados SINTÉTICOS para testar (opcional)
streamlit run streamlit_app.py
```

## Usar os dados reais
1. No Databricks, rode `databricks/06_export_gold_app.py` (depois dos notebooks Bronze/Silver/Gold).
2. Copie `agg_mercado.parquet` e `ref_rotas.parquet` para `data/`.
3. (Opcional) Copie `modelo.joblib` + `config.json` do MLflow para `model/`.
   `config.json` precisa ter `{"features": [...]}` com a ordem das colunas do modelo.
   Sem modelo, o risco usa a taxa histórica da rota (referência sem ML).

## Usar o Athena
`RAIOX_FONTE=athena streamlit run streamlit_app.py` + `.streamlit/secrets.toml`
(veja `secrets.toml.example`).

## Contrato de schema
| Tabela | Colunas |
|---|---|
| agg_mercado | categoria, uf_destino, pedidos, meses, pedidos_jan_ago_2017, pedidos_jan_ago_2018, vendedores_ativos, nota_media, pct_notas_1_2, nota_se_atrasar, preco_p25, preco_p50, preco_p75, parcelas_moda, queixa_top |
| ref_rotas | uf_origem, uf_destino, distancia_km, pedidos, taxa_atraso_hist, frete_base, frete_por_kg |

## Regras do semáforo (`src/config.py`)
| Sinal | Amarelo | Vermelho |
|---|---|---|
| Frete / ticket | ≥ 20% | ≥ 30% |
| Atraso (pior região) | ≥ 8% | ≥ 12% |
| Nota da categoria | < 4,0 | < 3,5 |
| Pedidos/mês | < 100 | < 20 |

Pontos: verde 0, amarelo 1, vermelho 2. Geral: 0–1 verde, 2–4 amarelo, ≥ 5 vermelho.

## Deploy
Streamlit Community Cloud apontando para este repositório (arquivo principal `streamlit_app.py`).
