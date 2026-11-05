"""
m1_regressao_logistica.py — Regressão Logística (família linear)

Por que entra: baseline interpretável. Cada coeficiente diz quanto uma variável aumenta ou reduz a chance de atraso.
Pré-processamento: one-hot nas categorias (raras agrupadas), log com sinal + padronização nos números,
imputação pela mediana com indicador de nulo (vendedor sem histórico).
Uso:  python m1_regressao_logistica.py compra|postagem
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from comum import log_com_sinal, modelo_base, momento_da_linha_de_comando, treinar_e_avaliar


def criar_pipeline(categoricas, numericas):
    numerico = Pipeline([
        ('imputa', SimpleImputer(strategy='median', add_indicator=True)),
        ('log', FunctionTransformer(log_com_sinal, feature_names_out='one-to-one')),
        ('padroniza', StandardScaler()),
    ])
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=50), categoricas),
        ('num', numerico, numericas),
    ])
    return Pipeline([('prep', prep), ('modelo', LogisticRegression(max_iter=3000))])


espaco = {
    'modelo__C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],   # força da regularização
    'modelo__class_weight': [None, 'balanced'],
}

if __name__ == '__main__':
    modelo, met = treinar_e_avaliar('m1_regressao_logistica', criar_pipeline, espaco, interpretabilidade=1.0,
                                    momento=momento_da_linha_de_comando())
    pipe = modelo_base(modelo)   # sem a camada de calibração
    nomes = pipe.named_steps['prep'].get_feature_names_out()
    coefs = pipe.named_steps['modelo'].coef_[0]
    print('  Maiores efeitos (razão de chances por desvio-padrão):')
    for n, c in sorted(zip(nomes, coefs), key=lambda t: -abs(t[1]))[:10]:
        print(f'    {n:45s} {c:+.3f}  (x{np.exp(c):.2f})')
