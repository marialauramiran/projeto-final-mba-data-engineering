"""
m3_hist_gradient_boosting.py — HistGradientBoosting do scikit-learn (boosting)

Por que entra: trata nulos e categorias nativamente e é rápido. Cada árvore corrige os erros das anteriores.
Uso:  python m3_hist_gradient_boosting.py compra|postagem
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from comum import SEED, ParaCategoria, momento_da_linha_de_comando, treinar_e_avaliar


def criar_pipeline(categoricas, numericas):
    return Pipeline([
        ('cat', ParaCategoria(categoricas)),
        ('modelo', HistGradientBoostingClassifier(categorical_features='from_dtype', early_stopping=False,
                                                  random_state=SEED)),
    ])


espaco = {
    'modelo__learning_rate': [0.02, 0.05, 0.1],
    'modelo__max_iter': [150, 300, 500],
    'modelo__max_leaf_nodes': [15, 31, 63],
    'modelo__min_samples_leaf': [20, 50, 100, 200],
    'modelo__l2_regularization': [0.0, 1.0, 5.0],
    'modelo__class_weight': [None, 'balanced'],
}

if __name__ == '__main__':
    treinar_e_avaliar('m3_hist_gradient_boosting', criar_pipeline, espaco, interpretabilidade=0.5,
                      momento=momento_da_linha_de_comando())
