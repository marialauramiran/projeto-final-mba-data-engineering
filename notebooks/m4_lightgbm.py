"""
m4_lightgbm.py — LightGBM (boosting otimizado, referência de mercado para dados tabulares)

Requer:  pip install lightgbm   (no Databricks, o runtime de ML costuma já trazer)
Uso:     python m4_lightgbm.py compra|postagem
Alternativa XGBoost: ver comentário no fim do arquivo.
"""
import sys

from sklearn.pipeline import Pipeline

from comum import SEED, ParaCategoria, momento_da_linha_de_comando, treinar_e_avaliar

try:
    from lightgbm import LGBMClassifier
except ImportError:
    sys.exit('LightGBM não instalado. Rode: pip install lightgbm')


def criar_pipeline(categoricas, numericas):
    return Pipeline([
        ('cat', ParaCategoria(categoricas)),
        ('modelo', LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)),
    ])


espaco = {
    'modelo__learning_rate': [0.02, 0.05, 0.1],
    'modelo__n_estimators': [150, 300, 500],
    'modelo__num_leaves': [15, 31, 63],
    'modelo__min_child_samples': [20, 50, 100, 200],
    'modelo__reg_lambda': [0.0, 1.0, 5.0],
    'modelo__colsample_bytree': [0.6, 0.8, 1.0],
    'modelo__class_weight': [None, 'balanced'],
}

if __name__ == '__main__':
    treinar_e_avaliar('m4_lightgbm', criar_pipeline, espaco, interpretabilidade=0.5,
                      momento=momento_da_linha_de_comando())

# XGBoost (alternativa):
#   from xgboost import XGBClassifier
#   XGBClassifier(enable_categorical=True, tree_method='hist', random_state=SEED, eval_metric='aucpr')
#   espaço: learning_rate, n_estimators, max_depth [4, 6, 8], min_child_weight, reg_lambda, scale_pos_weight [1, 12]
