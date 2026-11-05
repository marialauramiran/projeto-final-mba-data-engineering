"""
m2_random_forest.py — Random Forest (bagging de árvores)

Por que entra: robusto, pouco sensível a hiperparâmetros, captura não linearidades e interações.
Pré-processamento: categorias como inteiros (árvores não precisam de one-hot), nulos imputados.
Uso:  python m2_random_forest.py compra|postagem
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from comum import SEED, momento_da_linha_de_comando, treinar_e_avaliar


def criar_pipeline(categoricas, numericas):
    prep = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1), categoricas),
        ('num', SimpleImputer(strategy='median', add_indicator=True), numericas),
    ])
    return Pipeline([('prep', prep), ('modelo', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED))])


espaco = {
    'modelo__max_depth': [6, 8, 10, 14, None],
    'modelo__min_samples_leaf': [5, 20, 50, 100, 200],
    'modelo__max_features': ['sqrt', 0.3, 0.5],
    'modelo__class_weight': [None, 'balanced_subsample'],
}

if __name__ == '__main__':
    treinar_e_avaliar('m2_random_forest', criar_pipeline, espaco, interpretabilidade=0.5,
                      momento=momento_da_linha_de_comando())
