"""Formatação de textos exibidos na tela (não altera os valores usados nos filtros)."""
from . import config as C


def nome_categoria(cat) -> str:
    """agro_industria_e_comercio -> Agro Industria e Comercio"""
    if cat is None or str(cat).strip() == "":
        return "—"
    palavras = str(cat).replace("_", " ").split()
    saida = []
    for i, p in enumerate(palavras):
        p = p.lower()
        p = p if (i > 0 and p in C.PALAVRAS_MINUSCULAS) else p.capitalize()
        saida.append(C.CORRECOES_NOME.get(p, p))
    return " ".join(saida)
