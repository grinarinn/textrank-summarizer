"""
Модуль построения графа предложений.

Граф: вершины = предложения, ребро между двумя предложениями тем «толще»
(больше вес), чем сильнее они похожи по словам.

Реализованы две меры сходства:
  * overlap — как в оригинальной статье TextRank (Mihalcea & Tarau, 2004);
  * cosine  — косинусное сходство векторов TF-IDF.

Результат — матрица сходства W размером n x n, где W[i][j] — вес ребра.

Автор модуля: Участница 2.
"""

import math
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Мера 1. Пересечение слов (оригинальный TextRank)
# ---------------------------------------------------------------------------

def overlap_similarity(words_a, words_b):
    """sim(Si, Sj) = |общие слова| / (log|Si| + log|Sj|)

    Логарифмы в знаменателе нужны, чтобы длинные предложения
    не выигрывали только за счёт своей длины.
    """
    if len(words_a) < 2 or len(words_b) < 2:  # log(1) = 0 -> деление на ноль
        return 0.0
    common = len(set(words_a) & set(words_b))
    return common / (math.log(len(words_a)) + math.log(len(words_b)))


# ---------------------------------------------------------------------------
# Мера 2. Косинусное сходство TF-IDF
# ---------------------------------------------------------------------------

def tfidf_vectors(tokens):
    """Строит матрицу TF-IDF: строка = предложение, столбец = слово словаря.

    TF  (term frequency)  — как часто слово встречается в предложении;
    IDF (inverse document frequency) — насколько слово редкое в тексте.
    Слово, которое есть почти в каждом предложении, получает маленький вес.
    """
    vocab = sorted({w for sent in tokens for w in sent})
    index = {w: i for i, w in enumerate(vocab)}
    n = len(tokens)

    # в скольких предложениях встречается каждое слово
    df = Counter(w for sent in tokens for w in set(sent))
    idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in vocab])

    matrix = np.zeros((n, len(vocab)))
    for i, sent in enumerate(tokens):
        counts = Counter(sent)
        for w, c in counts.items():
            matrix[i, index[w]] = c / len(sent)  # TF
    return matrix * idf  # TF * IDF


def cosine_similarity_matrix(vectors):
    """cos(a, b) = (a · b) / (|a| * |b|) для всех пар строк сразу."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # пустые предложения не дают деления на ноль
    unit = vectors / norms
    return unit @ unit.T


# ---------------------------------------------------------------------------
# Сборка графа
# ---------------------------------------------------------------------------

def build_similarity_matrix(tokens, method="cosine", threshold=0.0):
    """Возвращает матрицу смежности взвешенного неориентированного графа.

    tokens    — список предложений, каждое — список основ слов;
    method    — 'cosine' или 'overlap';
    threshold — рёбра с весом ниже порога удаляются (граф становится реже).
    """
    n = len(tokens)
    if method == "cosine":
        W = cosine_similarity_matrix(tfidf_vectors(tokens))
    elif method == "overlap":
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                W[i, j] = W[j, i] = overlap_similarity(tokens[i], tokens[j])
    else:
        raise ValueError(f"Неизвестная мера сходства: {method}")

    np.fill_diagonal(W, 0.0)  # петли (предложение похоже само на себя) не нужны
    W[W < threshold] = 0.0
    return W


def graph_stats(W):
    """Простые характеристики графа — пригодятся для отчёта."""
    n = W.shape[0]
    edges = int(np.count_nonzero(np.triu(W, 1)))
    max_edges = n * (n - 1) // 2
    return {
        "вершин": n,
        "рёбер": edges,
        "плотность": round(edges / max_edges, 3) if max_edges else 0.0,
        "изолированных вершин": int(np.sum(W.sum(axis=1) == 0)),
    }
