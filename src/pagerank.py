"""
Модуль ранжирования вершин графа: взвешенный PageRank.

Формула TextRank (Mihalcea & Tarau, 2004):

    WS(Vi) = (1 - d) + d * Σ_j [ w_ji / Σ_k w_jk ] * WS(Vj)

где d — коэффициент затухания (обычно 0.85),
w_ji — вес ребра между предложениями j и i.

Смысл: предложение важное, если на него «голосуют» другие важные
предложения, причём голос тем сильнее, чем больше сходство.

Мы решаем это итеративно (степенной метод): начинаем с одинаковых
оценок и пересчитываем их, пока они не перестанут меняться.

Автор модуля: Участница 2.
"""

import numpy as np


def pagerank(W, damping=0.85, tol=1e-6, max_iter=200):
    """Взвешенный PageRank.

    W        — матрица сходства n x n (симметричная, с нулями на диагонали);
    damping  — коэффициент затухания d;
    tol      — точность: останавливаемся, когда оценки меняются меньше tol;
    max_iter — защита от бесконечного цикла.

    Возвращает (scores, history):
      scores  — массив оценок важности (нормированный, сумма = 1);
      history — список изменений на каждой итерации (для графика сходимости).
    """
    n = W.shape[0]
    if n == 0:
        return np.array([]), []

    # Нормируем каждую строку: сколько «голоса» вершина j отдаёт вершине i.
    out_weight = W.sum(axis=1)
    out_weight[out_weight == 0] = 1.0           # у изолированной вершины нет голосов
    M = (W / out_weight[:, None]).T             # M[i, j] = w_ji / Σ_k w_jk

    scores = np.ones(n)                         # стартуем с одинаковых оценок
    history = []
    for _ in range(max_iter):
        new_scores = (1 - damping) + damping * M @ scores
        delta = float(np.abs(new_scores - scores).sum())
        history.append(delta)
        scores = new_scores
        if delta < tol:
            break

    return scores / scores.sum(), history


def degree_centrality(W):
    """Упрощённая альтернатива: важность = сумма весов рёбер вершины.
    Используется в экспериментах для сравнения с PageRank."""
    s = W.sum(axis=1)
    return s / s.sum() if s.sum() > 0 else np.ones(len(s)) / len(s)
