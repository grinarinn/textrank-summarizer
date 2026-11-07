"""
Тесты модулей Участницы 2: граф, PageRank, метрика ROUGE.
Запуск из корня репозитория:  python -m unittest discover tests -v
"""

import unittest

import numpy as np

from src.graph import overlap_similarity, build_similarity_matrix
from src.pagerank import pagerank
from src.evaluation import rouge_n, rouge_l


class TestGraph(unittest.TestCase):
    def test_overlap_zero(self):
        self.assertEqual(overlap_similarity(["а", "б"], ["в", "г"]), 0.0)

    def test_matrix_symmetric(self):
        toks = [["кот", "спит", "дом"], ["кот", "ест", "дом"], ["солнце", "светит", "ярко"]]
        for method in ("cosine", "overlap"):
            W = build_similarity_matrix(toks, method)
            self.assertTrue(np.allclose(W, W.T))
            self.assertTrue(np.all(np.diag(W) == 0))
            self.assertGreater(W[0, 1], W[0, 2])  # похожие предложения ближе


class TestPageRank(unittest.TestCase):
    def test_star_graph(self):
        # «звезда»: вершина 0 соединена со всеми — она должна быть самой важной
        W = np.zeros((5, 5))
        for i in range(1, 5):
            W[0, i] = W[i, 0] = 1.0
        scores, _ = pagerank(W)
        self.assertEqual(int(np.argmax(scores)), 0)

    def test_sum_to_one(self):
        W = np.random.RandomState(0).rand(6, 6)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        scores, history = pagerank(W)
        self.assertAlmostEqual(scores.sum(), 1.0)
        self.assertLess(history[-1], 1e-6)  # алгоритм сошёлся


class TestRouge(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(rouge_n("кот спит", "кот спит")["f"], 1.0)
        self.assertAlmostEqual(rouge_l("кот спит", "кот спит")["f"], 1.0)

    def test_disjoint(self):
        self.assertEqual(rouge_n("кот спит", "пёс бежит")["f"], 0.0)


if __name__ == "__main__":
    unittest.main()
