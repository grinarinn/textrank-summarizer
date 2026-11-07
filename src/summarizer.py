"""
Главный модуль: TextRank-реферирование.

Конвейер (pipeline):
    текст -> предложения и слова -> граф -> PageRank -> топ-N предложений
    -> восстанавливаем исходный порядок -> реферат

Автор модуля: Участница 1 (сборка), использует модули Участницы 2.
"""

import math

from .preprocessing import preprocess
from .graph import build_similarity_matrix
from .pagerank import pagerank, degree_centrality


class TextRankSummarizer:
    def __init__(self, similarity="cosine", damping=0.85, threshold=0.0,
                 ranking="pagerank"):
        self.similarity = similarity   # 'cosine' или 'overlap'
        self.damping = damping         # коэффициент затухания PageRank
        self.threshold = threshold     # порог отсечения слабых рёбер
        self.ranking = ranking         # 'pagerank' или 'degree'

        # заполняются после вызова summarize() — удобно для отладки и графиков
        self.sentences = []
        self.scores = []
        self.matrix = None
        self.history = []

    def summarize(self, text, n_sentences=None, ratio=0.2):
        """Возвращает реферат — строку из выбранных предложений.

        n_sentences — сколько предложений взять (если задано);
        ratio       — иначе берём такую долю от исходного текста (20%).
        """
        sentences, tokens = preprocess(text)
        self.sentences = sentences
        if not sentences:
            return ""

        if n_sentences is None:
            n_sentences = max(1, math.ceil(len(sentences) * ratio))
        n_sentences = min(n_sentences, len(sentences))

        self.matrix = build_similarity_matrix(tokens, self.similarity, self.threshold)
        if self.ranking == "degree":
            self.scores, self.history = degree_centrality(self.matrix), []
        else:
            self.scores, self.history = pagerank(self.matrix, self.damping)

        # индексы лучших предложений
        best = sorted(range(len(sentences)), key=lambda i: self.scores[i],
                      reverse=True)[:n_sentences]
        # возвращаем их в том порядке, в каком они шли в тексте — так реферат читается связно
        best.sort()
        self.selected = best
        return " ".join(sentences[i] for i in best)

    def ranking_table(self):
        """Список (номер, оценка, предложение), отсортированный по важности."""
        rows = [(i + 1, float(self.scores[i]), s) for i, s in enumerate(self.sentences)]
        return sorted(rows, key=lambda r: r[1], reverse=True)
