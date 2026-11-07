"""
Тесты модулей Участницы 1: предобработка и сборка реферата.
Запуск из корня репозитория:  python -m unittest discover tests -v
"""

import unittest

from src.preprocessing import split_sentences, tokenize, stem, preprocess
from src.summarizer import TextRankSummarizer


class TestPreprocessing(unittest.TestCase):
    def test_split_simple(self):
        self.assertEqual(len(split_sentences("Первое. Второе! Третье?")), 3)

    def test_split_abbreviations(self):
        # «г.» и «т. е.» не должны разрывать предложение
        text = "В 1998 г. появился Google, т. е. новый поисковик. Это было давно."
        self.assertEqual(len(split_sentences(text)), 2)

    def test_split_initials(self):
        self.assertEqual(len(split_sentences("А. А. Марков изучал цепи. Он был математиком.")), 2)

    def test_tokenize(self):
        self.assertEqual(tokenize("Привет, Мир! Ёж 2024."), ["привет", "мир", "еж", "2024"])

    def test_stem_same_root(self):
        self.assertEqual(stem("графы"), stem("графом"))

    def test_preprocess_lengths(self):
        sents, toks = preprocess("Кошка спит. Собака бегает.")
        self.assertEqual(len(sents), len(toks))


class TestSummarizer(unittest.TestCase):
    def test_length_and_order(self):
        text = ("Графы состоят из вершин и рёбер. Вершины графа соединяются рёбрами. "
                "Сегодня хорошая погода. Рёбра графа могут иметь вес. Я люблю чай.")
        model = TextRankSummarizer()
        summary = model.summarize(text, n_sentences=2)
        self.assertEqual(len(model.selected), 2)
        self.assertEqual(model.selected, sorted(model.selected))  # исходный порядок сохранён
        self.assertNotIn("чай", summary)  # «лишнее» предложение не попало в реферат


if __name__ == "__main__":
    unittest.main()
