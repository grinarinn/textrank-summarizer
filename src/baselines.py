"""
Базовые (простые) методы реферирования — с ними сравниваем TextRank.
Если TextRank не лучше простых методов, значит, граф не помогает.

  * lead      — первые N предложений (так часто устроены новости);
  * random    — N случайных предложений (нижняя граница качества);
  * frequency — частотный метод Луна (1958): важны предложения
                с часто встречающимися в тексте словами.

Автор модуля: Участница 1.
"""

import random
from collections import Counter

from .preprocessing import preprocess


def lead_summary(text, n_sentences=3):
    sentences, _ = preprocess(text)
    return " ".join(sentences[:n_sentences])


def random_summary(text, n_sentences=3, seed=0):
    sentences, _ = preprocess(text)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(sentences)), min(n_sentences, len(sentences))))
    return " ".join(sentences[i] for i in idx)


def frequency_summary(text, n_sentences=3):
    """Оценка предложения = средняя частота его слов во всём тексте."""
    sentences, tokens = preprocess(text)
    freq = Counter(w for sent in tokens for w in sent)
    top = max(freq.values()) if freq else 1

    def score(sent):
        return sum(freq[w] / top for w in sent) / len(sent) if sent else 0.0

    best = sorted(range(len(sentences)), key=lambda i: score(tokens[i]),
                  reverse=True)[:n_sentences]
    return " ".join(sentences[i] for i in sorted(best))
