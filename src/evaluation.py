"""
Оценка качества рефератов метрикой ROUGE.

ROUGE (Lin, 2004) сравнивает автоматический реферат с эталонным,
написанным человеком, и считает, сколько слов/пар слов совпало.

  * ROUGE-1 — совпадение отдельных слов;
  * ROUGE-2 — совпадение пар соседних слов (биграмм);
  * ROUGE-L — длина наибольшей общей подпоследовательности (LCS).

Для каждой считаем:
  precision (точность) = совпало / слов в нашем реферате
  recall    (полнота)  = совпало / слов в эталоне
  F1 = 2·P·R / (P + R)

Слова перед сравнением приводим к основам, стоп-слова НЕ убираем
(так принято в ROUGE).

Автор модуля: Участница 2.
"""

from collections import Counter

from .preprocessing import tokenize, stem


def _words(text):
    return [stem(w) for w in tokenize(text)]


def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def _f1(overlap, cand_total, ref_total):
    p = overlap / cand_total if cand_total else 0.0
    r = overlap / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f": f}


def rouge_n(candidate, reference, n=1):
    cand, ref = _ngrams(_words(candidate), n), _ngrams(_words(reference), n)
    overlap = sum((cand & ref).values())  # & у Counter = минимум по каждому n-грамму
    return _f1(overlap, sum(cand.values()), sum(ref.values()))


def _lcs_length(a, b):
    """Длина наибольшей общей подпоследовательности (динамическое программирование)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(candidate, reference):
    cand, ref = _words(candidate), _words(reference)
    return _f1(_lcs_length(cand, ref), len(cand), len(ref))


def rouge_all(candidate, reference):
    """Возвращает словарь F1-мер: {'rouge-1': .., 'rouge-2': .., 'rouge-l': ..}."""
    return {
        "rouge-1": rouge_n(candidate, reference, 1)["f"],
        "rouge-2": rouge_n(candidate, reference, 2)["f"],
        "rouge-l": rouge_l(candidate, reference)["f"],
    }
