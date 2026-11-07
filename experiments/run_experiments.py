"""
Эксперименты проекта. Запуск из корня репозитория:

    python -m experiments.run_experiments

Что делаем:
  Эксперимент 1. Сравнение методов: Random, Lead-3, частотный (Луна),
                 TextRank-overlap, TextRank-cosine, Degree (без PageRank).
  Эксперимент 2. Влияние коэффициента затухания d.
  Эксперимент 3. Влияние порога отсечения слабых рёбер.
  Эксперимент 4. Сходимость PageRank (сколько итераций нужно).
  Эксперимент 5. Визуализация графа одного текста.

Все таблицы сохраняются в results/*.csv, графики — в results/*.png.

Автор: обе участницы (эксп. 1, 4 — Участница 1; эксп. 2, 3, 5 — Участница 2).
"""

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # рисуем в файл, без окна
import matplotlib.pyplot as plt

from src.summarizer import TextRankSummarizer
from src.baselines import lead_summary, random_summary, frequency_summary
from src.evaluation import rouge_all
from src.graph import graph_stats
from src.preprocessing import preprocess

ROOT = Path(__file__).resolve().parent.parent
TEXTS = ROOT / "data" / "texts"
REFS = ROOT / "data" / "references"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

N = 3  # длина реферата в предложениях (эталоны тоже из 3 предложений)
METRICS = ["rouge-1", "rouge-2", "rouge-l"]


def load_corpus():
    corpus = []
    for path in sorted(TEXTS.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        ref = (REFS / path.name).read_text(encoding="utf-8")
        corpus.append((path.stem, text, ref))
    return corpus


def evaluate(summarize_fn, corpus):
    """Средние ROUGE по корпусу + результаты по каждому тексту."""
    per_text = {}
    for name, text, ref in corpus:
        per_text[name] = rouge_all(summarize_fn(text), ref)
    mean = {m: float(np.mean([r[m] for r in per_text.values()])) for m in METRICS}
    return mean, per_text


def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # -sig: чтобы Excel открыл кириллицу
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------

def experiment_methods(corpus):
    print("\n=== Эксперимент 1: сравнение методов ===")

    methods = {
        "Random": None,  # считаем отдельно: усредняем по 20 запускам, иначе это лотерея
        "Lead-3": lambda t: lead_summary(t, N),
        "Частотный (Луна)": lambda t: frequency_summary(t, N),
        "Degree (без PageRank)": lambda t: TextRankSummarizer("cosine", ranking="degree").summarize(t, N),
        "TextRank-overlap": lambda t: TextRankSummarizer("overlap").summarize(t, N),
        "TextRank-cosine": lambda t: TextRankSummarizer("cosine").summarize(t, N),
    }

    rows, per_text_table = [], {}
    for name, fn in methods.items():
        if name == "Random":
            runs = [evaluate(lambda t, s=s: random_summary(t, N, seed=s), corpus) for s in range(20)]
            mean = {m: float(np.mean([r[0][m] for r in runs])) for m in METRICS}
            per_text = {k: {m: float(np.mean([r[1][k][m] for r in runs])) for m in METRICS}
                        for k in runs[0][1]}
        else:
            mean, per_text = evaluate(fn, corpus)
        per_text_table[name] = per_text
        rows.append([name] + [f"{mean[m]:.3f}" for m in METRICS])
        print(f"{name:<24}" + "  ".join(f"{m}={mean[m]:.3f}" for m in METRICS))

    save_csv(OUT / "exp1_methods.csv", ["Метод"] + METRICS, rows)

    # по текстам — ROUGE-1 для каждого метода
    names = [c[0] for c in corpus]
    save_csv(OUT / "exp1_per_text_rouge1.csv", ["Текст"] + list(methods),
             [[t] + [f"{per_text_table[m][t]['rouge-1']:.3f}" for m in methods] for t in names])

    # график
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    colors = ["#2a6f97", "#c2557a", "#7c9c5b"]
    for k, m in enumerate(METRICS):
        vals = [float(r[k + 1]) for r in rows]
        ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=m.upper(), color=colors[k])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=15)
    ax.set_ylabel("F1")
    ax.set_title("Сравнение методов реферирования (среднее по 6 текстам)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp1_methods.png", dpi=150)
    plt.close(fig)
    return rows


def experiment_convergence(corpus):
    print("\n=== Эксперимент 4: сходимость PageRank ===")
    name, text, _ = corpus[0]
    m = TextRankSummarizer("cosine")
    m.summarize(text, N)
    hist = m.history
    print(f"Текст {name}: сошлось за {len(hist)} итераций, последнее изменение {hist[-1]:.2e}")
    save_csv(OUT / "exp4_convergence.csv", ["Итерация", "Изменение оценок"],
             [[i + 1, f"{h:.2e}"] for i, h in enumerate(hist)])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(range(1, len(hist) + 1), hist, marker="o", color="#2a6f97")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Суммарное изменение оценок (лог. шкала)")
    ax.set_title("Сходимость PageRank на тексте «Байкал»")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp4_convergence.png", dpi=150)
    plt.close(fig)
    return hist


def corpus_stats(corpus):
    rows = []
    for name, text, ref in corpus:
        s, t = preprocess(text)
        rows.append([name, len(s), sum(len(x.split()) for x in s), len(ref.split())])
    save_csv(OUT / "corpus_stats.csv", ["Текст", "Предложений", "Слов", "Слов в эталоне"], rows)
    print("Корпус:", rows)


if __name__ == "__main__":
    corpus = load_corpus()
    corpus_stats(corpus)
    experiment_methods(corpus)
    experiment_convergence(corpus)
    print("\nГотово. Результаты в папке results/")
