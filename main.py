"""
Запуск реферирования из командной строки.

Примеры:
    python main.py data/texts/01_baikal.txt
    python main.py data/texts/01_baikal.txt -n 2 --similarity overlap
    python main.py data/texts/01_baikal.txt --show-ranking
"""

import argparse

from src.summarizer import TextRankSummarizer
from src.graph import graph_stats


def main():
    parser = argparse.ArgumentParser(description="Реферирование текста методом TextRank")
    parser.add_argument("file", help="путь к .txt файлу с текстом")
    parser.add_argument("-n", "--sentences", type=int, default=3,
                        help="сколько предложений оставить (по умолчанию 3)")
    parser.add_argument("--similarity", choices=["cosine", "overlap"], default="cosine",
                        help="мера сходства предложений")
    parser.add_argument("--damping", type=float, default=0.85,
                        help="коэффициент затухания d (по умолчанию 0.85)")
    parser.add_argument("--show-ranking", action="store_true",
                        help="показать оценки всех предложений")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        text = f.read()

    model = TextRankSummarizer(similarity=args.similarity, damping=args.damping)
    summary = model.summarize(text, n_sentences=args.sentences)

    print("=" * 70)
    print("ГРАФ:", graph_stats(model.matrix))
    print("Итераций PageRank до сходимости:", len(model.history))
    print("=" * 70)
    if args.show_ranking:
        print("Ранжирование предложений:")
        for num, score, sent in model.ranking_table():
            print(f"  [{num:>2}] {score:.4f}  {sent[:80]}...")
        print("=" * 70)
    print("РЕФЕРАТ:")
    print(summary)


if __name__ == "__main__":
    main()
