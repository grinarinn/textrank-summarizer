# Резюмирование текста на основе графов (TextRank)

Учебный проект: автоматическое экстрактивное реферирование русских текстов.
Текст представляется как граф предложений, важность предложений
вычисляется алгоритмом PageRank, в реферат попадают самые важные предложения.

## Как это работает

```
текст → предложения → слова (без стоп-слов, основы) → граф сходства → PageRank → топ-3 предложения → реферат
```

## Структура

| Папка / файл | Что внутри | Кто отвечает |
|---|---|---|
| `src/preprocessing.py` | деление на предложения, токенизация, стоп-слова, стеммер | Арина |
| `src/graph.py` | меры сходства (overlap, TF-IDF + косинус), матрица графа | Настя |
| `src/pagerank.py` | взвешенный PageRank (степенной метод) | Настя |
| `src/summarizer.py` | класс `TextRankSummarizer` — весь конвейер | Арина |
| `src/baselines.py` | простые методы для сравнения: Random, Lead-3, частотный | Арина |
| `src/evaluation.py` | метрики ROUGE-1, ROUGE-2, ROUGE-L | Настя |
| `data/texts/` | 6 научно-популярных текстов | обе |
| `data/references/` | эталонные рефераты (по 3 предложения, написаны вручную) | обе |
| `experiments/run_experiments.py` | 5 экспериментов, таблицы и графики | обе |
| `tests/test_project.py` | автоматические тесты | обе |
| `main.py` | запуск из командной строки | Арина |

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
# реферат одного текста
python main.py data/texts/01_baikal.txt
python main.py data/texts/01_baikal.txt -n 2 --similarity overlap --show-ranking

# все эксперименты (результаты появятся в results/)
python -m experiments.run_experiments

# тесты
python -m unittest discover tests -v
```

## Результаты (среднее по 6 текстам, F1)

| Метод | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Random | 0.320 | 0.134 | 0.238 |
| Lead-3 | 0.366 | 0.207 | 0.301 |
| Частотный (Луна) | 0.405 | 0.222 | 0.333 |
| Degree (без PageRank) | 0.428 | 0.238 | 0.343 |
| TextRank-overlap | 0.437 | 0.248 | 0.353 |
| **TextRank-cosine** | **0.444** | **0.263** | **0.368** |

## Литература

1. Mihalcea R., Tarau P. TextRank: Bringing Order into Texts // EMNLP 2004.
2. Brin S., Page L. The Anatomy of a Large-Scale Hypertextual Web Search Engine, 1998.
3. Erkan G., Radev D. LexRank: Graph-based Lexical Centrality as Salience in Text Summarization, 2004.
4. Lin C.-Y. ROUGE: A Package for Automatic Evaluation of Summaries, 2004.
5. Luhn H. P. The Automatic Creation of Literature Abstracts, 1958.
