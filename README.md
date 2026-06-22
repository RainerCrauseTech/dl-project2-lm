# Проект 2: Language Model - автодополнение текстов

Сравнение моделей языкового моделирования на датасете WikiText-2: LSTM, GRU и предобученный distilgpt2.

Основной файл решения: `project2_notebook.ipynb`

---

## Задача

Построить language model для предсказания следующего слова, обучить RNN с нуля, сравнить LSTM и GRU, оценить предобученный трансформер и сформировать выводы для продукта.

---

## Модели

| Модель     | Архитектура                         | Обучение             |
|------------|-------------------------------------|----------------------|
| LSTM       | Embedding -> LSTM -> Linear         | С нуля на WikiText-2 |
| GRU        | Embedding -> GRU -> Linear          | С нуля на WikiText-2 |
| distilgpt2 | Предобученный трансформер (HF)      | Без дообучения       |

---

## Результаты (test)

| Модель     | Параметры  | Val PPL | Test PPL | Время/эпоха   |
|------------|------------|---------|----------|---------------|
| LSTM       | 15,903,105 | 781.12  | 820.52   | ~1578 с (GPU) |
| GRU        | 15,672,705 | 1016.35 | 998.40   | ~1551 с (GPU) |
| distilgpt2 | 81,912,576 | n/a     | 79.77    | n/a           |

Perplexity LSTM/GRU и distilgpt2 напрямую не сравниваются (разные токенизаторы и словари).

Вывод: лучшее качество у distilgpt2; среди RNN лучше LSTM по perplexity, GRU чуть компактнее и быстрее.

---

## Структура репозитория

```
Project_2/
├── README.md
├── requirements_project2.txt
├── project2_notebook.ipynb
└── checkpoints/              # опционально
    ├── lstm_vm_best.pt
    └── gru_vm_best.pt
```

Чекпоинты (~60 МБ каждый). Если файлы лежат в `checkpoints/`, ноутбук пропускает обучение и загружает веса.

---

## Требования

- Python 3.10+
- PyTorch, datasets, transformers, matplotlib, numpy, tqdm

Установка:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_project2.txt
```

Для GPU (CUDA) установите PyTorch с https://pytorch.org под вашу версию CUDA.

---

## Запуск

1. Откройте `project2_notebook.ipynb` в Jupyter, VS Code.
2. Выберите локальное ядро Python (`.venv`).
3. Запускайте ноутбук из папки `Project_2/`, чтобы путь `checkpoints/` был виден.
4. Run All сверху вниз.

### Если чекпоинты уже есть

- Ячейки обучения LSTM/GRU выведут: `Загружен чекпоинт ... обучение пропущено`
- Ячейки с `evaluate_lm` на test все равно нужно выполнить (на CPU около 10-20 минут на модель)

### Если чекпоинтов нет

- Обучение LSTM/GRU на CPU займет много часов, лучше GPU (T4 и выше)
- WikiText-2 скачивается через `datasets` при первом запуске

---

## Данные

- Датасет: `Salesforce/wikitext`, конфиг `wikitext-2-raw-v1`
- Токенизация: пословная, словарь только по train, `min_freq=2`
- Длина окна: `SEQ_LEN=32`, батч: `64`
- `seed=42` для воспроизводимости

---

## Этапы в ноутбуке

0. Подготовка (imports, seed, устройство)
1. EDA и подготовка данных (vocab, Dataset, DataLoader)
2. LSTM: обучение, графики, test PPL
3. GRU vs LSTM: сравнение и выводы
4. distilgpt2: test PPL и примеры
5. Сводная таблица, сравнение предсказаний, рекомендации для PM

---