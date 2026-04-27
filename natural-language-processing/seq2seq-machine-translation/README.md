# Seq2Seq машинный перевод с Attention

Проект по нейронному машинному переводу описаний отелей и хостелов с русского языка на английский.

Реализованы GRU Encoder-Decoder модель, additive attention, bidirectional encoder и несколько стратегий декодирования. Качество перевода оценивается с помощью BLEU.

> **Стек:** `Python` · `PyTorch` · `NumPy` · `scikit-learn` · `NLTK` · `subword-nmt` · `Matplotlib` · `Bokeh` · `Jupyter Notebook`

## Описание проекта

В проекте построен NLP-пайплайн для машинного перевода:

- подготовка русско-английского параллельного корпуса;
- токенизация текста;
- Byte Pair Encoding для subword-токенизации;
- построение словарей;
- обучение GRU Encoder-Decoder модели;
- добавление additive attention;
- сравнение greedy decoding, top-p sampling и beam search;
- оценка качества перевода с помощью BLEU;
- визуализация attention-карт.

## Данные

Используется русско-английский параллельный корпус описаний отелей и хостелов.

После предобработки:

- train set: **47 000** пар предложений;
- validation set: **3 000** пар предложений;
- source vocabulary: **8 037** токенов;
- target vocabulary: **7 809** токенов.

Предобработка включает lowercasing, токенизацию, train/validation split, обучение BPE на train-части и преобразование предложений в последовательности токенов.

## Подход

Базовая модель построена на архитектуре **Encoder-Decoder**:

- **Encoder** кодирует русское предложение в скрытое представление;
- **Decoder** генерирует английский перевод токен за токеном.

Для улучшения качества перевода используется **additive attention**. На каждом шаге decoder вычисляет веса внимания по состояниям encoder и формирует контекстный вектор:

```math
a_t = linear_{out}(tanh(linear_e(h^e_t) + linear_d(h^d)))
```

```math
p_t = \frac{e^{a_t}}{\sum_\tau e^{a_\tau}}
```

```math
attn = \sum_t p_t \cdot h^e_t
```

Attention позволяет decoder использовать не только финальное состояние encoder, но и всю последовательность encoder states. Это особенно важно для более длинных предложений.

## Оценка качества

Качество перевода оценивается с помощью **BLEU** — метрики, основанной на совпадении n-грамм между предсказанным и эталонным переводом.

BLEU удобна для сравнения моделей машинного перевода, хотя не всегда полностью отражает смысловое качество перевода.

## Эксперименты

| Эксперимент | Основные параметры | BLEU | Краткий вывод |
|---|---|---:|---|
| Baseline attention | `emb_size=64`, `hid_size=128`, `attn_size=128`, `steps=800`, `lr=1e-3`, `max_len=50` | 10.79 | Базовая attention-модель обучается выравниванию, но качество ограничено небольшой ёмкостью модели и коротким обучением. |
| Larger attention | `emb_size=96`, `hid_size=192`, `attn_size=192`, `steps=1200`, `lr=1e-3`, `max_len=50` | 18.89 | Увеличение размерности эмбеддингов и скрытых состояний заметно улучшило качество перевода. |
| Bidirectional attention | `emb_size=128`, `hid_size=192`, `attn_size=192`, `steps=2500`, `lr=5e-4`, `max_len=50` | 23.12 | Двунаправленный encoder дал самый заметный архитектурный прирост за счёт контекста с обеих сторон входной фразы. |
| Bidirectional + top-p sampling | `emb_size=128`, `hid_size=192`, `attn_size=256`, `steps=3000`, `lr=4e-4`, `max_len=60`, `p=0.85`, `temperature=0.6` | 22.12 | Top-p sampling сделал генерацию более разнообразной, но снизил стабильность результата по BLEU. |
| Bidirectional + beam search | `emb_size=64`, `hid_size=128`, `attn_size=128`, `steps=10000`, `lr=1e-3`, `max_len=60`, `beam_size=4`, `alpha=0.7` | 28.07 | Beam search показал лучший BLEU, но результат также связан с более длинным обучением; напрямую сравнивать его с остальными экспериментами нужно осторожно. |

## Результаты

Лучший результат был получен в эксперименте с **bidirectional attention model + beam search decoding** — BLEU **28.07**.

Основные наблюдения:

- базовая attention-модель достигла BLEU **10.79** и показала способность обучаться выравниванию между русскими и английскими токенами;
- увеличение размера модели улучшило BLEU до **18.89**, что показывает важность ёмкости модели для машинного перевода;
- bidirectional encoder дал сильный прирост до **23.12 BLEU**, так как каждое состояние encoder содержит контекст с обеих сторон входной последовательности;
- top-p sampling не улучшил BLEU относительно обычной bidirectional attention-модели: итоговый результат составил **22.12**;
- beam search достиг BLEU **28.07**, однако этот эксперимент обучался дольше, поэтому прирост нельзя объяснять только стратегией декодирования;
- attention maps показывают, что модели учатся сопоставлять фрагменты русской фразы с английскими токенами перевода.

## Анализ attention maps

Визуализация attention показывает, что модели действительно учатся сопоставлять части русской фразы с английскими токенами перевода.

Для коротких предложений attention maps выглядят более чёткими: видны локальные соответствия между словами и subword-фрагментами. Для длинных предложений карты внимания становятся более разреженными и шумными, что частично объясняет ошибки перевода на длинных фразах.

Наиболее интерпретируемые attention maps получались у bidirectional attention моделей. Это согласуется с BLEU: более информативные encoder states помогают attention точнее выбирать релевантные части входной последовательности.

## Финальный вывод

В рамках серии экспериментов лучшей стала **bidirectional attention model with beam search decoding**. Она достигла наибольшего BLEU — **28.07**.

На качество перевода сильнее всего повлияли три фактора:

- использование attention-механизма;
- двунаправленное кодирование входной последовательности;
- более сильная стратегия декодирования по сравнению с greedy decoding.

При этом результат beam search следует интерпретировать осторожно: этот эксперимент обучался дольше остальных, поэтому улучшение связано не только с декодированием, но и с большим числом training steps.

## Материалы

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- [Re-evaluating the Role of BLEU in Machine Translation Research](https://www.cis.upenn.edu/~ccb/publications/re-evaluating-the-role-of-bleu-in-mt-research.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
