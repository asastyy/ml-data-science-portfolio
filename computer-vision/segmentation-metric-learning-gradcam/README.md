# Semantic Segmentation, Metric Learning и Grad-CAM

Проект объединяет три задачи компьютерного зрения: семантическую сегментацию, metric learning для поиска похожих изображений и интерпретацию модели с помощью Grad-CAM.

Реализованы fine-tuning U-Net для сегментации, обучение embedding-модели для image retrieval, оценка качества поиска по retrieval-метрикам и визуальный анализ ошибок модели.

> **Стек:** `Python` · `PyTorch` · `PyTorch Lightning` · `segmentation-models-pytorch` · `torchvision` · `FAISS` · `pandas` · `NumPy` · `Matplotlib` · `Grad-CAM` · `Jupyter Notebook`

## Описание проекта

В проекте реализованы несколько CV-пайплайнов:

- подготовка данных для семантической сегментации;
- fine-tuning предобученной U-Net модели;
- сравнение BCE loss, Dice loss и расширенных аугментаций;
- оценка сегментации с помощью mean IoU;
- обучение embedding-модели для metric learning;
- построение поиска похожих изображений через FAISS;
- оценка retrieval-качества с помощью Precision@k, Recall@k и mAP@k;
- визуализация областей внимания модели с помощью Grad-CAM.

## Данные

Для сегментации используется **Pascal VOC** — датасет с изображениями объектов и пиксельными масками классов.

Для metric learning используется **CARS196** — датасет изображений автомобилей, часто применяемый для задач image retrieval и metric learning.

## Semantic Segmentation

Для сегментации используется U-Net-based модель.  
Основная задача — выделить объекты на изображении и оценить качество предсказанных масок.

Сравнивались несколько вариантов обучения:

| Эксперимент | mean IoU | Краткий вывод |
|---|---:|---|
| Pretrained U-Net до fine-tuning | 0.11 | Модель почти не выделяла нужные области без дообучения на целевом датасете. |
| BCE + default augmentations | 0.74 | Fine-tuning резко улучшил качество сегментации. |
| BCE + extended augmentations | 0.75 | Дополнительные аугментации дали небольшой, но стабильный прирост. |
| BCE + Dice + extended augmentations | 0.76 | Лучший результат за счёт сочетания BCE, Dice loss и расширенных аугментаций. |

## Metric Learning

Для image retrieval обучается модель, которая переводит изображения в embedding-пространство.  
Идея состоит в том, чтобы изображения одного класса находились ближе друг к другу, а изображения разных классов — дальше.

В качестве backbone используется ResNet-based модель. Для обучения применяются metric learning loss-функции, включая ArcFace / Center Loss подходы.

Для поиска похожих изображений используется **FAISS**.

## Retrieval Evaluation

Качество поиска похожих изображений оценивается с помощью Precision@k, Recall@k и mAP@k.

| k | Precision@k | Recall@k | mAP@k |
|---:|---:|---:|---:|
| 1 | 0.7686 | 0.0238 | 0.7686 |
| 5 | 0.7260 | 0.1121 | 0.7065 |
| 10 | 0.6885 | 0.2124 | 0.6645 |

Модель показывает высокую точность на ближайших соседях: Precision@1 достигает **0.7686**.  
При увеличении k recall растёт, так как среди большего числа найденных изображений появляется больше релевантных объектов.

## Grad-CAM

Grad-CAM используется для визуальной интерпретации работы модели.

С помощью heatmap-визуализаций анализируется, на какие области изображения модель обращает внимание при принятии решения. Это помогает находить ошибки, связанные с фоном, нерелевантными деталями или недостаточной локализацией объекта.

## Результаты

Основные результаты проекта:

- fine-tuning U-Net повысил mean IoU с **0.11** до **0.74**;
- лучшая segmentation-конфигурация достигла mean IoU **0.76**;
- расширенные аугментации и Dice loss дали дополнительный прирост качества;
- metric learning модель достигла **Precision@1 = 0.7686**;
- FAISS позволил реализовать быстрый поиск похожих изображений в embedding-пространстве;
- Grad-CAM помог визуально проанализировать, какие области изображения влияют на предсказания модели.

## Финальный вывод

В проекте показаны три важных направления компьютерного зрения: сегментация, поиск похожих изображений и интерпретация моделей.

Лучший результат в задаче сегментации был получен при использовании BCE + Dice loss и расширенных аугментаций — mean IoU **0.76**. В задаче metric learning модель показала хорошее качество ближайшего поиска: Precision@1 составил **0.7686**.

Проект демонстрирует полный CV-пайплайн: от подготовки данных и обучения моделей до оценки качества, retrieval-поиска и визуального анализа ошибок.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [CARS196 Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [A Discriminative Feature Learning Approach for Deep Face Recognition / Center Loss](https://arxiv.org/abs/1707.07391)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
