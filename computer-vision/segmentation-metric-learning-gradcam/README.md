# Ball Object Detection with RetinaNet and DETR

Проект по детекции мячей на изображениях с использованием RetinaNet-style detector и fine-tuning DETR.

В проекте реализованы ключевые компоненты object detection pipeline: генерация anchors, feature pyramid, classification/regression heads, focal loss, fine-tuning модели и визуальная оценка предсказаний.

> **Стек:** `Python` · `PyTorch` · `torchvision` · `NumPy` · `pandas` · `OpenCV` · `Matplotlib` · `PIL` · `Jupyter Notebook`

## Описание проекта

В проекте рассматривается задача локализации объекта класса **Ball** на изображениях.

Реализованы и протестированы два подхода:

- RetinaNet-style object detector;
- DETR-based object detector.

Основной пайплайн включает:

- подготовку изображений и bounding boxes;
- генерацию anchors;
- построение Feature Pyramid Network;
- реализацию classification и box regression heads;
- использование focal loss для борьбы с дисбалансом классов;
- fine-tuning модели на целевом датасете;
- оценку качества детекции;
- визуализацию bounding boxes и анализ ошибок.

## Данные

Используется датасет изображений с разметкой объектов класса **Ball**.

Каждый объект размечен bounding box координатами.  
Задача модели — определить наличие мяча на изображении и корректно локализовать его с помощью прямоугольной рамки.

## Подход

### RetinaNet-style detector

RetinaNet используется как основной detection-подход.

В рамках проекта реализуются и применяются ключевые элементы RetinaNet:

- backbone для извлечения признаков;
- Feature Pyramid Network для работы с объектами разного масштаба;
- anchors разных размеров и пропорций;
- classification head для предсказания класса;
- regression head для уточнения координат bounding boxes;
- focal loss для снижения влияния easy negative examples.

### DETR fine-tuning

Дополнительно проводится эксперимент с **DETR** — transformer-based моделью для object detection.

DETR рассматривает детекцию как задачу прямого предсказания множества объектов и не требует ручной генерации anchors.  
В проекте модель дообучается на задаче детекции мячей и сравнивается с RetinaNet-style подходом по визуальному качеству предсказаний.

## Оценка качества

Качество object detection оценивается по корректности локализации объекта и визуальному анализу предсказанных bounding boxes.

Основной фокус сделан на:

- совпадении предсказанной рамки с истинной разметкой;
- наличии false positives;
- пропущенных объектах;
- ошибках на визуально похожих круглых объектах, бликах и сложном фоне.

## Результаты

RetinaNet-style detector показал устойчивое качество на задаче локализации мячей и корректно находил объект на большинстве визуальных примеров.

DETR также смог адаптироваться к задаче после fine-tuning, однако на отдельных изображениях модель была чувствительна к визуально похожим круглым объектам и бликам.

Основные наблюдения:

- RetinaNet-style подход хорошо подходит для задачи с небольшим числом классов и явно размеченными bounding boxes;
- focal loss помогает уменьшить влияние дисбаланса между объектом и фоном;
- Feature Pyramid Network улучшает работу с объектами разного масштаба;
- DETR удобен как anchor-free альтернатива, но требует аккуратного fine-tuning и достаточного числа обучающих примеров;
- визуальный анализ предсказаний важен для понимания ошибок object detection модели.

## Финальный вывод

В проекте реализован полный object detection pipeline для задачи детекции мячей.

RetinaNet-style detector оказался более интерпретируемым и контролируемым за счёт явной работы с anchors, heads и loss-функцией. DETR показал себя как современная anchor-free альтернатива, но оказался более чувствительным к качеству fine-tuning и визуально похожим объектам.

Проект демонстрирует ключевые навыки в computer vision: подготовку detection-датасета, работу с bounding boxes, реализацию компонентов RetinaNet, fine-tuning object detection моделей и анализ ошибок на изображениях.

## References

- [Focal Loss for Dense Object Detection / RetinaNet](https://arxiv.org/pdf/1708.02002.pdf)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)
- [RetinaNet explained and demystified](https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [torchvision transforms](https://pytorch.org/vision/stable/transforms.html)
- [torchvision detection models](https://pytorch.org/vision/stable/models.html)
- [Object Detection on COCO — Papers with Code](https://paperswithcode.com/sota/object-detection-on-coco)
- [Hugging Face Computer Vision Course: DETR](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/detr)
