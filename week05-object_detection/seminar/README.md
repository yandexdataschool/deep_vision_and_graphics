# Object Detection seminar

Семинар по object detection на COCO датасете (класс person).

## Подготовка данных

Скачивание и подготовка COCO val2017 датасета:

```bash
python tools/download_coco_mini.py
python tools/prepare_coco_mini.py
```

## Модели

### DETR

Query-based модель для object detection с transformer архитектурой.

**Архитектура (пошаговое построение):** [notebooks/resnet_fpn_to_detr.ipynb](notebooks/resnet_fpn_to_detr.ipynb)

**Обучение:** [notebooks/train_detr_person.ipynb](notebooks/train_detr_person.ipynb)

## Структура проекта

```
det/
├── configs/              # конфигурации моделей
├── data/coco/           # COCO датасет
├── notebooks/           # ноутбуки для обучения и экспериментов
├── src/
│   ├── models/         # реализации моделей
│   ├── dataset_coco.py # COCO dataset
│   ├── lightning_module.py
│   └── config.py
└── tools/              # утилиты для подготовки данных
```

