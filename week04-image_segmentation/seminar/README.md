# Segmentation seminar

Семинар по семантической и instance сегментации на COCO датасете (класс person).

## Подготовка данных

Скачивание и подготовка COCO val2017 датасета:

```bash
python tools/download_coco_mini.py
python tools/prepare_coco_mini.py
```

## Модели

### UNet

Baseline модель для семантической сегментации.

**Обучение:** [notebooks/train_unet_coco.ipynb](notebooks/train_unet_coco.ipynb)

### Mask2Former

Query-based модель для semantic и instance сегментации.

**Архитектура (пошаговое построение):** [notebooks/unet_to_mask2former_proper.ipynb](notebooks/unet_to_mask2former_proper.ipynb)

**Обучение:**
- Semantic сегментация: [notebooks/train_mask2former_semantic.ipynb](notebooks/train_mask2former_semantic.ipynb)
- Instance сегментация: [notebooks/train_mask2former_instance.ipynb](notebooks/train_mask2former_instance.ipynb)

## Структура проекта

```
seg/
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
