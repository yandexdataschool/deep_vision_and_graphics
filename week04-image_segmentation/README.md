# Сегментация через queries

[Ноутбук](notebooks/segmentation_transformers.ipynb) · [Пояснения и статьи](ADDITIONAL_NOTES.md)

В ноутбуке реализованы данные, модели, matching, loss, обучение и метрики. Сохранённые графики можно читать без установки зависимостей и повторного обучения.

## Содержание

1. Последняя свёртка U-Net: semantic и instance segmentation.
2. Набор масок: mask pooling, Hungarian matching, BCE, Dice и no-object.
3. Queries: динамические фильтры масок, attention и сравнение со свёрточным baseline.
4. Loss на промежуточных выходах.
5. Masked attention и обработка пустой разрешённой области.
6. Несколько разрешений и вычисление mask loss по точкам.
7. Сравнение качества, ошибок и стоимости вычислений.

В дополнении — проверки координат и симметрии queries, а также необязательный пример с COCO и ResNet. Самостоятельные PNG находятся в [figures](figures/).

## Запуск

Python 3.12. Команды выполняются из этой папки:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/start_notebook.py
```

Jupyter откроется на `127.0.0.1:8888`. Для повторения опытов выбрать **Restart Kernel and Run All Cells**. По умолчанию используется CPU; синтетические данные генерируются внутри ноутбука.

В парных сравнениях: 32 train / 32 validation изображения, 200 шагов, batch 4, три seeds. Отдельный опыт с восемью изображениями показывает переобучение за 100 шагов. Результаты сохраняются в `segmentation_outputs/` относительно рабочей папки kernel.

Выполнить весь ноутбук из терминала:

```bash
python tools/execute_notebook.py
```

COCO-пример выключен через `RUN_COCO = False`. Для него нужны дополнительные данные по инструкции в конце ноутбука; основной семинар работает без них.

## Как читать результаты

Semantic mIoU включает фон. Instance precision/recall считаются с one-to-one сопоставлением объектов одинакового класса при IoU ≥ 0.5 и score ≥ 0.4. Это не COCO AP. Высокий semantic mIoU не гарантирует правильного разделения экземпляров.

Короткие опыты на фигурах показывают поведение реализации. Их результаты зависят от данных и бюджета обучения; каждое сравнение в ноутбуке указывает, какие части модели меняются.

Материалы основаны на [семинаре ШАДа](https://github.com/yandexdataschool/deep_vision_and_graphics/tree/fall25/week04-image_segmentation/seminar). [Лицензия MIT](LICENSE).
