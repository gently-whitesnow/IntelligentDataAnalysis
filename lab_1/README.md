# ЛР 1 — вариант 7

## Задание
Регрессионный анализ прочности бетона на сжатие (`concrete_compressive_strength`) с:
- полносвязной нейросетевой моделью (Dense);
- одномерной сверточной сетью (1D CNN);
- стандартизацией признаков.

## Данные
Используйте набор данных **Concrete Compressive Strength**. Скачайте CSV/XLS файл и передайте путь через `--data`.

Ожидаемые столбцы (если заголовки отсутствуют):
`cement`, `blast_furnace_slag`, `fly_ash`, `water`, `superplasticizer`,
`coarse_aggregate`, `fine_aggregate`, `age`, `concrete_compressive_strength`.

## Запуск
```bash
python lab_1_variant_7.py --data /path/to/concrete.csv --artifacts artifacts
```

В папке `artifacts` сохраняются графики и `results.json`.