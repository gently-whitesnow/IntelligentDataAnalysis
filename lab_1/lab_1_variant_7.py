"""
Лабораторная работа 1, вариант 7: Регрессионный анализ прочности бетона на сжатие
Разведочный анализ и регрессия с использованием нейронных сетей

Датасет: Прочность бетона на сжатие
Модели: Полносвязная Dense НС, 1D сверточная НС
Преобразование данных: Стандартизация
Метрики качества регресии: RMSE, R²
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression

import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import layers, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Конфигурация
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
import random
random.seed(RANDOM_STATE)

# Настройка стиля графиков
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_data(file_path: str) -> pd.DataFrame:
    """Загрузить датасет с бетоном."""
    df = pd.read_csv(file_path, index_col=0)
    # Удалить пробелы из названий колонок
    df.columns = df.columns.str.strip()
    print("=" * 60)
    print("ДАТАСЕТ ЗАГРУЖЕН")
    print("=" * 60)
    print(f"Размерность: {df.shape}")
    print(f"\nКолонки: {list(df.columns)}")
    print(f"\nТипы данных:\n{df.dtypes}")
    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    """Показать описательную статистику."""
    print("\n" + "=" * 60)
    print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
    print("=" * 60)
    print(df.describe())

    print("\n" + "-" * 60)
    print("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
    print("-" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Количество пропусков': missing,
        'Процент': missing_pct
    })
    print(missing_df[missing_df['Количество пропусков'] > 0])

    if missing.sum() == 0:
        print("Пропущенных значений не найдено!")


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Построить одномерные распределения для всех признаков."""
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ОДНОМЕРНЫХ ВИЗУАЛИЗАЦИЙ")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Гистограммы и box-plot
    for col in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Гистограмма
        ax1.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(col)
        ax1.set_ylabel('Частота')
        ax1.set_title(f'Распределение {col}')
        ax1.grid(True, alpha=0.3)

        # Ящик с усами
        ax2.boxplot(df[col].dropna(), vert=True)
        ax2.set_ylabel(col)
        ax2.set_title(f'Box Plot для {col}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'dist_{col}.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f"Сохранено {len(df.columns)} графиков распределений в {output_dir}")


def plot_correlations(df: pd.DataFrame, output_dir: Path) -> None:
    """Построить тепловую карту корреляций и scatter-графики."""
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ МНОГОМЕРНЫХ ВИЗУАЛИЗАЦИЙ")
    print("=" * 60)

    # Тепловая карта корреляций
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Тепловая карта корреляций', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Тепловая карта корреляций сохранена")

    # Корреляции с целевой переменной
    target = 'concrete_compressive_strength'
    if target in df.columns:
        correlations_with_target = df.corr()[target].sort_values(ascending=False)
        print(f"\nКорреляции с {target}:")
        print(correlations_with_target)

        # Scatter-графики для признаков
        features = [col for col in df.columns if col != target]
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features):
            ax = axes[idx]
            ax.scatter(df[feature], df[target], alpha=0.5, s=20)
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            ax.set_title(f'{feature} vs {target}')
            ax.grid(True, alpha=0.3)

            # Добавить коэффициент корреляции
            corr = df[[feature, target]].corr().iloc[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5))

        # Скрыть неиспользуемые подграфики
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        # диаграмма рассеяния для каждой пары признаков
        plt.savefig(output_dir / 'scatter_plots.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Сохранены scatter-графики для {n_features} признаков")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистить датасет: обработать пропуски и дубликаты."""
    print("\n" + "=" * 60)
    print("ОЧИСТКА ДАННЫХ")
    print("=" * 60)

    df_clean = df.copy()

    # Обработка пропусков — заполнение медианой
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"Пропуски в {col} заполнены медианой: {median_val:.2f}")

    # Удаление дубликатов
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Удалено дубликатов строк: {n_duplicates}")
    else:
        print("Дубликаты строк не найдены")

    print(f"Итоговая форма датасета: {df_clean.shape}")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создать доменно-специфичные инженерные признаки."""
    print("\n" + "=" * 60)
    print("ИНЖИНИРИНГ ПРИЗНАКОВ")
    print("=" * 60)

    df_eng = df.copy()

    # Водоцементное отношение (критично для прочности бетона)
    df_eng['water_cement_ratio'] = df_eng['water'] / (df_eng['cement'] + 1e-6)

    # Суммарный заполнитель
    df_eng['total_aggregate'] = df_eng['coarse_aggregate'] + df_eng['fine_aggregate']

    # Общее содержание вяжущих (цементирующие материалы)
    df_eng['binder_content'] = (df_eng['cement'] +
                                  df_eng['blast_furnace_slag'] +
                                  df_eng['fly_ash'])

    # Водно-вяжущее отношение
    df_eng['water_binder_ratio'] = df_eng['water'] / (df_eng['binder_content'] + 1e-6)

    # Возраст в квадрате (нелинейный эффект возраста)
    df_eng['age_squared'] = df_eng['age'] ** 2

    # Логарифм возраста
    df_eng['age_log'] = np.log(df_eng['age'] + 1)

    new_features = ['water_cement_ratio', 'total_aggregate', 'binder_content',
                    'water_binder_ratio', 'age_squared', 'age_log']
    print(f"Создано новых признаков: {len(new_features)}")
    for feat in new_features:
        print(f"  - {feat}")

    print(f"Всего признаков: {df_eng.shape[1]}")
    return df_eng


def select_features(X: pd.DataFrame, y: pd.Series, top_fraction: float = 0.7) -> Tuple[pd.DataFrame, List[str]]:
    """Выбрать топ-признаки на основе взаимной информации."""
    print("\n" + "=" * 60)
    print("ОТБОР ПРИЗНАКОВ")
    print("=" * 60)

    # Посчитать взаимную информацию
    mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    print(f"\nОценки взаимной информации:")
    print(mi_scores)

    # Выбрать топ-признаки
    n_features = max(1, int(len(mi_scores) * top_fraction))
    selected_features = mi_scores.head(n_features).index.tolist()

    print(f"\nВыбрано топ {n_features} признаков ({top_fraction*100:.0f}%):")
    for feat in selected_features:
        print(f"  - {feat} (MI: {mi_scores[feat]:.4f})")

    return X[selected_features], selected_features


def prepare_datasets(df: pd.DataFrame, target_col: str) -> Dict:
    """Подготовить 4 набора данных: original, original_scaled, selected, selected_scaled."""
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА НАБОРОВ ДАННЫХ")
    print("=" * 60)

    # Разделить признаки и целевую переменную
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    print(f"Всего объектов: {len(y)}")
    print(f"Всего признаков (с инженерными): {X.shape[1]}")

    # Отбор признаков
    X_selected, _ = select_features(X, y, top_fraction=0.7)

    # Создать 4 версии датасета
    datasets = {}

    for name, features in [('original', X), ('selected', X_selected)]:
        # Разбиение на train/val/test (70/15/15)
        # Отделяем тестовые данные xtrain xtest ytrain ytest
        # отщепляем 15% для теста
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, y, test_size=0.15, random_state=RANDOM_STATE
        )

        # отделяем валидационные данные
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE
        )

        # Версия без масштабирования
        datasets[name] = {
            'X_train': X_train.values,
            'X_val': X_val.values,
            'X_test': X_test.values,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': features.columns.tolist(),
            'scaler': None
        }

        # Версия со стандартизацией
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        datasets[f'{name}_scaled'] = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': features.columns.tolist(),
            'scaler': scaler
        }

    print(f"\nСоздано 4 набора данных:")
    for name, data in datasets.items():
        print(f"  - {name}: {data['X_train'].shape[1]} признаков, "
              f"train={len(data['y_train'])}, val={len(data['y_val'])}, "
              f"test={len(data['y_test'])}")

    return datasets


def build_dense_model(input_dim: int, hidden_units: int = 128,
                      dropout: float = 0.3, learning_rate: float = 0.001) -> keras.Model:
    """Собрать полносвязную (Dense) нейросеть."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(hidden_units // 2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(hidden_units // 4, activation='relu'),
        layers.Dropout(dropout * 0.7),
        layers.Dense(1)  # Линейная активация для регрессии
    ], name='Dense_Model')

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model


def build_conv1d_model(input_dim: int, filters: int = 64, kernel_size: int = 3,
                       dropout: float = 0.3, learning_rate: float = 0.001) -> keras.Model:
    """Собрать 1D сверточную нейросеть."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(filters, kernel_size=kernel_size, activation='relu', padding='same'), # same - выход оставляем такой же дополняем нулями
        layers.BatchNormalization(),
        layers.Conv1D(filters * 2, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(), # среднее по всем фильтрам
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout * 0.7),
        layers.Dense(1)
    ], name='Conv1D_Model')

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Посчитать метрики регрессии."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'rmse': float(rmse),
        'r2': float(r2),
        'mae': float(mae)
    }


def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int = 200,
                batch_size: int = 32, verbose: int = 0) -> keras.callbacks.History:
    """Обучить модель нейросети."""
    # параметры регуляризации
    callbacks = [
        EarlyStopping( # ранний выход защита от переобучения
            monitor='val_loss',            # что отслеживаем: ошибка (loss) на валидации
            patience=20,                   # сколько ЭПОХ подряд ждать улучшения, прежде чем остановиться
            restore_best_weights=True,     # после остановки вернуть веса с ЛУЧШИМ val_loss (а не последние)
            verbose=0                      # уровень логов: 0 = тихо, 1 = печатать сообщения
        ),

        ReduceLROnPlateau( # снижение learning rate при застревании
            monitor='val_loss',            # снова следим за val_loss (важно для качества, а не для train)
            factor=0.5,                    # во сколько раз уменьшать lr: новый_lr = старый_lr * 0.5
            patience=10,                   # сколько эпох без улучшения ждать перед снижением lr
            min_lr=1e-7,                   # нижний предел lr: ниже этого lr опускаться не будет
            verbose=0                      # 0 = без сообщений, 1 = будет писать когда снизил lr
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )

    return history


def train_and_evaluate_all(datasets: Dict, epochs: int = 200) -> Dict:
    """Обучить обе модели на всех 4 наборах данных."""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ И ОЦЕНКА")
    print("=" * 60)

    results = {}
    all_models = {}

    for dataset_name, dataset in datasets.items():
        print(f"\n{'-' * 60}")
        print(f"Набор данных: {dataset_name}")
        print(f"{'-' * 60}")

        X_train = dataset['X_train']
        X_val = dataset['X_val']
        y_train = dataset['y_train']
        y_val = dataset['y_val']

        input_dim = X_train.shape[1]
        results[dataset_name] = {}
        all_models[dataset_name] = {}

        # Обучить Dense-модель
        print(f"\nОбучение Dense-модели...")
        dense_model = build_dense_model(input_dim)
        train_model(dense_model, X_train, y_train, X_val, y_val,
                   epochs=epochs, verbose=0)

        # Оценить Dense
        y_train_pred = dense_model.predict(X_train, verbose=0).flatten()
        y_val_pred = dense_model.predict(X_val, verbose=0).flatten()

        train_metrics = evaluate_metrics(y_train, y_train_pred)
        val_metrics = evaluate_metrics(y_val, y_val_pred)

        results[dataset_name]['dense'] = {
            'train': train_metrics,
            'val': val_metrics
        }
        all_models[dataset_name]['dense'] = dense_model

        print(f"Dense - Train RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"Dense - Val   RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")

        # Обучить Conv1D-модель (изменить форму данных)
        print(f"\nОбучение Conv1D-модели...")
        X_train_conv = X_train.reshape(-1, input_dim, 1)
        X_val_conv = X_val.reshape(-1, input_dim, 1)

        conv_model = build_conv1d_model(input_dim)
        train_model(conv_model, X_train_conv, y_train, X_val_conv, y_val,
                   epochs=epochs, verbose=0)

        # Оценить Conv1D
        y_train_pred = conv_model.predict(X_train_conv, verbose=0).flatten()
        y_val_pred = conv_model.predict(X_val_conv, verbose=0).flatten()

        train_metrics = evaluate_metrics(y_train, y_train_pred)
        val_metrics = evaluate_metrics(y_val, y_val_pred)

        results[dataset_name]['conv1d'] = {
            'train': train_metrics,
            'val': val_metrics
        }
        all_models[dataset_name]['conv1d'] = conv_model

        print(f"Conv1D - Train RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"Conv1D - Val   RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")

    return results, all_models


def find_best_model(results: Dict, datasets: Dict, all_models: Dict) -> Dict:
    """Найти лучшую модель по RMSE на валидации."""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    # Сформировать таблицу сравнения
    comparison = []
    for dataset_name in results:
        for model_name in results[dataset_name]:
            val_rmse = results[dataset_name][model_name]['val']['rmse']
            val_r2 = results[dataset_name][model_name]['val']['r2']
            train_rmse = results[dataset_name][model_name]['train']['rmse']
            train_r2 = results[dataset_name][model_name]['train']['r2']

            comparison.append({
                'dataset': dataset_name,
                'model': model_name,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('val_rmse')

    print("\nКачество всех моделей (отсортировано по RMSE на валидации):")
    print(comparison_df.to_string(index=False))

    # Найти лучшую
    best = comparison_df.iloc[0]
    best_dataset = best['dataset']
    best_model_name = best['model']

    print(f"\n{'=' * 60}")
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name} на {best_dataset}")
    print(f"{'=' * 60}")
    print(f"RMSE на валидации: {best['val_rmse']:.4f}")
    print(f"R² на валидации:   {best['val_r2']:.4f}")

    # Оценить на тестовой выборке
    dataset = datasets[best_dataset]
    model = all_models[best_dataset][best_model_name]

    X_test = dataset['X_test']
    y_test = dataset['y_test']

    if best_model_name == 'conv1d':
        X_test = X_test.reshape(-1, X_test.shape[1], 1)

    y_test_pred = model.predict(X_test, verbose=0).flatten()
    test_metrics = evaluate_metrics(y_test, y_test_pred)

    print(f"\nКачество на тесте:")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")

    return {
        'dataset_name': best_dataset,
        'model_name': best_model_name,
        'dataset': dataset,
        'model': model,
        'test_metrics': test_metrics,
        'val_metrics': {
            'rmse': best['val_rmse'],
            'r2': best['val_r2']
        },
        'comparison_df': comparison_df
    }


def grid_search(dataset: Dict, model_type: str, epochs: int = 150) -> Dict:
    """Выполнить grid search для подбора гиперпараметров."""
    print("\n" + "=" * 60)
    print(f"GRID SEARCH - {model_type.upper()}")
    print("=" * 60)

    X_train = dataset['X_train']
    X_val = dataset['X_val']
    y_train = dataset['y_train']
    y_val = dataset['y_val']
    input_dim = X_train.shape[1]

    if model_type == 'dense':
        param_grid = {
            'hidden_units': [64, 128],
            'dropout': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005]
        }
    else:  # conv1d
        param_grid = {
            'filters': [64, 128],
            'kernel_size': [3, 5],
            'dropout': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005]
        }
        X_train = X_train.reshape(-1, input_dim, 1)
        X_val = X_val.reshape(-1, input_dim, 1)

    best_score = float('inf')
    best_params = None
    best_model = None

    # Сгенерировать все комбинации
    import itertools
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    print(f"Тестируется комбинаций параметров: {len(combinations)}...")

    for idx, values in enumerate(combinations, 1):
        params = dict(zip(param_names, values))

        # Собрать модель с текущими параметрами
        if model_type == 'dense':
            model = build_dense_model(
                input_dim,
                hidden_units=params['hidden_units'],
                dropout=params['dropout'],
                learning_rate=params['learning_rate']
            )
        else:
            model = build_conv1d_model(
                input_dim,
                filters=params['filters'],
                kernel_size=params['kernel_size'],
                dropout=params['dropout'],
                learning_rate=params['learning_rate']
            )

        # Обучить
        train_model(model, X_train, y_train, X_val, y_val,
                   epochs=epochs, batch_size=32, verbose=0)

        # Оценить
        y_val_pred = model.predict(X_val, verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)

        print(f"  [{idx}/{len(combinations)}] {params} -> RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

        # Запомнить лучший вариант
        if val_rmse < best_score:
            best_score = val_rmse
            best_params = params
            best_model = model

    print(f"\n{'=' * 60}")
    print(f"ЛУЧШИЕ ПАРАМЕТРЫ НАЙДЕНЫ")
    print(f"{'=' * 60}")
    print(f"Параметры: {best_params}")
    print(f"RMSE на валидации: {best_score:.4f}")

    # Оценить на валидации
    y_val_pred = best_model.predict(X_val, verbose=0).flatten()
    val_metrics = evaluate_metrics(y_val, y_val_pred)

    return {
        'best_params': best_params,
        'best_model': best_model,
        'val_metrics': val_metrics
    }


def main():
    """Основная функция выполнения."""
    print(" " * 10 + "Регрессия прочности бетона на сжатие")

    # Подготовка
    output_dir = Path('lab_1_artifacts')
    output_dir.mkdir(exist_ok=True)

    # Шаг 1-2: Загрузка данных
    df = load_data('V7_dataset.csv')

    # Шаг 3: Разведочный анализ данных
    descriptive_stats(df)
    plot_distributions(df, output_dir)
    plot_correlations(df, output_dir)

    # Шаг 3c: Очистка данных
    df_clean = clean_data(df)

    # Шаг 3e: Инжиниринг признаков
    df_eng = engineer_features(df_clean)

    # Шаг 4: Подготовка 4 наборов данных
    datasets = prepare_datasets(df_eng, target_col='concrete_compressive_strength')

    # Шаг 5: Обучение и оценка всех моделей
    results, all_models = train_and_evaluate_all(datasets, epochs=200)

    # Шаг 6: Поиск лучшей модели и оценка на тесте
    best_info = find_best_model(results, datasets, all_models)

    # Шаг 7: Grid search по лучшей модели
    grid_results = grid_search(
        best_info['dataset'],
        best_info['model_name'],
        epochs=150
    )

    # Шаг 8: Оценка модели после grid search на тесте
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)

    X_test = best_info['dataset']['X_test']
    y_test = best_info['dataset']['y_test']

    if best_info['model_name'] == 'conv1d':
        X_test = X_test.reshape(-1, X_test.shape[1], 1)

    y_test_pred = grid_results['best_model'].predict(X_test, verbose=0).flatten()
    test_metrics_after = evaluate_metrics(y_test, y_test_pred)

    print(f"\nДо Grid Search:")
    print(f"  Test RMSE: {best_info['test_metrics']['rmse']:.4f}")
    print(f"  Test R²:   {best_info['test_metrics']['r2']:.4f}")

    print(f"\nПосле Grid Search:")
    print(f"  Test RMSE: {test_metrics_after['rmse']:.4f}")
    print(f"  Test R²:   {test_metrics_after['r2']:.4f}")

    improvement_rmse = ((best_info['test_metrics']['rmse'] - test_metrics_after['rmse']) /
                        best_info['test_metrics']['rmse'] * 100)
    improvement_r2 = ((test_metrics_after['r2'] - best_info['test_metrics']['r2']) /
                      abs(best_info['test_metrics']['r2']) * 100)

    print(f"\nУлучшение:")
    print(f"  RMSE: {improvement_rmse:+.2f}%")
    print(f"  R²:   {improvement_r2:+.2f}%")

    # Шаг 9: Сохранение результатов и выводов
    summary = {
        'best_model': best_info['model_name'],
        'best_dataset': best_info['dataset_name'],
        'results_before_grid_search': {
            'val_rmse': float(best_info['val_metrics']['rmse']),
            'val_r2': float(best_info['val_metrics']['r2']),
            'test_rmse': float(best_info['test_metrics']['rmse']),
            'test_r2': float(best_info['test_metrics']['r2']),
            'test_mae': float(best_info['test_metrics']['mae'])
        },
        'grid_search_params': grid_results['best_params'],
        'results_after_grid_search': {
            'val_rmse': float(grid_results['val_metrics']['rmse']),
            'val_r2': float(grid_results['val_metrics']['r2']),
            'test_rmse': float(test_metrics_after['rmse']),
            'test_r2': float(test_metrics_after['r2']),
            'test_mae': float(test_metrics_after['mae'])
        },
        'improvement': {
            'rmse_percentage': float(improvement_rmse),
            'r2_percentage': float(improvement_r2)
        },
        'all_models_comparison': best_info['comparison_df'].to_dict('records')
    }

    # Сохранить summary
    with open(output_dir / 'results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
