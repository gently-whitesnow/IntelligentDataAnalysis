"""
Лабораторная работа 2, вариант 7: Классификация уровней ожирения
Разведочный анализ и классификация с использованием нейронных сетей

Датасет: Уровни ожирения на основе привычек питания и физического состояния
Модели: Полносвязная Dense НС, Двунаправленная GRU НС
Преобразование данных: Min-max масштабирование
Метрики: Accuracy, Balanced Accuracy, F1-score
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

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.feature_selection import mutual_info_classif

# Try to configure JAX backend if available, otherwise use default
try:
    import os
    os.environ['KERAS_BACKEND'] = 'jax'
except:
    pass

# Try importing keras with available backend
try:
    import keras
except ModuleNotFoundError as e:
    # Keras requires TensorFlow, create minimal stub
    import sys
    from types import ModuleType

    # Create tensorflow stub
    tensorflow = ModuleType('tensorflow')
    tensorflow.python = ModuleType('tensorflow.python')
    tensorflow.python.trackable = ModuleType('tensorflow.python.trackable')
    tensorflow.python.trackable.data_structures = ModuleType('tensorflow.python.trackable.data_structures')

    class ListWrapper(list):
        pass

    tensorflow.python.trackable.data_structures.ListWrapper = ListWrapper
    sys.modules['tensorflow'] = tensorflow
    sys.modules['tensorflow.python'] = tensorflow.python
    sys.modules['tensorflow.python.trackable'] = tensorflow.python.trackable
    sys.modules['tensorflow.python.trackable.data_structures'] = tensorflow.python.trackable.data_structures

    import keras

import keras
from keras import layers, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

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
    """Загрузить датасет с уровнями ожирения."""
    df = pd.read_csv(file_path)
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

    # Численные признаки
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print("\nЧисленные признаки:")
    print(df[numerical_cols].describe())

    # Категориальные признаки
    categorical_cols = df.select_dtypes(include=['object']).columns
    print("\n" + "-" * 60)
    print("КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ")
    print("-" * 60)
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

    print("\n" + "-" * 60)
    print("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
    print("-" * 60)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("Пропущенных значений не найдено!")
    else:
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Количество пропусков': missing,
            'Процент': missing_pct
        })
        print(missing_df[missing_df['Количество пропусков'] > 0])


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Построить одномерные распределения для всех признаков."""
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ОДНОМЕРНЫХ ВИЗУАЛИЗАЦИЙ")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Численные признаки
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Гистограмма
        ax1.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(col)
        ax1.set_ylabel('Частота')
        ax1.set_title(f'Распределение {col}')
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(df[col].dropna(), vert=True)
        ax2.set_ylabel(col)
        ax2.set_title(f'Box Plot для {col}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'dist_{col}.png', dpi=100, bbox_inches='tight')
        plt.close()

    # Категориальные признаки
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        value_counts = df[col].value_counts()
        value_counts.plot(kind='bar', edgecolor='black', alpha=0.7)
        plt.xlabel(col)
        plt.ylabel('Частота')
        plt.title(f'Распределение {col}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / f'dist_{col}.png', dpi=100, bbox_inches='tight')
        plt.close()

    # Распределение целевой переменной
    if 'NObeyesdad' in df.columns:
        plt.figure(figsize=(12, 6))
        target_counts = df['NObeyesdad'].value_counts().sort_index()
        bars = plt.bar(range(len(target_counts)), target_counts.values,
                      edgecolor='black', alpha=0.7)
        plt.xlabel('Класс ожирения')
        plt.ylabel('Количество')
        plt.title('Распределение целевой переменной (NObeyesdad)')
        plt.xticks(range(len(target_counts)), target_counts.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Добавить значения на столбцы
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'target_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f"Сохранено {len(numerical_cols) + len(categorical_cols)} графиков распределений")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистить датасет: обработать пропуски и дубликаты."""
    print("\n" + "=" * 60)
    print("ОЧИСТКА ДАННЫХ")
    print("=" * 60)

    df_clean = df.copy()

    # Обработка пропусков
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in [np.float64, np.int64]:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"Пропуски в {col} заполнены медианой: {median_val:.2f}")
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"Пропуски в {col} заполнены модой: {mode_val}")

    # Удаление дубликатов
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Удалено дубликатов строк: {n_duplicates}")
    else:
        print("Дубликаты строк не найдены")

    print(f"Итоговая форма датасета: {df_clean.shape}")
    return df_clean


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Закодировать категориальные признаки."""
    print("\n" + "=" * 60)
    print("КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
    print("=" * 60)

    df_encoded = df.copy()
    encoders = {}

    # Бинарные признаки - Label Encoding
    binary_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for col in binary_features:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
            print(f"Label Encoding для {col}: {list(le.classes_)}")

    # Мультикатегориальные признаки - One-Hot Encoding
    multi_cat_features = ['CAEC', 'CALC', 'MTRANS']
    for col in multi_cat_features:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
            encoders[col] = list(df_encoded[col].unique())
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            print(f"One-Hot Encoding для {col}: {encoders[col]}")

    # Целевая переменная - сохранить в виде строк для стратификации
    # Закодируем позже при подготовке датасетов

    print(f"\nИтоговая размерность после кодирования: {df_encoded.shape}")
    return df_encoded, encoders


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создать доменно-специфичные инженерные признаки для предсказания ожирения."""
    print("\n" + "=" * 60)
    print("ИНЖИНИРИНГ ПРИЗНАКОВ")
    print("=" * 60)

    df_eng = df.copy()

    # BMI (Body Mass Index) - критический показатель для ожирения
    df_eng['BMI'] = df_eng['Weight'] / (df_eng['Height'] ** 2)

    # Взаимодействие возраста и веса
    df_eng['age_weight_interaction'] = df_eng['Age'] * df_eng['Weight']

    # Соотношение роста и веса
    df_eng['height_weight_ratio'] = df_eng['Height'] / (df_eng['Weight'] + 1e-6)

    # Соотношение воды и физической активности
    df_eng['water_activity_ratio'] = df_eng['CH2O'] / (df_eng['FAF'] + 1)

    # Оценка частоты питания
    df_eng['meal_frequency_score'] = df_eng['NCP'] * df_eng['FCVC']

    new_features = ['BMI', 'age_weight_interaction', 'height_weight_ratio',
                    'water_activity_ratio', 'meal_frequency_score']
    print(f"Создано новых признаков: {len(new_features)}")
    for feat in new_features:
        print(f"  - {feat}")

    print(f"Всего признаков: {df_eng.shape[1]}")
    return df_eng


def plot_correlations(df: pd.DataFrame, output_dir: Path, target_col: str = 'NObeyesdad') -> None:
    """Построить тепловую карту корреляций."""
    print("\n" + "=" * 60)
    print("АНАЛИЗ КОРРЕЛЯЦИЙ")
    print("=" * 60)

    # Временно закодировать целевую переменную для корреляции
    df_temp = df.copy()
    if df_temp[target_col].dtype == 'object':
        le = LabelEncoder()
        df_temp[target_col] = le.fit_transform(df_temp[target_col])

    # Только численные признаки
    numerical_cols = df_temp.select_dtypes(include=[np.number]).columns

    # Тепловая карта корреляций
    plt.figure(figsize=(14, 12))
    correlation = df_temp[numerical_cols].corr()
    sns.heatmap(correlation, annot=False, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Тепловая карта корреляций', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()

    # Корреляции с целевой переменной
    if target_col in numerical_cols:
        correlations_with_target = correlation[target_col].sort_values(ascending=False)
        print(f"\nТоп-10 корреляций с {target_col}:")
        print(correlations_with_target.head(10))


def select_features(X: pd.DataFrame, y: pd.Series, top_fraction: float = 0.7) -> Tuple[pd.DataFrame, List[str]]:
    """Выбрать топ-признаки на основе взаимной информации для классификации."""
    print("\n" + "=" * 60)
    print("ОТБОР ПРИЗНАКОВ")
    print("=" * 60)

    # Закодировать целевую переменную если нужно
    y_encoded = y
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

    # Посчитать взаимную информацию для классификации
    mi_scores = mutual_info_classif(X, y_encoded, random_state=RANDOM_STATE)
    mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    print(f"\nТоп-10 оценок взаимной информации:")
    print(mi_scores.head(10))

    # Выбрать топ-признаки
    n_features = max(1, int(len(mi_scores) * top_fraction))
    selected_features = mi_scores.head(n_features).index.tolist()

    print(f"\nВыбрано топ {n_features} признаков ({top_fraction*100:.0f}%):")
    for i, feat in enumerate(selected_features[:10], 1):
        print(f"  {i}. {feat} (MI: {mi_scores[feat]:.4f})")
    if len(selected_features) > 10:
        print(f"  ... и еще {len(selected_features) - 10} признаков")

    return X[selected_features], selected_features


def prepare_datasets(df: pd.DataFrame, target_col: str = 'NObeyesdad') -> Dict:
    """Подготовить 4 набора данных: original, original_scaled, selected, selected_scaled."""
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА НАБОРОВ ДАННЫХ")
    print("=" * 60)

    # Разделить признаки и целевую переменную
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    print(f"Всего объектов: {len(y)}")
    print(f"Всего признаков: {X.shape[1]}")
    print(f"Количество классов: {len(np.unique(y))}")

    # Закодировать целевую переменную
    le_target = LabelEncoder()
    y_labels = le_target.fit_transform(y)
    print(f"\nКлассы: {list(le_target.classes_)}")

    # Отбор признаков
    X_selected, selected_features = select_features(X, y, top_fraction=0.7)

    # Создать 4 версии датасета
    datasets = {}

    for name, features in [('original', X), ('selected', X_selected)]:
        # Стратифицированное разбиение на train/val/test (70/15/15)
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, y_labels, test_size=0.15, random_state=RANDOM_STATE,
            stratify=y_labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE,
            stratify=y_temp
        )

        # One-hot encode целевую переменную для обучения
        y_train_cat = to_categorical(y_train, num_classes=len(le_target.classes_))
        y_val_cat = to_categorical(y_val, num_classes=len(le_target.classes_))
        y_test_cat = to_categorical(y_test, num_classes=len(le_target.classes_))

        # Версия без масштабирования
        datasets[name] = {
            'X_train': X_train.values.astype(np.float32),
            'X_val': X_val.values.astype(np.float32),
            'X_test': X_test.values.astype(np.float32),
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'y_train_cat': y_train_cat,
            'y_val_cat': y_val_cat,
            'y_test_cat': y_test_cat,
            'feature_names': features.columns.tolist(),
            'scaler': None
        }

        # Версия с Min-Max масштабированием
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        datasets[f'{name}_scaled'] = {
            'X_train': X_train_scaled.astype(np.float32),
            'X_val': X_val_scaled.astype(np.float32),
            'X_test': X_test_scaled.astype(np.float32),
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'y_train_cat': y_train_cat,
            'y_val_cat': y_val_cat,
            'y_test_cat': y_test_cat,
            'feature_names': features.columns.tolist(),
            'scaler': scaler
        }

    print(f"\nСоздано 4 набора данных:")
    for name, data in datasets.items():
        print(f"  - {name}: {data['X_train'].shape[1]} признаков, "
              f"train={len(data['y_train'])}, val={len(data['y_val'])}, "
              f"test={len(data['y_test'])}")

    return datasets, le_target


def build_dense_model(input_dim: int, num_classes: int = 7,
                      hidden_units: int = 128, dropout: float = 0.3,
                      learning_rate: float = 0.001) -> keras.Model:
    """Собрать полносвязную (Dense) нейросеть для классификации."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(hidden_units // 2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(hidden_units // 4, activation='relu'),
        layers.Dropout(dropout * 0.7),
        layers.Dense(num_classes, activation='softmax')
    ], name='Dense_Model')

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_bidirectional_gru_model(input_dim: int, num_classes: int = 7,
                                   gru_units_1: int = 64, gru_units_2: int = 32,
                                   dropout: float = 0.3,
                                   learning_rate: float = 0.001) -> keras.Model:
    """Собрать двунаправленную GRU нейросеть для классификации табличных данных."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Bidirectional(layers.GRU(gru_units_1, return_sequences=True)),
        layers.Dropout(dropout),
        layers.Bidirectional(layers.GRU(gru_units_2)),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout * 0.7),
        layers.Dense(num_classes, activation='softmax')
    ], name='Bidirectional_GRU_Model')

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def evaluate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """Посчитать метрики классификации."""
    # Конвертировать one-hot в метки если нужно
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_per_class = f1_score(y_true, y_pred, average=None)

    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': [float(f) for f in f1_per_class]
    }


def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int = 200,
                batch_size: int = 32, verbose: int = 0) -> keras.callbacks.History:
    """Обучить модель нейросети."""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
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


def train_and_evaluate_all(datasets: Dict, num_classes: int = 7, epochs: int = 200) -> Dict:
    """Обучить обе модели на всех 4 наборах данных."""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("=" * 60)

    results = {}
    all_models = {}

    for dataset_name, dataset in datasets.items():
        print(f"\n{'-' * 60}")
        print(f"Набор данных: {dataset_name}")
        print(f"{'-' * 60}")

        X_train = dataset['X_train']
        X_val = dataset['X_val']
        X_test = dataset['X_test']
        y_train = dataset['y_train_cat']
        y_val = dataset['y_val_cat']

        input_dim = X_train.shape[1]
        results[dataset_name] = {}
        all_models[dataset_name] = {}

        # Обучить Dense-модель
        print(f"\nОбучение Dense-модели...")
        dense_model = build_dense_model(input_dim, num_classes=num_classes)
        train_model(dense_model, X_train, y_train, X_val, y_val,
                   epochs=epochs, verbose=0)

        # Оценить Dense
        y_train_pred = dense_model.predict(X_train, verbose=0)
        y_val_pred = dense_model.predict(X_val, verbose=0)

        train_metrics = evaluate_classification_metrics(y_train, y_train_pred)
        val_metrics = evaluate_classification_metrics(y_val, y_val_pred)

        results[dataset_name]['dense'] = {
            'train': train_metrics,
            'val': val_metrics
        }
        all_models[dataset_name]['dense'] = dense_model

        print(f"Dense - Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Balanced Acc: {train_metrics['balanced_accuracy']:.4f}, "
              f"F1: {train_metrics['f1_macro']:.4f}")
        print(f"Dense - Val   Acc: {val_metrics['accuracy']:.4f}, "
              f"Balanced Acc: {val_metrics['balanced_accuracy']:.4f}, "
              f"F1: {val_metrics['f1_macro']:.4f}")

        # Обучить Bidirectional GRU модель (изменить форму данных)
        print(f"\nОбучение Bidirectional GRU модели...")
        X_train_gru = X_train.reshape(-1, input_dim, 1)
        X_val_gru = X_val.reshape(-1, input_dim, 1)

        gru_model = build_bidirectional_gru_model(input_dim, num_classes=num_classes)
        train_model(gru_model, X_train_gru, y_train, X_val_gru, y_val,
                   epochs=epochs, verbose=0)

        # Оценить GRU
        y_train_pred = gru_model.predict(X_train_gru, verbose=0)
        y_val_pred = gru_model.predict(X_val_gru, verbose=0)

        train_metrics = evaluate_classification_metrics(y_train, y_train_pred)
        val_metrics = evaluate_classification_metrics(y_val, y_val_pred)

        results[dataset_name]['bidirectional_gru'] = {
            'train': train_metrics,
            'val': val_metrics
        }
        all_models[dataset_name]['bidirectional_gru'] = gru_model

        print(f"BiGRU - Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Balanced Acc: {train_metrics['balanced_accuracy']:.4f}, "
              f"F1: {train_metrics['f1_macro']:.4f}")
        print(f"BiGRU - Val   Acc: {val_metrics['accuracy']:.4f}, "
              f"Balanced Acc: {val_metrics['balanced_accuracy']:.4f}, "
              f"F1: {val_metrics['f1_macro']:.4f}")

    return results, all_models


def find_best_model(results: Dict, datasets: Dict, all_models: Dict,
                    le_target: LabelEncoder, output_dir: Path) -> Dict:
    """Найти лучшую модель по balanced accuracy на валидации."""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    # Сформировать таблицу сравнения
    comparison = []
    for dataset_name in results:
        for model_name in results[dataset_name]:
            val_metrics = results[dataset_name][model_name]['val']
            train_metrics = results[dataset_name][model_name]['train']

            comparison.append({
                'dataset': dataset_name,
                'model': model_name,
                'train_acc': train_metrics['accuracy'],
                'train_balanced_acc': train_metrics['balanced_accuracy'],
                'train_f1': train_metrics['f1_macro'],
                'val_acc': val_metrics['accuracy'],
                'val_balanced_acc': val_metrics['balanced_accuracy'],
                'val_f1': val_metrics['f1_macro']
            })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('val_balanced_acc', ascending=False)

    print("\nКачество всех моделей (отсортировано по Balanced Accuracy на валидации):")
    print(comparison_df.to_string(index=False))

    # Найти лучшую
    best = comparison_df.iloc[0]
    best_dataset = best['dataset']
    best_model_name = best['model']

    print(f"\n{'=' * 60}")
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name} на {best_dataset}")
    print(f"{'=' * 60}")
    print(f"Accuracy на валидации:          {best['val_acc']:.4f}")
    print(f"Balanced Accuracy на валидации: {best['val_balanced_acc']:.4f}")
    print(f"F1-score на валидации:          {best['val_f1']:.4f}")

    # Оценить на тестовой выборке
    dataset = datasets[best_dataset]
    model = all_models[best_dataset][best_model_name]

    X_test = dataset['X_test']
    y_test = dataset['y_test_cat']

    if best_model_name == 'bidirectional_gru':
        X_test = X_test.reshape(-1, X_test.shape[1], 1)

    y_test_pred = model.predict(X_test, verbose=0)
    test_metrics = evaluate_classification_metrics(y_test, y_test_pred)

    print(f"\nКачество на тестовой выборке:")
    print(f"Test Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test F1-score (macro):  {test_metrics['f1_macro']:.4f}")
    print(f"Test F1-score (weighted): {test_metrics['f1_weighted']:.4f}")

    # Детальный отчет по классам
    y_test_labels = np.argmax(y_test, axis=1)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)

    print(f"\nОтчет по классам:")
    print(classification_report(y_test_labels, y_test_pred_labels,
                               target_names=le_target.classes_))

    # Матрица ошибок
    cm = confusion_matrix(y_test_labels, y_test_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_)
    plt.title('Матрица ошибок')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_best.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Матрица ошибок сохранена в {output_dir / 'confusion_matrix_best.png'}")

    return {
        'dataset_name': best_dataset,
        'model_name': best_model_name,
        'dataset': dataset,
        'model': model,
        'test_metrics': test_metrics,
        'val_metrics': {
            'accuracy': best['val_acc'],
            'balanced_accuracy': best['val_balanced_acc'],
            'f1_macro': best['val_f1']
        },
        'comparison_df': comparison_df
    }


def grid_search(dataset: Dict, model_type: str, num_classes: int = 7,
                epochs: int = 150, n_random_samples: int = 9) -> Dict:
    """Выполнить random search для подбора гиперпараметров (сэмплируем n_random_samples случайных комбинаций)."""
    print("\n" + "=" * 60)
    print(f"RANDOM SEARCH - {model_type.upper()}")
    print("=" * 60)

    X_train = dataset['X_train']
    X_val = dataset['X_val']
    y_train = dataset['y_train_cat']
    y_val = dataset['y_val_cat']
    input_dim = X_train.shape[1]

    if model_type == 'dense':
        param_grid = {
            'hidden_units': [64, 128, 256],
            'dropout': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005, 0.0001]
        }
    else:  # bidirectional_gru
        param_grid = {
            'gru_units_1': [32, 64, 128],
            'gru_units_2': [16, 32, 64],
            'dropout': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005, 0.0001]
        }
        X_train = X_train.reshape(-1, input_dim, 1)
        X_val = X_val.reshape(-1, input_dim, 1)

    best_score = 0
    best_params = None
    best_model = None

    # Сгенерировать все комбинации и выбрать n_random_samples случайных
    import itertools
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    all_combinations = list(itertools.product(*param_values))

    # Случайный выбор комбинаций
    n_samples = min(n_random_samples, len(all_combinations))
    combinations = random.sample(all_combinations, n_samples)

    print(f"Тестируется комбинаций параметров: {len(combinations)} из {len(all_combinations)} возможных...")

    for idx, values in enumerate(combinations, 1):
        params = dict(zip(param_names, values))

        # Собрать модель с текущими параметрами
        if model_type == 'dense':
            model = build_dense_model(
                input_dim,
                num_classes=num_classes,
                hidden_units=params['hidden_units'],
                dropout=params['dropout'],
                learning_rate=params['learning_rate']
            )
        else:
            model = build_bidirectional_gru_model(
                input_dim,
                num_classes=num_classes,
                gru_units_1=params['gru_units_1'],
                gru_units_2=params['gru_units_2'],
                dropout=params['dropout'],
                learning_rate=params['learning_rate']
            )

        # Обучить
        train_model(model, X_train, y_train, X_val, y_val,
                   epochs=epochs, batch_size=32, verbose=0)

        # Оценить
        y_val_pred = model.predict(X_val, verbose=0)
        val_metrics = evaluate_classification_metrics(y_val, y_val_pred)
        val_balanced_acc = val_metrics['balanced_accuracy']

        print(f"  [{idx}/{len(combinations)}] {params}")
        print(f"    -> Balanced Acc: {val_balanced_acc:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1_macro']:.4f}")

        # Запомнить лучший вариант
        if val_balanced_acc > best_score:
            best_score = val_balanced_acc
            best_params = params
            best_model = model

    print(f"\n{'=' * 60}")
    print(f"ЛУЧШИЕ ПАРАМЕТРЫ НАЙДЕНЫ")
    print(f"{'=' * 60}")
    print(f"Параметры: {best_params}")
    print(f"Balanced Accuracy на валидации: {best_score:.4f}")

    # Оценить на валидации
    y_val_pred = best_model.predict(X_val, verbose=0)
    val_metrics = evaluate_classification_metrics(y_val, y_val_pred)

    return {
        'best_params': best_params,
        'best_model': best_model,
        'val_metrics': val_metrics
    }


def main():
    """Основная функция выполнения."""
    print("\n" + "=" * 70)
    print(" " * 20 + "ЛАБОРАТОРНАЯ 2 - ВАРИАНТ 7")
    print(" " * 10 + "Классификация уровней ожирения")
    print("=" * 70)

    # Подготовка
    output_dir = Path('lab_2_artifacts')
    output_dir.mkdir(exist_ok=True)

    # Шаг 1-2: Загрузка данных
    df = load_data('V7_classification_lr3.csv')

    # Шаг 3a: Описательная статистика
    descriptive_stats(df)

    # Шаг 3b: Визуализация данных
    plot_distributions(df, output_dir)

    # Шаг 3c: Очистка данных
    df_clean = clean_data(df)

    # Шаг 3g: Кодирование категориальных признаков (перед инжинирингом)
    df_encoded, encoders = encode_categoricals(df_clean)

    # Шаг 3d: Корреляционный анализ
    plot_correlations(df_encoded, output_dir, target_col='NObeyesdad')

    # Шаг 3e: Инжиниринг признаков
    df_eng = engineer_features(df_encoded)

    # Шаг 4: Подготовка 4 наборов данных (включает 3f - отбор признаков, 3h - масштабирование)
    datasets, le_target = prepare_datasets(df_eng, target_col='NObeyesdad')
    num_classes = len(le_target.classes_)

    # Шаг 5: Обучение и оценка всех моделей
    results, all_models = train_and_evaluate_all(datasets, num_classes=num_classes, epochs=200)

    # Шаг 6: Поиск лучшей модели и оценка на тесте
    best_info = find_best_model(results, datasets, all_models, le_target, output_dir)

    # Шаг 7: Grid search по лучшей модели
    grid_results = grid_search(
        best_info['dataset'],
        best_info['model_name'],
        num_classes=num_classes,
        epochs=150
    )

    # Шаг 8: Оценка модели после grid search на тесте
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)

    X_test = best_info['dataset']['X_test']
    y_test = best_info['dataset']['y_test_cat']

    if best_info['model_name'] == 'bidirectional_gru':
        X_test = X_test.reshape(-1, X_test.shape[1], 1)

    y_test_pred = grid_results['best_model'].predict(X_test, verbose=0)
    test_metrics_after = evaluate_classification_metrics(y_test, y_test_pred)

    print(f"\nДо Grid Search:")
    print(f"  Test Accuracy:          {best_info['test_metrics']['accuracy']:.4f}")
    print(f"  Test Balanced Accuracy: {best_info['test_metrics']['balanced_accuracy']:.4f}")
    print(f"  Test F1-score:          {best_info['test_metrics']['f1_macro']:.4f}")

    print(f"\nПосле Grid Search:")
    print(f"  Test Accuracy:          {test_metrics_after['accuracy']:.4f}")
    print(f"  Test Balanced Accuracy: {test_metrics_after['balanced_accuracy']:.4f}")
    print(f"  Test F1-score:          {test_metrics_after['f1_macro']:.4f}")

    improvement_acc = ((test_metrics_after['accuracy'] - best_info['test_metrics']['accuracy']) /
                       best_info['test_metrics']['accuracy'] * 100)
    improvement_bal_acc = ((test_metrics_after['balanced_accuracy'] -
                           best_info['test_metrics']['balanced_accuracy']) /
                          best_info['test_metrics']['balanced_accuracy'] * 100)
    improvement_f1 = ((test_metrics_after['f1_macro'] - best_info['test_metrics']['f1_macro']) /
                      best_info['test_metrics']['f1_macro'] * 100)

    print(f"\nУлучшение:")
    print(f"  Accuracy:          {improvement_acc:+.2f}%")
    print(f"  Balanced Accuracy: {improvement_bal_acc:+.2f}%")
    print(f"  F1-score:          {improvement_f1:+.2f}%")

    # Детальный отчет после grid search
    y_test_labels = np.argmax(y_test, axis=1)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)

    print(f"\nОтчет по классам после Grid Search:")
    print(classification_report(y_test_labels, y_test_pred_labels,
                               target_names=le_target.classes_))

    # Финальная матрица ошибок
    cm = confusion_matrix(y_test_labels, y_test_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_)
    plt.title('Матрица ошибок (после Grid Search)')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_final.png', dpi=100, bbox_inches='tight')
    plt.close()

    # Шаг 9: Сохранение результатов и выводов
    summary = {
        'best_model': best_info['model_name'],
        'best_dataset': best_info['dataset_name'],
        'num_classes': num_classes,
        'class_names': list(le_target.classes_),
        'results_before_grid_search': {
            'val_accuracy': float(best_info['val_metrics']['accuracy']),
            'val_balanced_accuracy': float(best_info['val_metrics']['balanced_accuracy']),
            'val_f1_macro': float(best_info['val_metrics']['f1_macro']),
            'test_accuracy': float(best_info['test_metrics']['accuracy']),
            'test_balanced_accuracy': float(best_info['test_metrics']['balanced_accuracy']),
            'test_f1_macro': float(best_info['test_metrics']['f1_macro']),
            'test_f1_weighted': float(best_info['test_metrics']['f1_weighted']),
            'test_f1_per_class': best_info['test_metrics']['f1_per_class']
        },
        'grid_search_params': grid_results['best_params'],
        'results_after_grid_search': {
            'val_accuracy': float(grid_results['val_metrics']['accuracy']),
            'val_balanced_accuracy': float(grid_results['val_metrics']['balanced_accuracy']),
            'val_f1_macro': float(grid_results['val_metrics']['f1_macro']),
            'test_accuracy': float(test_metrics_after['accuracy']),
            'test_balanced_accuracy': float(test_metrics_after['balanced_accuracy']),
            'test_f1_macro': float(test_metrics_after['f1_macro']),
            'test_f1_weighted': float(test_metrics_after['f1_weighted']),
            'test_f1_per_class': test_metrics_after['f1_per_class']
        },
        'improvement': {
            'accuracy_percentage': float(improvement_acc),
            'balanced_accuracy_percentage': float(improvement_bal_acc),
            'f1_macro_percentage': float(improvement_f1)
        },
        'all_models_comparison': best_info['comparison_df'].to_dict('records')
    }

    # Сохранить summary
    with open(output_dir / 'results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Результаты сохранены в {output_dir}/results_summary.json")
    print(f"Графики сохранены в {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
