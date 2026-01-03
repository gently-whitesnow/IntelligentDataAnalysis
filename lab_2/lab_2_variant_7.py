"""Lab 2, variant 7: Obesity level classification.

Steps:
1. Load data.
2. EDA with descriptive statistics and visualizations.
3. Optional cleaning and feature engineering.
4. Correlation analysis.
5. Feature selection.
6. One-hot encoding for categorical features.
7. Min-max scaling (variant requirement).
8. Train/evaluate Dense NN and Bidirectional GRU on 4 datasets.
9. Grid search hyperparameters for the best model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
TARGET_COL = "NObeyesdad"


@dataclass
class DatasetBundle:
    name: str
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(data_path: Path | None) -> pd.DataFrame:
    if data_path is None:
        raise FileNotFoundError(
            "Dataset path is required. Download the Obesity Levels dataset and pass --data /path/to/obesity.csv."
        )
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    if data_path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns: {df.columns}")

    return df


def basic_eda(df: pd.DataFrame, artifacts_dir: Path) -> None:
    print("\n=== Dataset info ===")
    print(df.info())
    print("\n=== Descriptive statistics (numeric) ===")
    print(df.describe().T)
    print("\n=== Missing values ===")
    print(df.isna().sum())

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(artifacts_dir / f"dist_{col}.png")
        plt.close()

    for col in categorical_cols:
        plt.figure(figsize=(7, 4))
        df[col].value_counts().plot(kind="bar")
        plt.title(f"Count: {col}")
        plt.tight_layout()
        plt.savefig(artifacts_dir / f"count_{col}.png")
        plt.close()

    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=False)
        plt.title("Numeric feature correlation")
        plt.tight_layout()
        plt.savefig(artifacts_dir / "correlation_heatmap.png")
        plt.close()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    for col in categorical_cols:
        mode_value = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_value)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"Height", "Weight"}.issubset(df.columns):
        df["BMI"] = df["Weight"] / (df["Height"] ** 2 + 1e-6)
    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df[TARGET_COL].astype(str).to_numpy()
    x = df.drop(columns=[TARGET_COL])
    x_encoded = pd.get_dummies(x, drop_first=False)
    return x_encoded, y


def select_features(
    x: pd.DataFrame, y: np.ndarray, top_fraction: float = 0.7
) -> Tuple[pd.DataFrame, List[str]]:
    k = max(10, int(x.shape[1] * top_fraction))
    scores = mutual_info_classif(x, y, discrete_features="auto", random_state=RANDOM_STATE)
    score_series = pd.Series(scores, index=x.columns).sort_values(ascending=False)
    selected_cols = score_series.head(k).index.tolist()
    return x[selected_cols].copy(), selected_cols


def split_dataset(
    x: pd.DataFrame,
    y: np.ndarray,
    scale: bool,
    scaler: MinMaxScaler | None = None,
    name: str = "",
) -> DatasetBundle:
    x_values = x.to_numpy(dtype=np.float32)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_values, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    if scale:
        scaler = scaler or MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    return DatasetBundle(
        name=name,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=list(x.columns),
    )


def build_dense_model(input_dim: int, hidden_units: int = 128, dropout: float = 0.3) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(hidden_units // 2, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    return model


def build_bigru_model(
    input_steps: int, gru_units: int = 64, dropout: float = 0.3
) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_steps, 1)),
            layers.Bidirectional(layers.GRU(gru_units, dropout=dropout, recurrent_dropout=dropout)),
            layers.Dense(gru_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    return model


def compile_and_train(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float,
    epochs: int,
) -> keras.callbacks.History:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0,
    )
    return history


def evaluate_model(
    model: keras.Model, x_data: np.ndarray, y_true: np.ndarray
) -> Dict[str, float]:
    probs = model.predict(x_data, verbose=0)
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def classification_report_dict(
    model: keras.Model, x_data: np.ndarray, y_true: np.ndarray
) -> Dict[str, Dict[str, float]]:
    probs = model.predict(x_data, verbose=0)
    y_pred = probs.argmax(axis=1)
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def train_and_evaluate_dataset(
    dataset: DatasetBundle, epochs: int
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    dense_model = build_dense_model(dataset.x_train.shape[1])
    compile_and_train(
        dense_model,
        dataset.x_train,
        dataset.y_train,
        dataset.x_val,
        dataset.y_val,
        learning_rate=1e-3,
        epochs=epochs,
    )
    results["dense_train"] = evaluate_model(dense_model, dataset.x_train, dataset.y_train)
    results["dense_val"] = evaluate_model(dense_model, dataset.x_val, dataset.y_val)
    results["dense_val_report"] = classification_report_dict(
        dense_model, dataset.x_val, dataset.y_val
    )

    x_train_seq = dataset.x_train.reshape(-1, dataset.x_train.shape[1], 1)
    x_val_seq = dataset.x_val.reshape(-1, dataset.x_val.shape[1], 1)
    bigru_model = build_bigru_model(dataset.x_train.shape[1])
    compile_and_train(
        bigru_model,
        x_train_seq,
        dataset.y_train,
        x_val_seq,
        dataset.y_val,
        learning_rate=1e-3,
        epochs=epochs,
    )
    results["bigru_train"] = evaluate_model(bigru_model, x_train_seq, dataset.y_train)
    results["bigru_val"] = evaluate_model(bigru_model, x_val_seq, dataset.y_val)
    results["bigru_val_report"] = classification_report_dict(
        bigru_model, x_val_seq, dataset.y_val
    )

    results["_models"] = {"dense": dense_model, "bigru": bigru_model}
    return results


def grid_search_best_model(
    dataset: DatasetBundle,
    model_name: str,
    epochs: int,
) -> Tuple[keras.Model, Dict[str, float]]:
    best_score = -1.0
    best_model = None
    best_params: Dict[str, float] = {}

    if model_name == "dense":
        param_grid = {
            "hidden_units": [64, 128],
            "dropout": [0.2, 0.4],
            "learning_rate": [1e-3, 5e-4],
        }
        for hidden_units in param_grid["hidden_units"]:
            for dropout in param_grid["dropout"]:
                for learning_rate in param_grid["learning_rate"]:
                    model = build_dense_model(dataset.x_train.shape[1], hidden_units, dropout)
                    compile_and_train(
                        model,
                        dataset.x_train,
                        dataset.y_train,
                        dataset.x_val,
                        dataset.y_val,
                        learning_rate=learning_rate,
                        epochs=epochs,
                    )
                    metrics = evaluate_model(model, dataset.x_val, dataset.y_val)
                    if metrics["f1_macro"] > best_score:
                        best_score = metrics["f1_macro"]
                        best_model = model
                        best_params = {
                            "hidden_units": hidden_units,
                            "dropout": dropout,
                            "learning_rate": learning_rate,
                        }
    else:
        param_grid = {
            "gru_units": [32, 64],
            "dropout": [0.2, 0.4],
            "learning_rate": [1e-3, 5e-4],
        }
        x_train_seq = dataset.x_train.reshape(-1, dataset.x_train.shape[1], 1)
        x_val_seq = dataset.x_val.reshape(-1, dataset.x_val.shape[1], 1)
        for gru_units in param_grid["gru_units"]:
            for dropout in param_grid["dropout"]:
                for learning_rate in param_grid["learning_rate"]:
                    model = build_bigru_model(dataset.x_train.shape[1], gru_units, dropout)
                    compile_and_train(
                        model,
                        x_train_seq,
                        dataset.y_train,
                        x_val_seq,
                        dataset.y_val,
                        learning_rate=learning_rate,
                        epochs=epochs,
                    )
                    metrics = evaluate_model(model, x_val_seq, dataset.y_val)
                    if metrics["f1_macro"] > best_score:
                        best_score = metrics["f1_macro"]
                        best_model = model
                        best_params = {
                            "gru_units": gru_units,
                            "dropout": dropout,
                            "learning_rate": learning_rate,
                        }

    if best_model is None:
        raise RuntimeError("Grid search failed to produce a model.")

    print(f"Best grid params for {model_name}: {best_params}")
    return best_model, best_params


def serialize_report(report: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {k: {metric: float(value) for metric, value in v.items()} for k, v in report.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 2 Variant 7: Obesity Level Classification")
    parser.add_argument("--data", type=Path, required=True, help="Path to obesity dataset CSV/XLSX")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("lab_2_artifacts"),
        help="Directory to save plots and reports",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    args = parser.parse_args()

    set_seed()

    df = load_dataset(args.data)
    basic_eda(df, args.artifacts_dir)
    df = clean_data(df)
    df = engineer_features(df)

    x_encoded, y_raw = encode_features(df)
    label_to_index = {label: idx for idx, label in enumerate(sorted(np.unique(y_raw)))}
    y = np.array([label_to_index[label] for label in y_raw])

    global NUM_CLASSES
    NUM_CLASSES = len(label_to_index)

    x_selected, selected_cols = select_features(x_encoded, y)
    print(f"Selected {len(selected_cols)} features out of {x_encoded.shape[1]}")

    datasets = [
        split_dataset(x_encoded, y, scale=False, name="original"),
        split_dataset(x_encoded, y, scale=True, name="original_scaled"),
        split_dataset(x_selected, y, scale=False, name="selected"),
        split_dataset(x_selected, y, scale=True, name="selected_scaled"),
    ]

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    best_entry = {
        "dataset": None,
        "model": None,
        "score": -1.0,
        "bundle": None,
        "trained_model": None,
    }

    for dataset in datasets:
        print(f"\n=== Training on dataset: {dataset.name} ===")
        results = train_and_evaluate_dataset(dataset, epochs=args.epochs)
        all_results[dataset.name] = {k: v for k, v in results.items() if k != "_models"}

        for model_name in ("dense", "bigru"):
            val_metrics = results[f"{model_name}_val"]
            if val_metrics["f1_macro"] > best_entry["score"]:
                best_entry.update(
                    {
                        "dataset": dataset.name,
                        "model": model_name,
                        "score": val_metrics["f1_macro"],
                        "bundle": dataset,
                        "trained_model": results["_models"][model_name],
                    }
                )

    print(
        f"\nBest model by val f1_macro: {best_entry['model']} on {best_entry['dataset']}"
    )

    best_bundle = best_entry["bundle"]
    best_model = best_entry["trained_model"]
    if best_entry["model"] == "bigru":
        x_test = best_bundle.x_test.reshape(-1, best_bundle.x_test.shape[1], 1)
    else:
        x_test = best_bundle.x_test
    test_metrics_before = evaluate_model(best_model, x_test, best_bundle.y_test)

    grid_model, grid_params = grid_search_best_model(
        best_bundle, best_entry["model"], epochs=args.epochs
    )
    if best_entry["model"] == "bigru":
        x_test_grid = best_bundle.x_test.reshape(-1, best_bundle.x_test.shape[1], 1)
    else:
        x_test_grid = best_bundle.x_test
    test_metrics_after = evaluate_model(grid_model, x_test_grid, best_bundle.y_test)

    summary = {
        "best_model": best_entry["model"],
        "best_dataset": best_entry["dataset"],
        "best_val_f1_macro": float(best_entry["score"]),
        "test_before_grid": test_metrics_before,
        "grid_params": grid_params,
        "test_after_grid": test_metrics_after,
        "all_results": all_results,
        "label_mapping": label_to_index,
    }

    report_path = args.artifacts_dir / "metrics_summary.json"
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved metrics summary to {report_path}")


if __name__ == "__main__":
    main()