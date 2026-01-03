"""Lab 1, variant 7: Concrete compressive strength regression.

Steps:
1. Load data.
2. EDA with descriptive statistics and visualizations.
3. Optional cleaning.
4. Correlation analysis.
5. Feature engineering.
6. Feature selection.
7. Standardization.
8. Train/evaluate Dense NN and 1D CNN on 4 datasets.
9. Grid search for CNN hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
TARGET_COL = "concrete_compressive_strength"

DEFAULT_COLUMN_NAMES = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    TARGET_COL,
]


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
    """Load dataset from a local file.

    Expected columns (in order) if headers are missing:
    cement, blast_furnace_slag, fly_ash, water, superplasticizer,
    coarse_aggregate, fine_aggregate, age, concrete_compressive_strength.
    """
    if data_path is None:
        raise FileNotFoundError(
            "Dataset path is required. Download the Concrete Compressive Strength dataset "
            "and pass --data /path/to/concrete.csv."
        )

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    if data_path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    if df.columns.tolist() == list(range(len(df.columns))):
        df.columns = DEFAULT_COLUMN_NAMES

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns: {df.columns}")

    return df


def basic_eda(df: pd.DataFrame, artifacts_dir: Path) -> None:
    print("\n=== Dataset info ===")
    print(df.info())
    print("\n=== Descriptive statistics ===")
    print(df.describe().T)
    print("\n=== Missing values ===")
    print(df.isna().sum())

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Univariate distributions
    for col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(artifacts_dir / f"dist_{col}.png")
        plt.close()

    # Pairplot for a subset of features
    subset_cols = df.columns[:5].tolist() + [TARGET_COL]
    sns.pairplot(df[subset_cols])
    plt.savefig(artifacts_dir / "pairplot_subset.png")
    plt.close()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, handle negative values if any."""
    df = df.drop_duplicates().copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)
    return df


def correlation_analysis(df: pd.DataFrame, artifacts_dir: Path) -> pd.Series:
    corr = df.corr(numeric_only=True)[TARGET_COL].sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "correlation_heatmap.png")
    plt.close()
    return corr


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional engineered features."""
    df = df.copy()
    df["water_binder_ratio"] = df["water"] / (df["cement"] + df["fly_ash"] + df["blast_furnace_slag"] + 1e-6)
    df["fine_to_coarse_ratio"] = df["fine_aggregate"] / (df["coarse_aggregate"] + 1e-6)
    df["cement_to_total_agg"] = df["cement"] / (
        df["coarse_aggregate"] + df["fine_aggregate"] + 1e-6
    )
    return df


def select_features(df: pd.DataFrame, threshold: float = 0.1) -> Tuple[pd.DataFrame, List[str]]:
    """Select features based on absolute correlation with target."""
    corr = df.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL)
    selected = corr[abs(corr) >= threshold].index.tolist()
    if not selected:
        selected = corr.index.tolist()
    return df[selected + [TARGET_COL]], selected


def split_dataset(
    df: pd.DataFrame,
    feature_names: List[str],
    scale: bool,
    scaler: StandardScaler | None = None,
) -> DatasetBundle:
    x = df[feature_names].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    if scale:
        scaler = scaler or StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    return DatasetBundle(
        name="",
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
    )


def build_dense_model(input_dim: int, hidden_units: int = 64, dropout: float = 0.2) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(hidden_units, activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_units // 2, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
    return model


def build_cnn_model(
    input_dim: int,
    filters: int = 32,
    kernel_size: int = 2,
    dense_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(input_dim, 1))
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu")(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def evaluate_model(model: keras.Model, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    preds = model.predict(x, verbose=0).ravel()
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)
    return {"rmse": rmse, "r2": r2}


def train_and_evaluate(
    dataset: DatasetBundle,
    model_builder,
    model_name: str,
    epochs: int = 200,
    batch_size: int = 32,
) -> Dict[str, Dict[str, float]]:
    if model_name == "cnn":
        x_train = dataset.x_train[..., None]
        x_val = dataset.x_val[..., None]
        x_test = dataset.x_test[..., None]
    else:
        x_train = dataset.x_train
        x_val = dataset.x_val
        x_test = dataset.x_test

    model = model_builder()
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    ]
    model.fit(
        x_train,
        dataset.y_train,
        validation_data=(x_val, dataset.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    return {
        "train": evaluate_model(model, x_train, dataset.y_train),
        "val": evaluate_model(model, x_val, dataset.y_val),
        "test": evaluate_model(model, x_test, dataset.y_test),
    }


def grid_search_cnn(dataset: DatasetBundle) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    param_grid = {
        "filters": [16, 32],
        "kernel_size": [2, 3],
        "dense_units": [32, 64],
        "dropout": [0.1, 0.3],
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [32, 64],
    }

    best_params = None
    best_val_rmse = float("inf")
    best_metrics = {}

    for filters in param_grid["filters"]:
        for kernel_size in param_grid["kernel_size"]:
            for dense_units in param_grid["dense_units"]:
                for dropout in param_grid["dropout"]:
                    for learning_rate in param_grid["learning_rate"]:
                        for batch_size in param_grid["batch_size"]:
                            builder = lambda: build_cnn_model(
                                input_dim=dataset.x_train.shape[1],
                                filters=filters,
                                kernel_size=kernel_size,
                                dense_units=dense_units,
                                dropout=dropout,
                                learning_rate=learning_rate,
                            )
                            metrics = train_and_evaluate(
                                dataset,
                                builder,
                                model_name="cnn",
                                epochs=150,
                                batch_size=batch_size,
                            )
                            val_rmse = metrics["val"]["rmse"]
                            if val_rmse < best_val_rmse:
                                best_val_rmse = val_rmse
                                best_params = {
                                    "filters": filters,
                                    "kernel_size": kernel_size,
                                    "dense_units": dense_units,
                                    "dropout": dropout,
                                    "learning_rate": learning_rate,
                                    "batch_size": batch_size,
                                }
                                best_metrics = metrics

    return best_params or {}, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 1, variant 7")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to Concrete Compressive Strength dataset (CSV/XLS/XLSX).",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    set_seed()
    df = load_dataset(args.data)
    basic_eda(df, args.artifacts)
    df = clean_data(df)

    corr = correlation_analysis(df, args.artifacts)
    print("\n=== Correlation with target ===")
    print(corr)

    df_engineered = engineer_features(df)
    df_selected, selected_features = select_features(df_engineered)

    original_features = [col for col in df.columns if col != TARGET_COL]

    datasets = {
        "original": (df, original_features, False),
        "original_scaled": (df, original_features, True),
        "engineered": (df_selected, selected_features, False),
        "engineered_scaled": (df_selected, selected_features, True),
    }

    results = {}
    for name, (dataset_df, feature_names, scale) in datasets.items():
        bundle = split_dataset(dataset_df, feature_names, scale=scale)
        bundle.name = name

        dense_builder = lambda: build_dense_model(input_dim=bundle.x_train.shape[1])
        cnn_builder = lambda: build_cnn_model(input_dim=bundle.x_train.shape[1])

        results[name] = {
            "dense": train_and_evaluate(bundle, dense_builder, model_name="dense"),
            "cnn": train_and_evaluate(bundle, cnn_builder, model_name="cnn"),
        }

    results_path = args.artifacts / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")

    # Find best model/dataset by validation RMSE
    best_entry = None
    best_val_rmse = float("inf")
    for dataset_name, models in results.items():
        for model_name, metrics in models.items():
            val_rmse = metrics["val"]["rmse"]
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_entry = (dataset_name, model_name)

    if best_entry is None:
        raise RuntimeError("No model results found.")

    best_dataset_name, best_model_name = best_entry
    print(f"\nBest model before grid search: {best_model_name} on {best_dataset_name}")

    # Grid search only for CNN on best dataset
    best_dataset_df, best_features, best_scaled = datasets[best_dataset_name]
    best_bundle = split_dataset(best_dataset_df, best_features, scale=best_scaled)
    best_bundle.name = best_dataset_name

    best_params, best_metrics = grid_search_cnn(best_bundle)
    print("\nBest CNN params after grid search:")
    print(best_params)
    print("\nMetrics after grid search (train/val/test):")
    print(best_metrics)


if __name__ == "__main__":
    main()