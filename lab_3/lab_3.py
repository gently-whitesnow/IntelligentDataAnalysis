"""Lab 3: Image classification with CNN and transfer learning."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RANDOM_STATE = 42
IMAGE_SIZE = (180, 180)


@dataclass
class DatasetPaths:
    train: Path
    val: Path
    test: Path


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_class_dirs(data_dir: Path) -> List[Path]:
    class_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in {data_dir}")
    return sorted(class_dirs)


def split_dataset(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DatasetPaths:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    class_dirs = list_class_dirs(data_dir)

    for class_dir in class_dirs:
        images = [p for p in class_dir.iterdir() if p.is_file()]
        if not images:
            raise ValueError(f"No images found in class folder {class_dir}")
        random.shuffle(images)
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        split_map = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in split_map.items():
            target_dir = output_dir / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, target_dir / img_path.name)

    return DatasetPaths(
        train=output_dir / "train",
        val=output_dir / "val",
        test=output_dir / "test",
    )


def build_base_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="base_cnn")
    return model


def build_transfer_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    unfreeze_top_layers: int = 0,
) -> keras.Model:
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    if unfreeze_top_layers > 0:
        for layer in base_model.layers[:-unfreeze_top_layers]:
            layer.trainable = False
    else:
        base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="transfer_mobilenetv2")
    return model


def build_generators(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    batch_size: int,
    augment: bool,
) -> Tuple[keras.utils.Sequence, keras.utils.Sequence, keras.utils.Sequence]:
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_STATE,
    )
    val_gen = test_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=RANDOM_STATE,
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def plot_history(history: keras.callbacks.History, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="val")
    ax[0].set_title(f"{title} accuracy")
    ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="val")
    ax[1].set_title(f"{title} loss")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_model(model: keras.Model, generator: keras.utils.Sequence) -> Dict[str, float]:
    predictions = model.predict(generator, verbose=0)
    y_true = generator.classes
    y_pred = predictions.argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    return {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def train_model(
    model: keras.Model,
    train_gen: keras.utils.Sequence,
    val_gen: keras.utils.Sequence,
    epochs: int,
    learning_rate: float,
) -> keras.callbacks.History:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
    )
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 3: Image classification")
    parser.add_argument("--data", type=Path, required=True, help="Path to folder with class subfolders.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--unfreeze-top", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()

    data_dir = args.data
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    artifacts_dir = args.artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = args.splits or data_dir.parent / "splits"
    dataset_paths = split_dataset(
        data_dir=data_dir,
        output_dir=splits_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_gen, val_gen, test_gen = build_generators(
        dataset_paths.train,
        dataset_paths.val,
        dataset_paths.test,
        batch_size=args.batch_size,
        augment=True,
    )
    num_classes = train_gen.num_classes
    input_shape = IMAGE_SIZE + (3,)

    base_model = build_base_cnn(input_shape, num_classes)
    base_history = train_model(
        base_model, train_gen, val_gen, epochs=args.epochs, learning_rate=args.learning_rate
    )
    plot_history(base_history, "Base CNN", artifacts_dir / "base_cnn_history.png")
    base_metrics = evaluate_model(base_model, test_gen)

    transfer_train_gen, transfer_val_gen, transfer_test_gen = build_generators(
        dataset_paths.train,
        dataset_paths.val,
        dataset_paths.test,
        batch_size=args.batch_size,
        augment=False,
    )
    transfer_model = build_transfer_model(
        input_shape=input_shape,
        num_classes=num_classes,
        unfreeze_top_layers=args.unfreeze_top,
    )
    transfer_history = train_model(
        transfer_model,
        transfer_train_gen,
        transfer_val_gen,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    plot_history(transfer_history, "Transfer MobileNetV2", artifacts_dir / "transfer_history.png")
    transfer_metrics = evaluate_model(transfer_model, transfer_test_gen)

    results = {
        "base_cnn": base_metrics,
        "transfer_mobilenetv2": transfer_metrics,
        "class_indices": train_gen.class_indices,
    }
    with (artifacts_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print("Base CNN:", base_metrics)
    print("Transfer MobileNetV2:", transfer_metrics)


if __name__ == "__main__":
    main()