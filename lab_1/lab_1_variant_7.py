"""
Lab 1 Variant 7: Concrete Compressive Strength Regression Analysis
Exploratory and Regression Analysis using Neural Networks

Dataset: Concrete compressive strength
Models: Fully-connected Dense NN, 1D Convolutional NN
Data transformation: Standardization
Metrics: RMSE, R²
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

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
import random
random.seed(RANDOM_STATE)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_data(file_path: str) -> pd.DataFrame:
    """Load the concrete dataset."""
    df = pd.read_csv(file_path, index_col=0)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print("=" * 60)
    print("DATASET LOADED")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    """Display descriptive statistics."""
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(df.describe())

    print("\n" + "-" * 60)
    print("MISSING VALUES")
    print("-" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])

    if missing.sum() == 0:
        print("No missing values found!")


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot univariate distributions for all features."""
    print("\n" + "=" * 60)
    print("GENERATING UNIVARIATE VISUALIZATIONS")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Histograms with KDE
    for col in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram with KDE
        ax1.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(col)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of {col}')
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(df[col].dropna(), vert=True)
        ax2.set_ylabel(col)
        ax2.set_title(f'Box Plot of {col}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'dist_{col}.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(df.columns)} distribution plots to {output_dir}")


def plot_correlations(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot correlation heatmap and scatter plots."""
    print("\n" + "=" * 60)
    print("GENERATING MULTIVARIATE VISUALIZATIONS")
    print("=" * 60)

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved correlation heatmap")

    # Correlation with target
    target = 'concrete_compressive_strength'
    if target in df.columns:
        correlations_with_target = df.corr()[target].sort_values(ascending=False)
        print(f"\nCorrelations with {target}:")
        print(correlations_with_target)

        # Scatter plots for top features
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

            # Add correlation coefficient
            corr = df[[feature, target]].corr().iloc[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5))

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / 'scatter_plots.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved scatter plots for {n_features} features")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by handling missing values and duplicates."""
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    df_clean = df.copy()

    # Handle missing values - fill with median
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val:.2f}")

    # Remove duplicates
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {n_duplicates} duplicate rows")
    else:
        print("No duplicate rows found")

    print(f"Final dataset shape: {df_clean.shape}")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific engineered features."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df_eng = df.copy()

    # Water-cement ratio (critical for concrete strength)
    df_eng['water_cement_ratio'] = df_eng['water'] / (df_eng['cement'] + 1e-6)

    # Total aggregate
    df_eng['total_aggregate'] = df_eng['coarse_aggregate'] + df_eng['fine_aggregate']

    # Total binder content (cementitious materials)
    df_eng['binder_content'] = (df_eng['cement'] +
                                  df_eng['blast_furnace_slag'] +
                                  df_eng['fly_ash'])

    # Water-binder ratio
    df_eng['water_binder_ratio'] = df_eng['water'] / (df_eng['binder_content'] + 1e-6)

    # Age squared (non-linear age effect)
    df_eng['age_squared'] = df_eng['age'] ** 2

    # Age log transform
    df_eng['age_log'] = np.log(df_eng['age'] + 1)

    new_features = ['water_cement_ratio', 'total_aggregate', 'binder_content',
                    'water_binder_ratio', 'age_squared', 'age_log']
    print(f"Created {len(new_features)} new features:")
    for feat in new_features:
        print(f"  - {feat}")

    print(f"Total features: {df_eng.shape[1]}")
    return df_eng


def select_features(X: pd.DataFrame, y: pd.Series, top_fraction: float = 0.7) -> Tuple[pd.DataFrame, List[str]]:
    """Select top features based on mutual information."""
    print("\n" + "=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)

    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    print(f"\nMutual Information Scores:")
    print(mi_scores)

    # Select top features
    n_features = max(1, int(len(mi_scores) * top_fraction))
    selected_features = mi_scores.head(n_features).index.tolist()

    print(f"\nSelected top {n_features} features ({top_fraction*100:.0f}%):")
    for feat in selected_features:
        print(f"  - {feat} (MI: {mi_scores[feat]:.4f})")

    return X[selected_features], selected_features


def prepare_datasets(df: pd.DataFrame, target_col: str) -> Dict:
    """Prepare 4 datasets: original, original_scaled, selected, selected_scaled."""
    print("\n" + "=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)

    # Separate features and target
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    print(f"Total samples: {len(y)}")
    print(f"Total features (with engineered): {X.shape[1]}")

    # Feature selection
    X_selected, selected_features = select_features(X, y, top_fraction=0.7)

    # Create 4 dataset versions
    datasets = {}

    for name, features in [('original', X), ('selected', X_selected)]:
        # Split into train/val/test (70/15/15)
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, y, test_size=0.15, random_state=RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE
        )

        # Unscaled version
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

        # Scaled version
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

    print(f"\nCreated 4 datasets:")
    for name, data in datasets.items():
        print(f"  - {name}: {data['X_train'].shape[1]} features, "
              f"train={len(data['y_train'])}, val={len(data['y_val'])}, "
              f"test={len(data['y_test'])}")

    return datasets


def build_dense_model(input_dim: int, hidden_units: int = 128,
                      dropout: float = 0.3, learning_rate: float = 0.001) -> keras.Model:
    """Build fully-connected dense neural network."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(hidden_units // 2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(hidden_units // 4, activation='relu'),
        layers.Dropout(dropout * 0.7),
        layers.Dense(1)  # Linear activation for regression
    ], name='Dense_Model')

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model


def build_conv1d_model(input_dim: int, filters: int = 64, kernel_size: int = 3,
                       dropout: float = 0.3, learning_rate: float = 0.001) -> keras.Model:
    """Build 1D Convolutional neural network."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(filters, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(filters * 2, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
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
    """Calculate regression metrics."""
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
    """Train a neural network model."""
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


def train_and_evaluate_all(datasets: Dict, epochs: int = 200) -> Dict:
    """Train both models on all 4 datasets."""
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION")
    print("=" * 60)

    results = {}
    all_models = {}

    for dataset_name, dataset in datasets.items():
        print(f"\n{'-' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'-' * 60}")

        X_train = dataset['X_train']
        X_val = dataset['X_val']
        X_test = dataset['X_test']
        y_train = dataset['y_train']
        y_val = dataset['y_val']
        y_test = dataset['y_test']

        input_dim = X_train.shape[1]
        results[dataset_name] = {}
        all_models[dataset_name] = {}

        # Train Dense model
        print(f"\nTraining Dense model...")
        dense_model = build_dense_model(input_dim)
        train_model(dense_model, X_train, y_train, X_val, y_val,
                   epochs=epochs, verbose=0)

        # Evaluate Dense
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

        # Train Conv1D model (reshape data)
        print(f"\nTraining Conv1D model...")
        X_train_conv = X_train.reshape(-1, input_dim, 1)
        X_val_conv = X_val.reshape(-1, input_dim, 1)

        conv_model = build_conv1d_model(input_dim)
        train_model(conv_model, X_train_conv, y_train, X_val_conv, y_val,
                   epochs=epochs, verbose=0)

        # Evaluate Conv1D
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
    """Find the best model based on validation RMSE."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # Create comparison table
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

    print("\nAll Models Performance (sorted by validation RMSE):")
    print(comparison_df.to_string(index=False))

    # Find best
    best = comparison_df.iloc[0]
    best_dataset = best['dataset']
    best_model_name = best['model']

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_model_name} on {best_dataset}")
    print(f"{'=' * 60}")
    print(f"Validation RMSE: {best['val_rmse']:.4f}")
    print(f"Validation R²:   {best['val_r2']:.4f}")

    # Evaluate on test set
    dataset = datasets[best_dataset]
    model = all_models[best_dataset][best_model_name]

    X_test = dataset['X_test']
    y_test = dataset['y_test']

    if best_model_name == 'conv1d':
        X_test = X_test.reshape(-1, X_test.shape[1], 1)

    y_test_pred = model.predict(X_test, verbose=0).flatten()
    test_metrics = evaluate_metrics(y_test, y_test_pred)

    print(f"\nTest Set Performance:")
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
    """Perform grid search for hyperparameter tuning."""
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

    # Generate all combinations
    import itertools
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    print(f"Testing {len(combinations)} parameter combinations...")

    for idx, values in enumerate(combinations, 1):
        params = dict(zip(param_names, values))

        # Build model with current params
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

        # Train
        train_model(model, X_train, y_train, X_val, y_val,
                   epochs=epochs, batch_size=32, verbose=0)

        # Evaluate
        y_val_pred = model.predict(X_val, verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)

        print(f"  [{idx}/{len(combinations)}] {params} -> RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

        # Track best
        if val_rmse < best_score:
            best_score = val_rmse
            best_params = params
            best_model = model

    print(f"\n{'=' * 60}")
    print(f"BEST PARAMETERS FOUND")
    print(f"{'=' * 60}")
    print(f"Parameters: {best_params}")
    print(f"Validation RMSE: {best_score:.4f}")

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val, verbose=0).flatten()
    val_metrics = evaluate_metrics(y_val, y_val_pred)

    return {
        'best_params': best_params,
        'best_model': best_model,
        'val_metrics': val_metrics
    }


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print(" " * 20 + "LAB 1 - VARIANT 7")
    print(" " * 10 + "Concrete Compressive Strength Regression")
    print("=" * 70)

    # Setup
    output_dir = Path('lab_1_artifacts')
    output_dir.mkdir(exist_ok=True)

    # Step 1-2: Load data
    df = load_data('V7_dataset.csv')

    # Step 3: Exploratory Data Analysis
    descriptive_stats(df)
    plot_distributions(df, output_dir)
    plot_correlations(df, output_dir)

    # Step 3c: Clean data
    df_clean = clean_data(df)

    # Step 3e: Feature engineering
    df_eng = engineer_features(df_clean)

    # Step 4: Prepare 4 datasets
    datasets = prepare_datasets(df_eng, target_col='concrete_compressive_strength')

    # Step 5: Train and evaluate all models
    results, all_models = train_and_evaluate_all(datasets, epochs=200)

    # Step 6: Find best model and evaluate on test set
    best_info = find_best_model(results, datasets, all_models)

    # Step 7: Grid search on best model
    grid_results = grid_search(
        best_info['dataset'],
        best_info['model_name'],
        epochs=150
    )

    # Step 8: Evaluate grid-searched model on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    X_test = best_info['dataset']['X_test']
    y_test = best_info['dataset']['y_test']

    if best_info['model_name'] == 'conv1d':
        X_test = X_test.reshape(-1, X_test.shape[1], 1)

    y_test_pred = grid_results['best_model'].predict(X_test, verbose=0).flatten()
    test_metrics_after = evaluate_metrics(y_test, y_test_pred)

    print(f"\nBefore Grid Search:")
    print(f"  Test RMSE: {best_info['test_metrics']['rmse']:.4f}")
    print(f"  Test R²:   {best_info['test_metrics']['r2']:.4f}")

    print(f"\nAfter Grid Search:")
    print(f"  Test RMSE: {test_metrics_after['rmse']:.4f}")
    print(f"  Test R²:   {test_metrics_after['r2']:.4f}")

    improvement_rmse = ((best_info['test_metrics']['rmse'] - test_metrics_after['rmse']) /
                        best_info['test_metrics']['rmse'] * 100)
    improvement_r2 = ((test_metrics_after['r2'] - best_info['test_metrics']['r2']) /
                      abs(best_info['test_metrics']['r2']) * 100)

    print(f"\nImprovement:")
    print(f"  RMSE: {improvement_rmse:+.2f}%")
    print(f"  R²:   {improvement_r2:+.2f}%")

    # Step 9: Save results and conclusions
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

    # Save summary
    with open(output_dir / 'results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_dir}/results_summary.json")
    print(f"Plots saved to {output_dir}/")
    print(f"{'=' * 60}")

    # Print conclusions
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print(f"""
1. Best Model: {best_info['model_name'].upper()} neural network
2. Best Dataset: {best_info['dataset_name']}
3. Final Test Performance:
   - RMSE: {test_metrics_after['rmse']:.4f} MPa
   - R² Score: {test_metrics_after['r2']:.4f}
   - MAE: {test_metrics_after['mae']:.4f} MPa

4. Impact of Standardization:
   - Standardization {'improved' if 'scaled' in best_info['dataset_name'] else 'did not improve'} model performance

5. Impact of Feature Selection:
   - {'Selected features performed better' if 'selected' in best_info['dataset_name'] else 'All features performed better'}

6. Impact of Hyperparameter Tuning:
   - RMSE improvement: {improvement_rmse:+.2f}%
   - R² improvement: {improvement_r2:+.2f}%

7. Model Architecture:
   - {'1D CNN captured feature patterns better than Dense NN' if best_info['model_name'] == 'conv1d' else 'Dense NN performed better than 1D CNN'}
   - Optimal hyperparameters: {grid_results['best_params']}

8. Recommendations:
   - The model can predict concrete strength with {test_metrics_after['r2']*100:.1f}% variance explained
   - Average prediction error: ±{test_metrics_after['mae']:.2f} MPa
   - {'Excellent' if test_metrics_after['r2'] > 0.9 else 'Good' if test_metrics_after['r2'] > 0.8 else 'Moderate'} predictive performance for practical applications
    """)

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
