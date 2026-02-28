"""
src/fintech_ml/explainability.py

Model explainability functions for the FinTech ML framework.

Includes utilities for computing SHAP values and visualizing feature
importance via summary bar charts and beeswarm plots.
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
except ImportError:
    raise ImportError("SHAP is required for explainability. Install with: pip install shap")


# Supported tree-based models for SHAP TreeExplainer
TREE_MODELS = [
    "GradientBoostingClassifier", "HistGradientBoostingClassifier",
    "GradientBoostingRegressor", "HistGradientBoostingRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
]


# ----------------------------------------
# SHAP Computation
# ----------------------------------------

def get_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute SHAP values for a fitted sklearn pipeline.

    Extracts the model step from the pipeline, transforms features
    through the preprocessor, and computes SHAP values on a sampled subset.

    Args:
        model: Fitted sklearn Pipeline with 'preprocessor' and 'model' steps.
        X (pd.DataFrame): Feature DataFrame (unprocessed).
        sample_size (int): Number of rows to sample for SHAP computation. Defaults to 1000.
        random_state (int): Random seed for sampling. Defaults to 42.

    Returns:
        tuple: (shap_values, X_transformed_sample, feature_names)

    Example:
        >>> shap_values, X_sample, feature_names = get_shap_values(gb_pipeline, X_test)
    """
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)

    # Extract pipeline steps
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    # Transform features
    X_transformed = preprocessor.transform(X_sample)
    feature_names = _get_feature_names(preprocessor, X_sample)

    # Determine SHAP explainer
    model_type = type(estimator).__name__
    if model_type in TREE_MODELS:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_transformed)
        # Handle binary classification returning a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(estimator, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    return shap_values, X_transformed, feature_names


def _get_feature_names(preprocessor, X: pd.DataFrame) -> List[str]:
    """
    Extract feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: Fitted sklearn ColumnTransformer.
        X (pd.DataFrame): Original feature DataFrame.

    Returns:
        list: Feature names after transformation.
    """
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(cols)
        elif hasattr(transformer[-1], "get_feature_names_out"):
            names = transformer[-1].get_feature_names_out(cols)
        else:
            names = cols
        feature_names.extend(names)

    return list(feature_names)


# ----------------------------------------
# SHAP Visualizations
# ----------------------------------------

def plot_shap_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a SHAP summary bar chart showing mean absolute feature importance.

    Args:
        shap_values (np.ndarray): SHAP values array.
        X_transformed (np.ndarray): Transformed feature matrix.
        feature_names (list): Feature names corresponding to columns.
        top_n (int): Number of top features to display. Defaults to 20.
        figsize (tuple): Figure size. Defaults to (10, 8).
        save_path (str, optional): Path to save figure. Defaults to None.

    Example:
        >>> plot_shap_summary(shap_values, X_sample, feature_names)
    """
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=True).tail(top_n)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(mean_abs_shap)))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(mean_abs_shap.index, mean_abs_shap.values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, mean_abs_shap.values):
        ax.text(bar.get_width() + mean_abs_shap.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=8)

    ax.set_title(f"Top {top_n} Features by Mean |SHAP| Value", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    sns.despine()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    seed: int = 42
) -> None:
    """
    Plot a SHAP beeswarm plot showing feature impact magnitude and direction.

    Args:
        shap_values (np.ndarray): SHAP values array.
        X_transformed (np.ndarray): Transformed feature matrix.
        feature_names (list): Feature names corresponding to columns.
        top_n (int): Number of top features to display. Defaults to 20.
        figsize (tuple): Figure size. Defaults to (10, 8).
        save_path (str, optional): Path to save figure. Defaults to None.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Example:
        >>> plot_shap_beeswarm(shap_values, X_sample, feature_names)
    """
    # Take top_n features by mean absolute SHAP
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[-top_n:]

    shap_top = shap_values[:, top_indices]
    X_top = X_transformed[:, top_indices]
    names_top = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=figsize)
    # Use the new rng parameter in SHAP (avoids FutureWarning)
    shap.summary_plot(
        shap_top,
        X_top,
        feature_names=names_top,
        show=False,
        plot_size=None,
        color_bar=True
    )

    ax.set_title(f"SHAP Beeswarm — Top {top_n} Features", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()