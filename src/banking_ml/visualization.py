# src/fintech_ml/visualization.py
"""
Visualization functions for the fintech ML framework
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional, Union

# -------------------------------------------------------------------
# Color palette for consistent styling
# -------------------------------------------------------------------
COLORS = {
    "train": "#3498db",
    "val": "#e74c3c",
    "default": "#e74c3c",
    "non_default": "#2ecc71",
    "highlight": "#f39c12",
}

# -------------------------------------------------------------------
# Helper function for saving and showing plots
# -------------------------------------------------------------------
def _save_and_show(fig, save_path: Optional[Union[str, Path]] = None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()

# -------------------------------------------------------------------
# ROC Curves from probability arrays
# -------------------------------------------------------------------
def plot_roc_curves_from_probs(
    probs_dict: dict,
    y_test: pd.Series,
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    from sklearn.metrics import roc_curve, auc

    colors = list(COLORS.values())
    fig, ax = plt.subplots(figsize=figsize)

    for (name, probs), color in zip(probs_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="lower right")
    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# ROC Curves from fitted models
# -------------------------------------------------------------------
def plot_roc_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: tuple = (8, 6),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    from sklearn.metrics import roc_curve, roc_auc_score

    colors = list(COLORS.values())
    fig, ax = plt.subplots(figsize=figsize)

    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="lower right")
    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# Training history plot
# -------------------------------------------------------------------
def plot_training_history(
    history: dict,
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, history["train_loss"], color=COLORS["train"], linewidth=2, label="Training Loss")
    ax.plot(epochs, history["val_loss"], color=COLORS["val"], linewidth=2, label="Validation Loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training History — Neural Credit Scorer", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# Class distribution plot
# -------------------------------------------------------------------
def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str,
    figsize: tuple = (6, 4),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    counts = df[target_col].value_counts().sort_index()
    pcts = df[target_col].value_counts(normalize=True).sort_index().mul(100)
    labels = ["Non-Default (0)", "Default (1)"]
    colors = [COLORS["non_default"], COLORS["default"]]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Loan Default Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Loans", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# Missingness plot
# -------------------------------------------------------------------
def plot_missingness(
    df: pd.DataFrame,
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Plot remaining missing value rates as a horizontal bar chart.

    Only columns with missing values are shown, sorted ascending.

    Args:
        df (pd.DataFrame): Input DataFrame.
        figsize (tuple): Figure size. Defaults to (10, 8).
        save_path (str, optional): Path to save the figure.
    """
    missing = df.isnull().mean().mul(100).round(2)
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        print("No missing values found in DataFrame.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn_r(missing.values / 50)  # normalized to 50% ceiling
    ax.barh(missing.index, missing.values, color=colors, edgecolor="white", linewidth=0.5)

    for i, (val, name) in enumerate(zip(missing.values, missing.index)):
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=8)

    ax.set_title(f"Remaining Missingness ({len(missing)} columns)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Missing Rate (%)", fontsize=11)
    ax.set_xlim(0, missing.max() * 1.15)
    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)
    
# -------------------------------------------------------------------
# Numeric feature distributions
# -------------------------------------------------------------------
def plot_numeric_distributions(
    df: pd.DataFrame,
    numeric_cols: list,
    target_col: str,
    figsize: tuple = (16, 12),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    n_cols = 3
    n_rows = -(-len(numeric_cols) // n_cols)
    colors = {0: COLORS["non_default"], 1: COLORS["default"]}
    labels = {0: "Non-Default", 1: "Default"}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for target_val in [0, 1]:
            subset = df[df[target_col] == target_val][col].dropna()
            sns.kdeplot(subset, ax=ax, label=labels[target_val],
                        color=colors[target_val], fill=True, alpha=0.3, linewidth=1.5)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Default Status", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# Default rate by categorical feature
# -------------------------------------------------------------------
def plot_default_rate_by_category(
    df: pd.DataFrame,
    cat_cols: list,
    target_col: str,
    figsize: tuple = (16, 4),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    is_binary = df[target_col].nunique() == 2
    label = "Default Rate" if is_binary else f"Mean {target_col}"

    fig, axes = plt.subplots(1, len(cat_cols), figsize=figsize)
    if len(cat_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        rates = df.groupby(col)[target_col].mean().sort_values(ascending=False)
        rates.plot(kind="bar", ax=ax, color=COLORS["train"], edgecolor="white")
        ax.set_title(f"{label} by {col}", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(axis="x", rotation=45)

    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# Correlation heatmap
# -------------------------------------------------------------------
def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_col: str,
    figsize: tuple = (16, 12),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    numeric_df = df.select_dtypes(include=[float, int])
    if target_col not in numeric_df.columns:
        numeric_df[target_col] = df[target_col]
    corr = numeric_df.corr()
    cols = [target_col] + [c for c in corr.columns if c != target_col]
    corr = corr.loc[cols, cols]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, ax=ax, cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.3, linecolor="white", cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# Feature-target correlation
# -------------------------------------------------------------------
def plot_feature_target_correlation(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 30,
    figsize: tuple = (10, 10),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    is_binary = df[target_col].nunique() == 2
    numeric_df = df.select_dtypes(include=[float, int])
    if target_col not in numeric_df.columns:
        numeric_df[target_col] = df[target_col]

    corr = numeric_df.corr()[target_col].drop(target_col).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n).sort_values()
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in corr.values]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(corr.index, corr.values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, corr.values):
        ax.text(val + (0.001 if val >= 0 else -0.001), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    pos_label = "Higher value → more default risk" if is_binary else f"Higher value → higher {target_col}"
    neg_label = "Higher value → less default risk" if is_binary else f"Higher value → lower {target_col}"
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="#e74c3c"), Patch(facecolor="#2ecc71")], [pos_label, neg_label],
              fontsize=9, loc="lower right")
    ax.set_title(f"Top {top_n} Feature Correlations with {target_col}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Pearson Correlation with Target", fontsize=11)
    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# Target distribution (continuous)
# -------------------------------------------------------------------
def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[target_col].dropna(), bins=50, color=COLORS["train"], edgecolor="white", linewidth=0.5)

    mean_val = df[target_col].mean()
    median_val = df[target_col].median()
    ax.axvline(mean_val, color=COLORS["val"], linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color=COLORS["non_default"], linestyle="--", linewidth=1.5, label=f"Median: {median_val:.2f}")

    ax.set_xlabel(f"{target_col}", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Distribution of {target_col}", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# Residuals plot
# -------------------------------------------------------------------
def plot_residuals(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, color=COLORS["train"])
    ax.axhline(0, color=COLORS["val"], linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted Interest Rate (%)", fontsize=11)
    ax.set_ylabel("Residual", fontsize=11)
    ax.set_title(f"Residual Plot — {model_name}", fontsize=14, fontweight="bold", pad=15)
    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)

# -------------------------------------------------------------------
# Predicted vs actual plot
# -------------------------------------------------------------------
def plot_predicted_vs_actual(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    y_pred = model.predict(X_test)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=COLORS["train"])
    ax.plot([min_val, max_val], [min_val, max_val], color=COLORS["val"], linestyle="--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Interest Rate (%)", fontsize=11)
    ax.set_ylabel("Predicted Interest Rate (%)", fontsize=11)
    ax.set_title(f"Predicted vs Actual — {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    _save_and_show(fig, save_path)