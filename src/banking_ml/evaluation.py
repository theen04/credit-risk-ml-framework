"""
src/fintech_ml/evaluation.py

Model evaluation functions for the FinTech ML framework.

Provides both regression and classification evaluation utilities,
including financial metrics like Gini coefficient and KS statistic.
"""

from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error
)


# ----------------------------------------
# Regression Evaluation
# ----------------------------------------

def evaluate_regressor(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a regression model on test data.

    Computes RMSE, MAE, R², and Max Error. Designed to complement
    `evaluate_classifier` for consistent pipeline evaluation.

    Args:
        model: Fitted sklearn pipeline or estimator.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True continuous target values.
        model_name (str): Display name for the model. Defaults to 'Model'.
        verbose (bool): If True, prints evaluation report. Defaults to True.

    Returns:
        dict: Metrics dictionary with keys: model, rmse, mae, r2, max_error.

    Example:
        >>> metrics = evaluate_regressor(xgb_pipeline, X_test, y_test, "XGBoost")
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    me = max_error(y_test, y_pred)

    metrics = {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "max_error": round(me, 4),
    }

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  {model_name} — Regression Evaluation")
        print(f"{'=' * 55}")
        print(f"  RMSE:       {rmse:.4f}  (lower is better)")
        print(f"  MAE:        {mae:.4f}  (lower is better)")
        print(f"  R²:         {r2:.4f}  (higher is better)")
        print(f"  Max Error:  {me:.4f}")
        print(f"{'=' * 55}\n")

    return metrics


def compare_regressors(metrics_list: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple regression models in a summary table.

    Args:
        metrics_list (list): List of metrics dictionaries returned by `evaluate_regressor`.

    Returns:
        pd.DataFrame: Models ranked by RMSE ascending.

    Example:
        >>> summary = compare_regressors([ridge_metrics, gb_metrics, xgb_metrics])
    """
    df = pd.DataFrame(metrics_list).set_index("model")
    return df.sort_values("rmse", ascending=True)


# ----------------------------------------
# Classification Evaluation
# ----------------------------------------

def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a binary classifier with standard and financial metrics.

    Metrics computed:
    - ROC-AUC: overall discrimination ability
    - Gini coefficient: 2 * AUC - 1, standard in credit risk
    - KS statistic: maximum separation between default/non-default score distributions
    - Average precision: area under precision-recall curve
    - Brier score: probability calibration quality
    - Confusion matrix and classification report (printed if verbose)

    Args:
        model: Fitted sklearn-compatible pipeline or estimator.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True binary labels (0/1).
        model_name (str): Display name for the model. Defaults to "Model".
        verbose (bool): If True, prints detailed evaluation report. Defaults to True.

    Returns:
        dict: Metrics dictionary with keys:
            model, roc_auc, gini, ks_statistic, avg_precision, brier_score.

    Example:
        >>> metrics = evaluate_classifier(lr_pipeline, X_test, y_test, "Logistic Regression")
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1
    avg_precision = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    ks_stat = _compute_ks(
        default_scores=y_prob[y_test == 1],
        non_default_scores=y_prob[y_test == 0]
    )

    metrics = {
        "model": model_name,
        "roc_auc": round(auc, 4),
        "gini": round(gini, 4),
        "ks_statistic": round(ks_stat, 4),
        "avg_precision": round(avg_precision, 4),
        "brier_score": round(brier, 4),
    }

    if verbose:
        print("=" * 55)
        print(f"  {model_name} — Classification Evaluation")
        print("=" * 55)
        print(f"  ROC-AUC:           {auc:.4f}")
        print(f"  Gini Coefficient:  {gini:.4f}")
        print(f"  KS Statistic:      {ks_stat:.4f}")
        print(f"  Avg Precision:     {avg_precision:.4f}")
        print(f"  Brier Score:       {brier:.4f}  (lower is better)")
        print("-" * 55)
        print(classification_report(y_test, y_pred, target_names=["Non-Default", "Default"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 55)

    return metrics


def _compute_ks(default_scores: np.ndarray, non_default_scores: np.ndarray) -> float:
    """
    Compute the KS (Kolmogorov-Smirnov) statistic between default
    and non-default score distributions.

    The KS statistic measures the maximum separation between the
    cumulative distributions of default and non-default predicted probabilities.
    Higher values indicate better model discrimination.

    Args:
        default_scores (np.ndarray): Predicted probabilities for default cases.
        non_default_scores (np.ndarray): Predicted probabilities for non-default cases.

    Returns:
        float: KS statistic (0–1).
    """
    thresholds = np.linspace(0, 1, 100)
    tpr = [np.mean(default_scores >= t) for t in thresholds]
    fpr = [np.mean(non_default_scores >= t) for t in thresholds]
    return float(np.max(np.abs(np.array(tpr) - np.array(fpr))))


def compare_classifiers(metrics_list: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple classifier evaluation results in a summary table.

    Args:
        metrics_list (list): List of metric dictionaries returned by `evaluate_classifier`.

    Returns:
        pd.DataFrame: Summary table sorted by ROC-AUC descending.

    Example:
        >>> summary = compare_classifiers([lr_metrics, gb_metrics])
    """
    df = pd.DataFrame(metrics_list).set_index("model")
    return df.sort_values("roc_auc", ascending=False)