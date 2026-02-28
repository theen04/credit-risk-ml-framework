"""
src/utils.py

Utility functions for the fintech ML framework.

Includes:
- Data loading and cleaning
- Data inspection and summaries
- Feature assessment
- Vintage filtering
- Skewness reporting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union


# -----------------------------
# Data Loading & Cleaning
# -----------------------------
def load_data(filepath: Union[str, Path], low_memory: bool = False, **kwargs) -> pd.DataFrame:
    """Load a dataset from CSV or Excel files."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    ext = filepath.suffix.lower()
    if ext in [".csv", ".gzip"]:
        return pd.read_csv(filepath, low_memory=low_memory, **kwargs)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Lending Club data: convert int_rate & revol_util to numeric."""
    df_clean = df.copy()
    for col in ["int_rate", "revol_util"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(
                df_clean[col].astype(str).str.replace("%", "", regex=False).str.strip(),
                errors="coerce"
            )
    return df_clean


# -----------------------------
# Data Inspection
# -----------------------------
def inspect_dataframe(df: pd.DataFrame) -> None:
    """Print a structured summary of a DataFrame."""
    print("=" * 50)
    print("DATAFRAME SUMMARY")
    print("=" * 50)
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nDtype breakdown:")
    print(df.dtypes.value_counts().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    missing = df.isnull().mean().mul(100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\nColumns with missing values: {len(missing)} of {df.shape[1]}")
    if not missing.empty:
        print(missing.head(20).to_string())
    print("=" * 50)


def save_data_summary_tables(df: pd.DataFrame, target_col: str, output_dir: Path, verbose: bool = True) -> None:
    """Save dataset summary tables: missing values, numeric stats, target distribution, categorical counts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Missing values
    missing = (
        df.isnull().mean().mul(100).round(2)
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("missing_pct", ascending=False)
    )
    missing.to_csv(output_dir / "missing_values.csv", index=False)

    # Numeric feature stats
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    stats = df[numeric_cols].agg(["mean", "std", "min", "max", "skew"]).T.reset_index().rename(columns={"index": "feature"})
    stats = stats.round(4)
    stats.to_csv(output_dir / "feature_stats.csv", index=False)

    # Target distribution
    counts = df[target_col].value_counts()
    pcts = df[target_col].value_counts(normalize=True).mul(100).round(2)
    target_dist = pd.DataFrame({"class": counts.index, "count": counts.values, "pct": pcts.values})
    target_dist.to_csv(output_dir / "target_distribution.csv", index=False)

    # Categorical counts
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_records = []
    for col in cat_cols:
        counts_col = df[col].value_counts()
        pcts_col = df[col].value_counts(normalize=True).mul(100).round(2)
        for val, count, pct in zip(counts_col.index, counts_col.values, pcts_col.values):
            cat_records.append({"feature": col, "value": val, "count": count, "pct": pct})
    pd.DataFrame(cat_records).to_csv(output_dir / "categorical_counts.csv", index=False)

    if verbose:
        print("Data summary tables saved:")
        for fname in ["missing_values.csv", "feature_stats.csv", "target_distribution.csv", "categorical_counts.csv"]:
            print(f"  {output_dir / fname}")


# -----------------------------
# Feature Assessment
# -----------------------------
def assess_features(df: pd.DataFrame, target_col: str, corr_threshold: float = 0.85, variance_threshold: float = 0.01, top_n_low_variance: int = 20, verbose: bool = True) -> dict:
    """Assess numeric features for redundancy, low variance, and zero variance."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    X = df[numeric_cols]

    zero_variance = X.columns[X.var() == 0].tolist()
    variance = X.var().sort_values()
    low_variance = variance[variance < variance_threshold]

    # High correlation pairs
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_pairs = [(col, row, round(upper.loc[row, col], 4))
                       for col in upper.columns for row in upper.index
                       if pd.notna(upper.loc[row, col]) and upper.loc[row, col] >= corr_threshold]
    redundant_pairs = sorted(redundant_pairs, key=lambda x: x[2], reverse=True)

    results = {"redundant_pairs": redundant_pairs, "low_variance": low_variance, "zero_variance": zero_variance}

    if verbose:
        print("=" * 55)
        print("  FEATURE ASSESSMENT REPORT")
        print("=" * 55)
        print(f"\nTotal numeric features assessed: {len(numeric_cols)}")
        print(f"\nZero variance features ({len(zero_variance)}): {zero_variance if zero_variance else 'None'}")
        print(f"\nLow variance features below {variance_threshold} ({len(low_variance)}):")
        print(low_variance.to_string() if not low_variance.empty else "  None")
        print(f"\nHighly correlated pairs above {corr_threshold} ({len(redundant_pairs)}):")
        if redundant_pairs:
            for a, b, corr in redundant_pairs:
                print(f"  {a} <-> {b}: {corr:.4f}")
        else:
            print("  None")
        print("=" * 55)

    return results


# -----------------------------
# Vintage Filtering
# -----------------------------
def filter_by_vintage(df: pd.DataFrame, date_col: str = "issue_d", start_year: int = None, end_year: int = None, date_format: str = "%b-%Y", verbose: bool = True) -> pd.DataFrame:
    """Filter dataset to a specific issuance vintage range."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    if start_year is not None:
        df = df[df[date_col].dt.year >= start_year]
    if end_year is not None:
        df = df[df[date_col].dt.year <= end_year]
    df = df.drop(columns=[date_col])
    if verbose:
        print(f"Vintage filter: {start_year} – {end_year}")
        print(f"Filtered dataset shape: {df.shape}")
    return df


# -----------------------------
# Skewness Reporting
# -----------------------------
def report_skewness(df: pd.DataFrame, target_col: str, thresholds: list = [1.0, 2.0, 5.0], verbose: bool = True) -> pd.Series:
    """Report skewness of numeric features above defined thresholds."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    skewed = df[numeric_cols].skew().sort_values(ascending=False)

    if verbose:
        for threshold in thresholds:
            count = (skewed > threshold).sum()
            print(f"Columns with skewness > {threshold}: {count}")
        print(f"\nFull list above {min(thresholds)}:")
        print(skewed[skewed > min(thresholds)].to_string())

    return skewed