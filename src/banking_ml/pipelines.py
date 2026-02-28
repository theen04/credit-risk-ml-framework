"""
src/fintech_ml/pipelines.py

Pipeline construction utilities for the FinTech ML framework.

Includes:
- ColumnTransformer for numeric, ordinal, and nominal features
- Full sklearn pipeline builder combining preprocessing and model
"""

from typing import Any
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from banking_ml.preprocessing import ORDINAL_FEATURES, NOMINAL_FEATURES


# ----------------------------------------
# Preprocessing Pipeline
# ----------------------------------------

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric, ordinal, and nominal features.

    Numeric features: median imputation + standard scaling  
    Ordinal features: most-frequent imputation + ordinal encoding  
    Nominal features: most-frequent imputation + one-hot encoding  

    Args:
        df (pd.DataFrame): Feature DataFrame (excluding target column).

    Returns:
        ColumnTransformer: Unfitted preprocessor for pipeline integration.

    Example:
        >>> preprocessor = build_preprocessor(X_train)
    """
    # Identify columns by type
    ordinal_cols = [c for c in ORDINAL_FEATURES.keys() if c in df.columns]
    nominal_cols = [c for c in NOMINAL_FEATURES if c in df.columns]
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ordinal_cols + nominal_cols
    ]

    # Pipelines per feature type
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[ORDINAL_FEATURES[c] for c in ordinal_cols],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("ordinal", ordinal_pipeline, ordinal_cols),
            ("nominal", nominal_pipeline, nominal_cols)
        ],
        remainder="drop"
    )

    return preprocessor


# ----------------------------------------
# Full Pipeline Builder
# ----------------------------------------

def build_pipeline(model: Any, df: pd.DataFrame) -> Pipeline:
    """
    Build a full sklearn Pipeline with preprocessing and model steps.

    Args:
        model (Any): Unfitted sklearn-compatible estimator.
        df (pd.DataFrame): Feature DataFrame to infer column types.

    Returns:
        Pipeline: Full pipeline with preprocessing and model.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> pipeline = build_pipeline(LogisticRegression(), X_train)
        >>> pipeline.fit(X_train, y_train)
    """
    preprocessor = build_preprocessor(df)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    

    return pipeline