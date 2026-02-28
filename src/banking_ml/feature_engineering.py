#/src/banking_ml/feature_engineering.py

from typing import List, Dict, Callable
import pandas as pd



# Dropped during engineer_features — high cardinality or replaced by engineered equivalents
DROP_FEATURES = [
    "emp_title",        # too high cardinality
    "earliest_cr_line", # replaced by credit_age_months
    "grade",            # Lending Club proprietary score — excluded by design
    "sub_grade",        # Lending Club proprietary score — excluded by design
]

# =============================================================================
# Feature Engineering — Private Helpers
# =============================================================================

def _add_burden_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add debt and payment burden ratio features.

    Captures affordability relationships between loan obligations
    and borrower income that raw features don't express individually.

    New features:
        - loan_to_income: loan amount relative to annual income
        - payment_to_income: monthly installment relative to monthly income
        - revol_bal_to_income: revolving balance relative to annual income
    """
    annual_inc = df["annual_inc"].clip(lower=1)

    df["loan_to_income"] = df["loan_amnt"] / annual_inc
    df["payment_to_income"] = df["installment"] / (annual_inc / 12)
    df["revol_bal_to_income"] = df["revol_bal"] / annual_inc
    
    added_cols = ["loan_to_income", "payment_to_income", "revol_bal_to_income"]
    return df, added_cols


def _add_utilization_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Add a composite credit utilization stress indicator.

    Combines revolving, bankcard, and all-trade utilization into a
    single weighted composite. High utilization across multiple credit
    types is a stronger default signal than any single utilization metric.

    New features:
        - utilization_composite: weighted average of available utilization metrics
    """
    util_cols = {
        "revol_util": 0.4,  # revolving — most predictive for consumer credit
        "bc_util": 0.4,     # bankcard — highly predictive for consumer credit
        "all_util": 0.2     # all trades — broader but less specific
    }

    available = {col: weight for col, weight in util_cols.items() if col in df.columns}

    if available:
        total_weight = sum(available.values())
        df["utilization_composite"] = sum(
            df[col].fillna(df[col].median()) * (weight / total_weight)
            for col, weight in available.items()
        )

    added_cols = ["utilization_composite"]
    return df, added_cols


def _add_derogatory_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a composite derogatory history score.

    Combines delinquency, public record, bankruptcy, and collection
    indicators into a single score. Multiple derogatory marks together
    are a stronger default signal than any individual indicator.

    New features:
        - derogatory_score: weighted sum of derogatory history indicators
    """
    components = {
        "delinq_2yrs": 1.0,               # recent delinquency
        "pub_rec": 1.0,                    # public records
        "pub_rec_bankruptcies": 2.0,       # bankruptcy — double weight
        "collections_12_mths_ex_med": 1.0  # recent collections
    }

    available = {col: weight for col, weight in components.items() if col in df.columns}

    if available:
        df["derogatory_score"] = sum(
            df[col].fillna(0) * weight
            for col, weight in available.items()
        )

    added_cols = ["derogatory_score"]
    return df, added_cols


def _add_credit_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Add credit age and velocity features.

    Combines credit history length with recent credit-seeking behavior
    to capture the 'young credit with high recent activity' risk pattern.

    New features:
        - credit_velocity: ratio of recent new accounts to credit age —
          higher values indicate aggressive recent credit seeking
        - seasoning_ratio: inverse of velocity — longer established credit
          relative to recent activity indicates more stable credit behavior
    """
    if "credit_age_months" in df.columns and "num_tl_op_past_12m" in df.columns:
        credit_age = df["credit_age_months"].clip(lower=1)
        recent_accounts = df["num_tl_op_past_12m"].fillna(0)

        df["credit_velocity"] = recent_accounts / credit_age
        df["seasoning_ratio"] = credit_age / (recent_accounts + 1)

    added_cols = ["credit_velocity", "seasoning_ratio"]
    return df, added_cols

def _add_rate_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add features capturing risk relative to assigned interest rate."""
    added_cols = []

    if "int_rate" not in df.columns:
        return df, added_cols

    int_rate = df["int_rate"]

    if "dti" in df.columns:
        df["rate_to_dti"] = int_rate * df["dti"].clip(lower=0.01)
        added_cols.append("rate_to_dti")

    if "annual_inc" in df.columns:
        df["rate_to_income"] = int_rate / df["annual_inc"].clip(lower=1) * 10000
        added_cols.append("rate_to_income")

    if "utilization_composite" in df.columns:
        df["rate_utilization_stress"] = int_rate * df["utilization_composite"].fillna(0)
        added_cols.append("rate_utilization_stress")

    if "payment_to_income" in df.columns:
        df["affordability_gap"] = df["payment_to_income"] * (1 + int_rate / 100)
        added_cols.append("affordability_gap")

    return df, added_cols

def _add_fico_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FICO-based credit quality features.

    Uses origination FICO range to engineer credit quality indicators
    that complement the grade-free feature set.

    New features:
        - fico_score: midpoint of origination FICO range
        - fico_dti_risk: FICO score inversely weighted by DTI —
          low FICO combined with high DTI is a compounding risk signal
    """
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

        if "dti" in df.columns:
            df["fico_dti_risk"] = df["dti"].clip(lower=0.01) / df["fico_score"].clip(lower=300)
    added_cols = ["fico_dti_risk"]
    return df, added_cols

# Registry: each project maps to a list of functions to apply in order
FEATURE_REGISTRY: Dict[str, List[Callable[[pd.DataFrame], pd.DataFrame]]] = {
    "credit_risk": [
        _add_burden_ratios,
        _add_utilization_composite,
        _add_derogatory_score,
        _add_credit_velocity,
        _add_rate_features,
        _add_fico_features,
    ],
    # future projects can just add new lists of functions
    # "deposit_conversion": [fn1, fn2, ...],
}

# =============================================================================
# Feature Engineering — Public Interface
# =============================================================================

def engineer_features(
    df: pd.DataFrame,
    project_name: str = "credit_risk",
    registry: Dict[str, List[Callable[[pd.DataFrame], pd.DataFrame]]] = FEATURE_REGISTRY,
    reference_date: str = "2020-01-01",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Engineer features dynamically using the registry for the specified project.
    """
    df = df.copy()
    reference_date = pd.Timestamp(reference_date)

    # Credit age
    if "earliest_cr_line" in df.columns:
        df["credit_age_months"] = (
            reference_date - pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
        ).dt.days // 30
        if verbose:
            print("Created: credit_age_months")

    if project_name not in registry:
        raise ValueError(f"Project '{project_name}' not found in the feature registry.")

    # Apply each function
    for func in registry[project_name]:
        df, added_cols = func(df)
        if verbose:
            print(f"Applied: {func.__name__} -> added features: {added_cols}")

    # Drop unwanted features
    cols_to_drop = [c for c in DROP_FEATURES if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    if verbose and cols_to_drop:
        print(f"Dropped {len(cols_to_drop)} features: {cols_to_drop}")
        print(f"Shape after feature engineering: {df.shape}")

    return df
