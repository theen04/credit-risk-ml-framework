"""
Data preprocessing functions for the fintech ML framework.

Pipeline order:
    1. drop_high_missingness     — remove sparse columns
    2. inspect_target            — audit target distribution
    3. encode_binary_target      — classification only
    4. drop_leakage_columns      — remove post-origination features
    5. engineer_features         — domain-specific feature construction
    6. select_core_features      — curated 33-feature classification set
       select_core_features_regression  — 29-feature pricing set
       select_core_neural_features      — full feature set for neural network
    7. split_data                — stratified train/test split
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# =============================================================================
# Constants
# =============================================================================

LEAKAGE_COLUMNS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "out_prncp", "out_prncp_inv",
    "last_pymnt_amnt", "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "hardship_flag", "debt_settlement_flag", "pymnt_plan"
]

ADMINISTRATIVE_COLUMNS = [
    "id", "Unnamed: 0",
    "funded_amnt", "funded_amnt_inv",
    "issue_d", "zip_code", "title", "policy_code",
]

ORDINAL_FEATURES = {
    "emp_length": [
        "< 1 year", "1 year", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
    ],
    "grade": ["A", "B", "C", "D", "E", "F", "G"],
    "sub_grade": [
        "A1", "A2", "A3", "A4", "A5",
        "B1", "B2", "B3", "B4", "B5",
        "C1", "C2", "C3", "C4", "C5",
        "D1", "D2", "D3", "D4", "D5",
        "E1", "E2", "E3", "E4", "E5",
        "F1", "F2", "F3", "F4", "F5",
        "G1", "G2", "G3", "G4", "G5"
    ]
}

NOMINAL_FEATURES = [
    "home_ownership", "verification_status", "purpose",
    "addr_state", "application_type", "initial_list_status",
    "term", "disbursement_method"
]

# Dropped during engineer_features — high cardinality or replaced by engineered equivalents
DROP_FEATURES = [
    "emp_title",        # too high cardinality
    "earliest_cr_line", # replaced by credit_age_months
    "grade",            # Lending Club proprietary score — excluded by design
    "sub_grade",        # Lending Club proprietary score — excluded by design
]

# Classification: 33 curated features + target
# Validated against full 92-feature set in notebook 3 — curated set outperforms by 0.0105 AUC
CORE_FEATURES = [
    # Core loan characteristics
    "loan_amnt", "int_rate", "term", "addr_state",

    # Borrower fundamentals
    "annual_inc", "dti", "emp_length", "home_ownership",
    "verification_status",

    # Credit profile
    "fico_score", "open_acc", "revol_bal", "revol_util",
    "total_acc", "pub_rec", "pub_rec_bankruptcies", "mort_acc",
    "inq_last_6mths",
    "acc_open_past_24mths",
    "num_actv_rev_tl",
    "mths_since_recent_inq",
    "bc_open_to_buy",
    "tot_cur_bal",
    "mths_since_recent_bc",

    # Loan purpose and type
    "purpose", "application_type", "initial_list_status",

    # Engineered features
    "payment_to_income", "derogatory_score", "utilization_composite",
    "rate_to_dti", "credit_age_months", "seasoning_ratio",

    # Target
    "loan_status"
]

# Pricing: 29 curated features + target
# Excludes payment_to_income and rate_to_dti — both derived from int_rate (leakage)
CORE_FEATURES_REGRESSION = [
    # Core loan characteristics
    "loan_amnt", "term",

    # Borrower fundamentals
    "annual_inc", "dti", "emp_length", "home_ownership",
    "verification_status", "all_util",

    # Credit profile
    "fico_score", "open_acc", "revol_bal",
    "total_acc", "pub_rec", "pub_rec_bankruptcies", "mort_acc",
    "inq_last_6mths", "acc_open_past_24mths", "mths_since_recent_inq",
    "bc_open_to_buy", "percent_bc_gt_75",

    # Loan purpose and type
    "purpose", "application_type", "initial_list_status",

    # Engineered features
    "derogatory_score", "utilization_composite", "credit_age_months",

    # Target
    "int_rate"
]

# Neural network: full feature set + target
# No redundancy removal — network learns to suppress irrelevant features internally
# Validated in notebook 3: 92 features outperforms 33-feature set by 0.0024 AUC
CORE_NEURAL_FEATURES = [
    # Core loan characteristics
    "loan_amnt", "int_rate", "term", "installment",
    "addr_state", "purpose", "initial_list_status", "application_type",

    # Borrower fundamentals
    "annual_inc", "emp_length", "home_ownership",
    "verification_status", "dti",

    # Credit profile — depth and history
    "fico_score", "fico_range_low", "fico_range_high", "credit_age_months",
    "open_acc", "total_acc", "mort_acc",
    "num_rev_accts", "num_bc_tl", "num_il_tl",
    "num_sats", "num_bc_sats", "num_op_rev_tl",
    "num_rev_tl_bal_gt_0", "num_actv_bc_tl", "num_actv_rev_tl",

    # Revolving and utilization
    "revol_bal", "revol_util", "bc_util", "all_util",
    "percent_bc_gt_75", "bc_open_to_buy",
    "total_rev_hi_lim", "total_bc_limit",
    "total_bal_ex_mort", "total_bal_il",
    "tot_cur_bal", "tot_hi_cred_lim", "avg_cur_bal",

    # Credit seeking behavior
    "inq_last_6mths", "inq_last_12m", "inq_fi",
    "mths_since_recent_inq", "acc_open_past_24mths", "num_tl_op_past_12m",

    # Derogatory and risk signals
    "delinq_2yrs", "acc_now_delinq",
    "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_120dpd_2m",
    "num_accts_ever_120_pd", "chargeoff_within_12_mths",
    "collections_12_mths_ex_med", "pub_rec", "pub_rec_bankruptcies",
    "tax_liens", "delinq_amnt", "pct_tl_nvr_dlq",

    # Recency and aging signals
    "mths_since_recent_bc", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "mo_sin_old_il_acct", "mths_since_rcnt_il",

    # Installment and trade mix
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
    "il_util", "open_rv_12m", "open_rv_24m",
    "max_bal_bc", "total_cu_tl", "tot_coll_amt",
    "total_il_high_credit_limit",

    # Engineered features
    "loan_to_income", "payment_to_income", "revol_bal_to_income",
    "utilization_composite", "derogatory_score",
    "credit_velocity", "seasoning_ratio",
    "rate_to_dti", "rate_to_income",
    "rate_utilization_stress", "affordability_gap", "fico_dti_risk",

    # Target
    "loan_status"
]


# =============================================================================
# Preprocessing Functions
# =============================================================================

def drop_high_missingness(
    df: pd.DataFrame,
    threshold: float = 0.5,
    verbose: bool = True
) -> pd.DataFrame:
    """Drop columns exceeding a missing value threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Maximum allowed missing rate (0.0 to 1.0).
            Defaults to 0.5.
        verbose (bool): If True, prints dropped columns. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with high-missingness columns removed.

    Example:
        >>> df_clean = drop_high_missingness(df, threshold=0.5)
    """
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()

    if verbose:
        print(f"Dropping {len(cols_to_drop)} columns above {threshold:.0%} missing threshold:")
        for col in cols_to_drop:
            print(f"  {col}: {missing_rate[col]:.1%}")

    return df.drop(columns=cols_to_drop)


def inspect_target(df: pd.DataFrame, target_col: str) -> None:
    """Print value distribution of a target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target column.

    Example:
        >>> inspect_target(df, "loan_status")
    """
    counts = df[target_col].value_counts()
    pct = df[target_col].value_counts(normalize=True).mul(100).round(2)
    summary = pd.DataFrame({"count": counts, "pct": pct})

    print(f"Target column: '{target_col}'")
    print(f"Unique values: {df[target_col].nunique()}")
    print(f"\n{summary.to_string()}")


def encode_binary_target(
    df: pd.DataFrame,
    target_col: str,
    positive_classes: list,
    negative_classes: list,
    verbose: bool = True
) -> pd.DataFrame:
    """Encode a multi-class target column into a binary target.

    Rows not belonging to positive or negative classes are dropped,
    as they represent unresolved or ambiguous outcomes.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target column.
        positive_classes (list): Class labels mapped to 1 (default).
        negative_classes (list): Class labels mapped to 0 (non-default).
        verbose (bool): If True, prints encoding summary. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with binary target and ambiguous rows removed.

    Example:
        >>> df = encode_binary_target(
        ...     df,
        ...     target_col="loan_status",
        ...     positive_classes=["Charged Off", "Default",
        ...         "Does not meet the credit policy. Status:Charged Off"],
        ...     negative_classes=["Fully Paid",
        ...         "Does not meet the credit policy. Status:Fully Paid"]
        ... )
    """
    keep = positive_classes + negative_classes
    df = df[df[target_col].isin(keep)].copy()

    df[target_col] = df[target_col].apply(
        lambda x: 1 if x in positive_classes else 0
    )

    if verbose:
        total = len(df)
        defaults = df[target_col].sum()
        print(f"Retained rows: {total:,}")
        print(f"Default (1): {defaults:,} ({defaults/total:.1%})")
        print(f"Non-default (0): {total-defaults:,} ({(total-defaults)/total:.1%})")

    return df


def drop_leakage_columns(
    df: pd.DataFrame,
    leakage_cols: list = LEAKAGE_COLUMNS,
    admin_cols: list = ADMINISTRATIVE_COLUMNS,
    verbose: bool = True
) -> pd.DataFrame:
    """Remove post-origination leakage and administrative columns.

    Leakage columns contain information only available after a loan
    outcome is known, which would inflate model performance unrealistically.
    Administrative columns carry no predictive signal.

    Args:
        df (pd.DataFrame): Input DataFrame.
        leakage_cols (list): Post-origination columns to drop.
            Defaults to LEAKAGE_COLUMNS.
        admin_cols (list): Administrative columns to drop.
            Defaults to ADMINISTRATIVE_COLUMNS.
        verbose (bool): If True, prints summary. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with leakage and admin columns removed.

    Example:
        >>> df_clean = drop_leakage_columns(df)
    """
    all_cols_to_drop = [c for c in leakage_cols + admin_cols if c in df.columns]
    dropped_leakage = [c for c in leakage_cols if c in df.columns]
    dropped_admin = [c for c in admin_cols if c in df.columns]

    if verbose:
        print(f"Dropping {len(dropped_leakage)} leakage columns")
        print(f"Dropping {len(dropped_admin)} administrative columns")
        print(f"Total dropped: {len(all_cols_to_drop)}")
        print(f"Remaining columns: {df.shape[1] - len(all_cols_to_drop)}")

    return df.drop(columns=all_cols_to_drop)


# =============================================================================
# Feature Selection
# =============================================================================

def select_core_features(
    df: pd.DataFrame,
    core_features: list = CORE_FEATURES,
    verbose: bool = True
) -> pd.DataFrame:
    """Select the curated 33-feature set for credit risk classification.

    Reduces the feature set to loan characteristics, borrower fundamentals,
    credit profile metrics, and engineered features. Validated against the
    full 92-feature set — curated selection outperforms by 0.0105 AUC for
    XGBoost (notebook 3).

    Args:
        df (pd.DataFrame): Input DataFrame after engineer_features.
        core_features (list): Columns to retain. Defaults to CORE_FEATURES.
        verbose (bool): If True, prints summary. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with only core features and target retained.

    Example:
        >>> df_clean = select_core_features(df_clean)
    """
    available = [c for c in core_features if c in df.columns]
    missing = [c for c in core_features if c not in df.columns]

    if verbose:
        print(f"Selecting {len(available) - 1} core features and the target")
        if missing:
            print(f"Requested but not available: {missing}")

    return df[available]


def select_core_features_regression(
    df: pd.DataFrame,
    core_features: list = CORE_FEATURES_REGRESSION,
    verbose: bool = True
) -> pd.DataFrame:
    """Select the curated 29-feature set for interest rate regression.

    Adapted from select_core_features for the pricing task — int_rate
    becomes the target. Excludes payment_to_income and rate_to_dti since
    both are mathematically derived from int_rate and would constitute
    leakage. loan_status is excluded as it's not available at origination.

    Args:
        df (pd.DataFrame): Input DataFrame after engineer_features.
        core_features (list): Columns to retain. Defaults to
            CORE_FEATURES_REGRESSION.
        verbose (bool): If True, prints summary. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with only regression core features retained.

    Example:
        >>> df_clean = select_core_features_regression(df_clean)
    """
    available = [c for c in core_features if c in df.columns]
    missing = [c for c in core_features if c not in df.columns]

    if verbose:
        print(f"Selecting {len(available) - 1} core features and the target")
        if missing:
            print(f"Requested but not available: {missing}")

    return df[available]


def select_core_features_neural(
    df: pd.DataFrame,
    core_features: list = CORE_NEURAL_FEATURES,
    verbose: bool = True
) -> pd.DataFrame:
    """Select the full feature set for neural network classification.

    Retains all engineered and raw features after leakage removal — no
    redundancy removal applied. The neural network learns to suppress
    irrelevant features internally through its weights. Validated in
    notebook 3: 92 features outperforms the 33-feature curated set by
    0.0024 AUC for the neural network architecture.

    Args:
        df (pd.DataFrame): Input DataFrame after engineer_features.
        core_features (list): Columns to retain. Defaults to
            CORE_NEURAL_FEATURES.
        verbose (bool): If True, prints summary. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with full neural feature set retained.

    Example:
        >>> df_clean = select_core_neural_features(df_clean)
    """
    available = [c for c in core_features if c in df.columns]
    missing = [c for c in core_features if c not in df.columns]

    if verbose:
        print(f"Selecting {len(available) - 1} core features and the target")
        if missing:
            print(f"Requested but not available: {missing}")

    return df[available]


# =============================================================================
# Train / Test Split
# =============================================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """Split features and target into stratified training and test sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Fraction of data for test set. Defaults to 0.2.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: X_train, X_test, y_train, y_test

    Example:
        >>> X_train, X_test, y_train, y_test = split_data(X, y)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )