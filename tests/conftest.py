import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def raw_loan_df():
    """Minimal synthetic loan DataFrame mimicking raw Lending Club data.
    Covers the key columns used across all three notebooks.
    """
    np.random.seed(42)
    n = 200

    return pd.DataFrame({
        # Loan characteristics
        "loan_amnt": np.random.uniform(1000, 40000, n),
        "int_rate": np.random.uniform(5, 30, n),
        "installment": np.random.uniform(50, 1500, n),
        "term": np.random.choice([" 36 months", " 60 months"], n),
        "purpose": np.random.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "other"], n
        ),
        "application_type": np.random.choice(["Individual", "Joint App"], n),
        "initial_list_status": np.random.choice(["w", "f"], n),

        # Borrower fundamentals
        "annual_inc": np.random.uniform(20000, 200000, n),
        "dti": np.random.uniform(0, 40, n),
        "emp_length": np.random.choice(
            ["< 1 year", "1 year", "5 years", "10+ years", np.nan], n
        ),
        "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n),
        "verification_status": np.random.choice(
            ["Verified", "Source Verified", "Not Verified"], n
        ),

        # Credit profile
        "fico_range_low": np.random.randint(620, 800, n),
        "fico_range_high": np.random.randint(625, 805, n),
        "open_acc": np.random.randint(1, 20, n),
        "revol_bal": np.random.uniform(0, 50000, n),
        "revol_util": np.random.uniform(0, 100, n),
        "bc_util": np.random.uniform(0, 100, n),
        "all_util": np.random.uniform(0, 100, n),
        "total_acc": np.random.randint(5, 40, n),
        "pub_rec": np.random.randint(0, 3, n),
        "pub_rec_bankruptcies": np.random.randint(0, 2, n),
        "mort_acc": np.random.randint(0, 5, n),
        "inq_last_6mths": np.random.randint(0, 5, n),
        "acc_open_past_24mths": np.random.randint(0, 10, n),
        "mths_since_recent_inq": np.random.uniform(0, 24, n),
        "bc_open_to_buy": np.random.uniform(0, 20000, n),
        "percent_bc_gt_75": np.random.uniform(0, 100, n),
        "num_actv_rev_tl": np.random.randint(0, 10, n),
        "tot_cur_bal": np.random.uniform(0, 100000, n),
        "mths_since_recent_bc": np.random.uniform(0, 24, n),
        "delinq_2yrs": np.random.randint(0, 3, n),
        "collections_12_mths_ex_med": np.random.randint(0, 2, n),
        "num_tl_op_past_12m": np.random.randint(0, 5, n),
        "addr_state": np.random.choice(["CA", "NY", "TX", "FL"], n),

        # Credit age
        "earliest_cr_line": np.random.choice(
            ["Jan-2000", "Mar-2005", "Jun-2010", "Sep-2015"], n
        ),

        # Target
        "loan_status": np.random.choice(
            ["Fully Paid", "Charged Off"], n, p=[0.8, 0.2]
        ),

        # Leakage columns
        "total_pymnt": np.random.uniform(0, 40000, n),
        "recoveries": np.random.uniform(0, 1000, n),
        "last_fico_range_high": np.random.randint(625, 805, n),
        "last_fico_range_low": np.random.randint(620, 800, n),

        # Administrative columns
        "id": range(n),
        "url": [f"https://lc.com/{i}" for i in range(n)],
        "funded_amnt": np.random.uniform(1000, 40000, n),

        # Sparse / high missingness column
        "hardship_flag": [np.nan] * n,
    })


@pytest.fixture
def engineered_loan_df(raw_loan_df):
    """Raw DataFrame after leakage removal and feature engineering applied."""
    from banking_ml.preprocessing import drop_leakage_columns
    from banking_ml.feature_engineering import engineer_features
    df = drop_leakage_columns(raw_loan_df, verbose=False)
    df = engineer_features(df, verbose=False)
    return df


@pytest.fixture
def classification_df(engineered_loan_df):
    """Engineered DataFrame after core feature selection for classification."""
    from banking_ml.preprocessing import encode_binary_target, select_core_features
    df = encode_binary_target(
        engineered_loan_df,
        target_col="loan_status",
        positive_classes=["Charged Off", "Default"],
        negative_classes=["Fully Paid"],
        verbose=False
    )
    return select_core_features(df, verbose=False)


@pytest.fixture
def X_y_split(classification_df):
    """Feature matrix and target vector ready for modeling."""
    X = classification_df.drop(columns=["loan_status"])
    y = classification_df["loan_status"]
    return X, y