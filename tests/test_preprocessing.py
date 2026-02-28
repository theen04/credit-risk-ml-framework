import pytest
import pandas as pd
import numpy as np
from banking_ml.preprocessing import (
    drop_high_missingness,
    inspect_target,
    encode_binary_target,
    drop_leakage_columns,
    select_core_features,
    select_core_features_regression,
    select_core_features_neural,
    split_data,
    LEAKAGE_COLUMNS,
    ADMINISTRATIVE_COLUMNS,
    CORE_FEATURES,
    CORE_FEATURES_REGRESSION,
    CORE_NEURAL_FEATURES,
)
from banking_ml.feature_engineering import engineer_features

class TestDropHighMissingness:

    def test_drops_columns_above_threshold(self, raw_loan_df):
        result = drop_high_missingness(raw_loan_df, threshold=0.5, verbose=False)
        assert "hardship_flag" not in result.columns

    def test_retains_columns_below_threshold(self, raw_loan_df):
        result = drop_high_missingness(raw_loan_df, threshold=0.5, verbose=False)
        assert "loan_amnt" in result.columns

    def test_returns_dataframe(self, raw_loan_df):
        result = drop_high_missingness(raw_loan_df, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_threshold_zero_drops_any_missing(self, raw_loan_df):
        result = drop_high_missingness(raw_loan_df, threshold=0.0, verbose=False)
        assert result.isnull().any().sum() == 0

    def test_does_not_modify_original(self, raw_loan_df):
        original_cols = set(raw_loan_df.columns)
        drop_high_missingness(raw_loan_df, verbose=False)
        assert set(raw_loan_df.columns) == original_cols


class TestEncodeBinaryTarget:

    def test_binary_output(self, raw_loan_df):
        result = encode_binary_target(
            raw_loan_df,
            target_col="loan_status",
            positive_classes=["Charged Off"],
            negative_classes=["Fully Paid"],
            verbose=False
        )
        assert set(result["loan_status"].unique()).issubset({0, 1})

    def test_drops_ambiguous_classes(self, raw_loan_df):
        raw_loan_df = raw_loan_df.copy()
        raw_loan_df.loc[0, "loan_status"] = "Current"
        result = encode_binary_target(
            raw_loan_df,
            target_col="loan_status",
            positive_classes=["Charged Off"],
            negative_classes=["Fully Paid"],
            verbose=False
        )
        assert len(result) < len(raw_loan_df)

    def test_positive_class_maps_to_one(self, raw_loan_df):
        result = encode_binary_target(
            raw_loan_df,
            target_col="loan_status",
            positive_classes=["Charged Off"],
            negative_classes=["Fully Paid"],
            verbose=False
        )
        # Every row that was Charged Off should be 1
        assert result["loan_status"].isin([0, 1]).all()

    def test_does_not_modify_original(self, raw_loan_df):
        original_values = raw_loan_df["loan_status"].copy()
        encode_binary_target(
            raw_loan_df,
            target_col="loan_status",
            positive_classes=["Charged Off"],
            negative_classes=["Fully Paid"],
            verbose=False
        )
        pd.testing.assert_series_equal(raw_loan_df["loan_status"], original_values)


class TestDropLeakageColumns:

    def test_drops_leakage_columns(self, raw_loan_df):
        result = drop_leakage_columns(raw_loan_df, verbose=False)
        for col in LEAKAGE_COLUMNS:
            assert col not in result.columns

    def test_drops_administrative_columns(self, raw_loan_df):
        result = drop_leakage_columns(raw_loan_df, verbose=False)
        for col in ADMINISTRATIVE_COLUMNS:
            assert col not in result.columns

    def test_retains_feature_columns(self, raw_loan_df):
        result = drop_leakage_columns(raw_loan_df, verbose=False)
        assert "loan_amnt" in result.columns
        assert "annual_inc" in result.columns
        assert "int_rate" in result.columns

    def test_handles_missing_leakage_columns_gracefully(self, raw_loan_df):
        df = raw_loan_df.drop(columns=["total_pymnt"], errors="ignore")
        result = drop_leakage_columns(df, verbose=False)
        assert isinstance(result, pd.DataFrame)


class TestSelectCoreFeatures:

    def test_retains_target_column(self, engineered_loan_df):
        df = encode_binary_target(
            engineered_loan_df,
            target_col="loan_status",
            positive_classes=["Charged Off"],
            negative_classes=["Fully Paid"],
            verbose=False
        )
        result = select_core_features(df, verbose=False)
        assert "loan_status" in result.columns

    def test_output_columns_subset_of_core_features(self, engineered_loan_df):
        df = encode_binary_target(
            engineered_loan_df,
            target_col="loan_status",
            positive_classes=["Charged Off"],
            negative_classes=["Fully Paid"],
            verbose=False
        )
        result = select_core_features(df, verbose=False)
        assert set(result.columns).issubset(set(CORE_FEATURES))

    def test_handles_missing_columns_gracefully(self, engineered_loan_df):
        # Drop a column that's in CORE_FEATURES
        df = engineered_loan_df.drop(columns=["mort_acc"], errors="ignore")
        df = encode_binary_target(
            df,
            target_col="loan_status",
            positive_classes=["Charged Off"],
            negative_classes=["Fully Paid"],
            verbose=False
        )
        result = select_core_features(df, verbose=False)
        assert isinstance(result, pd.DataFrame)


class TestSelectCoreFeaturesRegression:

    def test_retains_int_rate_as_target(self, engineered_loan_df):
        result = select_core_features_regression(engineered_loan_df, verbose=False)
        assert "int_rate" in result.columns

    def test_excludes_leakage_features(self, engineered_loan_df):
        result = select_core_features_regression(engineered_loan_df, verbose=False)
        assert "payment_to_income" not in result.columns
        assert "rate_to_dti" not in result.columns

    def test_excludes_loan_status(self, engineered_loan_df):
        result = select_core_features_regression(engineered_loan_df, verbose=False)
        assert "loan_status" not in result.columns


class TestSplitData:

    def test_output_shapes(self, X_y_split):
        X, y = X_y_split
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_default_test_size(self, X_y_split):
        X, y = X_y_split
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert abs(len(X_test) / len(X) - 0.2) < 0.05

    def test_stratification_preserves_class_ratio(self, X_y_split):
        X, y = X_y_split
        _, _, y_train, y_test = split_data(X, y)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.02

    def test_reproducibility(self, X_y_split):
        X, y = X_y_split
        split1 = split_data(X, y, random_state=42)
        split2 = split_data(X, y, random_state=42)
        pd.testing.assert_frame_equal(split1[0], split2[0])

    def test_different_seeds_produce_different_splits(self, X_y_split):
        X, y = X_y_split
        X_train_1, _, _, _ = split_data(X, y, random_state=42)
        X_train_2, _, _, _ = split_data(X, y, random_state=99)
        assert not X_train_1.index.equals(X_train_2.index)