import pytest
import pandas as pd
import numpy as np
from banking_ml.feature_engineering import engineer_features

from banking_ml.preprocessing import drop_leakage_columns

class TestEngineerFeatures:

    def test_creates_fico_score(self, raw_loan_df):
        df = drop_leakage_columns(raw_loan_df, verbose=False)
        result = engineer_features(df, verbose=False)
        assert "fico_score" in result.columns

    def test_fico_score_is_midpoint(self, raw_loan_df):
        df = drop_leakage_columns(raw_loan_df, verbose=False)
        result = engineer_features(df, verbose=False)
        expected = (df["fico_range_low"] + df["fico_range_high"]) / 2
        pd.testing.assert_series_equal(
            result["fico_score"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_creates_burden_ratios(self, raw_loan_df):
        df = drop_leakage_columns(raw_loan_df, verbose=False)
        result = engineer_features(df, verbose=False)
        for col in ["loan_to_income", "payment_to_income", "revol_bal_to_income"]:
            assert col in result.columns

    def test_creates_utilization_composite(self, raw_loan_df):
        df = drop_leakage_columns(raw_loan_df, verbose=False)
        result = engineer_features(df, verbose=False)
        assert "utilization_composite" in result.columns

    def test_creates_derogatory_score(self, raw_loan_df):
        df = drop_leakage_columns(raw_loan_df, verbose=False)
        result = engineer_features(df, verbose=False)
        assert "derogatory_score" in result.columns
        assert (result["derogatory_score"] >= 0).all()

    def test_creates_credit_age_months(self, raw_loan_df):
        df = drop_leakage_columns(raw_loan_df, verbose=False)
        result = engineer_features(df, verbose=False)
        assert "credit_age_months" in result.columns
        assert (result["credit_age_months"] > 0).all()

    def test_creates_rate_features(self, raw_loan_df):
        df = drop_leakage_columns(raw_loan_df, verbose=False)
        result = engineer_features(df, verbose=False)
        for col in ["rate_to_dti", "rate_to_income", "rate_utilization_stress"]:
            assert col in result.columns

    def test_drops_grade_and_subgrade(self, raw_loan_df):
        df = raw_loan_df.copy()
        df["grade"] = "A"
        df["sub_grade"] = "A1"
        df = drop_leakage_columns(df, verbose=False)
        result = engineer_features(df, verbose=False)
        assert "grade" not in result.columns
        assert "sub_grade" not in result.columns

    def test_does_not_modify_original(self, raw_loan_df):
        df = drop_leakage_columns(raw_loan_df, verbose=False)
        original_shape = df.shape
        engineer_features(df, verbose=False)
        assert df.shape == original_shape
