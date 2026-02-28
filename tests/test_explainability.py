import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from banking_ml.explainability import get_shap_values
from banking_ml.pipelines import build_pipeline


class TestGetShapValues:

    @pytest.fixture
    def fitted_xgb(self, X_y_split):
        X, y = X_y_split
        pipeline = build_pipeline(
            XGBClassifier(n_estimators=10, random_state=42, verbosity=0), X
        )
        pipeline.fit(X, y)
        return pipeline, X

    def test_returns_three_items(self, fitted_xgb):
        pipeline, X = fitted_xgb
        result = get_shap_values(pipeline, X)
        assert len(result) == 3

    def test_shap_values_shape(self, fitted_xgb):
        pipeline, X = fitted_xgb
        shap_values, X_sample, feature_names = get_shap_values(pipeline, X)
        assert shap_values.shape[0] == X_sample.shape[0]
        assert shap_values.shape[1] == len(feature_names)

    def test_feature_names_are_strings(self, fitted_xgb):
        pipeline, X = fitted_xgb
        _, _, feature_names = get_shap_values(pipeline, X)
        assert all(isinstance(name, str) for name in feature_names)

    def test_sample_size_reasonable(self, fitted_xgb):
        pipeline, X = fitted_xgb
        _, X_sample, _ = get_shap_values(pipeline, X)
        assert len(X_sample) <= len(X)
        assert len(X_sample) >= 1