import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from banking_ml.pipelines import build_pipeline


class TestBuildPipeline:

    def test_returns_sklearn_pipeline(self, X_y_split):
        X, y = X_y_split
        pipeline = build_pipeline(LogisticRegression(max_iter=200), X)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_preprocessor_step(self, X_y_split):
        X, y = X_y_split
        pipeline = build_pipeline(LogisticRegression(max_iter=200), X)
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_has_model_step(self, X_y_split):
        X, y = X_y_split
        pipeline = build_pipeline(LogisticRegression(max_iter=200), X)
        assert "model" in pipeline.named_steps

    def test_pipeline_fits_without_error(self, X_y_split):
        X, y = X_y_split
        pipeline = build_pipeline(LogisticRegression(max_iter=200), X)
        pipeline.fit(X, y)

    def test_fitted_pipeline_predicts(self, X_y_split):
        X, y = X_y_split
        pipeline = build_pipeline(LogisticRegression(max_iter=200), X)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(X)

    def test_fitted_pipeline_predict_proba(self, X_y_split):
        X, y = X_y_split
        pipeline = build_pipeline(LogisticRegression(max_iter=200), X)
        pipeline.fit(X, y)
        probs = pipeline.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_handles_numeric_only_features(self, X_y_split):
        X, y = X_y_split
        X_numeric = X.select_dtypes(include="number")
        pipeline = build_pipeline(LogisticRegression(max_iter=200), X_numeric)
        pipeline.fit(X_numeric, y)