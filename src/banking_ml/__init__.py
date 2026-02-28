#src/fintech_ml/__init__.py

from .utils import *
from .preprocessing import *
from .pipelines import *
from .models import *
from .evaluation import *
from .visualization import *
from .explainability import *
from .feature_engineering import *

# Utilities
__all__ = [
    "load_data",
    "inspect_dataframe",
    "save_data_summary_tables",
    "clean_raw_data",
    "assess_features",
    "filter_by_vintage",
    "report_skewness",
]

# Preprocessing
__all__ = [
    "split_data",
    "drop_high_missingness",
    "inspect_target",
    "encode_binary_target",
    "drop_leakage_columns",
    "engineer_features",
    "select_core_features",
    "select_core_features_regression",
    "select_core_features_neural",
]

# Feature Engineering
__all__ = [
    "engineer_features",
]

# Pipelines
__all__ = [
    "build_preprocessor",
    "build_pipeline",
]

# Models
__all__ = [
    "get_xgb_param_grid",
    "tune_model",
    "display_best_params",
    "get_xgb_regression_param_grid",
    "NeuralCreditScorer",
    "NeuralClassifierWrapper",
    "train_neural_model",
]

# Evaluation
__all__ = [
    "evaluate_classifier",
    "compare_classifiers",
    "evaluate_regressor",
    "compare_regressors",
]

# Explainability
__all__ = [
    "get_shap_values",
    "plot_shap_summary",
    "plot_shap_beeswarm",
]

# Visualization
__all__ = [
    "plot_class_distribution",
    "plot_numeric_distributions",
    "plot_default_rate_by_category",
    "plot_correlation_heatmap",
    "plot_missingness",
    "plot_feature_target_correlation",
    "plot_roc_curves",
    "plot_target_distribution",
    "plot_residuals",
    "plot_predicted_vs_actual",
    "plot_training_history",
    "plot_roc_curves_from_probs",

]

