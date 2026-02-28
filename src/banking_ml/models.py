"""
src/fintech_ml/models.py

Model definition, configuration, and tuning utilities for the FinTech ML framework.

Includes:
- PyTorch feedforward classifier for tabular credit data
- Sklearn wrapper for neural models
- XGBoost hyperparameter grids
- RandomizedSearchCV tuning utility
"""

from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------
# PyTorch Neural Model Wrapper
# ----------------------------------------

class NeuralClassifierWrapper:
    """Sklearn-compatible wrapper for PyTorch neural networks.

    Allows usage of PyTorch models in sklearn pipelines and evaluation
    utilities by implementing `predict` and `predict_proba`.

    Args:
        model (nn.Module): Trained PyTorch model.
        device (str): Device the model is located on.

    Example:
        >>> wrapper = NeuralClassifierWrapper(model, device)
        >>> metrics = evaluate_classifier(wrapper, X_test_transformed, y_test)
    """
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

# ----------------------------------------
# PyTorch Feedforward Neural Network
# ----------------------------------------

class NeuralCreditScorer(nn.Module):
    """Feedforward neural network for binary credit classification.

    Three hidden layers with batch normalization and dropout. Designed
    for preprocessed tabular credit data.

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list[int]): Sizes of hidden layers. Defaults to [256, 128, 64].
        dropout_rates (list[float]): Dropout per hidden layer. Defaults to [0.3, 0.3, 0.2].

    Example:
        >>> model = NeuralCreditScorer(input_dim=103)
        >>> model = NeuralCreditScorer(input_dim=103, hidden_dims=[512, 256, 128], dropout_rates=[0.4, 0.3, 0.2])
    """
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, dropout_rates: Optional[List[float]] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        dropout_rates = dropout_rates or [0.3, 0.3, 0.2]

        layers = []
        in_dim = input_dim
        for h_dim, d_rate in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(d_rate)
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))  # logits output
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# ----------------------------------------
# Training Utility for Neural Network
# ----------------------------------------

def train_neural_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 2048,
    learning_rate: float = 1e-4,
    pos_weight: Optional[float] = None,
    patience: int = 10,
    random_state: int = 42,
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train a PyTorch binary classifier with BCEWithLogitsLoss and early stopping.

    Args:
        model (nn.Module): Initialized PyTorch model.
        X_train, X_val (np.ndarray): Feature matrices.
        y_train, y_val (np.ndarray): Binary labels.
        epochs (int): Max training epochs.
        batch_size (int): Mini-batch size.
        learning_rate (float): Adam optimizer learning rate.
        pos_weight (float, optional): Weight for positive class; computed if None.
        patience (int): Early stopping patience.
        random_state (int): Seed for reproducibility.
        device (str, optional): Training device; auto-detect if None.
        verbose (bool): Print progress if True.

    Returns:
        dict: Training history with keys 'train_loss' and 'val_loss'.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    if verbose:
        print(f"Training on device: {device}")

    # Class weight
    if pos_weight is None:
        pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t.to(device)), y_val_t.to(device)).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and verbose:
                print(f"Early stopping at epoch {epoch + 1} (best val loss: {best_val_loss:.4f})")
                break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if best_weights:
        model.load_state_dict(best_weights)
        if verbose:
            print("Restored best model weights.")

    return history

# ----------------------------------------
# XGBoost Parameter Grids
# ----------------------------------------

def get_xgb_param_grid() -> dict:
    """Hyperparameter grid for XGBClassifier."""
    return {
        "model__n_estimators": [600, 800, 1000],
        "model__learning_rate": [0.03, 0.05, 0.07],
        "model__max_depth": [5, 6, 7],
        "model__subsample": [0.85, 0.9, 0.95],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__min_child_weight": [2, 3, 4],
        "model__gamma": [0.2, 0.3, 0.4],
        "model__reg_alpha": [0.05, 0.1, 0.15],
        "model__reg_lambda": [1.2, 1.5, 1.8],
    }

def get_xgb_regression_param_grid() -> dict:
    """Hyperparameter grid for XGBRegressor."""
    return {
        "model__n_estimators": [300, 500, 700],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 4, 5],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 5, 10],
        "model__gamma": [0, 0.1, 0.2],
        "model__reg_alpha": [0, 0.01, 0.1],
        "model__reg_lambda": [0.5, 1.0, 1.5],
    }

# ----------------------------------------
# Hyperparameter Tuning Utility
# ----------------------------------------

def tune_model(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    n_iter: int = 25,
    cv: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = 1,
    verbose: int = 1,
    use_sample_weights: bool = True,
) -> RandomizedSearchCV:
    """
    Tune a sklearn pipeline using RandomizedSearchCV with optional sample weighting.

    Args:
        pipeline: sklearn Pipeline object.
        X_train, y_train: Training data.
        param_grid: Hyperparameter search space.
        n_iter, cv, scoring, random_state, n_jobs, verbose: RandomizedSearchCV params.
        use_sample_weights (bool): Apply balanced sample weights if True.

    Returns:
        RandomizedSearchCV: Fitted search object.
    """
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )

    fit_params = {}
    if use_sample_weights:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        fit_params = {"model__sample_weight": sample_weights}

    search.fit(X_train, y_train, **fit_params)

    # Display summary
    print("\n" + "=" * 55)
    print("  TUNING RESULTS")
    print("=" * 55)
    print(f"  Best CV {scoring}: {search.best_score_:.4f}")
    print(f"  Best parameters:")
    for param, value in search.best_params_.items():
        print(f"    {param}: {value}")
    print("=" * 55)

    return search

def display_best_params(best_params: dict, prefix: str = "model__") -> pd.DataFrame:
    """
    Convert best_params dict to a readable DataFrame.

    Args:
        best_params (dict): search.best_params_.
        prefix (str): Optional prefix to remove from param names.

    Returns:
        pd.DataFrame: Parameter names and values.
    """
    clean_params = {k.replace(prefix, ""): v for k, v in best_params.items()}
    return pd.DataFrame.from_dict(clean_params, orient="index", columns=["Value"])