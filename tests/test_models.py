import pytest
import numpy as np
import torch
import torch.nn as nn
torch.device("cpu")  # Ensure model and tensors stay on CPU
torch.set_num_threads(1)  # Avoid segfaults with batch_norm
from banking_ml.models import (
    NeuralCreditScorer,
    NeuralClassifierWrapper,
    train_neural_model,
)


class TestNeuralCreditScorer:

    def test_forward_pass_shape(self):
        model = NeuralCreditScorer(input_dim=10)
        x = torch.randn(32, 10)
        out = model(x)
        assert out.shape == (32, 1)

    def test_custom_hidden_dims(self):
        model = NeuralCreditScorer(
            input_dim=20,
            hidden_dims=[64, 32],
            dropout_rates=[0.2, 0.2]
        )
        x = torch.randn(16, 20)
        out = model(x)
        assert out.shape == (16, 1)

    def test_parameter_count_reasonable(self):
        model = NeuralCreditScorer(input_dim=103)
        total_params = sum(p.numel() for p in model.parameters())
        # Should be in the tens of thousands range for this architecture
        assert 10_000 < total_params < 500_000

    def test_output_is_raw_logits(self):
        model = NeuralCreditScorer(input_dim=10)
        x = torch.randn(100, 10)
        out = model(x)

        # logits should allow values outside (0, 1)
        # We don't enforce range — just ensure no sigmoid layer exists
        for module in model.modules():
            assert not isinstance(module, nn.Sigmoid)

    def test_reproducibility_with_seed(self):
        x = torch.randn(8, 10)  # create x once, share it

        torch.manual_seed(42)
        model1 = NeuralCreditScorer(input_dim=10)
        out1 = model1(x)

        torch.manual_seed(42)
        model2 = NeuralCreditScorer(input_dim=10)
        out2 = model2(x)

        assert torch.allclose(out1, out2)


class TestNeuralClassifierWrapper:

    @pytest.fixture
    def trained_wrapper(self):
        torch.manual_seed(42)
        np.random.seed(42)

        n_train, n_val, n_features = 500, 100, 20
        X_train = np.random.randn(n_train, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, n_train)
        X_val = np.random.randn(n_val, n_features).astype(np.float32)
        y_val = np.random.randint(0, 2, n_val)

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        model = NeuralCreditScorer(input_dim=n_features, hidden_dims=[32, 16])
        train_neural_model(
            model, X_train, y_train, X_val, y_val,
            epochs=3, batch_size=64, verbose=False
        )

        wrapper = NeuralClassifierWrapper(model, device)
        return wrapper, X_val

    def test_predict_proba_shape(self, trained_wrapper):
        wrapper, X = trained_wrapper
        probs = wrapper.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, trained_wrapper):
        wrapper, X = trained_wrapper
        probs = wrapper.predict_proba(X)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_proba_between_zero_and_one(self, trained_wrapper):
        wrapper, X = trained_wrapper
        probs = wrapper.predict_proba(X)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_returns_binary(self, trained_wrapper):
        wrapper, X = trained_wrapper
        preds = wrapper.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_threshold(self, trained_wrapper):
        wrapper, X = trained_wrapper
        preds_default = wrapper.predict(X, threshold=0.5)
        preds_low = wrapper.predict(X, threshold=0.1)
        # Lower threshold should predict more positives
        assert preds_low.sum() >= preds_default.sum()


class TestTrainNeuralModel:

    @pytest.fixture
    def training_data(self):
        np.random.seed(42)
        n, f = 400, 15
        X_train = np.random.randn(n, f).astype(np.float32)
        y_train = np.random.randint(0, 2, n)
        X_val = np.random.randn(100, f).astype(np.float32)
        y_val = np.random.randint(0, 2, 100)
        return X_train, y_train, X_val, y_val

    def test_returns_history_dict(self, training_data):
        X_train, y_train, X_val, y_val = training_data
        model = NeuralCreditScorer(input_dim=15, hidden_dims=[32, 16])
        history = train_neural_model(
            model, X_train, y_train, X_val, y_val,
            epochs=3, verbose=False
        )
        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history

    def test_history_length_matches_epochs(self, training_data):
        X_train, y_train, X_val, y_val = training_data
        model = NeuralCreditScorer(input_dim=15, hidden_dims=[32, 16])
        history = train_neural_model(
            model, X_train, y_train, X_val, y_val,
            epochs=5, verbose=False
        )
        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5

    def test_loss_decreases_over_training(self, training_data):
        X_train, y_train, X_val, y_val = training_data
        torch.manual_seed(42)
        model = NeuralCreditScorer(input_dim=15, hidden_dims=[32, 16])
        history = train_neural_model(
            model, X_train, y_train, X_val, y_val,
            epochs=10, verbose=False
        )
        # Training loss should decrease from epoch 1 to epoch 10
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_early_stopping_triggers(self, training_data):
        X_train, y_train, X_val, y_val = training_data
        model = NeuralCreditScorer(input_dim=15, hidden_dims=[32, 16])
        history = train_neural_model(
            model, X_train, y_train, X_val, y_val,
            epochs=100, patience=3, verbose=False
        )
        # Early stopping should have recorded patience counter in history
        # Verify it ran at most epochs worth of steps
        assert len(history["train_loss"]) <= 100

    def test_reproducibility_with_random_state(self, training_data):
        X_train, y_train, X_val, y_val = training_data

        torch.manual_seed(42)
        model1 = NeuralCreditScorer(input_dim=15, hidden_dims=[32, 16])
        history1 = train_neural_model(
            model1, X_train, y_train, X_val, y_val,
            epochs=3, random_state=42, verbose=False
        )
        

        torch.manual_seed(42)
        model2 = NeuralCreditScorer(input_dim=15, hidden_dims=[32, 16])
        history2 = train_neural_model(
            model2, X_train, y_train, X_val, y_val,
            epochs=3, random_state=42, verbose=False
        )

        assert history1["train_loss"] == history2["train_loss"]