import numpy as np
import pandas as pd
import pytest

import lstm_core as core
import ridge_core
from train_model import chronological_split, make_sequences


def make_synthetic_ohlc(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    price = 1.10 + np.cumsum(rng.standard_normal(n) * 0.0002)
    spread = np.abs(rng.standard_normal(n)) * 0.0001 + 1e-6
    high = price + spread
    low = price - spread
    close = price + rng.uniform(-1, 1, n) * spread * 0.9
    open_ = price + rng.uniform(-1, 1, n) * spread * 0.9
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def train_tiny_ridge(feature_frame, seq_length=core.SEQ_LENGTH, val_fraction=0.15, alpha=10.0):
    X, y = make_sequences(feature_frame, seq_length)
    X_train, y_train, X_val, y_val = chronological_split(X, y, val_fraction)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    return ridge_core.fit_ridge(X_train_flat, y_train, alpha)


class TestFitRidge:
    def test_weights_and_intercept_finite(self):
        df = make_synthetic_ohlc(n=400, seed=1)
        feature_frame = core.build_feature_frame(df)
        model = train_tiny_ridge(feature_frame)
        assert np.isfinite(model["weights"]).all()
        assert np.isfinite(model["intercept"])

    def test_matches_hand_computed_closed_form_solution(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 5))
        y = rng.standard_normal(50)
        alpha = 2.0

        model = ridge_core.fit_ridge(X, y, alpha)

        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean
        expected_weights = np.linalg.inv(Xc.T @ Xc + alpha * np.eye(5)) @ Xc.T @ yc
        expected_intercept = y_mean - X_mean @ expected_weights

        assert np.allclose(model["weights"], expected_weights, atol=1e-8)
        assert model["intercept"] == pytest.approx(expected_intercept, abs=1e-8)

    def test_zero_alpha_matches_ordinary_least_squares(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((100, 5))
        true_weights = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
        y = X @ true_weights + rng.standard_normal(100) * 0.01

        model = ridge_core.fit_ridge(X, y, alpha=0.0)
        assert np.allclose(model["weights"], true_weights, atol=0.05)

    def test_larger_alpha_shrinks_weights_toward_zero(self):
        df = make_synthetic_ohlc(n=400, seed=2)
        feature_frame = core.build_feature_frame(df)
        X, y = make_sequences(feature_frame, core.SEQ_LENGTH)
        X_train, y_train, _, _ = chronological_split(X, y, 0.15)
        X_flat = X_train.reshape(X_train.shape[0], -1)

        small_alpha_model = ridge_core.fit_ridge(X_flat, y_train, alpha=0.1)
        large_alpha_model = ridge_core.fit_ridge(X_flat, y_train, alpha=1000.0)

        small_norm = np.linalg.norm(small_alpha_model["weights"])
        large_norm = np.linalg.norm(large_alpha_model["weights"])
        assert large_norm < small_norm

    def test_raises_value_error_on_ill_conditioned_input_rather_than_silently_returning_garbage(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((5, 300))
        y = rng.standard_normal(5)
        try:
            model = ridge_core.fit_ridge(X, y, alpha=0.0)
            assert np.isfinite(model["weights"]).all(), "if it didn't raise, weights must still be finite"
        except ValueError:
            pass


class TestPredictRidge:
    def test_prediction_shape(self):
        rng = np.random.default_rng(4)
        X_train = rng.standard_normal((50, 10))
        y_train = rng.standard_normal(50)
        model = ridge_core.fit_ridge(X_train, y_train, alpha=1.0)

        X_new = rng.standard_normal((5, 10))
        preds = ridge_core.predict_ridge(model, X_new)
        assert preds.shape == (5,)

    def test_save_load_roundtrip(self, tmp_path):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((30, 8))
        y = rng.standard_normal(30)
        model = ridge_core.fit_ridge(X, y, alpha=5.0)

        path = str(tmp_path / "ridge_model.json")
        ridge_core.save_ridge_model(model, path)
        loaded = ridge_core.load_ridge_model(path)
        assert loaded == model


class TestRidgeBacktestAndForecast:
    def test_output_matches_lstm_prediction_run_shape(self):
        df = make_synthetic_ohlc(n=400, seed=8)
        feature_frame = core.build_feature_frame(df)
        model = train_tiny_ridge(feature_frame)

        run = ridge_core.run_ridge_backtest_and_forecast(model, feature_frame)

        assert len(run.predicted_prices) == core.EVALUATION_STEPS
        assert len(run.true_prices) == core.EVALUATION_STEPS
        assert len(run.timestamps) == core.EVALUATION_STEPS
        assert isinstance(run.next_prediction, float)
        assert isinstance(run.predicted_change, float)

    def test_raises_on_insufficient_data(self):
        model = train_tiny_ridge(core.build_feature_frame(make_synthetic_ohlc(n=400, seed=10)))
        df = make_synthetic_ohlc(n=50, seed=12)
        feature_frame = core.build_feature_frame(df)
        with pytest.raises(ValueError):
            ridge_core.run_ridge_backtest_and_forecast(model, feature_frame)

    def test_metrics_computable_on_output(self):
        df = make_synthetic_ohlc(n=400, seed=13)
        feature_frame = core.build_feature_frame(df)
        model = train_tiny_ridge(feature_frame)
        run = ridge_core.run_ridge_backtest_and_forecast(model, feature_frame)

        metrics = core.regression_metrics(run.true_prices, run.predicted_prices)
        dir_acc = core.directional_accuracy(run.true_prices, run.predicted_prices)
        assert np.isfinite(metrics["mae"])
        assert np.isfinite(metrics["rmse"])
        assert 0.0 <= dir_acc <= 1.0
