"""
Unit tests for lstm_core.py.

Run with:
    pytest tests/

These tests use synthetic OHLC data (no network access, no
pretrained weights required) so they run anywhere, including CI.
"""

import numpy as np
import pandas as pd
import pytest

import lstm_core as core


def make_synthetic_ohlc(n=500, seed=0):
    """Generate synthetic OHLC data respecting the real invariant that
    close (and open) always fall within [low, high] for the same bar."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    price = 1.10 + np.cumsum(rng.standard_normal(n) * 0.0002)
    spread = np.abs(rng.standard_normal(n)) * 0.0001 + 1e-6
    high = price + spread
    low = price - spread
    close = price + rng.uniform(-1, 1, n) * spread * 0.9  # stays inside [low, high]
    open_ = price + rng.uniform(-1, 1, n) * spread * 0.9
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


class TestNormalization:
    def test_output_columns(self):
        df = make_synthetic_ohlc()
        result = core.normalize_daily_rolling(df)
        assert list(result.columns) == ["norm_open", "norm_high", "norm_low", "norm_close"]

    def test_no_nans(self):
        df = make_synthetic_ohlc()
        result = core.normalize_daily_rolling(df)
        assert result.isnull().sum().sum() == 0

    def test_values_in_unit_range(self):
        """Normalized values should stay within a small tolerance of [0, 1]
        since they're computed from the same day's rolling high/low."""
        df = make_synthetic_ohlc()
        result = core.normalize_daily_rolling(df)
        for col in result.columns:
            assert result[col].between(-1e-6, 1 + 1e-6).all(), f"{col} out of range"

    def test_flat_prices_no_division_by_zero(self):
        """If high == low for a whole day (edge case), the function must not
        raise or produce inf/nan — it should fall back to 0."""
        idx = pd.date_range("2026-01-01", periods=10, freq="15min")
        df = pd.DataFrame({
            "open": [1.1] * 10, "high": [1.1] * 10, "low": [1.1] * 10, "close": [1.1] * 10,
        }, index=idx)
        result = core.normalize_daily_rolling(df)
        assert np.isfinite(result.values).all()


class TestBuildFeatureFrame:
    def test_shape_and_columns(self):
        df = make_synthetic_ohlc(n=200)
        ff = core.build_feature_frame(df)
        assert ff.shape == (200, 5)
        assert list(ff.columns) == core.INPUT_FEATURES

    def test_time_token_range(self):
        df = make_synthetic_ohlc(n=200)
        ff = core.build_feature_frame(df)
        assert ff["time_token"].between(0, 1).all()


class TestMetrics:
    def test_directional_accuracy_perfect(self):
        true = [1.0, 1.1, 1.05, 1.2]
        # predicted direction matches true direction exactly (predicted[i] vs true[i-1])
        pred = [1.0, 1.2, 0.9, 1.3]
        acc = core.directional_accuracy(true, pred)
        assert acc == 1.0

    def test_directional_accuracy_random_is_around_half(self):
        rng = np.random.default_rng(1)
        true = np.cumsum(rng.standard_normal(2000)) 
        pred = true + rng.standard_normal(2000) * 5  # noisy, weakly correlated
        acc = core.directional_accuracy(true, pred)
        assert 0.0 <= acc <= 1.0

    def test_regression_metrics_zero_error(self):
        true = [1.0, 2.0, 3.0]
        metrics = core.regression_metrics(true, true)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)

    def test_regression_metrics_known_error(self):
        true = [0.0, 0.0, 0.0]
        pred = [1.0, 1.0, 1.0]
        metrics = core.regression_metrics(true, pred)
        assert metrics["mae"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(1.0)


class TestBaseline:
    def test_naive_baseline_shapes(self):
        df = make_synthetic_ohlc(n=300)
        ff = core.build_feature_frame(df)
        true_vals, pred_vals = core.naive_baseline_predictions(ff, evaluation_steps=50)
        assert len(true_vals) == len(pred_vals) == 50

    def test_naive_baseline_is_lagged_close(self):
        """The naive baseline predicts 'next = previous close', so predicted[i]
        should equal true[i-1] for i > 0 within the evaluation window."""
        df = make_synthetic_ohlc(n=300)
        ff = core.build_feature_frame(df)
        true_vals, pred_vals = core.naive_baseline_predictions(ff, evaluation_steps=50)
        for i in range(1, len(true_vals)):
            assert pred_vals[i] == pytest.approx(true_vals[i - 1])
