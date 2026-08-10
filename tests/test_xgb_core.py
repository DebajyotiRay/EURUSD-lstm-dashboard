import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import lstm_core as core
import xgb_core
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


def train_tiny_booster(feature_frame, seq_length=core.SEQ_LENGTH, val_fraction=0.15):
    X, y = make_sequences(feature_frame, seq_length)
    X_train, y_train, X_val, y_val = chronological_split(X, y, val_fraction)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    dtrain = xgb.DMatrix(X_train_flat, label=y_train)
    dval = xgb.DMatrix(X_val_flat, label=y_val)
    params = {"max_depth": 2, "eta": 0.1, "objective": "reg:squarederror", "eval_metric": "mae"}
    return xgb.train(params, dtrain, num_boost_round=10, evals=[(dval, "val")],
                      early_stopping_rounds=5, verbose_eval=False)


class TestFlattenWindow:
    def test_shape(self):
        seq = np.random.randn(core.SEQ_LENGTH, len(core.INPUT_FEATURES))
        flat = xgb_core.flatten_window(seq)
        assert flat.shape == (core.SEQ_LENGTH * len(core.INPUT_FEATURES),)

    def test_preserves_values_in_row_major_order(self):
        seq = np.arange(12).reshape(3, 4)
        flat = xgb_core.flatten_window(seq)
        assert list(flat) == list(range(12))

    def test_matches_train_val_flatten_shape(self):
        df = make_synthetic_ohlc(n=300)
        feature_frame = core.build_feature_frame(df)
        X, _ = make_sequences(feature_frame, core.SEQ_LENGTH)
        training_time_shape = X.reshape(X.shape[0], -1).shape[1]

        single_window = X[0]
        inference_time_shape = xgb_core.flatten_window(single_window).shape[0]
        assert training_time_shape == inference_time_shape


class TestXGBoostBacktestAndForecast:
    def test_output_matches_lstm_prediction_run_shape(self):
        df = make_synthetic_ohlc(n=400, seed=1)
        feature_frame = core.build_feature_frame(df)
        booster = train_tiny_booster(feature_frame)

        run = xgb_core.run_xgboost_backtest_and_forecast(booster, feature_frame)

        assert len(run.predicted_prices) == core.EVALUATION_STEPS
        assert len(run.true_prices) == core.EVALUATION_STEPS
        assert len(run.timestamps) == core.EVALUATION_STEPS
        assert isinstance(run.next_prediction, float)
        assert isinstance(run.predicted_change, float)

    def test_raises_on_insufficient_data(self):
        df = make_synthetic_ohlc(n=50, seed=2)
        feature_frame = core.build_feature_frame(df)
        booster = train_tiny_booster(
            core.build_feature_frame(make_synthetic_ohlc(n=400, seed=3))
        )
        with pytest.raises(ValueError):
            xgb_core.run_xgboost_backtest_and_forecast(booster, feature_frame)

    def test_metrics_computable_on_output(self):
        df = make_synthetic_ohlc(n=400, seed=4)
        feature_frame = core.build_feature_frame(df)
        booster = train_tiny_booster(feature_frame)
        run = xgb_core.run_xgboost_backtest_and_forecast(booster, feature_frame)

        metrics = core.regression_metrics(run.true_prices, run.predicted_prices)
        dir_acc = core.directional_accuracy(run.true_prices, run.predicted_prices)
        assert np.isfinite(metrics["mae"])
        assert np.isfinite(metrics["rmse"])
        assert 0.0 <= dir_acc <= 1.0
