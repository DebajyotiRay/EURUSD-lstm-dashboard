from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

import lstm_core as core

logger = logging.getLogger("xgb_core")

XGBOOST_MODEL_FILENAME = "xgboost_model.json"


def flatten_window(seq: np.ndarray) -> np.ndarray:
    return seq.reshape(-1)


def load_xgboost_model(model_path: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(model_path)
    logger.info("Loaded XGBoost model weights from %s", model_path)
    return booster


def run_xgboost_backtest_and_forecast(
    model: xgb.Booster,
    feature_frame: pd.DataFrame,
    seq_length: int = core.SEQ_LENGTH,
    evaluation_steps: int = core.EVALUATION_STEPS,
) -> core.PredictionRun:
    if len(feature_frame) < seq_length + evaluation_steps:
        raise ValueError(
            f"Not enough rows after preprocessing: {len(feature_frame)} "
            f"< {seq_length + evaluation_steps}"
        )

    values = feature_frame.values
    all_true, all_pred, timestamps = [], [], []

    for step in range(evaluation_steps, 0, -1):
        seq = values[-step - seq_length:-step]
        flat = flatten_window(seq).reshape(1, -1)
        prediction = float(model.predict(xgb.DMatrix(flat))[0])

        all_true.append(float(feature_frame["norm_close"].values[-step]))
        all_pred.append(prediction)
        timestamps.append(str(feature_frame.index[-step]))

    next_seq = values[-seq_length:]
    next_flat = flatten_window(next_seq).reshape(1, -1)
    next_prediction = float(model.predict(xgb.DMatrix(next_flat))[0])

    last_pred = all_pred[-1]
    pct_change = ((next_prediction - last_pred) / (abs(last_pred) + 1e-10)) * 100

    return core.PredictionRun(
        timestamps=timestamps,
        true_prices=all_true,
        predicted_prices=all_pred,
        next_prediction=next_prediction,
        percent_change=pct_change,
    )
