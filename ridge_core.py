from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

import lstm_core as core

logger = logging.getLogger("ridge_core")

RIDGE_MODEL_FILENAME = "ridge_model.json"


def flatten_window(seq: np.ndarray) -> np.ndarray:
    return seq.reshape(-1)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    X_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - X_mean
    yc = y - y_mean

    n_features = X.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(n_features)
    b = Xc.T @ yc
    try:
        weights = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"Ridge fit failed to solve (alpha={alpha}): {exc}. "
            "The feature matrix is exactly singular — try a larger alpha."
        ) from exc
    intercept = y_mean - float(X_mean @ weights)

    if not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise ValueError(
            f"Ridge fit produced non-finite weights (alpha={alpha}). "
            "This usually means the feature matrix is close to singular "
            "and alpha is too small to stabilise it — try a larger alpha."
        )

    return {"weights": weights.tolist(), "intercept": intercept, "alpha": alpha}


def predict_ridge(model: dict, X: np.ndarray) -> np.ndarray:
    weights = np.array(model["weights"])
    intercept = model["intercept"]
    return X @ weights + intercept


def save_ridge_model(model: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(model, f)


def load_ridge_model(path: str) -> dict:
    with open(path) as f:
        model = json.load(f)
    logger.info("Loaded Ridge model weights from %s (alpha=%s)", path, model.get("alpha"))
    return model


def run_ridge_backtest_and_forecast(
    model: dict,
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
        prediction = float(predict_ridge(model, flat)[0])

        all_true.append(float(feature_frame["norm_close"].values[-step]))
        all_pred.append(prediction)
        timestamps.append(str(feature_frame.index[-step]))

    next_seq = values[-seq_length:]
    next_flat = flatten_window(next_seq).reshape(1, -1)
    next_prediction = float(predict_ridge(model, next_flat)[0])

    last_true = all_true[-1]
    predicted_change = next_prediction - last_true

    return core.PredictionRun(
        timestamps=timestamps,
        true_prices=all_true,
        predicted_prices=all_pred,
        next_prediction=next_prediction,
        predicted_change=predicted_change,
    )
