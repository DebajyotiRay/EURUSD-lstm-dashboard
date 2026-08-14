from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

import lstm_core as core
import ridge_core
from train_model import chronological_split, make_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("train_ridge_model")


def evaluate_ridge(model: dict, X: np.ndarray, y: np.ndarray) -> dict:
    X_flat = X.reshape(X.shape[0], -1)
    preds = ridge_core.predict_ridge(model, X_flat)
    metrics = core.regression_metrics(y, preds)
    metrics["directional_accuracy"] = core.directional_accuracy(y, preds)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the EURUSD Ridge regression comparison model.")
    parser.add_argument("--period", type=str, default="60d")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seq-length", type=int, default=core.SEQ_LENGTH)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, ridge_core.RIDGE_MODEL_FILENAME))
    args = parser.parse_args()

    core.configure_logging()

    logger.info("Fetching ~%s of EURUSD 15-min history via yfinance...", args.period)
    raw = core.fetch_yfinance(period=args.period, bars=0)
    if raw is None or raw.empty:
        raise SystemExit(
            "Could not fetch training data from yfinance. Check your internet "
            "connection, or supply your own OHLC CSV and adapt this script."
        )
    logger.info("Fetched %d raw bars.", len(raw))

    feature_frame = core.build_feature_frame(raw)
    X, y = make_sequences(feature_frame, args.seq_length)
    logger.info("Built %d training sequences (seq_length=%d).", len(X), args.seq_length)

    X_train, y_train, X_val, y_val = chronological_split(X, y, args.val_fraction)
    logger.info("Train/val split: %d train, %d val (chronological, no shuffling).", len(X_train), len(X_val))

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    logger.info("Flattened shape: %s (was %s)", X_train_flat.shape, X_train.shape)

    logger.info("── Alpha search ─────────────────────────────────")
    best_alpha = None
    best_val_mae = float("inf")
    best_model = None
    search_results = []

    for alpha in args.alphas:
        try:
            model = ridge_core.fit_ridge(X_train_flat, y_train, alpha)
        except ValueError as exc:
            logger.warning("alpha=%s failed: %s", alpha, exc)
            continue

        train_metrics = evaluate_ridge(model, X_train, y_train)
        val_metrics = evaluate_ridge(model, X_val, y_val)
        gap = val_metrics["mae"] - train_metrics["mae"]

        logger.info(
            "alpha=%-8g  train MAE=%.5f  val MAE=%.5f  val DirAcc=%.1f%%  gap=%.5f",
            alpha, train_metrics["mae"], val_metrics["mae"],
            val_metrics["directional_accuracy"] * 100, gap
        )
        search_results.append({
            "alpha": alpha, "train_metrics": train_metrics,
            "val_metrics": val_metrics, "train_val_mae_gap": gap,
        })

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_alpha = alpha
            best_model = model

    if best_model is None:
        raise SystemExit("Every alpha candidate failed to produce a finite fit — try larger alpha values.")

    logger.info("── Selected alpha=%s (lowest validation MAE) ─────", best_alpha)
    train_metrics = evaluate_ridge(best_model, X_train, y_train)
    val_metrics = evaluate_ridge(best_model, X_val, y_val)
    gap = val_metrics["mae"] - train_metrics["mae"]

    logger.info("Train  MAE=%.5f RMSE=%.5f DirAcc=%.1f%%",
                train_metrics["mae"], train_metrics["rmse"], train_metrics["directional_accuracy"] * 100)
    logger.info("Val    MAE=%.5f RMSE=%.5f DirAcc=%.1f%%",
                val_metrics["mae"], val_metrics["rmse"], val_metrics["directional_accuracy"] * 100)

    if gap > 0.05:
        logger.warning(
            "Train/val MAE gap is %.5f — this is a real overfitting warning sign. "
            "Consider a larger alpha.", gap
        )
    else:
        logger.info("Train/val MAE gap is %.5f — no strong overfitting signal.", gap)

    ridge_core.save_ridge_model(best_model, args.output)
    logger.info("Saved trained weights to %s", args.output)

    report = {
        "selected_alpha": best_alpha,
        "alpha_search": search_results,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_val_mae_gap": gap,
    }
    report_path = os.path.join(BASE_DIR, "ridge_training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved training report to %s", report_path)


if __name__ == "__main__":
    main()
