from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import xgboost as xgb

import lstm_core as core
import xgb_core
from train_model import chronological_split, make_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("train_xgboost_model")


def evaluate_xgboost(booster: xgb.Booster, X: np.ndarray, y: np.ndarray) -> dict:
    X_flat = X.reshape(X.shape[0], -1)
    preds = booster.predict(xgb.DMatrix(X_flat))
    metrics = core.regression_metrics(y, preds)
    metrics["directional_accuracy"] = core.directional_accuracy(y, preds)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the EURUSD XGBoost comparison model.")
    parser.add_argument("--period", type=str, default="60d",
                         help="yfinance history window, e.g. 7d, 30d, 60d (60d is the yfinance 15m cap)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seq-length", type=int, default=core.SEQ_LENGTH)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4,
                         help="Kept shallow deliberately — 300 flattened features on a few "
                              "thousand rows is a lot of room to overfit; shallower trees "
                              "and early stopping (below) are the main defenses against that.")
    parser.add_argument("--early-stopping-rounds", type=int, default=15)
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, xgb_core.XGBOOST_MODEL_FILENAME))
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
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    logger.info("Flattened shape: %s (was %s)", X_train_flat.shape, X_train.shape)

    dtrain = xgb.DMatrix(X_train_flat, label=y_train)
    dval = xgb.DMatrix(X_val_flat, label=y_val)

    params = {
        "max_depth": args.max_depth,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    }
    booster = xgb.train(
        params, dtrain, num_boost_round=args.num_boost_round,
        evals=[(dval, "val")], early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
    )
    logger.info("Training stopped at %d boosting rounds (best iteration %d).",
                booster.num_boosted_rounds(), booster.best_iteration)


    train_metrics = evaluate_xgboost(booster, X_train, y_train)
    val_metrics = evaluate_xgboost(booster, X_val, y_val)

    logger.info("── Results ──────────────────────────────────────")
    logger.info("Train  MAE=%.5f RMSE=%.5f DirAcc=%.1f%%",
                train_metrics["mae"], train_metrics["rmse"], train_metrics["directional_accuracy"] * 100)
    logger.info("Val    MAE=%.5f RMSE=%.5f DirAcc=%.1f%%",
                val_metrics["mae"], val_metrics["rmse"], val_metrics["directional_accuracy"] * 100)

    gap = val_metrics["mae"] - train_metrics["mae"]
    if gap > 0.05:
        logger.warning(
            "Train/val MAE gap is %.5f — this is a real overfitting warning sign, "
            "not just a number to note. Consider lowering --max-depth or --num-boost-round.",
            gap
        )
    else:
        logger.info("Train/val MAE gap is %.5f — no strong overfitting signal.", gap)

    booster.save_model(args.output)
    logger.info("Saved trained weights to %s", args.output)

    report = {
        "num_boost_round_requested": args.num_boost_round,
        "num_boost_round_used": booster.num_boosted_rounds(),
        "best_iteration": booster.best_iteration,
        "max_depth": args.max_depth,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_val_mae_gap": gap,
    }
    report_path = os.path.join(BASE_DIR, "xgboost_training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved training report to %s", report_path)


if __name__ == "__main__":
    main()
