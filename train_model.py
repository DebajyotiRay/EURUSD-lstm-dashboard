from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn

import lstm_core as core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("train_model")


def make_sequences(feature_frame, seq_length: int):
    values = feature_frame.values
    close_idx = feature_frame.columns.get_loc("norm_close")

    X, y = [], []
    for i in range(seq_length, len(values)):
        X.append(values[i - seq_length:i])
        y.append(values[i, close_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def chronological_split(X, y, val_fraction: float = 0.15):
    split_idx = int(len(X) * (1 - val_fraction))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def train(
    model: nn.Module,
    X_train, y_train, X_val, y_val,
    epochs: int = 40,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 8,
    grad_clip: float = 1.0,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val, device=device)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()


        train_preds = torch.stack([model(seq) for seq in X_train_t]).squeeze()
        loss = loss_fn(train_preds, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = torch.stack([model(seq) for seq in X_val_t]).squeeze()
            val_loss = loss_fn(val_preds, y_val_t)

        scheduler.step(val_loss)
        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss.item()))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
                epoch, epochs, loss.item(), val_loss.item(), optimizer.param_groups[0]["lr"]
            )

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d (no val improvement for %d epochs). "
                "Restoring best weights (val_loss=%.6f).",
                epoch, patience, best_val_loss
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate(model, X_val, y_val, device: str = "cpu"):
    model.eval()
    with torch.no_grad():
        preds = torch.stack([
            model(torch.tensor(seq, device=device)) for seq in X_val
        ]).squeeze().cpu().numpy()

    metrics = core.regression_metrics(y_val, preds)
    dir_acc = core.directional_accuracy(y_val, preds)
    metrics["directional_accuracy"] = dir_acc
    return metrics, preds


def evaluate_naive_baseline(X_val, y_val, close_col_idx: int):
    baseline_preds = np.array([seq[-1, close_col_idx] for seq in X_val])
    metrics = core.regression_metrics(y_val, baseline_preds)
    dir_acc = core.directional_accuracy(y_val, baseline_preds)
    metrics["directional_accuracy"] = dir_acc
    return metrics


def plot_loss_curve(history, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train the EURUSD LSTM model.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--period", type=str, default="60d",
                         help="yfinance history window, e.g. 7d, 30d, 60d (60d is the yfinance 15m cap)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seq-length", type=int, default=core.SEQ_LENGTH)
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "lstm_model.pth"))
    args = parser.parse_args()

    core.configure_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

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

    model = core.LSTMModel().to(device)
    history = train(model, X_train, y_train, X_val, y_val, epochs=args.epochs, lr=args.lr, device=device)

    close_col_idx = feature_frame.columns.get_loc("norm_close")
    model_metrics, _ = evaluate(model, X_val, y_val, device=device)
    baseline_metrics = evaluate_naive_baseline(X_val, y_val, close_col_idx)

    logger.info("── Validation results ──────────────────────────")
    logger.info("LSTM      MAE=%.5f RMSE=%.5f DirAcc=%.1f%%",
                model_metrics["mae"], model_metrics["rmse"], model_metrics["directional_accuracy"] * 100)
    logger.info("Baseline  MAE=%.5f RMSE=%.5f DirAcc=n/a (a flat/no-change predictor never calls a direction, "
                "so this metric can't score it meaningfully)",
                baseline_metrics["mae"], baseline_metrics["rmse"])
    logger.info(
        "NOTE: judge the LSTM's directional accuracy (%.1f%%) against 50%% (random chance), "
        "not against the baseline's undefined figure above.",
        model_metrics["directional_accuracy"] * 100
    )

    torch.save(model.state_dict(), args.output)
    logger.info("Saved trained weights to %s", args.output)

    report = {
        "epochs": args.epochs,
        "seq_length": args.seq_length,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "history": history,
        "lstm_metrics": model_metrics,
        "naive_baseline_metrics": baseline_metrics,
    }
    report_path = os.path.join(BASE_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved training report to %s", report_path)

    try:
        plot_loss_curve(history, os.path.join(BASE_DIR, "loss_curve.png"))
        logger.info("Saved loss curve plot to loss_curve.png")
    except ImportError:
        logger.warning("matplotlib not installed — skipping loss_curve.png (pip install matplotlib)")


if __name__ == "__main__":
    main()
