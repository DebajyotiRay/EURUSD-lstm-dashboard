from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger("lstm_core")


INPUT_FEATURES   = ["time_token", "norm_open", "norm_high", "norm_low", "norm_close"]
HIDDEN_SIZE      = 100
OUTPUT_SIZE      = 1
SEQ_LENGTH       = 60
EVALUATION_STEPS = 100
REQUIRED_BARS    = SEQ_LENGTH + EVALUATION_STEPS + 10


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = len(INPUT_FEATURES),
                 hidden_layer_size: int = HIDDEN_SIZE,
                 output_size: int = OUTPUT_SIZE):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def _init_hidden(self, device: torch.device):
        return (
            torch.zeros(1, 1, self.hidden_layer_size, device=device),
            torch.zeros(1, 1, self.hidden_layer_size, device=device),
        )

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        hidden_cell = self._init_hidden(input_seq.device)
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1), hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def load_model(model_path: str, device: str = "cpu") -> LSTMModel:
    model = LSTMModel()

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded model weights from %s", model_path)
    return model


def normalize_daily_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df.index.date
    df["rolling_high"] = df.groupby("date")["high"].transform(
        lambda x: x.expanding(min_periods=1).max()
    )
    df["rolling_low"] = df.groupby("date")["low"].transform(
        lambda x: x.expanding(min_periods=1).min()
    )
    denom = (df["rolling_high"] - df["rolling_low"]).replace(0, np.nan)
    df["norm_open"] = (df["open"] - df["rolling_low"]) / denom
    df["norm_high"] = (df["high"] - df["rolling_low"]) / denom
    df["norm_low"] = (df["low"] - df["rolling_low"]) / denom
    df["norm_close"] = (df["close"] - df["rolling_low"]) / denom
    df.fillna(0, inplace=True)
    return df[["norm_open", "norm_high", "norm_low", "norm_close"]]


def add_time_token(df: pd.DataFrame) -> pd.Series:
    return (df.index.hour * 3600 + df.index.minute * 60 + df.index.second) / 86400


def build_feature_frame(raw_ohlc: pd.DataFrame) -> pd.DataFrame:
    data = raw_ohlc.copy()
    data[["norm_open", "norm_high", "norm_low", "norm_close"]] = normalize_daily_rolling(data)
    data["time_token"] = add_time_token(data)
    return data[INPUT_FEATURES]


def fetch_yfinance(symbol: str = "EURUSD=X", bars: int = REQUIRED_BARS,
                    period: str = "7d", interval: str = "15m") -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return None

    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)

    if df is None or df.empty:
        logger.warning("yfinance returned no data.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[["open", "high", "low", "close"]].dropna()
    logger.info("Fetched %d bars of EURUSD historical data from yfinance.", len(df))
    return df


def get_market_data(required_bars: int = REQUIRED_BARS) -> tuple[pd.DataFrame, str]:
    df = fetch_yfinance(bars=required_bars)
    if df is not None and len(df) >= required_bars:
        return df, "yfinance"

    raise RuntimeError(
        "Could not obtain enough market data from yfinance. "
        "Check your internet connection, or that Yahoo Finance is reachable "
        "(pip install --upgrade yfinance if this used to work — see README Troubleshooting)."
    )


@dataclass
class PredictionRun:
    timestamps: list[str]
    true_prices: list[float]
    predicted_prices: list[float]
    next_prediction: float
    percent_change: float


def run_backtest_and_forecast(
    model: LSTMModel,
    feature_frame: pd.DataFrame,
    seq_length: int = SEQ_LENGTH,
    evaluation_steps: int = EVALUATION_STEPS,
    device: str = "cpu",
) -> PredictionRun:
    if len(feature_frame) < seq_length + evaluation_steps:
        raise ValueError(
            f"Not enough rows after preprocessing: {len(feature_frame)} "
            f"< {seq_length + evaluation_steps}"
        )

    values = feature_frame.values
    all_true, all_pred, timestamps = [], [], []

    with torch.no_grad():
        for step in range(evaluation_steps, 0, -1):
            seq = values[-step - seq_length:-step]
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
            prediction = model(seq_tensor).item()

            all_true.append(float(feature_frame["norm_close"].values[-step]))
            all_pred.append(float(prediction))
            timestamps.append(str(feature_frame.index[-step]))

        next_seq = torch.tensor(values[-seq_length:], dtype=torch.float32, device=device)
        next_prediction = float(model(next_seq).item())

    last_pred = all_pred[-1]
    pct_change = ((next_prediction - last_pred) / (abs(last_pred) + 1e-10)) * 100

    return PredictionRun(
        timestamps=timestamps,
        true_prices=all_true,
        predicted_prices=all_pred,
        next_prediction=next_prediction,
        percent_change=pct_change,
    )


def naive_baseline_predictions(feature_frame: pd.DataFrame, evaluation_steps: int = EVALUATION_STEPS):
    close = feature_frame["norm_close"].values
    true_vals, pred_vals = [], []
    for step in range(evaluation_steps, 0, -1):
        true_vals.append(float(close[-step]))
        pred_vals.append(float(close[-step - 1]))
    return true_vals, pred_vals


def directional_accuracy(true_prices, predicted_prices) -> float:
    true_prices = np.asarray(true_prices, dtype=float)
    predicted_prices = np.asarray(predicted_prices, dtype=float)
    if len(true_prices) < 2:
        return float("nan")

    true_dir = np.sign(np.diff(true_prices))
    pred_dir = np.sign(predicted_prices[1:] - true_prices[:-1])
    matches = (true_dir == pred_dir) & (true_dir != 0)
    valid = true_dir != 0
    return float(matches.sum() / valid.sum()) if valid.sum() > 0 else float("nan")


def regression_metrics(true_prices, predicted_prices) -> dict:
    true_prices = np.asarray(true_prices, dtype=float)
    predicted_prices = np.asarray(predicted_prices, dtype=float)
    errors = predicted_prices - true_prices
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {"mae": mae, "rmse": rmse}


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
