"""
LSTM_model_prediction.py
------------------------
CLI entry point: fetches EURUSD 15-min candle data, runs the trained LSTM
model, and writes prediction_results.json for the Flask dashboard.

All the real logic (model, data fetching, preprocessing, inference) lives
in lstm_core.py so it can be reused and unit tested. This script is
intentionally a thin wrapper.

Usage:
    python LSTM_model_prediction.py
"""

import json
import logging
import os
import sys

import lstm_core as core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("prediction_script")


def main() -> int:
    core.configure_logging()

    model_path = os.path.join(BASE_DIR, "lstm_model.pth")
    if not os.path.exists(model_path):
        logger.error("Model file not found: %s", model_path)
        logger.error("Make sure lstm_model.pth is in the same folder as this script.")
        return 1

    try:
        model = core.load_model(model_path)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return 1

    try:
        raw_data, data_source = core.get_market_data()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    raw_data = raw_data.iloc[-core.REQUIRED_BARS:]

    try:
        feature_frame = core.build_feature_frame(raw_data)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        return 1

    try:
        run = core.run_backtest_and_forecast(model, feature_frame)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    metrics = core.regression_metrics(run.true_prices, run.predicted_prices)
    dir_acc = core.directional_accuracy(run.true_prices, run.predicted_prices)

    results = {
        "predicted_prices": run.predicted_prices,
        "true_prices": run.true_prices,
        "timestamps": run.timestamps,
        "percent_change": run.percent_change,
        "next_prediction": run.next_prediction,
        "data_source": data_source,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "directional_accuracy": dir_acc,
    }

    output_path = os.path.join(BASE_DIR, "prediction_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f)

    logger.info("Prediction complete. Source: %s", data_source)
    logger.info(
        "Backtest MAE=%.5f RMSE=%.5f DirAcc=%.1f%%",
        metrics["mae"], metrics["rmse"], dir_acc * 100
    )
    logger.info("Results saved to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
