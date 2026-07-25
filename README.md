# EURUSD LSTM Market Prediction Dashboard

A forex forecasting dashboard that trains an LSTM neural network (PyTorch) on EURUSD 15-minute candles and serves live-updating predictions through a Flask + Plotly web UI.

**[Live demo](https://eurusd-lstm-dashboard.onrender.com)** — hosted on Render's free tier, so it sleeps after 15 minutes of inactivity and the first load can take 10-30 seconds to wake up.

![Dashboard screenshot](docs/screenshot.png)

> No broker account needed. The dashboard runs entirely on free historical data from Yahoo Finance — no login, no install, works anywhere.

---

## What this project demonstrates

- Training a sequence model (LSTM) on real-world time-series data with a proper chronological train/validation split (no shuffling, since that would leak the future into training).
- Evaluating the model against a naive baseline ("predict next = last observed close") rather than reporting accuracy in isolation.
- Serving the trained model behind a Flask dashboard with background job scheduling, health checks, and Docker packaging.
- Structuring the ML code (`lstm_core.py`) as unit-tested, reusable functions rather than a single monolithic script.

## Features

- LSTM model forecasts the next 15-min normalized close price
- Directional forecast (up/down) with backtest accuracy shown alongside it
- Interactive Plotly charts: true vs. predicted price, and bar-by-bar point change
- Real market data via yfinance
- Background scheduler refreshes predictions on an interval, independent of page views
- Dockerized, with a `render.yaml` blueprint for deployment

## Tech Stack

| Layer | Tech |
|---|---|
| Model | PyTorch (single-layer LSTM) |
| Backend | Python / Flask + APScheduler |
| Data | yfinance (Yahoo Finance) |
| Frontend | Plotly (server-rendered, embedded via Jinja2) |
| Testing | pytest |
| Container | Docker / gunicorn |

---

## Quickstart

```bash
git clone https://github.com/DebajyotiRay/EURUSD-lstm-dashboard.git
cd EURUSD-lstm-dashboard

pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5001**. The first prediction runs automatically in the background within a few seconds; refresh the page or click **Refresh Now**.

## Run with Docker

```bash
docker build -t eurusd-dashboard .
docker run -p 5001:5001 eurusd-dashboard
```

Open **http://localhost:5001**.

---

## Training

`lstm_model.pth` ships pretrained.

```bash
python train_model.py --epochs 40 --period 60d
```

This fetches ~60 days of EURUSD 15-min history via yfinance (Yahoo's cap for that interval), builds sliding-window sequences, splits them chronologically (train on the earlier portion, validate on the later portion), trains the LSTM, and writes:

- `lstm_model.pth` — updated weights
- `training_report.json` — full metrics + loss history
- `loss_curve.png` — train/validation loss plot

## Evaluation

The model is evaluated on directional accuracy (did it correctly call up vs. down?) and MAE/RMSE on the normalized close price, alongside a naive baseline. Forex at 15-minute resolution is close to a random walk, so beating the naive baseline by any real margin is a meaningful result.

| Metric | LSTM | Naive baseline |
|---|---|---|
| Directional accuracy | 59.7% | n/a — a flat/no-change predictor never calls a direction, so this metric doesn't apply to it. The LSTM's 59.7% should be judged against 50% (random chance). |
| MAE (normalized price) | 0.191 | 0.109 |
| RMSE (normalized price) | 0.230 | 0.179 |

From a training run on ~5,600 bars of EURUSD 15-min history (July 2026). Across several training runs during development, directional accuracy ranged 57-60%, and MAE/RMSE stayed consistently above the naive baseline.

The LSTM does not beat the naive baseline on raw regression error (MAE/RMSE) — a real and expected result for short-horizon forex (see Modeling Experiments below). It does show a modest, reproducible edge over random chance on directional accuracy, which is the more relevant metric for a directional forecast. This should be read as a research finding, not a trading edge — see Limitations.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

Covers the normalization math (including the flat-price division-by-zero edge case), feature-frame construction, and both metrics functions, using synthetic OHLC data.

---

## Model Architecture

| Parameter | Value |
|---|---|
| Type | Single-layer LSTM |
| Input features | 5 (`time_token`, `norm_open`, `norm_high`, `norm_low`, `norm_close`) |
| Hidden size | 100 |
| Output | 1 (predicted normalized close) |
| Sequence length | 60 bars (15 hours of M15 data) |
| Backtest window | 100 steps |

Prices are normalized with a rolling daily min-max (expanding window, resets at midnight) so the model sees relative intraday movement rather than raw pip values. Using an expanding rather than a full-day window avoids leaking the day's future high/low into earlier predictions. A time-of-day token (seconds since midnight / 86400) is appended as a fifth feature.

Training uses gradient clipping (max norm 1.0), `ReduceLROnPlateau` learning-rate scheduling, and early stopping on validation loss (patience 8 epochs, best weights restored).

## Modeling Experiments

A more elaborate version of this model was built and evaluated before settling on the architecture above.

I expanded the 5 input features to 8, adding momentum, rolling volatility, and a cyclical sin/cos time-of-day encoding instead of a raw linear token, plus a deeper 2-layer LSTM with dropout. On identical train/validation splits, this did not outperform the simpler model — validation MAE and RMSE were both modestly worse (0.20 vs 0.164 MAE), and directional accuracy was statistically indistinguishable (56-58% across three separate training runs). While building it I also found and fixed a data-leakage bug (a feature scaler was initially fit on the full dataset instead of the training split only); fixing it barely moved the results, which confirmed the leak wasn't the cause of the underperformance — the added complexity itself just didn't help.

Neither version beat the naive baseline on raw regression error. I kept the simpler architecture and documented the experiment here rather than leaving it out.

## Project Structure

```
eurusd-lstm-dashboard/
├── app.py                      # Flask backend + background prediction scheduler
├── lstm_core.py                # Model, data fetching, preprocessing, inference, metrics (unit-tested)
├── LSTM_model_prediction.py    # Thin CLI wrapper around lstm_core (one-off prediction run)
├── train_model.py              # Training script: fetch data, train, evaluate vs. baseline
├── lstm_model.pth              # Trained model weights
├── requirements.txt
├── Dockerfile
├── render.yaml                 # Render deployment blueprint
├── tests/
│   └── test_lstm_core.py
└── README.md
```

## Limitations

- This is not a trading system. No position sizing, risk management, spread/slippage/commission modeling, or execution logic — it's a directional forecast, nothing more.
- Predictions are on normalized prices (0-1 within each trading day), not raw pips — they aren't directly tradable numbers.
- 15-minute forex is close to a random walk; any accuracy improvement over the naive baseline should be treated as a research result, not a signal to act on.
- The dashboard re-runs the model on a fixed interval and shows the most recent backtest window — it does not track live P&L or compare predictions against what actually happened after the fact.

---

## Troubleshooting

**`JSONDecodeError: Expecting value: line 1 column 1 (char 0)` when fetching data:** Yahoo Finance periodically changes its server-side auth/cookie handling, which breaks older yfinance versions with this exact error.
```bash
pip install --upgrade yfinance
```
Verify with a quick standalone check:
```bash
python -c "import yfinance as yf; print(yf.download('EURUSD=X', period='5d', interval='15m').tail())"
```

**Bottom chart y-axis showing absurd values like "1T" (trillion):** this was a real bug I hit in production. The normalized price sits at exactly 0 at each day's rolling low by construction, so a naive percent-change calculation divides by near-zero there and produces meaningless numbers. Fixed by switching that chart to absolute point change instead of percentage.

## Deployment

### Render

1. Push this repo to GitHub.
2. On [render.com](https://render.com), choose **New → Blueprint**, connect the repo — Render auto-detects `render.yaml` and builds the Dockerfile.
3. Free-tier web services spin down after 15 minutes idle; the first request afterward takes ~10-30s to wake up.

### Hugging Face Spaces

1. Create a new Space, SDK = Docker.
2. Push this repo's contents to the Space's git remote.
3. HF Spaces reads the `Dockerfile` directly — add this to the top of the Space's own README:
   ```yaml
   ---
   title: EURUSD LSTM Dashboard
   sdk: docker
   app_port: 5001
   ---
   ```

### Any Docker host (Fly.io, a VPS, etc.)

```bash
docker build -t eurusd-dashboard .
docker run -p 5001:5001 -e PORT=5001 eurusd-dashboard
```

---

## Disclaimer

This project is for educational and portfolio purposes only. It is not financial advice and should not be used for real trading decisions.
