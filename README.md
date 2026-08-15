# EURUSD Forecasting Dashboard — LSTM vs. XGBoost vs. Ridge Regression

A forex forecasting dashboard that trains and compares three fundamentally different machine learning approaches — a recurrent neural network (LSTM), gradient-boosted trees (XGBoost), and regularized linear regression (Ridge) — on the same EUR/USD 15-minute candle data, and serves live-updating predictions from all three through a Flask + Plotly web dashboard with a model comparison toggle.

**[Live demo](https://eurusd-lstm-dashboard.onrender.com)** — hosted on Render's free tier, so it sleeps after 15 minutes of inactivity and the first load can take 10-30 seconds to wake up.

![Dashboard screenshot](docs/screenshot.png)

> No broker account needed. The dashboard runs entirely on free historical data from Yahoo Finance — no login, no install, works anywhere.

---

## What this project demonstrates

- Comparing three genuinely different modeling approaches — a sequence-aware neural network, a non-sequential tree ensemble, and a plain linear model — on identical data, identical chronological train/validation splits, and identical evaluation functions, so the comparison is actually fair rather than apples-to-oranges.
- Evaluating every model against a naive baseline ("predict next = last observed close") rather than reporting accuracy in isolation — a model that can't beat that baseline hasn't demonstrated it learned anything useful about price movement specifically.
- Treating a negative or inconclusive result as worth reporting, not hiding — an earlier, more complex LSTM variant was tested and didn't outperform the simpler one, and that's documented here rather than left out.
- Serving all three trained models behind a single Flask dashboard with background job scheduling, health checks, and Docker packaging, with each model toggle degrading gracefully if that model hasn't been trained yet rather than crashing.
- Structuring the ML code as unit-tested, reusable functions (`lstm_core.py`, `xgb_core.py`, `ridge_core.py`) instead of one monolithic script, with 28 tests covering the parts most likely to have a subtle correctness bug.

## Features

- Three trained models — LSTM, XGBoost, and Ridge Regression — forecasting the next 15-min normalized close price from the same input data
- A model comparison toggle on the dashboard: switch between all three instantly, client-side, with no page reload, since all three predictions are already computed in the background before anyone clicks anything
- Directional forecast (up/down) with backtest MAE, RMSE, and directional accuracy shown for whichever model is selected
- Interactive Plotly charts per model: true vs. predicted price, and bar-by-bar point change
- Real market data via yfinance
- Background scheduler refreshes all three models' predictions on an interval, independent of page views
- Dockerized, with a `render.yaml` blueprint for deployment

## Tech Stack

| Layer | Tech |
|---|---|
| Models | PyTorch (LSTM) · XGBoost (gradient-boosted trees) · NumPy (Ridge regression, closed-form solution, no ML library needed) |
| Backend | Python / Flask + APScheduler |
| Data | yfinance (Yahoo Finance) |
| Frontend | Plotly (server-rendered, embedded via Jinja2), vanilla JS for the model toggle |
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

Open **http://127.0.0.1:5001**. The LSTM prediction runs automatically in the background within a few seconds; refresh the page or click **Refresh Now**. `lstm_model.pth` ships pretrained, so this works immediately — the XGBoost and Ridge tabs will show "not trained yet" until you run their training scripts below, which is expected, not an error.

## Run with Docker

```bash
docker build -t eurusd-dashboard .
docker run -p 5001:5001 eurusd-dashboard
```

Open **http://localhost:5001**.

---

## Training

Each model has its own training script, but all three share the exact same data pipeline, feature construction, and chronological train/validation split — reused directly from `train_model.py` rather than reimplemented — so results are directly comparable.

### LSTM

```bash
python train_model.py --epochs 40 --period 60d
```

Fetches ~60 days of EURUSD 15-min history via yfinance (Yahoo's cap for that interval), builds sliding-window sequences, splits them chronologically, trains the LSTM, and writes `lstm_model.pth`, `training_report.json`, and `loss_curve.png`.

### XGBoost (optional)

```bash
python train_xgboost_model.py --period 60d
```

Same windows as the LSTM, flattened from `(60, 5)` into a 300-length feature vector, since XGBoost has no built-in concept of sequence order. Uses XGBoost's native `Booster` API rather than the sklearn-style wrapper — `XGBRegressor.save_model()` calls into an internal scikit-learn API that changed in scikit-learn 1.6+, breaking on any reasonably current environment across every version I tested. The native API sidesteps that entirely. Writes `xgboost_model.json` and `xgboost_training_report.json`.

### Ridge Regression (optional)

```bash
python train_ridge_model.py --period 60d
```

Same flattened windows as XGBoost, fit with a closed-form solution (no training loop, no epochs — solved directly via `np.linalg.solve`). The script searches over several regularization strengths (`alpha`) and picks whichever gives the lowest validation MAE, logging train and validation metrics at every alpha tried, not just the winner. 300 flattened features on a few thousand rows is enough to make the underlying matrix nearly singular without regularization, so the fit function explicitly checks for non-finite weights and raises a clear error rather than silently returning garbage. Writes `ridge_model.json` and `ridge_training_report.json`.

Both XGBoost and Ridge are optional in the sense that the dashboard works fine without them — their toggle buttons just show "not trained yet" until you run the corresponding script, and the background job logs that it's skipping them rather than failing.

## Evaluation

Every model is evaluated on directional accuracy (did it correctly call up vs. down?) and MAE/RMSE on the normalized close price, alongside a naive baseline. Forex at 15-minute resolution is close to a random walk, so beating the naive baseline by any real margin is a meaningful result.

| Metric | LSTM | XGBoost | Ridge | Naive baseline |
|---|---|---|---|---|
| Directional accuracy | 59.7% | 60.3% | 62.0% | n/a — a flat/no-change predictor never calls a direction, so this metric doesn't apply to it. Judge every model's accuracy against 50% (random chance) instead. |
| MAE (normalized price) | 0.191 | 0.109 | 0.124 | 0.109 |
| RMSE (normalized price) | 0.230 | 0.162 | 0.185 | 0.179 |

From actual training runs on real EURUSD 15-min history. XGBoost's RMSE was the first result in this project to actually beat the naive baseline, not just tie or lose to it — its MAE ties the baseline exactly. Ridge Regression, the simplest model in the comparison, achieved the best directional accuracy of all three (62.0%), which is a genuinely interesting result: the most complex model here (the LSTM) has the weakest directional accuracy of the three, and a plain linear model edges out both the LSTM and XGBoost on that specific metric. None of the three models beat the naive baseline on raw regression error by a meaningful margin, which is a real and expected result for short-horizon forex (see Modeling Experiments below). Numbers shift somewhat between runs since yfinance only serves the trailing ~60 days at 15-min resolution, so the exact window fetched changes day to day — directional accuracy for the LSTM has ranged 57-60% across separate runs.

This should be read as a research finding, not a trading edge — see Limitations.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

28 tests across three files. `test_lstm_core.py` covers the normalization math (including the flat-price division-by-zero edge case) and feature-frame construction. `test_xgb_core.py` covers the window-flattening logic, including a test that the flatten shape at inference time matches the shape at training time — a silent mismatch there wouldn't raise an error, it would just make predictions wrong. `test_ridge_core.py` includes a test that the fitted weights exactly match a hand-derived closed-form solution, which is the strongest kind of correctness check available for this model — not just "does it run," but "is the math actually right." All using synthetic OHLC data, no network access or pretrained weights required.

---

## Model Architecture

### LSTM

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

### XGBoost

Trained on the identical 5-feature windows as the LSTM, flattened into a 300-length vector. Shallow trees (`max_depth=4`) and early stopping against the validation set guard against overfitting — the training script explicitly logs the train/validation MAE gap as a direct check, not just the validation number in isolation.

### Ridge Regression

The same flattened 300-feature windows, fit with L2-regularized linear regression. No iterative training — the closed-form solution is computed once:

```
w = (XᵀX + αI)⁻¹ Xᵀy
```

with the feature matrix and target mean-centered first so the intercept isn't itself penalized. `alpha` controls how strongly large weights are discouraged; the training script searches several values and keeps whichever generalizes best to the validation set. This is the simplest model in the comparison — the point of including it is to test whether the problem needs nonlinearity at all, or whether even a straight line captures most of the learnable signal.

## Modeling Experiments

A more elaborate version of the LSTM was built and evaluated before settling on the architecture above.

I expanded the 5 input features to 8, adding momentum, rolling volatility, and a cyclical sin/cos time-of-day encoding instead of a raw linear token, plus a deeper 2-layer LSTM with dropout. On identical train/validation splits, this did not outperform the simpler model — validation MAE and RMSE were both modestly worse (0.20 vs 0.164 MAE), and directional accuracy was statistically indistinguishable (56-58% across three separate training runs). While building it I also found and fixed a data-leakage bug (a feature scaler was initially fit on the full dataset instead of the training split only); fixing it barely moved the results, which confirmed the leak wasn't the cause of the underperformance — the added complexity itself just didn't help.

Neither version beat the naive baseline on raw regression error. I kept the simpler LSTM architecture and documented the experiment here rather than leaving it out.

Adding XGBoost and Ridge Regression as comparison models later reframed this finding further. Both simpler, non-sequential models slightly outperformed the LSTM on this same data — Ridge, the simplest model in the entire comparison, actually achieved the best directional accuracy of the three. That raises a real question worth sitting with: is explicit sequence-awareness earning its added complexity here at all, or is the 15-minute EUR/USD signal better captured by a flat set of feature interactions (or even a straight line) than by temporal order? I don't think this project answers that question definitively — one data window, one train/validation split — but it's a more interesting and more honest place to land than any single model's number in isolation would suggest.

## Project Structure

```
eurusd-lstm-dashboard/
├── app.py                      # Flask backend + background scheduler + model comparison toggle
├── lstm_core.py                # LSTM model, data fetching, preprocessing, inference, metrics (unit-tested)
├── xgb_core.py                 # XGBoost inference — reuses lstm_core's data pipeline and metrics
├── ridge_core.py               # Ridge regression fit/inference — closed-form, pure NumPy
├── LSTM_model_prediction.py    # Thin CLI wrapper around lstm_core (one-off prediction run)
├── train_model.py              # LSTM training: fetch data, train, evaluate vs. baseline
├── train_xgboost_model.py      # XGBoost training: same data/split, different model
├── train_ridge_model.py        # Ridge training: same data/split, alpha search
├── lstm_model.pth              # Trained LSTM weights
├── xgboost_model.json          # Trained XGBoost weights (optional — generated by train_xgboost_model.py)
├── ridge_model.json            # Trained Ridge weights (optional — generated by train_ridge_model.py)
├── requirements.txt
├── Dockerfile
├── render.yaml                 # Render deployment blueprint
├── tests/
│   ├── test_lstm_core.py
│   ├── test_xgb_core.py
│   └── test_ridge_core.py
└── README.md
```

## Limitations

- This is not a trading system. No position sizing, risk management, spread/slippage/commission modeling, or execution logic — every model here produces a directional forecast, nothing more.
- Predictions are on normalized prices (0-1 within each trading day), not raw pips — they aren't directly tradable numbers.
- 15-minute forex is close to a random walk; any accuracy improvement over the naive baseline should be treated as a research result, not a signal to act on.
- The dashboard re-runs all three models on a fixed interval and shows the most recent backtest window — it does not track live P&L or compare predictions against what actually happened after the fact.
- The three models are compared on identical data and identical splits, but that comparison is still a single train/validation split on one historical window — not a statistically rigorous multi-seed, multi-period benchmark.

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

**Bottom chart y-axis showing absurd values like "1T" (trillion), or the "next bar" figure showing something like +342931091785%:** both were real bugs I hit in production, same root cause. The normalized price sits at exactly 0 at each day's rolling low by construction, so any percent-change calculation dividing by a value near zero — whether that's the model's last prediction or the last actual price — can produce meaningless numbers. I didn't find a "safer" number to divide by; I removed the division entirely. Both the chart and the "next bar" figure now show absolute point change instead of a percentage.

**A second model's chart renders as an empty box when using the comparison toggle:** this was a real bug from switching the toggle to a no-reload design — both models' full HTML, including their Plotly chart `<div>` elements, get rendered into the page at once (one hidden via CSS), and the chart-building code originally gave every model's charts the same fixed HTML id. Browsers only resolve the first matching id, so the second model's chart never received its data. Fixed by giving each model's charts a unique id.

## Deployment

### Render

1. Push this repo to GitHub.
2. On [render.com](https://render.com), choose **New → Blueprint**, connect the repo — Render auto-detects `render.yaml` and builds the Dockerfile.
3. Free-tier web services spin down after 15 minutes idle; the first request afterward takes ~10-30s to wake up.
4. `xgboost_model.json` and `ridge_model.json` are optional — the Dockerfile copies whatever's present without failing the build if they're missing, so a fresh deploy works fine even before you've trained the comparison models.

### Hugging Face Spaces

1. Create a new Space, SDK = Docker.
2. Push this repo's contents to the Space's git remote.
3. HF Spaces reads the `Dockerfile` directly — add this to the top of the Space's own README:
   ```yaml
   ---
   title: EURUSD Forecasting Dashboard
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
