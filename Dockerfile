# ── EURUSD LSTM Dashboard — Dockerfile ───────────────────────────────────────
# Runs the dashboard in demo mode (yfinance data).
# MT5 live data is not available inside Docker (MT5 is Windows-only).
#
# Local build & run:
#   docker build -t eurusd-dashboard .
#   docker run -p 5001:5001 eurusd-dashboard
#   -> Open http://localhost:5001
#
# Deploy platforms (Render, Fly.io, Hugging Face Spaces) set the PORT env var
# for you automatically; gunicorn below reads it via the shell form CMD.

FROM python:3.11-slim

WORKDIR /app

# System deps needed to build some scientific-python wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lstm_core.py .
COPY app.py .
COPY LSTM_model_prediction.py .
COPY train_model.py .
COPY lstm_model.pth .

# Refresh interval for the background scheduler (seconds). Override at deploy
# time with -e REFRESH_INTERVAL_SECONDS=120 if you want less frequent refreshes.
ENV REFRESH_INTERVAL_SECONDS=60
ENV PORT=5001

EXPOSE 5001

# Single worker: the background APScheduler job runs in-process, so multiple
# gunicorn workers would each run their own scheduler and duplicate work.
# For more concurrency, move the scheduler into a separate worker process.
CMD gunicorn --workers 1 --threads 4 --bind 0.0.0.0:${PORT} --timeout 120 app:app
