import json
import logging
import os
import sys
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, redirect, render_template_string, url_for

import lstm_core as core
import xgb_core

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("app")

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.pth")
XGBOOST_MODEL_PATH = os.path.join(BASE_DIR, xgb_core.XGBOOST_MODEL_FILENAME)
RESULTS_PATH = os.path.join(BASE_DIR, "prediction_results.json")


REFRESH_INTERVAL_SECONDS = int(os.environ.get("REFRESH_INTERVAL_SECONDS", "60"))

_model = None
_xgboost_model = None


def get_model():
    global _model
    if _model is None:
        _model = core.load_model(MODEL_PATH)
    return _model


def get_xgboost_model():
    global _xgboost_model
    if _xgboost_model is None and os.path.exists(XGBOOST_MODEL_PATH):
        _xgboost_model = xgb_core.load_xgboost_model(XGBOOST_MODEL_PATH)
    return _xgboost_model


def _build_result_entry(run, data_source):
    metrics = core.regression_metrics(run.true_prices, run.predicted_prices)
    dir_acc = core.directional_accuracy(run.true_prices, run.predicted_prices)
    return {
        "predicted_prices": run.predicted_prices,
        "true_prices": run.true_prices,
        "timestamps": run.timestamps,
        "predicted_change": run.predicted_change,
        "next_prediction": run.next_prediction,
        "data_source": data_source,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "directional_accuracy": dir_acc,
    }


def run_prediction_job():
    try:
        raw_data, data_source = core.get_market_data()
        raw_data = raw_data.iloc[-core.REQUIRED_BARS:]
        feature_frame = core.build_feature_frame(raw_data)

        lstm_model = get_model()
        lstm_run = core.run_backtest_and_forecast(lstm_model, feature_frame)

        results = {
            "lstm": _build_result_entry(lstm_run, data_source),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

        xgboost_model = get_xgboost_model()
        if xgboost_model is not None:
            try:
                xgb_run = xgb_core.run_xgboost_backtest_and_forecast(xgboost_model, feature_frame)
                results["xgboost"] = _build_result_entry(xgb_run, data_source)
            except Exception:


                logger.exception("XGBoost prediction failed this cycle; LSTM result is unaffected.")
        else:
            logger.info("No xgboost_model.json found — skipping XGBoost prediction "
                         "(run train_xgboost_model.py to enable the comparison toggle).")

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f)
        logger.info("Prediction refreshed. Source: %s", data_source)
    except Exception:
        logger.exception("Prediction job failed; keeping last known results.")


def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None, "No predictions generated yet. Click Refresh Now or wait for the first scheduled run."
    try:
        with open(RESULTS_PATH) as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"Failed to parse prediction_results.json: {exc}"


def build_charts(results, key, load_plotlyjs):
    import plotly.graph_objects as go

    timestamps = results["timestamps"]
    true_prices = results["true_prices"]
    pred_prices = results["predicted_prices"]


    PANEL_BG = "#12181F"
    GRID = "#1E2630"
    TEXT = "#9AA5B1"
    FONT = dict(family="Inter, -apple-system, sans-serif", color=TEXT, size=12)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=timestamps, y=true_prices, mode="lines", name="Actual",
        line=dict(color="#3DDC84", width=2),
    ))
    fig_price.add_trace(go.Scatter(
        x=timestamps, y=pred_prices, mode="lines", name="Predicted",
        line=dict(color="#4FA8E0", width=2, dash="dot"),
    ))
    fig_price.update_layout(
        title=dict(text="ACTUAL VS PREDICTED CLOSE (normalised 0-1)", font=dict(size=13, color=TEXT), x=0, y=0.98, yanchor="top"),
        xaxis_title=None, yaxis_title=None,
        hovermode="x unified", legend=dict(orientation="h", y=-0.18, x=0, font=FONT),
        margin=dict(l=45, r=20, t=45, b=55),
        plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
        font=FONT,
    )
    fig_price.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig_price.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID)


    delta_true = [true_prices[i] - true_prices[i - 1] for i in range(1, len(true_prices))]
    delta_pred = [pred_prices[i] - pred_prices[i - 1] for i in range(1, len(pred_prices))]

    fig_pct = go.Figure()
    fig_pct.add_trace(go.Bar(x=timestamps[1:], y=delta_true, name="Actual", marker_color="#3DDC84", opacity=0.85))
    fig_pct.add_trace(go.Bar(x=timestamps[1:], y=delta_pred, name="Predicted", marker_color="#4FA8E0", opacity=0.85))
    fig_pct.update_layout(
        title=dict(text="BAR-BY-BAR CHANGE", font=dict(size=13, color=TEXT), x=0, y=0.98, yanchor="top"),
        xaxis_title=None, yaxis_title=None, barmode="group",
        hovermode="x unified", legend=dict(orientation="h", y=-0.18, x=0, font=FONT),
        margin=dict(l=45, r=20, t=45, b=55),
        plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
        font=FONT,
    )
    fig_pct.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig_pct.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor="#3A4552", linecolor=GRID)

    chart_config = {"displayModeBar": False, "responsive": True}
    price_html = fig_price.to_html(full_html=False, include_plotlyjs="cdn" if load_plotlyjs else False,
                                    div_id=f"priceChart-{key}", config=chart_config)
    pct_html = fig_pct.to_html(full_html=False, include_plotlyjs=False,
                                div_id=f"pctChart-{key}", config=chart_config)
    return price_html, pct_html


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>EURUSD LSTM Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0B0F14;
      --panel: #12181F;
      --panel-raised: #161D26;
      --border: #1E2630;
      --text: #E8ECEF;
      --text-muted: #6B7785;
      --text-dim: #9AA5B1;
      --up: #3DDC84;
      --down: #FF5C5C;
      --accent: #4FA8E0;
      --mono: 'JetBrains Mono', ui-monospace, monospace;
      --sans: 'Inter', -apple-system, sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: var(--sans); background: var(--bg); color: var(--text);
      margin: 0; padding: 32px 16px; display: flex; flex-direction: column; align-items: center;
      -webkit-font-smoothing: antialiased;
    }
    .terminal { width: 100%; max-width: 1040px; }

    .topbar {
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 10px; padding-bottom: 18px; margin-bottom: 22px;
      border-bottom: 1px solid var(--border);
    }
    .ticker { display: flex; align-items: baseline; gap: 10px; }
    .ticker-symbol { font-family: var(--mono); font-size: 1.05em; font-weight: 700; letter-spacing: 0.04em; color: var(--text); }
    .ticker-label { font-size: 0.78em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }

    .badge {
      display: inline-flex; align-items: center; gap: 6px;
      font-family: var(--mono); font-size: 0.72em; font-weight: 500; letter-spacing: 0.05em;
      padding: 4px 10px; border-radius: 20px; text-transform: uppercase;
      background: var(--panel-raised); border: 1px solid var(--border); color: var(--text-dim);
    }
    .live-dot {
      width: 7px; height: 7px; border-radius: 50%; background: var(--up);
      box-shadow: 0 0 0 0 rgba(61, 220, 132, 0.6);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%   { box-shadow: 0 0 0 0 rgba(61, 220, 132, 0.55); }
      70%  { box-shadow: 0 0 0 6px rgba(61, 220, 132, 0); }
      100% { box-shadow: 0 0 0 0 rgba(61, 220, 132, 0); }
    }
    @media (prefers-reduced-motion: reduce) {
      .live-dot { animation: none; }
    }

    .model-toggle {
      display: flex; gap: 8px; margin-bottom: 20px;
    }
    .model-tab {
      font-family: var(--mono); font-size: 0.78em; font-weight: 600; letter-spacing: 0.04em;
      padding: 7px 16px; border-radius: 6px; text-decoration: none; text-transform: uppercase;
      background: var(--panel); border: 1px solid var(--border); color: var(--text-muted);
      transition: border-color 0.15s, color 0.15s;
    }
    .model-tab:hover { border-color: var(--accent); color: var(--text); }
    .model-tab.active {
      background: var(--panel-raised); border-color: var(--accent); color: var(--accent);
    }
    .model-tab.disabled {
      opacity: 0.45; cursor: not-allowed; pointer-events: none;
    }

    .hero { margin-bottom: 20px; }
    .hero-label { font-size: 0.78em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
    .hero-signal {
      font-family: var(--mono); font-weight: 700; font-size: 2.1em; letter-spacing: -0.01em;
      display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
    }
    .hero-signal.up   { color: var(--up); }
    .hero-signal.down { color: var(--down); }
    .hero-signal.err  { color: var(--text-dim); font-size: 1.3em; }
    .hero-change { font-size: 0.55em; color: var(--text-dim); font-weight: 500; }
    .arrow { font-size: 0.75em; }

    .stat-strip {
      display: flex; gap: 1px; background: var(--border);
      border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
      margin-bottom: 18px;
    }
    .stat {
      flex: 1; background: var(--panel); padding: 12px 16px; min-width: 120px;
    }
    .stat-label { font-size: 0.68em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
    .stat-value { font-family: var(--mono); font-size: 1.05em; font-weight: 500; color: var(--text); }

    .meta-row {
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      font-size: 0.8em; color: var(--text-muted); margin-bottom: 22px;
    }
    .meta-row .dot-sep { color: var(--border); }

    button {
      font-family: var(--sans); padding: 7px 16px; font-size: 0.82em; font-weight: 600;
      border: 1px solid var(--border); border-radius: 6px;
      background: var(--panel-raised); color: var(--text); cursor: pointer;
      transition: border-color 0.15s, color 0.15s;
    }
    button:hover { border-color: var(--accent); color: var(--accent); }

    .error-box {
      background: rgba(255, 92, 92, 0.08); color: #ff8585; border: 1px solid rgba(255, 92, 92, 0.3);
      border-radius: 8px; padding: 16px; margin: 16px 0; white-space: pre-wrap; font-size: 0.85em;
      font-family: var(--mono);
    }

    .model-view { display: none; }
    .model-view.active { display: block; }

    .chart-wrap {
      background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
      padding: 12px 8px 4px; margin-bottom: 14px;
    }

    .disclaimer {
      font-size: 0.72em; color: var(--text-muted); text-align: center;
      margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border); line-height: 1.6;
    }
    .disclaimer em { color: var(--text-dim); font-style: normal; }

    @media (max-width: 560px) {
      .hero-signal { font-size: 1.5em; }
      .stat-strip { flex-direction: column; }
    }
  </style>
  <meta http-equiv="refresh" content="30">
</head>
<body>
<div class="terminal">
  <div class="topbar">
    <div class="ticker">
      <span class="ticker-symbol">EUR/USD</span>
      <span class="ticker-label">Forecast Terminal</span>
    </div>
    <span class="badge"><span class="live-dot"></span>{{ source_label }}</span>
  </div>

  {% if error %}
    <div class="hero">
      <div class="hero-label">Status</div>
      <div class="hero-signal err">Prediction unavailable</div>
    </div>
    <div class="error-box">{{ error }}</div>
    <form method="post" action="/refresh"><button type="submit">Run Prediction Now</button></form>
  {% else %}
    <div class="model-toggle">
      <button type="button" class="model-tab" data-model="lstm" onclick="showModel('lstm')">LSTM</button>
      <button type="button" class="model-tab{{ ' disabled' if not xgboost_available else '' }}" data-model="xgboost" onclick="showModel('xgboost')">
        XGBoost{{ '' if xgboost_available else ' (not trained yet)' }}
      </button>
    </div>

    {% for m in models %}
    <div class="model-view" id="view-{{ m.key }}">
      <div class="hero">
        <div class="hero-label">Next-bar forecast &middot; {{ m.key|upper }}</div>
        <div class="hero-signal {{ m.signal_class }}">
          <span><span class="arrow">{{ m.arrow }}</span> {{ m.signal_word }}</span>
          <span class="hero-change">{{ m.signal_change }}</span>
        </div>
      </div>

      <div class="stat-strip">
        <div class="stat"><div class="stat-label">Backtest MAE</div><div class="stat-value">{{ m.mae }}</div></div>
        <div class="stat"><div class="stat-label">Backtest RMSE</div><div class="stat-value">{{ m.rmse }}</div></div>
        <div class="stat"><div class="stat-label">Directional Acc.</div><div class="stat-value">{{ m.dir_acc }}</div></div>
      </div>

      <div class="meta-row">
        <span>Generated {{ generated_at }}</span>
        <span class="dot-sep">&middot;</span>
        <span>Refreshes every {{ refresh_interval }}s</span>
        <span class="dot-sep">&middot;</span>
        <form method="post" action="/refresh"><button type="submit">Refresh now</button></form>
      </div>

      <div class="chart-wrap">{{ m.price_chart | safe }}</div>
      <div class="chart-wrap">{{ m.pct_chart   | safe }}</div>
    </div>
    {% endfor %}
  {% endif %}

  <div class="disclaimer">
    <em>LSTM</em> vs <em>XGBoost</em> comparison &middot; trained on EUR/USD M15 &nbsp;&middot;&nbsp;
    Prices shown are <em>normalised</em> (0-1 within each trading day) &nbsp;&middot;&nbsp;
    Educational / portfolio project only &mdash; not financial advice.
  </div>
</div>
<script>
function showModel(key) {
  document.querySelectorAll('.model-view').forEach(function(el) {
    el.classList.toggle('active', el.id === 'view-' + key);
  });
  document.querySelectorAll('.model-tab').forEach(function(el) {
    el.classList.toggle('active', el.dataset.model === key);
  });
  if (!document.querySelector('.model-tab[data-model="' + key + '"]').classList.contains('disabled')) {
    localStorage.setItem('selectedModel', key);
  }
}
(function () {
  var saved = localStorage.getItem('selectedModel') || 'lstm';
  var tab = document.querySelector('.model-tab[data-model="' + saved + '"]');
  if (!tab || tab.classList.contains('disabled')) saved = 'lstm';
  showModel(saved);
})();
</script>
</body>
</html>
"""


def _build_view_context(key, results, load_plotlyjs):
    change = float(results.get("predicted_change", 0.0))
    is_buy = change > 0
    price_chart, pct_chart = build_charts(results, key, load_plotlyjs)
    mae = results.get("mae")
    rmse = results.get("rmse")
    dir_acc = results.get("directional_accuracy")
    return {
        "key": key,
        "signal_class": "up" if is_buy else "down",
        "arrow": "\u25b2" if is_buy else "\u25bc",
        "signal_word": "UP" if is_buy else "DOWN",
        "signal_change": f"{'+' if is_buy else ''}{change:.5f} next bar (normalised price)",
        "mae": f"{mae:.5f}" if mae is not None else "n/a",
        "rmse": f"{rmse:.5f}" if rmse is not None else "n/a",
        "dir_acc": f"{dir_acc * 100:.1f}%" if dir_acc is not None else "n/a",
        "price_chart": price_chart,
        "pct_chart": pct_chart,
    }


def render_page(error=None):
    all_results, load_err = load_results()
    error = error or load_err

    if error or all_results is None or "lstm" not in all_results:
        return render_template_string(
            PAGE_TEMPLATE,
            error=error or "No predictions generated yet. Click Refresh Now or wait for the first scheduled run.",
            source_label="no data", models=[], generated_at="",
            refresh_interval=REFRESH_INTERVAL_SECONDS, xgboost_available=False,
        )

    xgboost_available = "xgboost" in all_results
    models = [_build_view_context("lstm", all_results["lstm"], load_plotlyjs=True)]
    if xgboost_available:
        models.append(_build_view_context("xgboost", all_results["xgboost"], load_plotlyjs=False))

    source = all_results["lstm"].get("data_source", "")
    return render_template_string(
        PAGE_TEMPLATE,
        error=None,
        source_label=source.lower(),
        models=models,
        generated_at=all_results.get("generated_at", ""),
        refresh_interval=REFRESH_INTERVAL_SECONDS,
        xgboost_available=xgboost_available,
    )


@app.route("/")
def index():
    return render_page()


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/refresh", methods=["POST"])
def refresh():
    run_prediction_job()
    return redirect(url_for("index"))


def start_scheduler():
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(run_prediction_job, "interval", seconds=REFRESH_INTERVAL_SECONDS, next_run_time=datetime.now())
    scheduler.start()
    return scheduler


_scheduler = start_scheduler()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 55)
    print("  EURUSD LSTM Dashboard (Plotly edition)")
    print(f"  Open http://127.0.0.1:{port} in your browser")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=port)
