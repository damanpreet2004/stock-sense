"""
app.py — Stock Market Prediction PRO (Modular Wizard Edition)
================================================================
Run with:  python -m streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

# ── project modules ───────────────────────────────────────────────────────────
from data_utils        import fetch_ohlcv, fetch_live_price, fetch_ticker_info, compute_indicators, parse_csv_upload
from ml_models         import train_models, predict_next
from sentiment_utils   import analyse_sentiment
from portfolio         import init_portfolio, add_holding, remove_holding, compute_pnl, portfolio_summary
from alerts            import init_alerts, check_thresholds, get_alerts_log, clear_alerts
from report_generator  import build_pdf

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stock Sense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def scroll_to_top():
    components.html(
        """<script>
            function scrollAppToTop() {
                const doc = window.parent.document;
                const targets = [
                    doc.querySelector('[data-testid="stMain"]'),
                    doc.querySelector('[data-testid="stAppViewContainer"]'),
                    doc.documentElement,
                    doc.body,
                ];
                targets.forEach(el => { if (el) el.scrollTo(0, 0); });
                window.parent.scrollTo(0, 0);
            }
            scrollAppToTop();
            setTimeout(scrollAppToTop, 50);
            setTimeout(scrollAppToTop, 200);
        </script>""",
        height=0,
    )

st.markdown("""
<style>
/* ── global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1d23 0%, #22262e 100%);
    border: 1px solid #2d3139;
    border-radius: 12px;
    padding: 16px 20px;
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,255,140,0.12);
}

/* ── tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #0e1117;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #2d3139;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    color: #aab0bb;
    background: transparent;
    border: none;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00c87a, #00a86b) !important;
    color: #fff !important;
    box-shadow: 0 4px 12px rgba(0,200,122,0.3);
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117 0%, #141820 100%);
    border-right: 1px solid #2d3139;
}

/* ── View segmented control (pill buttons) ── */
[data-testid="stButtonGroup"] {
    background: #0e1117;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #2d3139;
    gap: 6px;
}
[data-testid="stButtonGroup"] button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
    color: #aab0bb !important;
    transition: all 0.2s !important;
}
[data-testid="stButtonGroup"] button:focus,
[data-testid="stButtonGroup"] button:focus-visible,
[data-testid="stButtonGroup"] button:active {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"],
[data-testid="stButtonGroup"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #00c87a, #00a86b) !important;
    color: #fff !important;
    box-shadow: 0 4px 12px rgba(0,200,122,0.3) !important;
}

/* ── button ── */
.stButton > button {
    background: linear-gradient(135deg, #00c87a, #00a86b);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,200,122,0.4);
}

/* ── alert banners ── */
.alert-up   { background:#0a2e1a; border:2px solid #00c87a; border-radius:10px; padding:14px 20px; }
.alert-down { background:#2e0a0a; border:2px solid #ff4d4d; border-radius:10px; padding:14px 20px; }

/* ── section header ── */
.section-hdr {
    font-size: 1.15rem;
    font-weight: 700;
    color: #00c87a;
    border-left: 4px solid #00c87a;
    padding-left: 12px;
    margin: 18px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

init_portfolio()
init_alerts()

for key, default in [
    ("df", None),
    ("ticker", "AAPL"),
    ("ticker_info", {}),
    ("start_date", date.today() - timedelta(days=365 * 2)),
    ("end_date",   date.today() - timedelta(days=1)),
    ("pred_fig",   None),
    ("last_metrics", None),
    ("last_sentiment", (None, None)),
    ("last_sentiment_ticker", ""),
    ("analysis_step", 1),
    ("scroll_to_top", False),
    ("prev_view", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 Stock Sense")
    try:
        view_mode = st.segmented_control(
            "View",
            ["📊 Stock Analysis", "💼 Portfolio & Alerts"],
            default="📊 Stock Analysis",
            key="view_mode_control",
        )
    except AttributeError:
        view_mode = st.radio(
            "View",
            ["📊 Stock Analysis", "💼 Portfolio & Alerts"],
            index=0,
            horizontal=True,
            key="view_mode_control",
        )
    if not view_mode:
        view_mode = "📊 Stock Analysis"

    if st.session_state.get("prev_view") is not None and st.session_state.prev_view != view_mode:
        st.session_state.scroll_to_top = True
    st.session_state.prev_view = view_mode

    st.markdown("---")

    # ── ticker & dates ────────────────────────────────────────────────────────
    st.markdown("### 🔍 Data Selection")
    ticker_input = st.text_input("Ticker Symbol", value=st.session_state.ticker,
                                  placeholder="e.g. AAPL, RELIANCE.NS, BTC-USD").upper().strip()
    start_date = st.date_input("Start Date", value=st.session_state.start_date, max_value=date.today() - timedelta(days=30))
    end_date   = st.date_input("End Date",   value=st.session_state.end_date,   max_value=date.today())

    fetch_btn = st.button("🔄 Fetch / Refresh Data", use_container_width=True)
    st.markdown("---")

    # ── CSV upload ────────────────────────────────────────────────────────────
    st.markdown("### 📂 Or Upload CSV")
    uploaded_csv = st.file_uploader("OHLCV CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded_csv:
        csv_df, csv_err = parse_csv_upload(uploaded_csv)
        if csv_err:
            st.error(csv_err)
        else:
            st.session_state.df = csv_df
            st.session_state.ticker = "CSV"
            st.success(f"Loaded {len(csv_df)} rows from CSV.")
    st.markdown("---")

    # ── alerts ────────────────────────────────────────────────────────────────
    st.markdown("### 🔔 Price Alerts")
    upper_target = st.number_input("Upper Target ($)", min_value=0.0, value=0.0, step=0.5,
                                    help="Alert if price rises above this level.")
    lower_stop   = st.number_input("Lower Stop ($)",   min_value=0.0, value=0.0, step=0.5,
                                    help="Alert if price falls below this level.")
    st.markdown("---")

    # ── news settings ─────────────────────────────────────────────────────────
    st.markdown("### 📰 News & Sentiment")
    newsapi_key   = st.text_input("NewsAPI Key (optional)", type="password",
                                   help="Get a free key at newsapi.org for fallback news.")
    use_finbert   = st.toggle("Use FinBERT (slower, more accurate)", value=False,
                               help="Downloads ~500 MB of model weights on first use.")
    st.markdown("---")

    st.markdown("<small style='color:#555'>Stock Sense · Modular Edition</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

if fetch_btn or st.session_state.df is None:
    if ticker_input:
        with st.spinner(f"Fetching {ticker_input} data…"):
            df, err = fetch_ohlcv(ticker_input, start_date, end_date)
        if err:
            st.error(err)
        else:
            st.session_state.df            = df
            st.session_state.ticker        = ticker_input
            st.session_state.start_date    = start_date
            st.session_state.end_date      = end_date
            st.session_state.ticker_info   = fetch_ticker_info(ticker_input)
            st.session_state.pred_fig      = None   # invalidate cached chart
            st.session_state.last_metrics  = None

# ── convenience aliases ───────────────────────────────────────────────────────

df          = st.session_state.df
ticker      = st.session_state.ticker
ticker_info = st.session_state.ticker_info
live_price  = fetch_live_price(ticker) if ticker != "CSV" else None

# ── check alerts ──────────────────────────────────────────────────────────────

alerts_triggered = check_thresholds(
    ticker,
    live_price,
    upper_target if upper_target > 0 else None,
    lower_stop   if lower_stop   > 0 else None,
)

# ── App title ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex; align-items:center; gap:16px; margin-bottom:4px;'>
  <span style='font-size:2.4rem;'>📈</span>
  <div>
    <h1 style='margin:0; font-size:1.9rem; font-weight:700;'>Stock Sense</h1>
    <p style='margin:0; color:#6b7280; font-size:0.95rem;'>
        {ticker_info.get('name', ticker)}
        {'&nbsp;·&nbsp;' + ticker_info.get('sector','') if ticker_info.get('sector','') not in ('','N/A') else ''}
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── alert banners (top of page) ───────────────────────────────────────────────
for al in alerts_triggered:
    if "Upper" in al["type"]:
        st.markdown(f"<div class='alert-up'>🟢 <b>UPPER TARGET HIT</b> — {ticker} reached <b>${al['price']}</b> ≥ target <b>${al['threshold']}</b></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alert-down'>🔴 <b>LOWER STOP HIT</b> — {ticker} fell to <b>${al['price']}</b> ≤ stop <b>${al['threshold']}</b></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Scroll to top hook ────────────────────────────────────────────────────────
if st.session_state.get("scroll_to_top"):
    scroll_to_top()
    st.session_state.scroll_to_top = False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN NAVIGATION & CONTENT
# ══════════════════════════════════════════════════════════════════════════════

if view_mode == "📊 Stock Analysis":
    step = st.session_state.get("analysis_step", 1)

    # ── Step progress indicator ────────────────────────────────────────────────
    st.progress(step / 3.0)
    st.markdown(f"<small style='color:#00c87a; font-weight:600;'>Step {step} of 3</small>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — OVERVIEW & EDA
    # ──────────────────────────────────────────────────────────────────────────
    if step == 1:
        if df is None:
            st.info("👈 Enter a ticker in the sidebar and click **Fetch / Refresh Data**.")
            st.stop()

        # ── key metrics row ───────────────────────────────────────────────────
        df_ind = compute_indicators(df)

        c1, c2, c3, c4, c5 = st.columns(5)
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
        day_chg    = last_close - prev_close
        day_chg_pct = (day_chg / prev_close * 100) if prev_close else 0

        c1.metric("Last Close",     f"${last_close:,.2f}",  f"{day_chg:+.2f} ({day_chg_pct:+.2f}%)")
        c2.metric("52-W High",      f"${df['Close'].max():,.2f}")
        c3.metric("52-W Low",       f"${df['Close'].min():,.2f}")
        c4.metric("Avg Volume",
                  f"{int(df['Volume'].mean()):,}" if "Volume" in df.columns else "N/A")
        rsi_last = df_ind["RSI"].dropna().iloc[-1] if "RSI" in df_ind.columns and not df_ind["RSI"].dropna().empty else None
        c5.metric("RSI (14)",       f"{rsi_last:.1f}" if rsi_last else "N/A")

        st.markdown("---")

        # ── price chart + MAs ─────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>Price History with Moving Averages</div>", unsafe_allow_html=True)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df_ind["Date"], y=df_ind["Close"],
                                       name="Close", line=dict(color="#00ff9d", width=2)))
        for ma, col in [("MA20", "orange"), ("MA50", "cyan"), ("MA5", "#ff7eb3"), ("MA10", "#a78bfa")]:
            if ma in df_ind.columns:
                fig_price.add_trace(go.Scatter(x=df_ind["Date"], y=df_ind[ma],
                                               name=ma, line=dict(color=col, width=1, dash="dot"),
                                               opacity=0.8))
        fig_price.update_layout(template="plotly_dark", height=420,
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                 margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_price, use_container_width=True)

        # ── RSI ───────────────────────────────────────────────────────────────
        col_rsi, col_vol = st.columns(2)
        with col_rsi:
            st.markdown("<div class='section-hdr'>RSI (14)</div>", unsafe_allow_html=True)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_ind["Date"], y=df_ind["RSI"],
                                         name="RSI", line=dict(color="#f59e0b", width=1.5)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought 70")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold 30")
            fig_rsi.update_layout(template="plotly_dark", height=280,
                                   yaxis=dict(range=[0, 100]),
                                   margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_rsi, use_container_width=True)

        # ── Volume ────────────────────────────────────────────────────────────
        with col_vol:
            if "Volume" in df.columns:
                st.markdown("<div class='section-hdr'>Trading Volume</div>", unsafe_allow_html=True)
                vol_colors = ["#00c87a" if c >= o else "#ff4d4d"
                              for c, o in zip(df["Close"], df["Open"] if "Open" in df.columns else df["Close"])]
                fig_vol = go.Figure(go.Bar(x=df_ind["Date"], y=df_ind["Volume"],
                                           marker_color=vol_colors, name="Volume"))
                fig_vol.update_layout(template="plotly_dark", height=280,
                                       margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_vol, use_container_width=True)

        # ── Rolling Volatility ────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>20-Day Rolling Annualised Volatility (%)</div>", unsafe_allow_html=True)
        fig_vol2 = go.Figure(go.Scatter(x=df_ind["Date"], y=df_ind["rolling_vol_20"],
                                        fill="tozeroy", line=dict(color="#818cf8"),
                                        name="Volatility %"))
        fig_vol2.update_layout(template="plotly_dark", height=250,
                                margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_vol2, use_container_width=True)

        # ── EDA Stats table ───────────────────────────────────────────────────
        with st.expander("📋 Descriptive Statistics"):
            desc = df[["Close", "Volume"]].describe().round(2) if "Volume" in df.columns else df[["Close"]].describe().round(2)
            st.dataframe(desc, use_container_width=True)

        # ── Company info ──────────────────────────────────────────────────────
        if ticker_info.get("description"):
            with st.expander("🏢 About the Company"):
                st.write(ticker_info["description"][:1200] + "…" if len(ticker_info.get("description","")) > 1200 else ticker_info.get("description",""))

        st.markdown("---")
        if st.button("Proceed to ML & Sentiment ➡️", use_container_width=True):
            st.session_state.analysis_step = 2
            st.session_state.scroll_to_top = True
            st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — ML PREDICTIONS + NEWS & SENTIMENT
    # ──────────────────────────────────────────────────────────────────────────
    elif step == 2:
        if df is None:
            st.info("Fetch data first.")
            st.stop()

        # ── SUB-SECTION 1: ML PREDICTIONS ──────────────────────────────────────
        st.markdown("### 🤖 ML Price Prediction")

        col_cfg, col_info = st.columns([1, 2])
        with col_cfg:
            horizon = st.selectbox("Forecast Horizon (days)", [1, 3, 7], index=0,
                                   help="Number of trading days ahead to predict.")
            train_btn = st.button("⚙️ Train / Refresh Models", use_container_width=True)

        with col_info:
            st.info(
                "Models are trained on **80 % of the historical data** "
                "(chronological split, no shuffling to avoid lookahead bias).\n\n"
                "The remaining **20 %** is used as a held-out test set "
                "to compute MAE, RMSE, MAPE, Accuracy, and R²."
            )

        # ── train ─────────────────────────────────────────────────────────────
        try:
            with st.spinner("Training models… (cached after first run)"):
                model_result = train_models(
                    ticker, str(st.session_state.start_date), str(st.session_state.end_date),
                    horizon, df
                )
            st.session_state.last_metrics = model_result["metrics"]
        except ValueError as ve:
            st.warning(f"⚠️ {ve}")
            st.stop()
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.stop()

        metrics = model_result["metrics"]
        y_test  = model_result["y_test"]
        dates_test = pd.to_datetime(model_result["dates_test"])

        # ── Metrics comparison table ──────────────────────────────────────────
        st.markdown("<div class='section-hdr'>Test-Set Evaluation Metrics</div>", unsafe_allow_html=True)
        metrics_rows = []
        for mname, short in [("LinearRegression", "Linear Regression"), ("RandomForestRegressor", "Random Forest")]:
            m = metrics[mname]
            mape_val = m.get("MAPE", np.nan)
            acc_val  = m.get("Accuracy_pct", max(0.0, round(100.0 - mape_val, 2)) if not np.isnan(mape_val) else "N/A")
            metrics_rows.append({
                "Model": short,
                "MAE ($)": f"${m['MAE']:.4f}",
                "RMSE ($)": f"${m['RMSE']:.4f}",
                "MAPE (%)": f"{mape_val:.2f}%" if isinstance(mape_val, (int, float)) else "N/A",
                "Accuracy (%)": f"{acc_val:.2f}%" if isinstance(acc_val, (int, float)) else "N/A",
                "R²": f"{m['R2']:.4f}",
            })
        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption("Note: Accuracy (%) is estimated as 100% − MAPE (min 0%), representing average percentage precision, distinct from R² which measures goodness of fit.")

        st.markdown("---")

        # ── Predicted vs Actual chart ──────────────────────────────────────────
        st.markdown("<div class='section-hdr'>Predicted vs Actual (Test Period)</div>", unsafe_allow_html=True)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=dates_test, y=y_test,
                                      name="Actual Close", line=dict(color="#aab0bb", width=2)))
        colors_pred = {"LinearRegression": "#f59e0b", "RandomForestRegressor": "#00ff9d"}
        for mname, col in colors_pred.items():
            fig_pred.add_trace(go.Scatter(x=dates_test, y=metrics[mname]["pred"],
                                          name=mname, line=dict(color=col, width=1.5, dash="dot")))
        fig_pred.update_layout(template="plotly_dark", height=420,
                                xaxis_title="Date", yaxis_title="Price ($)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_pred, use_container_width=True)
        st.session_state.pred_fig = fig_pred

        # ── Next-day forecast ──────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>Next-Horizon Price Forecast</div>", unsafe_allow_html=True)
        last_close = float(df["Close"].iloc[-1])
        fc_cols = st.columns(3)
        fc_cols[0].metric("Current Price", f"${last_close:,.2f}")
        for i, (mname, short) in enumerate([("LinearRegression", "Lin. Reg."), ("RandomForestRegressor", "Rnd. Forest")]):
            forecast = predict_next(model_result, mname, df, horizon)
            if forecast:
                delta = forecast - last_close
                fc_cols[i+1].metric(f"{short} Forecast (+{horizon}d)",
                                     f"${forecast:,.2f}", f"{delta:+.2f}")
            else:
                fc_cols[i+1].metric(f"{short} Forecast (+{horizon}d)", "N/A")

        # ── Methodology expander ──────────────────────────────────────────────
        with st.expander("📖 Methodology & Features Used"):
            st.markdown("""
**Feature Engineering**

| Feature Group      | Columns Generated |
|--------------------|-------------------|
| Lagged Returns     | `ret_lag1`, `ret_lag2`, `ret_lag3`, `ret_lag5`, `ret_lag10` |
| Moving Averages    | `ma5`, `ma10`, `ma20`, `ma50` |
| MA Ratios          | `ma5_ratio`, …, `ma50_ratio` (Close ÷ MA) |
| RSI                | 14-period Wilder RSI |
| Rolling Volatility | 20-day std of daily returns |
| Volume Features    | `vol_chg`, `vol_ma5` (5-d avg vol ÷ 20-d avg vol) |
| OHLC Features      | `hl_range` (H-L / Close), `oc_change` (Close-Open / Open) |

**Train / Test Split**

The data is split **chronologically** at the 80th percentile of rows — the earliest 80 % form the training set and the latest 20 % are the held-out test set.
Shuffling is deliberately avoided to prevent **lookahead bias** (a common mistake in time-series modelling).

**Models**

- **LinearRegression** — establishes a baseline; interpretable and fast.
- **RandomForestRegressor** — 200 trees, random_state=42; captures non-linear interactions between features.

**Target**

`target_close` = `Close` shifted back by *horizon* rows, so the model learns to predict the price *horizon* trading days into the future.

**Metrics (test set only)**

| Metric | Formula |
|--------|---------|
| MAE | Mean Absolute Error — average absolute difference in dollars |
| RMSE | Root Mean Squared Error — penalises large errors in dollars |
| MAPE | Mean Absolute Percentage Error — average relative error (%) |
| Accuracy (%) | 100 − MAPE (min 0%) — approximate percentage-based prediction accuracy |
| R² | Coefficient of determination — 1.0 = perfect fit |
            """)

        # ── VISUAL DIVIDER ────────────────────────────────────────────────────
        st.markdown("---")

        # ── SUB-SECTION 2: NEWS & SENTIMENT ────────────────────────────────────
        st.markdown(f"### 📰 News & Sentiment for **{ticker}**")
        col_btn, col_tog = st.columns([2, 3])
        run_sentiment = col_btn.button("🔍 Fetch Headlines & Score", use_container_width=True)

        if run_sentiment or (st.session_state.last_sentiment_ticker == ticker and st.session_state.last_sentiment[0] is not None):
            if run_sentiment or st.session_state.last_sentiment_ticker != ticker:
                with st.spinner("Fetching news and scoring sentiment…"):
                    sent_df, sent_summary = analyse_sentiment(ticker, newsapi_key, use_finbert)
                st.session_state.last_sentiment = (sent_df, sent_summary)
                st.session_state.last_sentiment_ticker = ticker
            else:
                sent_df, sent_summary = st.session_state.last_sentiment

            if sent_df is None or sent_df.empty:
                st.warning(f"No news headlines found for **{ticker}**. Try adding a NewsAPI key in the sidebar.")
            else:
                # ── Aggregate gauge bar ───────────────────────────────────────
                st.markdown("<div class='section-hdr'>Aggregate Sentiment</div>", unsafe_allow_html=True)
                g1, g2, g3, g4 = st.columns(4)
                compound = sent_summary["mean_compound"]
                overall  = "Positive 😊" if compound >= 0.05 else ("Negative 😟" if compound <= -0.05 else "Neutral 😐")
                g1.metric("Overall Sentiment", overall)
                g2.metric("Mean Score",  f"{compound:.4f}")
                g3.metric("Positive %",  f"{sent_summary['positive_pct']} %")
                g4.metric("Negative %",  f"{sent_summary['negative_pct']} %")

                # stacked bar chart
                fig_bar = go.Figure(go.Bar(
                    x=["Positive", "Neutral", "Negative"],
                    y=[sent_summary["positive_pct"], sent_summary["neutral_pct"], sent_summary["negative_pct"]],
                    marker_color=["#00c87a", "#a78bfa", "#ff4d4d"],
                    text=[f"{v:.1f}%" for v in [sent_summary["positive_pct"], sent_summary["neutral_pct"], sent_summary["negative_pct"]]],
                    textposition="auto",
                ))
                fig_bar.update_layout(template="plotly_dark", height=260, showlegend=False,
                                      yaxis_title="%", margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_bar, use_container_width=True)

                # ── headline table ────────────────────────────────────────────
                st.markdown("<div class='section-hdr'>Headlines</div>", unsafe_allow_html=True)
                label_color = {"Positive": "#00c87a", "Negative": "#ff4d4d", "Neutral": "#a78bfa"}
                for _, row in sent_df.iterrows():
                    lc = label_color.get(row["label"], "#aab0bb")
                    link_html = f'<a href="{row["link"]}" target="_blank" style="color:#60a5fa; text-decoration:none;">{row["title"]}</a>' if row.get("link") else row["title"]
                    st.markdown(
                        f"""<div style='border-left:4px solid {lc}; padding:8px 12px; margin:6px 0;
                        background:#1a1d23; border-radius:6px;'>
                        <div style='font-size:0.95rem;'>{link_html}</div>
                        <div style='font-size:0.78rem; color:#6b7280; margin-top:4px;'>
                            {row['publisher']} · {row['published']} &nbsp;|&nbsp;
                            <span style='color:{lc}; font-weight:600;'>{row['label']}</span>
                            &nbsp;({row['compound']:+.4f})
                        </div></div>""",
                        unsafe_allow_html=True
                    )
        else:
            st.info("Click **Fetch Headlines & Score** to load news for the selected ticker.")

        # ── BOTTOM NAVIGATION BUTTONS ──────────────────────────────────────────
        st.markdown("---")
        bcol1, bcol2 = st.columns(2)
        if bcol1.button("⬅ Back to Overview", use_container_width=True):
            st.session_state.analysis_step = 1
            st.session_state.scroll_to_top = True
            st.rerun()
        if bcol2.button("Proceed to Final Report ➡️", use_container_width=True):
            st.session_state.analysis_step = 3
            st.session_state.scroll_to_top = True
            st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — FINAL REPORT
    # ──────────────────────────────────────────────────────────────────────────
    elif step == 3:
        st.markdown("### 📄 Automated PDF Report")
        st.write("Generate a comprehensive PDF combining EDA stats, ML metrics, sentiment, and portfolio data.")

        gen_col, dl_col = st.columns([1, 2])
        with gen_col:
            gen_btn = st.button("📑 Generate PDF Report", use_container_width=True)

        if gen_btn:
            if df is None:
                st.warning("Fetch data first before generating a report.")
            else:
                pnl_df_r  = compute_pnl()
                summ_r    = portfolio_summary(pnl_df_r)
                sent_df_r, sent_summ_r = st.session_state.last_sentiment
                eda_stats = df[["Close", "Volume"]].describe().round(2) if "Volume" in df.columns else df[["Close"]].describe().round(2)

                with st.spinner("Generating PDF…"):
                    try:
                        pdf_bytes = build_pdf(
                            ticker      = ticker,
                            start       = st.session_state.start_date,
                            end         = st.session_state.end_date,
                            ticker_info = ticker_info,
                            eda_stats   = eda_stats,
                            metrics     = st.session_state.last_metrics,
                            pred_fig    = st.session_state.pred_fig,
                            sentiment_summary = sent_summ_r,
                            portfolio_df      = pnl_df_r if not pnl_df_r.empty else None,
                            portfolio_summary = summ_r   if not pnl_df_r.empty else None,
                        )
                        st.success("✅ Report ready! Click below to download.")
                        st.download_button(
                            label    = f"⬇️ Download {ticker}_report.pdf",
                            data     = pdf_bytes,
                            file_name= f"{ticker}_stock_report.pdf",
                            mime     = "application/pdf",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.error(f"PDF generation failed: {exc}")

        st.markdown("""
**Report sections include:**
1. 🏢 Company Overview (name, sector, market cap, P/E, 52-week range)
2. 📋 EDA Descriptive Statistics
3. 📈 ML Prediction Chart (PNG embedded via kaleido)
4. 📊 Model Evaluation Metrics Table (MAE, RMSE, R² for both models)
5. 📰 News Sentiment Summary
6. 💼 Portfolio Holdings & P&L (if any positions are held)

> **Tip:** Visit the ML Predictions step and run training *before* generating the report so that the prediction chart is embedded.
        """)

        st.markdown("---")
        bcol1, bcol2 = st.columns(2)
        if bcol1.button("⬅ Back to ML & Sentiment", use_container_width=True):
            st.session_state.analysis_step = 2
            st.session_state.scroll_to_top = True
            st.rerun()
        if bcol2.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.analysis_step = 1
            st.session_state.scroll_to_top = True
            st.rerun()

else:
    # ══════════════════════════════════════════════════════════════════════════
    # PORTFOLIO & ALERTS PAGE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 💼 Portfolio Tracker")

    # ── Add holding form ──────────────────────────────────────────────────────
    with st.expander("➕ Add a Holding", expanded=(len(st.session_state.holdings) == 0)):
        with st.form("add_holding_form", clear_on_submit=True):
            f1, f2, f3, f4 = st.columns(4)
            h_ticker   = f1.text_input("Ticker", placeholder="AAPL").upper().strip()
            h_qty      = f2.number_input("Quantity", min_value=0.001, value=1.0, step=0.001, format="%.3f")
            h_price    = f3.number_input("Buy Price ($)", min_value=0.01, value=100.0, step=0.01)
            h_date     = f4.date_input("Buy Date", value=date.today() - timedelta(days=30))
            submitted  = st.form_submit_button("Add Holding", use_container_width=True)
            if submitted and h_ticker:
                add_holding(h_ticker, h_qty, h_price, h_date)
                st.success(f"Added {h_qty:.3f} × {h_ticker} @ ${h_price:.2f}")

    # ── P&L table ──────────────────────────────────────────────────────────────
    pnl_df = compute_pnl()
    if pnl_df.empty:
        st.info("No holdings yet. Add your first position above.")
    else:
        summ = portfolio_summary(pnl_df)

        # totals
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total Invested",       f"${summ['total_invested']:,.2f}")
        t2.metric("Total Current Value",  f"${summ['total_current']:,.2f}")
        t3.metric("Total P&L ($)",        f"${summ['total_pnl']:,.2f}",
                  delta=f"{summ['total_pnl_pct']:+.2f} %")
        t4.metric("Overall Return",       f"{summ['total_pnl_pct']:+.2f} %")

        st.markdown("---")

        # colour-coded dataframe
        def _colour_pnl(val):
            try:
                v = float(str(val).replace("$","").replace(",","").replace("%",""))
                return "color: #00c87a" if v >= 0 else "color: #ff4d4d"
            except Exception:
                return ""

        styled = (pnl_df.style
                  .map(_colour_pnl, subset=["P&L ($)", "P&L (%)"])
                  .format({"Buy Price": "${:,.2f}", "Live Price": "${:,.2f}",
                           "Invested ($)": "${:,.2f}", "Current Val ($)": "${:,.2f}",
                           "P&L ($)": "${:,.2f}", "P&L (%)": "{:+.2f}%"}))
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── remove buttons ─────────────────────────────────────────────────────
        st.markdown("**Remove a holding:**")
        rcols = st.columns(min(len(pnl_df), 5))
        for i, row in pnl_df.iterrows():
            with rcols[i % 5]:
                if st.button(f"🗑 {row['Ticker']}", key=f"rm_{i}"):
                    remove_holding(i)
                    st.rerun()

        # ── allocation pie ─────────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>Allocation by Current Value</div>", unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=pnl_df["Ticker"],
            values=pnl_df["Current Val ($)"],
            hole=0.5,
            marker=dict(colors=px.colors.qualitative.Plotly),
            textinfo="label+percent",
        ))
        fig_pie.update_layout(template="plotly_dark", height=380,
                               showlegend=True, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── VISUAL DIVIDER ────────────────────────────────────────────────────────
    st.markdown("---")

    # ── ALERTS SUB-SECTION ───────────────────────────────────────────────────
    st.markdown("### 🔔 Alerts Log")
    if live_price:
        lp_col, up_col, ls_col = st.columns(3)
        lp_col.metric("Live Price", f"${live_price:,.2f}")
        up_col.metric("Upper Target", f"${upper_target:,.2f}" if upper_target > 0 else "Not set")
        ls_col.metric("Lower Stop",   f"${lower_stop:,.2f}"   if lower_stop   > 0 else "Not set")

    alerts_df = get_alerts_log()
    if alerts_df.empty:
        st.info("No alerts triggered yet. Set thresholds in the sidebar.")
    else:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)
        if st.button("🗑 Clear Alert Log"):
            clear_alerts()
            st.rerun()

    st.markdown("---")
    st.markdown("""
**How alerts work**

- Set an **Upper Target** price in the sidebar. A green banner appears and an entry is logged whenever the live price ≥ the target.
- Set a **Lower Stop** price. A red banner appears and an entry is logged whenever the live price ≤ the stop.
- The log is session-persisted — it resets when you refresh the browser.
    """)