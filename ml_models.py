"""
ml_models.py — Feature engineering, model training/inference, and evaluation.

All models are regression models predicting the closing price N days ahead.
Training is chronological (no data leakage) using the earliest 80 % of rows.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Build a feature matrix from OHLCV data.

    Parameters
    ----------
    df       : DataFrame with at least [Date, Open, High, Low, Close, Volume]
    horizon  : forecast horizon in trading days

    Returns
    -------
    DataFrame that includes all feature columns AND the target column
    'target_close' (= Close shifted back by *horizon* rows so that row i
    shows the price *horizon* days *after* row i).  Rows with NaN are dropped.
    """
    f = df[["Date", "Close"]].copy()
    close = df["Close"]

    # --- lagged returns ---
    for lag in (1, 2, 3, 5, 10):
        f[f"ret_lag{lag}"] = close.pct_change(lag)

    # --- moving averages (raw price level) ---
    for w in (5, 10, 20, 50):
        f[f"ma{w}"] = close.rolling(w).mean()

    # --- MA ratios (price relative to its MA) ---
    for w in (5, 10, 20, 50):
        f[f"ma{w}_ratio"] = close / close.rolling(w).mean()

    # --- RSI ---
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    f["rsi"] = 100 - (100 / (1 + rs))

    # --- rolling volatility ---
    f["vol_20"] = close.pct_change().rolling(20).std()

    # --- volume change ---
    if "Volume" in df.columns:
        f["vol_chg"] = df["Volume"].pct_change()
        f["vol_ma5"] = df["Volume"].rolling(5).mean() / df["Volume"].rolling(20).mean()

    # --- OHLC features ---
    if all(c in df.columns for c in ("Open", "High", "Low")):
        f["hl_range"]  = (df["High"] - df["Low"]) / close
        f["oc_change"] = (close - df["Open"]) / df["Open"]

    # --- target: close price N days in the future ---
    f["target_close"] = close.shift(-horizon)

    f = f.dropna().reset_index(drop=True)
    return f


# ── train / evaluate ──────────────────────────────────────────────────────────

FEATURE_COLS_EXCLUDE = {"Date", "target_close"}


def _get_xy(feat_df: pd.DataFrame):
    """Split feature DataFrame into X (numpy) and y (numpy)."""
    feature_cols = [c for c in feat_df.columns if c not in FEATURE_COLS_EXCLUDE]
    X = feat_df[feature_cols].values
    y = feat_df["target_close"].values
    return X, y, feature_cols


@st.cache_resource(show_spinner=False)
def train_models(ticker: str, start_str: str, end_str: str, horizon: int,
                 _df: pd.DataFrame):
    """
    Train LinearRegression and RandomForestRegressor on *_df*.

    The leading underscore on *_df* tells Streamlit not to hash the DataFrame
    itself (DataFrames are unhashable by cache_resource); the other scalar
    args serve as the cache key.

    Returns
    -------
    dict with keys: 'lr', 'rf', 'feature_cols', 'X_test', 'y_test',
                    'dates_test', 'metrics'
    """
    feat_df = engineer_features(_df, horizon=horizon)
    if len(feat_df) < 30:
        raise ValueError("Not enough data rows to train (need at least 30 after feature engineering).")

    X, y, feature_cols = _get_xy(feat_df)
    dates = feat_df["Date"].values

    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    def _metrics(model, X_t, y_t):
        pred = model.predict(X_t)
        return {
            "MAE":  round(mean_absolute_error(y_t, pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_t, pred)), 4),
            "R2":   round(r2_score(y_t, pred), 4),
            "pred": pred,
        }

    metrics = {
        "LinearRegression":     _metrics(lr, X_test, y_test),
        "RandomForestRegressor": _metrics(rf, X_test, y_test),
    }

    return {
        "lr": lr,
        "rf": rf,
        "feature_cols": feature_cols,
        "X_test":  X_test,
        "y_test":  y_test,
        "dates_test": dates_test,
        "metrics": metrics,
        "feat_df": feat_df,
    }


def predict_next(model_result: dict, model_name: str,
                 df: pd.DataFrame, horizon: int) -> float | None:
    """
    Produce a single price forecast for *horizon* days from the last known row.

    Uses the last available feature row (before NaN-dropping due to horizon
    shift) so we can forecast beyond the test set.
    """
    try:
        feat_df = engineer_features(df, horizon=1)   # horizon=1 to keep last row
        if feat_df.empty:
            return None
        feature_cols = model_result["feature_cols"]
        # Make sure every column is present
        missing = [c for c in feature_cols if c not in feat_df.columns]
        if missing:
            return None
        X_last = feat_df[feature_cols].iloc[[-1]].values
        model = model_result["lr"] if model_name == "LinearRegression" else model_result["rf"]
        return float(model.predict(X_last)[0])
    except Exception:
        return None
