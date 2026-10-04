"""
data_utils.py — Data fetching, caching, and OHLCV preprocessing.
All external calls are wrapped in try/except; callers receive (df, error_str).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta


# ── helpers ──────────────────────────────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the second level of a MultiIndex column header produced by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance returns a DatetimeIndex. Reset it to a plain 'Date' column so the
    rest of the app can always do  df['Date']  without worrying about the index.
    """
    if "Date" not in df.columns:
        df = df.reset_index()
    # Make sure it is tz-naive for consistent comparisons
    if pd.api.types.is_datetime64tz_dtype(df["Date"]):
        df["Date"] = df["Date"].dt.tz_localize(None)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ── main fetch ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker: str, start: date, end: date) -> tuple[pd.DataFrame | None, str]:
    """
    Download OHLCV data from Yahoo Finance for *ticker* between *start* and *end*.

    Returns
    -------
    (df, "")        on success — df has columns Date, Open, High, Low, Close, Volume
    (None, msg)     on failure — msg is a human-readable error string
    """
    try:
        raw = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None, f"No data returned for **{ticker}** in the selected date range."
        raw = _flatten_columns(raw)
        raw = _ensure_date_column(raw)
        # Require at minimum Close & Volume
        for col in ("Close", "Volume"):
            if col not in raw.columns:
                return None, f"Data for **{ticker}** is missing the '{col}' column."
        raw = raw.sort_values("Date").reset_index(drop=True)
        return raw, ""
    except Exception as exc:
        return None, f"Error fetching **{ticker}**: {exc}"


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_price(ticker: str) -> float | None:
    """Return the most-recent closing price, or None on failure."""
    try:
        info = yf.Ticker(ticker).fast_info
        price = info.get("last_price") or info.get("regularMarketPrice")
        return float(price) if price else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    """Return a dict of basic company info; empty dict on failure."""
    try:
        obj = yf.Ticker(ticker)
        info = obj.info or {}
        fast = obj.fast_info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "last_price": fast.get("last_price"),
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception:
        return {}


# ── technical indicators ──────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard technical indicators to a copy of *df*.

    Added columns
    -------------
    MA20, MA50, MA5, MA10, RSI, rolling_vol_20
    """
    df = df.copy()
    close = df["Close"]

    df["MA5"]  = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    # RSI (14-period Wilder smoothing)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # 20-day rolling volatility (annualised %)
    df["rolling_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    return df


def parse_csv_upload(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """
    Parse a user-uploaded CSV file into a standard OHLCV DataFrame.

    Returns (df, "") on success, (None, error_msg) on failure.
    """
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().title() for c in df.columns]

        # Try to locate a date column
        date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if not date_candidates:
            return None, "CSV has no recognisable date/time column."
        df = df.rename(columns={date_candidates[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # Locate Close column
        close_candidates = [c for c in df.columns if "close" in c.lower()]
        if not close_candidates:
            return None, "CSV has no recognisable 'Close' column."
        if close_candidates[0] != "Close":
            df = df.rename(columns={close_candidates[0]: "Close"})

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])

        if df.empty:
            return None, "CSV contained no valid rows after parsing."

        return df, ""
    except Exception as exc:
        return None, f"Failed to parse CSV: {exc}"
