"""
alerts.py — Price threshold monitoring and alert log management.

Alerts are stored in st.session_state['alerts_log'] as a list of dicts.
"""

import streamlit as st
import pandas as pd
from datetime import datetime


# ── state initialisation ──────────────────────────────────────────────────────

def init_alerts():
    if "alerts_log" not in st.session_state:
        st.session_state.alerts_log = []


# ── threshold check ───────────────────────────────────────────────────────────

def check_thresholds(
    ticker: str,
    live_price: float | None,
    upper_target: float | None,
    lower_stop: float | None,
) -> list[dict]:
    """
    Compare *live_price* against *upper_target* and *lower_stop*.

    Returns a list of triggered alert dicts (may be empty).
    Each dict: {timestamp, ticker, type, threshold, price}
    """
    init_alerts()
    if live_price is None:
        return []

    triggered = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if upper_target and live_price >= upper_target:
        triggered.append({
            "timestamp":  now,
            "ticker":     ticker,
            "type":       "Upper Target ↑",
            "threshold":  upper_target,
            "price":      round(live_price, 2),
        })

    if lower_stop and live_price <= lower_stop:
        triggered.append({
            "timestamp":  now,
            "ticker":     ticker,
            "type":       "Lower Stop ↓",
            "threshold":  lower_stop,
            "price":      round(live_price, 2),
        })

    # Append to session log (avoid duplicate consecutive entries)
    for alert in triggered:
        last = st.session_state.alerts_log[-1] if st.session_state.alerts_log else {}
        if not (last.get("ticker") == alert["ticker"] and
                last.get("type") == alert["type"] and
                last.get("threshold") == alert["threshold"]):
            st.session_state.alerts_log.append(alert)

    return triggered


def get_alerts_log() -> pd.DataFrame:
    """Return the session alert log as a DataFrame."""
    init_alerts()
    return pd.DataFrame(st.session_state.alerts_log) if st.session_state.alerts_log else pd.DataFrame()


def clear_alerts():
    st.session_state.alerts_log = []
