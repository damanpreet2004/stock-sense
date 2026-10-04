"""
portfolio.py — Portfolio holdings management and P&L computation.

Holdings are stored in st.session_state['holdings'] as a list of dicts:
    {ticker, quantity, buy_price, buy_date}

All live prices are fetched via data_utils.fetch_live_price.
"""

import streamlit as st
import pandas as pd
from datetime import date

from data_utils import fetch_live_price


# ── state initialisation ──────────────────────────────────────────────────────

def init_portfolio():
    if "holdings" not in st.session_state:
        st.session_state.holdings = []


# ── CRUD ──────────────────────────────────────────────────────────────────────

def add_holding(ticker: str, quantity: float, buy_price: float, buy_date: date):
    st.session_state.holdings.append({
        "ticker":    ticker.upper().strip(),
        "quantity":  quantity,
        "buy_price": buy_price,
        "buy_date":  str(buy_date),
    })


def remove_holding(index: int):
    try:
        st.session_state.holdings.pop(index)
    except IndexError:
        pass


# ── P&L computation ───────────────────────────────────────────────────────────

def compute_pnl() -> pd.DataFrame:
    """
    Build a DataFrame of current holdings with live prices and P&L columns.

    Columns
    -------
    ticker, quantity, buy_price, buy_date,
    current_price, invested_value, current_value,
    pnl_dollar, pnl_pct
    """
    init_portfolio()
    rows = []
    for h in st.session_state.holdings:
        ticker      = h["ticker"]
        qty         = h["quantity"]
        buy_price   = h["buy_price"]
        live_price  = fetch_live_price(ticker) or buy_price   # fallback = cost

        invested    = qty * buy_price
        current_val = qty * live_price
        pnl_dollar  = current_val - invested
        pnl_pct     = (pnl_dollar / invested * 100) if invested else 0.0

        rows.append({
            "Ticker":          ticker,
            "Qty":             qty,
            "Buy Price":       round(buy_price, 2),
            "Buy Date":        h["buy_date"],
            "Live Price":      round(live_price, 2),
            "Invested ($)":    round(invested, 2),
            "Current Val ($)": round(current_val, 2),
            "P&L ($)":         round(pnl_dollar, 2),
            "P&L (%)":         round(pnl_pct, 2),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def portfolio_summary(pnl_df: pd.DataFrame) -> dict:
    """Return totals dict from a compute_pnl() DataFrame."""
    if pnl_df.empty:
        return {"total_invested": 0, "total_current": 0, "total_pnl": 0, "total_pnl_pct": 0}
    invested  = pnl_df["Invested ($)"].sum()
    current   = pnl_df["Current Val ($)"].sum()
    pnl       = current - invested
    pnl_pct   = (pnl / invested * 100) if invested else 0.0
    return {
        "total_invested": round(invested, 2),
        "total_current":  round(current, 2),
        "total_pnl":      round(pnl, 2),
        "total_pnl_pct":  round(pnl_pct, 2),
    }
