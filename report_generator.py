"""
report_generator.py — PDF report generation using fpdf2.

Combines: ticker info, EDA statistics, ML metrics, sentiment summary,
portfolio summary, and an embedded prediction chart PNG (via kaleido).
"""

import io
import tempfile
import os
from datetime import date

import pandas as pd
from fpdf import FPDF

# ── PDF class ─────────────────────────────────────────────────────────────────

class StockReport(FPDF):
    def __init__(self, ticker: str, start: date, end: date):
        super().__init__()
        self.ticker = ticker
        self.start  = start
        self.end    = end
        self.set_auto_page_break(auto=True, margin=15)

    # ── header / footer ──────────────────────────────────────────────────────

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 180, 120)
        self.cell(0, 10, f"Stock Market Prediction PRO - {self.ticker}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Report period: {self.start} to {self.end}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_draw_color(0, 180, 120)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # ── helpers ───────────────────────────────────────────────────────────────

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 180, 120)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(50, 50, 50)
        self.ln(2)

    def kv_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(60, 7, key + ":", new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")

    def add_dataframe_table(self, df: pd.DataFrame):
        """Render a DataFrame as a simple grid table."""
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 180, 120)
        self.set_text_color(255, 255, 255)
        col_w = 180 / max(len(df.columns), 1)
        for col in df.columns:
            self.cell(col_w, 7, str(col)[:22], border=1, fill=True, align="C")
        self.ln()

        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        fill = False
        for _, row in df.iterrows():
            self.set_fill_color(240, 248, 245) if fill else self.set_fill_color(255, 255, 255)
            for val in row:
                self.cell(col_w, 6, str(val)[:22], border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(4)


# ── figure → PNG helper ───────────────────────────────────────────────────────

def _fig_to_png_bytes(fig) -> bytes | None:
    """Export a Plotly figure to PNG bytes via kaleido."""
    try:
        return fig.to_image(format="png", width=900, height=400, scale=2)
    except Exception:
        return None


# ── public builder ────────────────────────────────────────────────────────────

def build_pdf(
    ticker: str,
    start: date,
    end: date,
    ticker_info: dict,
    eda_stats: pd.DataFrame | None,
    metrics: dict | None,
    pred_fig=None,
    sentiment_summary: dict | None = None,
    portfolio_df: pd.DataFrame | None = None,
    portfolio_summary: dict | None = None,
) -> bytes:
    """
    Build and return a PDF report as raw bytes.

    Parameters mirror what the tabs already have available.
    """
    pdf = StockReport(ticker, start, end)
    pdf.add_page()

    # ── 1. Company Overview ──────────────────────────────────────────────────
    pdf.section_title("1. Company Overview")
    fields = [
        ("Name",           ticker_info.get("name", ticker)),
        ("Sector",         ticker_info.get("sector", "N/A")),
        ("Industry",       ticker_info.get("industry", "N/A")),
        ("Currency",       ticker_info.get("currency", "USD")),
        ("Last Price",     f"{ticker_info.get('last_price', 'N/A')}"),
        ("Market Cap",     f"{ticker_info.get('market_cap', 'N/A'):,}" if isinstance(ticker_info.get('market_cap'), (int, float)) else "N/A"),
        ("52-Week High",   f"{ticker_info.get('fifty_two_week_high', 'N/A')}"),
        ("52-Week Low",    f"{ticker_info.get('fifty_two_week_low', 'N/A')}"),
        ("P/E Ratio",      f"{ticker_info.get('pe_ratio', 'N/A')}"),
    ]
    for k, v in fields:
        pdf.kv_row(k, v)
    pdf.ln(4)

    # ── 2. EDA Statistics ────────────────────────────────────────────────────
    if eda_stats is not None and not eda_stats.empty:
        pdf.section_title("2. Key EDA Statistics")
        # eda_stats is a describe() DataFrame — transpose for readability
        display = eda_stats.reset_index().rename(columns={"index": "Stat"})
        pdf.add_dataframe_table(display)

    # ── 3. Prediction Chart ──────────────────────────────────────────────────
    if pred_fig is not None:
        png = _fig_to_png_bytes(pred_fig)
        if png:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(png)
                tmp_path = tmp.name
            try:
                pdf.section_title("3. ML Prediction Chart")
                pdf.image(tmp_path, w=180)
                pdf.ln(4)
            finally:
                os.unlink(tmp_path)

    # ── 4. Model Metrics ─────────────────────────────────────────────────────
    if metrics:
        pdf.section_title("4. Model Evaluation Metrics (Test Set)")
        rows = []
        for model_name, m in metrics.items():
            rows.append({
                "Model": model_name,
                "MAE":   m["MAE"],
                "RMSE":  m["RMSE"],
                "R²":    m["R2"],
            })
        pdf.add_dataframe_table(pd.DataFrame(rows))

    # ── 5. Sentiment Summary ─────────────────────────────────────────────────
    if sentiment_summary:
        pdf.section_title("5. News Sentiment Summary")
        s = sentiment_summary
        pdf.kv_row("Total Headlines",  str(s.get("total", "N/A")))
        pdf.kv_row("Mean Compound",    str(s.get("mean_compound", "N/A")))
        pdf.kv_row("Positive %",       f"{s.get('positive_pct', 0)} %")
        pdf.kv_row("Neutral %",        f"{s.get('neutral_pct', 0)} %")
        pdf.kv_row("Negative %",       f"{s.get('negative_pct', 0)} %")
        pdf.ln(4)

    # ── 6. Portfolio Summary ─────────────────────────────────────────────────
    if portfolio_df is not None and not portfolio_df.empty and portfolio_summary:
        pdf.section_title("6. Portfolio Summary")
        ps = portfolio_summary
        pdf.kv_row("Total Invested",   f"${ps.get('total_invested', 0):,.2f}")
        pdf.kv_row("Total Current Value", f"${ps.get('total_current', 0):,.2f}")
        pdf.kv_row("Total P&L",        f"${ps.get('total_pnl', 0):,.2f}  ({ps.get('total_pnl_pct', 0):.2f} %)")
        pdf.ln(3)
        # Show holdings table (subset of columns to fit)
        cols = ["Ticker", "Qty", "Buy Price", "Live Price", "P&L ($)", "P&L (%)"]
        cols_present = [c for c in cols if c in portfolio_df.columns]
        pdf.add_dataframe_table(portfolio_df[cols_present])

    # ── Serialize ────────────────────────────────────────────────────────────
    return bytes(pdf.output())
