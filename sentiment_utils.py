"""
sentiment_utils.py — News fetching and NLP-based sentiment scoring.

Primary scorer : VADER (no GPU, no API key needed)
Optional scorer: FinBERT (transformers pipeline, requires model download)
News source    : yfinance Ticker.news (free, no API key)
Fallback       : NewsAPI (requires user-supplied key)
"""

import datetime
import streamlit as st
import pandas as pd

# ── VADER initialisation ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_vader():
    """Load VADER analyser once per session."""
    try:
        import nltk
        # Download the lexicon silently if absent
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def _get_finbert():
    """Load FinBERT pipeline once per session (heavy: ~500 MB download)."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        return pipe
    except Exception as exc:
        return None


# ── sentiment scoring ─────────────────────────────────────────────────────────

def _vader_score(sia, text: str) -> dict:
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"compound": round(compound, 4), "label": label}


def _finbert_score(pipe, text: str) -> dict:
    result = pipe(text[:512])[0]
    label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    label = label_map.get(result["label"].lower(), "Neutral")
    # Normalise score to [-1, +1] like compound
    score = result["score"]
    compound = score if label == "Positive" else (-score if label == "Negative" else 0.0)
    return {"compound": round(compound, 4), "label": label}


# ── news fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_yfinance(ticker: str) -> list[dict]:
    """
    Fetch recent news from yfinance.  Returns a list of dicts with keys
    'title', 'publisher', 'link', 'published'.
    """
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
        results = []
        for item in items:
            content = item.get("content", {})
            title = (
                content.get("title")
                or item.get("title", "")
            )
            publisher = (
                content.get("provider", {}).get("displayName", "")
                or item.get("publisher", "")
            )
            link = (
                content.get("canonicalUrl", {}).get("url", "")
                or item.get("link", "")
            )
            pub_ts = (
                content.get("pubDate")
                or item.get("providerPublishTime")
            )
            if isinstance(pub_ts, (int, float)):
                pub_date = datetime.datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M")
            elif isinstance(pub_ts, str):
                pub_date = pub_ts[:16]
            else:
                pub_date = "N/A"
            if title:
                results.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "published": pub_date,
                })
        return results
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_newsapi(ticker: str, api_key: str) -> list[dict]:
    """Fallback: fetch via NewsAPI (requires a free API key)."""
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}&sortBy=publishedAt&pageSize=20&apiKey={api_key}"
        )
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        results = []
        for a in articles:
            results.append({
                "title": a.get("title", ""),
                "publisher": a.get("source", {}).get("name", ""),
                "link": a.get("url", ""),
                "published": (a.get("publishedAt") or "")[:16],
            })
        return results
    except Exception as exc:
        st.warning(f"NewsAPI fetch failed: {exc}")
        return []


# ── main public function ──────────────────────────────────────────────────────

def analyse_sentiment(
    ticker: str,
    newsapi_key: str = "",
    use_finbert: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Fetch headlines for *ticker* and score each one.

    Returns
    -------
    (df, summary) where df has columns
        title, publisher, published, compound, label
    and summary has keys:
        mean_compound, positive_pct, negative_pct, neutral_pct, total
    """
    headlines = fetch_news_yfinance(ticker)
    api_key_clean = newsapi_key.strip() if newsapi_key else ""
    if len(headlines) < 3 and api_key_clean:
        headlines += fetch_news_newsapi(ticker, api_key_clean)
    if not headlines:
        return pd.DataFrame(), {}

    if use_finbert:
        pipe = _get_finbert()
        if pipe is None:
            st.warning("FinBERT failed to load — falling back to VADER.")
            use_finbert = False

    sia = None
    if not use_finbert:
        sia = _get_vader()
        if sia is None:
            return pd.DataFrame(), {}

    rows = []
    for h in headlines:
        text = h["title"]
        if use_finbert:
            scores = _finbert_score(pipe, text)
        else:
            scores = _vader_score(sia, text)
        rows.append({
            "title":     text,
            "publisher": h["publisher"],
            "published": h["published"],
            "link":      h["link"],
            "compound":  scores["compound"],
            "label":     scores["label"],
        })

    df = pd.DataFrame(rows)
    total = len(df)
    counts = df["label"].value_counts()
    summary = {
        "mean_compound":  round(df["compound"].mean(), 4),
        "positive_pct":   round(counts.get("Positive", 0) / total * 100, 1),
        "negative_pct":   round(counts.get("Negative", 0) / total * 100, 1),
        "neutral_pct":    round(counts.get("Neutral",  0) / total * 100, 1),
        "total":          total,
    }
    return df, summary
