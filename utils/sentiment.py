"""News fetching and VADER sentiment analysis."""

from __future__ import annotations

import os
from typing import Any

from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NEWS_API_KEY_ENV


def fetch_news(query: str, page_size: int = 5) -> list[dict[str, Any]]:
    """Fetch recent English news articles from NewsAPI.

    If NEWS_API_KEY is not configured, an empty list is returned so the app can
    still run with a neutral sentiment score.
    """
    api_key = os.getenv(NEWS_API_KEY_ENV)
    if not api_key:
        return []

    newsapi = NewsApiClient(api_key=api_key)
    response = newsapi.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=page_size,
    )
    return response.get("articles", [])


def analyze_sentiment(articles: list[dict[str, Any]]) -> dict[str, float | str]:
    """Analyze article title and description sentiment using VADER."""
    if not articles:
        return {"score": 0.0, "label": "Neutral"}

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        text = f"{title}. {description}".strip()
        if text:
            scores.append(analyzer.polarity_scores(text)["compound"])

    average_score = sum(scores) / len(scores) if scores else 0.0
    return {"score": float(average_score), "label": sentiment_label(average_score)}


def get_sentiment(stock_name: str) -> float:
    """Return aggregate title sentiment compatible with the notebook logic."""
    articles = fetch_news(stock_name, page_size=5)
    if not articles:
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    sentiment = 0.0
    for article in articles:
        sentiment += analyzer.polarity_scores(article.get("title") or "")["compound"]
    return float(sentiment)


def sentiment_label(score: float) -> str:
    """Convert a compound sentiment score into a readable label."""
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"
