"""Stock prediction orchestration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import FEATURE_COLUMNS
from utils.fetch_data import fetch_stock_data, get_current_price, normalize_stock_symbol
from utils.indicators import add_technical_indicators
from utils.model import get_or_train_model
from utils.sentiment import analyze_sentiment, fetch_news, get_sentiment


def predict_stock(stock_symbol: str) -> dict[str, Any]:
    """Predict BUY, HOLD, or SELL for a stock and return display artifacts."""
    ticker = normalize_stock_symbol(stock_symbol)
    historical_data = fetch_stock_data(ticker)
    engineered_data = add_technical_indicators(historical_data)

    news_query = ticker.replace(".NS", "")
    articles = fetch_news(news_query, page_size=5)
    sentiment = analyze_sentiment(articles)
    sentiment_score = get_sentiment(news_query) if articles else 0.0
    engineered_data["Sentiment"] = sentiment_score

    latest_features = engineered_data[FEATURE_COLUMNS].dropna().tail(1)
    if latest_features.empty:
        raise ValueError(
            f"Not enough historical data for {ticker} to calculate all indicators."
        )

    model_bundle = get_or_train_model()
    model = model_bundle.model
    prediction_class = int(model.predict(latest_features)[0])
    prediction = model_bundle.label_mapping.get(prediction_class, "HOLD")
    probabilities = _class_probabilities(model, latest_features, model_bundle.label_mapping)
    confidence = probabilities.get(prediction, 0.0)
    current_info = get_current_price(ticker)

    return {
        "symbol": ticker,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": latest_features.iloc[0].to_dict(),
        "historical_data": engineered_data,
        "current_info": current_info,
        "news": articles,
        "sentiment": sentiment,
        "model_accuracy": model_bundle.accuracy,
        "model_report": model_bundle.class_report,
        "feature_importance": _feature_importance(model),
        "horizon_days": model_bundle.horizon_days,
        "buy_threshold": model_bundle.buy_threshold,
        "sell_threshold": model_bundle.sell_threshold,
    }


def build_prediction_report(result: dict[str, Any]) -> pd.DataFrame:
    """Create a single-row CSV-friendly report for a prediction result."""
    current_info = result["current_info"]
    row = {
        "symbol": result["symbol"],
        "company": current_info.get("name"),
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "buy_probability": result["probabilities"].get("BUY"),
        "hold_probability": result["probabilities"].get("HOLD"),
        "sell_probability": result["probabilities"].get("SELL"),
        "horizon_days": result["horizon_days"],
        "buy_threshold": result["buy_threshold"],
        "sell_threshold": result["sell_threshold"],
        "sentiment_score": result["sentiment"]["score"],
        "sentiment_label": result["sentiment"]["label"],
        "current_price": current_info.get("current_price"),
        "daily_change": current_info.get("daily_change"),
        "daily_change_percent": current_info.get("daily_change_percent"),
    }
    row.update(result["features"])
    return pd.DataFrame([row])


def _class_probabilities(
    model: Any,
    features: pd.DataFrame,
    label_mapping: dict[int, str],
) -> dict[str, float]:
    """Return class probabilities as label-to-percentage values."""
    probabilities = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
    if not hasattr(model, "predict_proba"):
        return probabilities

    predicted_probabilities = model.predict_proba(features)[0]
    for class_value, probability in zip(model.classes_, predicted_probabilities, strict=False):
        label = label_mapping.get(int(class_value))
        if label:
            probabilities[label] = float(probability * 100)
    return probabilities


def _feature_importance(model: Any) -> pd.DataFrame:
    """Return model feature importances as a DataFrame."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    return pd.DataFrame(
        {"Feature": FEATURE_COLUMNS, "Importance": importances}
    ).sort_values("Importance", ascending=False)
