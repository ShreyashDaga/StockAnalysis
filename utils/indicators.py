"""Technical indicator calculations."""

from __future__ import annotations

import pandas as pd


def calculate_sma(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate a simple moving average for the Close price."""
    return data["Close"].rolling(window=period).mean()


def calculate_sma50(data: pd.DataFrame) -> pd.DataFrame:
    """Add the 50-day simple moving average."""
    output = data.copy()
    output["SMA50"] = calculate_sma(output, 50)
    return output


def calculate_sma200(data: pd.DataFrame) -> pd.DataFrame:
    """Add the 200-day simple moving average."""
    output = data.copy()
    output["SMA200"] = calculate_sma(output, 200)
    return output


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add the relative strength index."""
    output = data.copy()
    delta = output["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    output["RSI"] = 100 - (100 / (1 + rs))
    return output


def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
    """Add MACD, signal line, and histogram columns."""
    output = data.copy()
    short_ema = output["Close"].ewm(span=12, adjust=False).mean()
    long_ema = output["Close"].ewm(span=26, adjust=False).mean()
    output["MACD"] = short_ema - long_ema
    output["MACD_Signal"] = output["MACD"].ewm(span=9, adjust=False).mean()
    output["MACD_Histogram"] = output["MACD"] - output["MACD_Signal"]
    return output


def add_market_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight market-derived features for the classifier."""
    output = data.copy()
    output["Daily_Return"] = output["Close"].pct_change()
    output["Volatility_20"] = output["Daily_Return"].rolling(window=20).std()
    output["Volume_Ratio"] = output["Volume"] / output["Volume"].rolling(window=20).mean()
    return output


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators used by the model."""
    output = calculate_sma50(data)
    output = calculate_sma200(output)
    output = calculate_rsi(output)
    output = calculate_macd(output)
    output = add_market_features(output)
    return output
