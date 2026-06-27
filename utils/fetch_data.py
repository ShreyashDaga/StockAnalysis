"""Yahoo Finance data access utilities."""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import yfinance as yf

from config import DEFAULT_INTERVAL, DEFAULT_PERIOD


def normalize_stock_symbol(stock_symbol: str) -> str:
    """Return a cleaned Yahoo Finance stock symbol."""
    return stock_symbol.strip().upper()


def fetch_stock_data(
    tickers: str | Iterable[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    pause_seconds: float = 0.0,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Fetch historical OHLCV stock data from Yahoo Finance.

    A single ticker returns a DataFrame. An iterable returns a dictionary keyed
    by ticker, preserving the notebook's multi-stock workflow.
    """
    if isinstance(tickers, str):
        ticker = normalize_stock_symbol(tickers)
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data.empty:
            raise ValueError(f"No historical data returned for {ticker}.")
        data = _flatten_columns(data)
        data.index = pd.to_datetime(data.index)
        return data

    stock_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        normalized = normalize_stock_symbol(ticker)
        try:
            stock_data[normalized] = fetch_stock_data(
                normalized,
                period=period,
                interval=interval,
            )
        except Exception as exc:
            print(f"Error fetching data for {normalized}: {exc}")
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return stock_data


def get_current_price(stock_symbol: str) -> dict[str, float | str | None]:
    """Return current market price and company metadata."""
    ticker = normalize_stock_symbol(stock_symbol)
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info or {}
    history = yf_ticker.history(period="5d", interval="1d", auto_adjust=False)

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

    if current_price is None and not history.empty:
        current_price = float(history["Close"].iloc[-1])
    if previous_close is None and len(history) >= 2:
        previous_close = float(history["Close"].iloc[-2])

    daily_change = None
    daily_change_percent = None
    if current_price is not None and previous_close:
        daily_change = float(current_price) - float(previous_close)
        daily_change_percent = daily_change / float(previous_close) * 100

    return {
        "symbol": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": _to_float(current_price),
        "previous_close": _to_float(previous_close),
        "daily_change": _to_float(daily_change),
        "daily_change_percent": _to_float(daily_change_percent),
        "market_cap": _to_float(info.get("marketCap")),
        "fifty_two_week_high": _to_float(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _to_float(info.get("fiftyTwoWeekLow")),
        "dividend_yield": _to_float(info.get("dividendYield")),
        "currency": info.get("currency", ""),
    }


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns when present."""
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    return data


def _to_float(value: object) -> float | None:
    """Convert numeric-like values to float while preserving missing values."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
