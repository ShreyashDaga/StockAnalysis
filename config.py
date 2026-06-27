"""Application configuration for StockAnalysis."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

NEWS_API_KEY_ENV = "NEWS_API_KEY"

DEFAULT_PERIOD = os.getenv("STOCK_PERIOD", "2y")
DEFAULT_INTERVAL = os.getenv("STOCK_INTERVAL", "1d")

PREDICTION_HORIZON_DAYS = int(os.getenv("PREDICTION_HORIZON_DAYS", "5"))
BUY_RETURN_THRESHOLD = float(os.getenv("BUY_RETURN_THRESHOLD", "0.05"))
SELL_RETURN_THRESHOLD = float(os.getenv("SELL_RETURN_THRESHOLD", "-0.03"))

LABEL_SELL = "SELL"
LABEL_HOLD = "HOLD"
LABEL_BUY = "BUY"
LABEL_TO_CLASS = {LABEL_SELL: 0, LABEL_HOLD: 1, LABEL_BUY: 2}
CLASS_TO_LABEL = {value: key for key, value in LABEL_TO_CLASS.items()}

NIFTY_50_TICKERS: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "BAJFINANCE.NS",
    "SBIN.NS",
    "LT.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "ITC.NS",
    "MARUTI.NS",
    "BHARTIARTL.NS",
    "TATAMOTORS.NS",
    "TITAN.NS",
    "M&M.NS",
    "ASIANPAINT.NS",
    "SUNPHARMA.NS",
    "ULTRACEMCO.NS",
]

FEATURE_COLUMNS: list[str] = [
    "SMA50",
    "SMA200",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Sentiment",
    "Daily_Return",
    "Volatility_20",
    "Volume_Ratio",
]
