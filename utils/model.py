"""Machine learning training and persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from config import (
    BUY_RETURN_THRESHOLD,
    CLASS_TO_LABEL,
    FEATURE_COLUMNS,
    LABEL_TO_CLASS,
    MODEL_PATH,
    NIFTY_50_TICKERS,
    PREDICTION_HORIZON_DAYS,
    SELL_RETURN_THRESHOLD,
)
from utils.fetch_data import fetch_stock_data
from utils.indicators import add_technical_indicators
from utils.sentiment import get_sentiment


@dataclass
class ModelBundle:
    """Container for a trained model and its metadata."""

    model: RandomForestClassifier
    accuracy: float | None
    feature_columns: list[str]
    label_mapping: dict[int, str]
    horizon_days: int
    buy_threshold: float
    sell_threshold: float
    class_report: dict[str, Any] | None = None


def create_target_variable(
    data: pd.DataFrame,
    horizon_days: int = PREDICTION_HORIZON_DAYS,
    buy_threshold: float = BUY_RETURN_THRESHOLD,
    sell_threshold: float = SELL_RETURN_THRESHOLD,
) -> pd.DataFrame:
    """Create BUY/HOLD/SELL target labels from future returns."""
    if sell_threshold >= buy_threshold:
        raise ValueError("SELL_RETURN_THRESHOLD must be lower than BUY_RETURN_THRESHOLD.")

    output = data.copy()
    future_return = output["Close"].shift(-horizon_days) / output["Close"] - 1
    output["Future_Return"] = future_return
    output["Signal"] = LABEL_TO_CLASS["HOLD"]
    output.loc[future_return >= buy_threshold, "Signal"] = LABEL_TO_CLASS["BUY"]
    output.loc[future_return <= sell_threshold, "Signal"] = LABEL_TO_CLASS["SELL"]
    return output


def prepare_data_for_training(
    stock_data: dict[str, pd.DataFrame],
    sentiment_by_ticker: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector for model training."""
    features: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    sentiment_by_ticker = sentiment_by_ticker or {}

    for ticker, data in stock_data.items():
        engineered = add_technical_indicators(data)
        engineered["Sentiment"] = sentiment_by_ticker.get(ticker, 0.0)
        engineered = create_target_variable(engineered)

        ticker_features = engineered[FEATURE_COLUMNS].iloc[:-PREDICTION_HORIZON_DAYS]
        ticker_target = engineered["Signal"].iloc[:-PREDICTION_HORIZON_DAYS]
        ticker_features = ticker_features.replace([float("inf"), float("-inf")], pd.NA)
        valid_rows = ticker_features.notna().all(axis=1)

        if valid_rows.any():
            features.append(ticker_features.loc[valid_rows])
            targets.append(ticker_target.loc[valid_rows])

    if not features or not targets:
        raise ValueError("Not enough clean historical data to train the model.")

    return pd.concat(features), pd.concat(targets)


def train_model(tickers: list[str] | None = None, save_path=MODEL_PATH) -> ModelBundle:
    """Train and save the RandomForestClassifier."""
    selected_tickers = tickers or NIFTY_50_TICKERS
    stock_data = fetch_stock_data(selected_tickers, pause_seconds=0.2)
    sentiment_by_ticker = {ticker: get_sentiment(ticker) for ticker in stock_data}
    x, y = prepare_data_for_training(stock_data, sentiment_by_ticker)

    if y.nunique() < 2:
        raise ValueError(
            "Training data has only one target class. Adjust thresholds, period, or tickers."
        )

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test,
        predictions,
        labels=sorted(CLASS_TO_LABEL),
        target_names=[CLASS_TO_LABEL[label] for label in sorted(CLASS_TO_LABEL)],
        output_dict=True,
        zero_division=0,
    )

    bundle = ModelBundle(
        model=classifier,
        accuracy=accuracy,
        feature_columns=FEATURE_COLUMNS.copy(),
        label_mapping=CLASS_TO_LABEL.copy(),
        horizon_days=PREDICTION_HORIZON_DAYS,
        buy_threshold=BUY_RETURN_THRESHOLD,
        sell_threshold=SELL_RETURN_THRESHOLD,
        class_report=report,
    )
    save_model(bundle, save_path)
    return bundle


def save_model(bundle: ModelBundle, save_path=MODEL_PATH) -> None:
    """Persist the trained model bundle with Joblib."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path)


def load_model(model_path=MODEL_PATH) -> ModelBundle | None:
    """Load an existing model bundle if available and compatible."""
    if not model_path.exists():
        return None

    loaded = joblib.load(model_path)
    if isinstance(loaded, ModelBundle) and _is_model_compatible(loaded):
        return loaded

    return None


def get_or_train_model() -> ModelBundle:
    """Load an existing compatible model or train a new one."""
    bundle = load_model()
    if bundle is not None:
        return bundle
    return train_model()


def _is_model_compatible(bundle: ModelBundle) -> bool:
    """Return whether a persisted model matches current runtime settings."""
    return (
        bundle.feature_columns == FEATURE_COLUMNS
        and bundle.horizon_days == PREDICTION_HORIZON_DAYS
        and bundle.buy_threshold == BUY_RETURN_THRESHOLD
        and bundle.sell_threshold == SELL_RETURN_THRESHOLD
    )
