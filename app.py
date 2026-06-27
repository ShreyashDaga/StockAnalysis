"""Streamlit frontend for AI-powered stock market analysis."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    BUY_RETURN_THRESHOLD,
    NIFTY_50_TICKERS,
    PREDICTION_HORIZON_DAYS,
    SELL_RETURN_THRESHOLD,
)
from utils.predict import build_prediction_report, predict_stock


st.set_page_config(
    page_title="StockAnalysis AI",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Run the Streamlit application."""
    inject_styles()
    render_sidebar()

    st.title("AI-Powered Stock Market Analysis")
    st.caption(
        "BUY, HOLD, and SELL recommendations from Yahoo Finance prices, "
        "technical indicators, news sentiment, and a Random Forest classifier."
    )

    selected_stock = st.session_state.get("selected_stock", NIFTY_50_TICKERS[0])
    analyze = st.session_state.get("analyze", False)

    if not analyze:
        render_empty_state()
        return

    with st.spinner(f"Analyzing {selected_stock}..."):
        try:
            result = predict_stock(selected_stock)
        except Exception as exc:
            st.error(f"Unable to analyze {selected_stock}: {exc}")
            st.info(
                "Check your internet connection, stock symbol, and optional "
                "NEWS_API_KEY environment variable."
            )
            return

    render_analysis(result)


def render_sidebar() -> None:
    """Render sidebar controls."""
    with st.sidebar:
        st.header("Analysis Controls")
        st.selectbox(
            "Stock Symbol",
            options=NIFTY_50_TICKERS,
            index=0,
            key="selected_stock",
        )
        custom_symbol = st.text_input(
            "Or enter a Yahoo Finance symbol",
            placeholder="AAPL, MSFT, RELIANCE.NS",
        )
        if custom_symbol.strip():
            st.session_state["selected_stock"] = custom_symbol.strip().upper()

        if st.button("Analyze Stock", type="primary", use_container_width=True):
            st.session_state["analyze"] = True

        st.divider()
        st.write("Model Settings")
        st.caption(
            f"{PREDICTION_HORIZON_DAYS}-day horizon | BUY >= "
            f"{BUY_RETURN_THRESHOLD:.1%} | SELL <= {SELL_RETURN_THRESHOLD:.1%}"
        )
        st.caption("Override with environment variables before deployment.")
        st.divider()
        st.write("Set `NEWS_API_KEY` for live news sentiment.")


def render_empty_state() -> None:
    """Render first-load content."""
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Data Source", "Yahoo Finance")
    col_b.metric("Model", "Random Forest")
    col_c.metric("Signal", "BUY / HOLD / SELL")
    st.info("Choose a stock in the sidebar and click Analyze Stock.")


def render_analysis(result: dict) -> None:
    """Render all analysis panels."""
    current_info = result["current_info"]
    historical_data = result["historical_data"]
    features = result["features"]

    render_company_snapshot(current_info)
    render_prediction_cards(result)

    st.subheader("Technical Indicators")
    indicator_cols = st.columns(4)
    indicator_cols[0].metric("SMA50", format_number(features["SMA50"]))
    indicator_cols[1].metric("SMA200", format_number(features["SMA200"]))
    indicator_cols[2].metric("RSI", format_number(features["RSI"]))
    indicator_cols[3].metric("MACD", format_number(features["MACD"]))

    chart_tab, rsi_tab, macd_tab, volume_tab = st.tabs(
        ["Price", "RSI", "MACD", "Volume"]
    )
    with chart_tab:
        st.plotly_chart(price_chart(historical_data, result["symbol"]), use_container_width=True)
    with rsi_tab:
        st.plotly_chart(rsi_chart(historical_data), use_container_width=True)
    with macd_tab:
        st.plotly_chart(macd_chart(historical_data), use_container_width=True)
    with volume_tab:
        st.plotly_chart(volume_chart(historical_data), use_container_width=True)

    lower_left, lower_right = st.columns([1.25, 1])
    with lower_left:
        render_news(result)
    with lower_right:
        render_probability_chart(result)
        render_feature_importance(result)
        render_download(result)


def render_company_snapshot(current_info: dict) -> None:
    """Render company and market information."""
    st.subheader(current_info.get("name") or current_info.get("symbol"))
    cols = st.columns(5)
    cols[0].metric("Current Price", money(current_info.get("current_price"), current_info))
    cols[1].metric(
        "Daily Change",
        money(current_info.get("daily_change"), current_info),
        format_percent(current_info.get("daily_change_percent")),
    )
    cols[2].metric("Market Cap", compact_number(current_info.get("market_cap")))
    cols[3].metric("52W High", money(current_info.get("fifty_two_week_high"), current_info))
    cols[4].metric("52W Low", money(current_info.get("fifty_two_week_low"), current_info))

    dividend_yield = current_info.get("dividend_yield")
    if dividend_yield:
        st.caption(f"Dividend Yield: {dividend_yield * 100:.2f}%")


def render_prediction_cards(result: dict) -> None:
    """Render prediction, confidence, and sentiment cards."""
    prediction = result["prediction"]
    prediction_class = {
        "BUY": "buy-card",
        "HOLD": "hold-card",
        "SELL": "sell-card",
    }.get(prediction, "hold-card")
    sentiment = result["sentiment"]

    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card {prediction_class}">
                <span>Recommendation</span>
                <strong>{prediction}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <span>Confidence</span>
                <strong>{result["confidence"]:.1f}%</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.plotly_chart(
            sentiment_gauge(sentiment["score"], sentiment["label"]),
            use_container_width=True,
        )

    st.caption(
        f"Prediction horizon: {result['horizon_days']} trading days. "
        f"BUY >= {result['buy_threshold']:.1%}, SELL <= {result['sell_threshold']:.1%}."
    )
    if result.get("model_accuracy") is not None:
        st.caption(f"Latest model validation accuracy: {result['model_accuracy'] * 100:.2f}%")


def render_probability_chart(result: dict) -> None:
    """Render class probability chart."""
    probabilities = result["probabilities"]
    colors = {"SELL": "#ef4444", "HOLD": "#f59e0b", "BUY": "#22c55e"}
    labels = ["SELL", "HOLD", "BUY"]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=[probabilities.get(label, 0.0) for label in labels],
            marker_color=[colors[label] for label in labels],
            text=[f"{probabilities.get(label, 0.0):.1f}%" for label in labels],
            textposition="auto",
        )
    )
    figure.update_layout(
        title="Class Probabilities",
        template="plotly_dark",
        height=300,
        yaxis_title="Probability (%)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_news(result: dict) -> None:
    """Render latest news articles."""
    st.subheader("Latest News")
    articles = result["news"]
    if not articles:
        st.warning("No news articles available. Sentiment is neutral.")
        return

    for article in articles:
        title = article.get("title") or "Untitled"
        source = (article.get("source") or {}).get("name", "Unknown source")
        url = article.get("url")
        published = article.get("publishedAt", "")[:10]
        if url:
            st.markdown(f"**[{title}]({url})**")
        else:
            st.markdown(f"**{title}**")
        st.caption(f"{source} | {published}")


def render_feature_importance(result: dict) -> None:
    """Render feature importance graph."""
    importance = result["feature_importance"]
    if importance.empty:
        return

    st.subheader("Feature Importance")
    figure = go.Figure(
        go.Bar(
            x=importance["Importance"],
            y=importance["Feature"],
            orientation="h",
            marker_color="#38bdf8",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_download(result: dict) -> None:
    """Render CSV report download button."""
    report = build_prediction_report(result)
    st.download_button(
        "Download Prediction Report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name=f"{result['symbol']}_prediction_report.csv",
        mime="text/csv",
        use_container_width=True,
    )


def price_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create candlestick and close price chart."""
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC",
        )
    )
    figure.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line_color="#22c55e"))
    figure.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA50", line_color="#f59e0b"))
    figure.add_trace(go.Scatter(x=data.index, y=data["SMA200"], name="SMA200", line_color="#ef4444"))
    figure.update_layout(
        title=f"{symbol} Price Chart",
        template="plotly_dark",
        height=560,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return figure


def rsi_chart(data: pd.DataFrame) -> go.Figure:
    """Create RSI chart."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line_color="#a78bfa"))
    figure.add_hline(y=70, line_dash="dash", line_color="#ef4444")
    figure.add_hline(y=30, line_dash="dash", line_color="#22c55e")
    figure.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10))
    return figure


def macd_chart(data: pd.DataFrame) -> go.Figure:
    """Create MACD chart."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line_color="#38bdf8"))
    figure.add_trace(
        go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal", line_color="#f97316")
    )
    figure.add_trace(
        go.Bar(x=data.index, y=data["MACD_Histogram"], name="Histogram", marker_color="#64748b")
    )
    figure.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10))
    return figure


def volume_chart(data: pd.DataFrame) -> go.Figure:
    """Create volume chart."""
    figure = go.Figure(
        go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color="#14b8a6")
    )
    figure.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10))
    return figure


def sentiment_gauge(score: float, label: str) -> go.Figure:
    """Create a sentiment gauge chart."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"News Sentiment: {label}"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [-1, -0.05], "color": "#7f1d1d"},
                    {"range": [-0.05, 0.05], "color": "#334155"},
                    {"range": [0.05, 1], "color": "#14532d"},
                ],
            },
        )
    )
    figure.update_layout(template="plotly_dark", height=230, margin=dict(l=10, r=10, t=35, b=10))
    return figure


def inject_styles() -> None:
    """Inject app-specific CSS."""
    st.markdown(
        """
        <style>
        .metric-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            padding: 22px;
            min-height: 132px;
            background: #0f172a;
        }
        .metric-card span {
            color: #94a3b8;
            display: block;
            font-size: 0.88rem;
            margin-bottom: 16px;
        }
        .metric-card strong {
            color: #e2e8f0;
            font-size: 2.3rem;
            line-height: 1;
        }
        .buy-card {
            border-color: rgba(34, 197, 94, 0.65);
            background: #052e16;
        }
        .hold-card {
            border-color: rgba(245, 158, 11, 0.65);
            background: #451a03;
        }
        .sell-card {
            border-color: rgba(239, 68, 68, 0.65);
            background: #450a0a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_number(value: float | int | None) -> str:
    """Format a numeric indicator value."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.2f}"


def format_percent(value: float | int | None) -> str | None:
    """Format a percent delta for Streamlit metrics."""
    if value is None or pd.isna(value):
        return None
    return f"{float(value):.2f}%"


def money(value: float | int | None, current_info: dict) -> str:
    """Format a currency value."""
    if value is None or pd.isna(value):
        return "N/A"
    currency = current_info.get("currency") or ""
    return f"{currency} {float(value):,.2f}".strip()


def compact_number(value: float | int | None) -> str:
    """Format large numeric values compactly."""
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    for suffix, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(value) >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    return f"{value:,.0f}"


if __name__ == "__main__":
    main()
