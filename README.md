# StockAnalysis AI

AI-powered stock market analysis web app that generates BUY, HOLD, and SELL recommendations from historical Yahoo Finance prices, technical indicators, NewsAPI sentiment, VADER, and a Random Forest classifier.

## Live Demo:
https://stockanalysis-pzrgylmvgvdlohkcnurq42.streamlit.app/

## Features

- Live historical OHLCV data from Yahoo Finance.
- Technical indicators: SMA50, SMA200, RSI, MACD, MACD signal, and MACD histogram.
- Additional ML features: daily return, 20-day volatility, and volume ratio.
- News sentiment analysis with NewsAPI and VADER.
- Configurable BUY/HOLD/SELL labels based on future returns.
- RandomForestClassifier with balanced class weights and Joblib persistence.
- Automatic model loading and retraining when feature or threshold settings change.
- Confidence score and class probability chart.
- Interactive Plotly candlestick, close price, volume, RSI, and MACD charts.
- Company snapshot with current price, daily change, market cap, 52-week high/low, and dividend yield.
- Feature importance graph.
- Downloadable CSV prediction report.
- Dark Streamlit UI ready for Streamlit Community Cloud.

## Screenshots

Add screenshots of the running Streamlit app here after deployment.

## Machine Learning Logic

The model predicts a signal from future returns over a configurable horizon.

- `BUY`: future return is greater than or equal to `BUY_RETURN_THRESHOLD`
- `SELL`: future return is less than or equal to `SELL_RETURN_THRESHOLD`
- `HOLD`: future return is between the sell and buy thresholds

Defaults:

```text
PREDICTION_HORIZON_DAYS=5
BUY_RETURN_THRESHOLD=0.05
SELL_RETURN_THRESHOLD=-0.03
```

You can override these values with environment variables before running or deploying the app.

## Installation

```bash
git clone https://github.com/ShreyashDaga/StockAnalysis.git
cd StockAnalysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For live news sentiment, create a `.env` file locally or configure Streamlit secrets:

```bash
NEWS_API_KEY=your_newsapi_key
```

The app still runs without `NEWS_API_KEY`; sentiment defaults to neutral when live news is unavailable.

## Usage

```bash
streamlit run app.py
```

Choose a stock from the sidebar, optionally enter a custom Yahoo Finance symbol, and click **Analyze Stock**.

On first run, the app trains `models/model.pkl` if no compatible model exists. Later runs load the saved model automatically.

## Project Structure

```text
StockAnalysis/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── models/
│   └── .gitkeep
├── utils/
│   ├── __init__.py
│   ├── fetch_data.py
│   ├── indicators.py
│   ├── sentiment.py
│   ├── model.py
│   └── predict.py
├── notebooks/
│   └── trade1.ipynb
└── assets/
    └── .gitkeep
```

## Technologies Used

- Python
- Streamlit
- Yahoo Finance via `yfinance`
- NewsAPI
- VADER Sentiment
- pandas
- scikit-learn
- Joblib
- Plotly

## Deployment

To deploy on Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new Streamlit app.
3. Set `app.py` as the entrypoint.
4. Add `NEWS_API_KEY` in Streamlit app secrets if news sentiment is required.
5. Optionally configure `PREDICTION_HORIZON_DAYS`, `BUY_RETURN_THRESHOLD`, and `SELL_RETURN_THRESHOLD`.
6. Deploy.

Because `models/model.pkl` is ignored by Git, the model trains on first deployment run and is loaded afterward when the deployment filesystem preserves it.

## Future Improvements

- Add walk-forward validation for time-series evaluation.
- Add model training logs and metric history.
- Include additional indicators such as Bollinger Bands, ATR, ADX, and stochastic oscillator.
- Add SHAP-based explainability.
- Add portfolio watchlists and alerting.
- Add cached data storage for faster repeated analysis.

## Author

Built by Shreyash Daga.
