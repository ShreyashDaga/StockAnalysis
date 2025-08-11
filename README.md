# StockAnalysis

## Overview
This project predicts short-term stock movement for the **Top Nifty 50 stocks** by combining **technical indicators** with **news sentiment analysis**.  
It fetches historical stock data from Yahoo Finance, retrieves the latest news headlines using NewsAPI, performs sentiment analysis with VADER, calculates technical indicators, and trains a **Random Forest Classifier** to generate **BUY** or **HOLD** signals.

---

## Features
- **Fetches Stock Data** for top Nifty 50 companies from Yahoo Finance.
- **Retrieves Latest News** using NewsAPI.
- **Performs Sentiment Analysis** using the VADER sentiment analyzer.
- **Calculates Technical Indicators**:
  - Simple Moving Average (SMA50, SMA200)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
- **Generates Target Labels** based on 10–15% price increase in the next 5 days.
- **Trains a Machine Learning Model** (Random Forest Classifier) to predict BUY or HOLD signals.
- **Predicts Current Stock Signals** for all top Nifty 50 stocks.

---

## Project Structure
