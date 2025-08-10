# Algo-Trading-Automation
This project is an end-to-end algorithmic trading signal generator that uses technical indicators and a machine learning classifier (SVM) to identify buy and sell opportunities for selected stocks.

Key Features:
Data Collection: Fetches up-to-date stock data using Yahoo Finance API (yfinance).

Technical Indicators:

RSI (Relative Strength Index)

MACD (Moving Average Convergence Divergence)

Moving Averages (10, 20, 50 days)

Bollinger Bands

Daily Returns & Volatility

Signal Generation:

Buy: RSI < 30 & short-term trend up

Sell: RSI > 70 & short-term trend down

Machine Learning Prediction: Uses Support Vector Machine (SVM) to predict next-day price movement.

Performance Tracking: Calculates total PnL and win ratio for executed trades.

Google Sheets Logging: Stores trade logs, summary statistics, and win ratios in separate tabs.

Telegram Alerts: Sends buy/sell signals with ML accuracy score directly to your Telegram chat.

Scheduled Execution: Automatically runs daily scans at a set time.

