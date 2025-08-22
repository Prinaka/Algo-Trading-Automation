# Algo-Trading-Automation
This project is an end-to-end algorithmic trading signal generator that uses technical indicators and a machine learning classifier (XGBClassifier) to identify buy and sell opportunities for selected stocks. The pipeline is automated with GitHub Actions, so that it runs every day at 9:00 AM IST (03:30 UTC).

Key Features:
* Data Collection: Fetches up-to-date stock data using Yahoo Finance API (yfinance).
* Technical Indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Daily Returns & Volatility
* Signal Generation:
  - Buy: RSI < 30 & short-term trend up
  - Sell: RSI > 70 & short-term trend down
* Machine Learning Prediction: Uses Extreme Gradient Boosting (XGBoost) with Time-Series CV for buy/sell signal prediction.
* Performance Tracking: Calculates total PnL and win ratio for executed trades.
* Google Sheets Logging: Stores trade logs, summary statistics, and win ratios in separate tabs.
* Telegram Alerts: Sends buy/sell signals with ML accuracy score directly to your Telegram chat.
* Scheduled Execution: Scheduled with GitHub Actions (cron job). The workflow file is in .github/workflows/daily-run.yml. It runs the bot automatically at 03:30 UTC (9:00 AM IST) daily. You can also trigger a manual run from the Actions tab in GitHub.

Outputs:
* Google Sheets:
  - Trade_Log → Daily trade signals
  - Summary_PnL → Total PnL and trades
  - Win_Ratio → Win ratio per ticker
  - Model_Metrics → Train/Test performance metrics
* Telegram Alerts: Buy/Sell signals with model accuracy delivered to your Telegram
