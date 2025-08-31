# Algo Trading Automation
This project is an end-to-end algorithmic trading signal generator that uses technical indicators and a machine learning classifier (XGBClassifier) to identify buy and sell opportunities for selected stocks. The pipeline is automated with GitHub Actions, so that it runs every day at 9:00 AM IST (03:30 UTC).

**Key Features:**

* Data Collection: Fetches up-to-date stock data using Yahoo Finance API (yfinance).
* Momentum Indicators:
  - RSI (Relative Strength Index) → Detects overbought (>70) and oversold (<30) conditions.
  - MACD (Moving Average Convergence Divergence) → Captures trend strength and momentum shifts.
  - Stochastic Oscillator (STOCH_K, STOCH_D) → Measures the closing price relative to recent highs/lows.
  - ROC (Rate of Change) → Quantifies momentum by comparing today’s price to earlier values.
  - Williams %R → Similar to Stochastic; identifies overbought/oversold signals.
* Trend Indicators:
  - EMA10, EMA20, EMA50 → Short, medium, and longer-term exponential moving averages for trend direction.
  - ADX (Average Directional Index) → Measures trend strength (not direction).
* Volatility Indicators:
  - Bollinger Bands (Upper, Lower, %B) → Tracks volatility expansion/contraction and price extremes.
  - ATR (Average True Range) → Measures volatility magnitude.
  - Z-Score of Close → Standardised measure of how far the price deviates from its mean.
  - Volatility (10-day rolling) → Captures recent fluctuations.
* Volume-based Indicators:
  - OBV (On-Balance Volume) → Cumulative volume confirming price moves.
  - CMF (Chaikin Money Flow) → Combines price and volume to gauge buying/selling pressure.
* Statistical Features:
  - 1-day and 5-day Returns → Captures short-term momentum.
* Signal Generation:
  - Buy: RSI < 30 & short-term trend up
  - Sell: RSI > 70 & short-term trend down
* Machine Learning Prediction: Uses Extreme Gradient Boosting (XGBoost) with Time-Series CV for buy/sell signal prediction.
* Performance Tracking: Calculates total PnL and win ratio for executed trades.
* Google Sheets Logging: Stores trade logs, summary statistics, and win ratios in separate tabs.
* Telegram Alerts: Sends buy/sell signals with ML accuracy score directly to your Telegram chat.
* Scheduled Execution: Scheduled with GitHub Actions (cron job). The workflow file is in .github/workflows/daily-run.yml. It runs the bot automatically at 03:30 UTC (9:00 AM IST) daily. You can also trigger a manual run from the Actions tab in GitHub.

**Outputs:**

* Google Sheets:
  - Trade_Log → Daily trade signals
  - Summary_PnL → Total PnL and trades
  - Win_Ratio → Win ratio per ticker
  - Model_Metrics → Train/Test performance metrics
<img width="1331" height="624" alt="Image" src="https://github.com/user-attachments/assets/ce488e3d-9e07-403d-8ff7-ed7a167ad5b3" />
<img width="953" height="252" alt="Image" src="https://github.com/user-attachments/assets/026f614d-8d46-4420-b4e8-692749f715ab" />

* Telegram Alerts: Buy/Sell signals with model accuracy delivered to your Telegram

**Installation:**

1. Clone the repository:
```
gh repo clone Prinaka/Algo-Trading-Automation
cd algo-trading-automation
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```

**License:**

This project is licensed under the MIT License – see the LICENSE file for details.
