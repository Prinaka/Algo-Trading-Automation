# library imports
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import logging
import requests
import schedule
import time
from dotenv import load_dotenv
import os

# load data from .env
load_dotenv()

# configuration
tickers = ['INFY.NS', 'SBIN.NS', 'ADANIENT.NS', 
           'POWERGRID.NS', 'BPCL.NS', 'IOC.NS', 'SUNPHARMA.NS']
sheet_name = "AlgoTradeLogs"
telegram_token = os.getenv("telegram_token")
telegram_chat_id = os.getenv("telegram_chat_id")
trade_log_tab = "Trade_Log"
summary_tab = "Summary_PnL"
win_ratio_tab = "Win_Ratio"

# logging
logging.basicConfig(filename='trading.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# data ingestion
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df.dropna(inplace=True)
    return df

# indicators
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# strategy signals
def apply_strategy(df):
    df['RSI'] = rsi(df['Close'])
    df['10DMA'] = df['Close'].rolling(window=10).mean()
    df['20DMA'] = df['Close'].rolling(window=20).mean()
    df['50DMA'] = df['Close'].rolling(window=50).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    df['MACD'], df['MACD_signal'] = macd(df['Close'])
    bb_mid = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Mid'] = bb_mid
    df['BB_Upper'] = bb_mid + (2 * bb_std)
    df['BB_Lower'] = bb_mid - (2 * bb_std)
    df['RSI_1d_ago'] = df['RSI'].shift(1)
    df['MACD_1d_ago'] = df['MACD'].shift(1)
    df['Return_1d_ago'] = df['Returns'].shift(1)
    df['Buy_Signal'] = (df['RSI'] < 30) & (df['20DMA'] > df['50DMA']) # Buy Signal: oversold + short-term trend up
    df['Sell_Signal'] = (df['RSI'] > 70) & (df['20DMA'] < df['50DMA']) # Sell Signal: overbought + short-term trend down
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

# ml prediction
def predict_movement(df):
    features = [
        'RSI', 'MACD',
        '10DMA', '20DMA', '50DMA',
        'Returns', 'Volatility',
        'RSI_1d_ago', 'MACD_1d_ago', 'Return_1d_ago'
    ]
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = DecisionTreeClassifier(max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return accuracy

def calculate_performance(trade_log):
    if trade_log.empty:
        return pd.DataFrame(), pd.DataFrame()
    trade_log['Close'] = pd.to_numeric(trade_log['Close'], errors='coerce')
    trade_log['PnL'] = trade_log['Close'].shift(-1) - trade_log['Close']
    trade_log.dropna(subset=['PnL'], inplace=True)
    trade_log['PnL'] = pd.to_numeric(trade_log['PnL'], errors='coerce')
    total_pnl = trade_log['PnL'].sum()
    win_ratio = (trade_log['PnL'] > 0).mean() * 100
    summary_df = pd.DataFrame({
        'Total Trades': [len(trade_log)],
        'Total PnL': [total_pnl],
        'Win Ratio (%)': [win_ratio]
    })
    win_ratio_df = (
        trade_log.groupby('Ticker')['PnL']
        .apply(lambda x: (x > 0).mean() * 100)
        .reset_index(name='Win Ratio (%)')
    )
    return summary_df, win_ratio_df

# google sheets setup
def connect_to_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name)


def update_sheet(sheet, tab, df):
    try:
        worksheet = sheet.worksheet(tab)
    except:
        worksheet = sheet.add_worksheet(title=tab, rows="1000", cols="20")
    set_with_dataframe(worksheet, df)

# telegram alerts
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage?chat_id={telegram_chat_id}&text={message}"
    requests.get(url)

# main workflow
def run():
    logging.info("Starting daily scan...")
    sheet = connect_to_sheet()
    all_trades = []

    for ticker in tickers:
        df = fetch_data(ticker)
        df = apply_strategy(df)
        last_date = pd.Timestamp.today().normalize()
        lookback_days = 7
        # Buy and Sell signals for today
        buy_signals = df[(df['Buy_Signal'] == True) & (df.index >= last_date - pd.Timedelta(days=lookback_days))]
        sell_signals = df[(df['Sell_Signal'] == True) & (df.index >= last_date - pd.Timedelta(days=lookback_days))]
        logging.info(f"{ticker}: {len(buy_signals)} buy signals, {len(sell_signals)} sell signals in last {lookback_days} days")
        ticker_signals = []
        if not buy_signals.empty:
            latest_buy = buy_signals.iloc[-1]
            ticker_signals.append({
                'Ticker': ticker,
                'Date': latest_buy.name.date(),
                'Close': latest_buy['Close'],
                'Type': 'BUY'
            })
            all_trades.append(ticker_signals[-1])
        if not sell_signals.empty:
            latest_sell = sell_signals.iloc[-1]
            ticker_signals.append({
                'Ticker': ticker,
                'Date': latest_sell.name.date(),
                'Close': latest_sell['Close'],
                'Type': 'SELL'
            })
            all_trades.append(ticker_signals[-1])
        acc = predict_movement(df)
        logging.info(f"{ticker} ML Accuracy: {acc:.2f}")
        if not buy_signals.empty and acc >= 0.5:
            send_telegram_alert(f"Buy Signal for {ticker} at {buy_signals.index[-1].date()} - Accuracy: {acc*100:.0f}%")
        if not sell_signals.empty and acc >= 0.5:
            send_telegram_alert(f"Sell Signal for {ticker} at {sell_signals.index[-1].date()} - Accuracy: {acc*100:.0f}%")
        if ticker_signals:
            ticker_df = pd.DataFrame(ticker_signals)
            update_sheet(sheet, ticker, ticker_df)
    trade_log_df = pd.DataFrame(all_trades)
    update_sheet(sheet, trade_log_tab, trade_log_df)
    summary_df, win_ratio_df = calculate_performance(trade_log_df)
    update_sheet(sheet, summary_tab, summary_df)
    update_sheet(sheet, win_ratio_tab, win_ratio_df)

# scheduler
def start_scheduler():
    schedule.every().day.at("09:00:00").do(run)
    while True:
        schedule.run_pending()
        time.sleep(60)

# entry point
if __name__ == "__main__":
    run()
    start_scheduler()
