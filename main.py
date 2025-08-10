# library imports
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
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
tickers = ['ONGC.NS', 'HCLTECH.NS', 'HEROMOTOCO.NS', 'TECHM.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'SUNPHARMA.NS']
sheet_name = "AlgoTradeLogs"
telegram_token = os.getenv("telegram_token")
telegram_chat_id = os.getenv("telegram_chat_id")
trade_log_tab = "Trade_Log"
summary_tab = "Summary_PnL"
win_ratio_tab = "Win_Ratio"
previous_data = {t: None for t in tickers}


# logging
logging.basicConfig(filename='trading.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# data ingestion
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    if df is None or df.empty:
        return pd.DataFrame()
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
    df['RSI_1d_ago'] = df['RSI'].shift(1)
    df['MACD_1d_ago'] = df['MACD'].shift(1)
    df['Return_1d_ago'] = df['Returns'].shift(1)
    df['Buy_Signal'] = (df['RSI'] < 30) & (df['20DMA'] > df['50DMA'])  # Buy Signal: oversold + short-term trend up
    df['Sell_Signal'] = (df['RSI'] > 70) & (df['20DMA'] < df['50DMA'])  # Sell Signal: overbought + short-term trend down
    df['Buy_Signal_Future'] = df['Buy_Signal'].shift(-1)
    df['Sell_Signal_Future'] = df['Sell_Signal'].shift(-1)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df


# ml prediction
def predict_movement(df, target_column):
    features = [
        'RSI', 'MACD', '10DMA', '20DMA', '50DMA',
        'Returns', 'Volatility',
        'RSI_1d_ago', 'MACD_1d_ago', 'Return_1d_ago'
    ]
    missing = [c for c in features + [target_column] if c not in df.columns]
    if missing:
        logging.warning(f"predict_movement: missing columns {missing}")
        return 0.0, 0
    data = df[features + [target_column]].dropna()

    if data.shape[0] < 30:
        logging.info(f"predict_movement: insufficient rows ({data.shape[0]}) for target {target_column}")
        return 0.0, 0
    X = data[features]
    y = data[target_column].astype(int)
    if len(np.unique(y)) < 2:
        logging.info(f"predict_movement: only one class found for {target_column}, relaxing for training")
        if 'Buy_Signal_Future' in target_column:
            y = (df['RSI'] < 35).astype(int).iloc[-len(X):]  # relaxed RSI for training only
        elif 'Sell_Signal_Future' in target_column:
            y = (df['RSI'] > 65).astype(int).iloc[-len(X):]
        if len(np.unique(y)) < 2:
            return 0.0, 0  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    try:
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            objective='binary:logistic',
            base_score=0.5
        )
        model.fit(X_train, y_train)
    except Exception as e:
        logging.exception(f"predict_movement: model training failed: {e}")
        return 0.0, 0
    y_pred_test = model.predict(X_test)
    acc = round(float(accuracy_score(y_test, y_pred_test)), 2) if len(y_test) > 0 else 0.0
    try:
        tomorrow_pred = int(model.predict(X.iloc[[-1]])[0])
    except Exception:
        tomorrow_pred = 0

    return acc, tomorrow_pred


# pnl and win ratio calculation
def calculate_performance(trade_log):
    if trade_log is None or trade_log.empty:
        return pd.DataFrame(), pd.DataFrame()
    trade_log['Close'] = trade_log['Close'].astype(str).str.extract(r'(\d+\.\d+)').astype(float)
    trade_log = trade_log.copy()
    trade_log.sort_values(['Ticker', 'Date'], inplace=True)
    results = []
    win_ratio_dict = {}

    for ticker, group in trade_log.groupby('Ticker'):
        group = group.sort_values('Date').reset_index(drop=True)
        open_trade = None
        trades_pnl = []
        for _, row in group.iterrows():
            if row['Signal'] == 'BUY' and open_trade is None:
                open_trade = row
            elif row['Signal'] == 'SELL' and open_trade is not None:
                pnl = row['Close'] - open_trade['Close']
                results.append({
                    "Ticker": ticker,
                    "Entry Date": open_trade['Date'],
                    "Entry Price": open_trade['Close'],
                    "Exit Date": row['Date'],
                    "Exit Price": row['Close'],
                    "PnL": pnl
                })
                trades_pnl.append(pnl)
                open_trade = None  # reset after closing
        win_ratio_dict[ticker] = (sum(1 for p in trades_pnl if p > 0) / len(trades_pnl) * 100) if trades_pnl else 0.0
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        total_pnl = results_df['PnL'].sum()
        win_ratio = (results_df['PnL'] > 0).mean() * 100
        summary_df = pd.DataFrame({
            'Total Trades': [len(results_df)],
            'Total PnL': [total_pnl],
            'Win Ratio (%)': [win_ratio]
        })
    else:
        summary_df = pd.DataFrame(columns=['Total Trades', 'Total PnL', 'Win Ratio (%)'])
    win_ratio_df = pd.DataFrame(list(win_ratio_dict.items()), columns=['Ticker', 'Win Ratio (%)'])
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
    except Exception:
        worksheet = sheet.add_worksheet(title=tab, rows="1000", cols="20")
    set_with_dataframe(worksheet, df)


# telegram alerts
def send_telegram_alert(message):
    if not telegram_token or not telegram_chat_id:
        logging.debug("Telegram token/chat id missing; skipping alert.")
        return
    try:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        requests.post(url, data={"chat_id": telegram_chat_id, "text": message})
    except Exception:
        logging.exception("Failed to send telegram alert")


# main workflow
def run():
    global previous_data
    logging.info("Starting run process")

    try:
        sheet = connect_to_sheet()
        logging.info("Connected to Google Sheet")
    except Exception as e:
        logging.exception(f"Failed to connect to Google Sheet: {e}")
        return

    trade_log_list = []  

    for ticker in tickers:
        try:
            logging.info(f"Processing ticker: {ticker}")

            current_df = fetch_data(ticker)
            if current_df.empty:
                logging.warning(f"No data for {ticker}. Skipping.")
                continue
            last_close = float(current_df['Close'].iloc[-1])
            if previous_data.get(ticker) == last_close:
                logging.info(f"No change in last close for {ticker}. Skipping.")
                continue
            df = apply_strategy(current_df.copy())
            if df.empty:
                logging.warning(f"Strategy returned empty DataFrame for {ticker}")
                previous_data[ticker] = last_close
                continue
            buy_acc, tomorrow_buy = predict_movement(df, 'Buy_Signal_Future')
            sell_acc, tomorrow_sell = predict_movement(df, 'Sell_Signal_Future')
            if tomorrow_buy > 0:
                trade_log_list.append({
                    "Date": current_df.index[-1].strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Signal": "BUY",
                    "Close": last_close,
                    "Accuracy": buy_acc
                })
            elif tomorrow_sell > 0:
                trade_log_list.append({
                    "Date": current_df.index[-1].strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Signal": "SELL",
                    "Close": last_close,
                    "Accuracy": sell_acc
                })
            previous_data[ticker] = last_close
            if tomorrow_buy > 0 and buy_acc > 0.5 and tomorrow_sell <= 0:
                alert_message = f"""
                {ticker}
                Close: {last_close}
                Tomorrow Buy: {tomorrow_buy} (Accuracy: {buy_acc:.2f})
                """
                send_telegram_alert(alert_message)
            elif tomorrow_sell > 0 and sell_acc > 0.5 and tomorrow_buy <= 0:
                alert_message = f"""
                {ticker}
                Close: {last_close}
                Tomorrow Sell: {tomorrow_sell} (Accuracy: {sell_acc:.2f})
                """
                send_telegram_alert(alert_message)
            elif tomorrow_buy > 0 and buy_acc > 0.5 and tomorrow_sell > 0 and sell_acc > 0.5:
                alert_message = f"""
                {ticker}
                Close: {last_close}
                Tomorrow Buy: {tomorrow_buy} (Accuracy: {buy_acc:.2f})
                Tomorrow Sell: {tomorrow_sell} (Accuracy: {sell_acc:.2f})
                """
                send_telegram_alert(alert_message)
        except Exception as e:
            logging.exception(f"Error processing {ticker}: {e}")

    if trade_log_list:
        try:
            trade_log_df = pd.DataFrame(trade_log_list)
            summary_df, win_ratio_df = calculate_performance(trade_log_df)
            update_sheet(sheet, trade_log_tab, trade_log_df)
            update_sheet(sheet, summary_tab, summary_df)
            update_sheet(sheet, win_ratio_tab, win_ratio_df)
            logging.info("Google Sheet updated successfully")
        except Exception as e:
            logging.exception(f"Failed to update Google Sheet: {e}")
    else:
        logging.info("No new data to update in Google Sheet")


# function to see pnl and win ratio of past trades
def backfill_trades():
    logging.info("Starting backfill process for historical trades")
    try:
        sheet = connect_to_sheet()
        logging.info("Connected to Google Sheet for backfill")
    except Exception as e:
        logging.exception(f"Failed to connect to Google Sheet: {e}")
        return
    trade_log_list = []
    for ticker in tickers:
        try:
            logging.info(f"Backfilling ticker: {ticker}")
            df = fetch_data(ticker)
            if df.empty:
                logging.warning(f"No data for {ticker}, skipping.")
                continue
            df = apply_strategy(df.copy())
            if df.empty:
                continue
            for i in range(len(df) - 1):  
                date_str = df.index[i].strftime('%Y-%m-%d')
                close_price = df['Close'].iloc[i]

                if df['Buy_Signal'].iloc[i]:
                    trade_log_list.append({
                        "Date": date_str,
                        "Ticker": ticker,
                        "Signal": "BUY",
                        "Close": close_price,
                        "Accuracy": None  
                    })
                elif df['Sell_Signal'].iloc[i]:
                    trade_log_list.append({
                        "Date": date_str,
                        "Ticker": ticker,
                        "Signal": "SELL",
                        "Close": close_price,
                        "Accuracy": None
                    })
        except Exception as e:
            logging.exception(f"Error backfilling {ticker}: {e}")
    trade_log_df = pd.DataFrame(trade_log_list)
    summary_df, win_ratio_df = calculate_performance(trade_log_df)
    try:
        update_sheet(sheet, trade_log_tab, trade_log_df)
        update_sheet(sheet, summary_tab, summary_df)
        update_sheet(sheet, win_ratio_tab, win_ratio_df)
        logging.info("Backfill Google Sheet updated successfully")
    except Exception as e:
        logging.exception(f"Failed to update Google Sheet during backfill: {e}")
    logging.info("Backfill process completed")

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
