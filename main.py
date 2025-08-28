# library imports
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import logging
import requests
import schedule
import time
from dotenv import load_dotenv
import os
import json

# load data from .env
load_dotenv()


# configuration
tickers = ['MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'XOM']
sheet_name = "AlgoTradeLogs"
telegram_token = os.getenv("TELEGRAM_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
trade_log_tab = "Trade_Log"
summary_tab = "Summary_PnL"
win_ratio_tab = "Win_Ratio"
previous_data = {t: None for t in tickers}


# logging
logging.basicConfig(filename='trading.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# data ingestion
def fetch_data(ticker):
    df = yf.download(ticker, period="48mo", interval="1d")
    if df is None or df.empty:
        return pd.DataFrame()
    df.dropna(inplace=True)
    return df

# indicators
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi_vals = 100 - (100 / (1 + rs))
    # ensure 1-D
    rsi_vals = np.ravel(rsi_vals.to_numpy())
    return pd.Series(rsi_vals, index=series.index, name="RSI")

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return (pd.Series(np.ravel(macd_line.to_numpy()), index=series.index, name="MACD"),
            pd.Series(np.ravel(signal_line.to_numpy()), index=series.index, name="MACD_signal"),
            pd.Series(np.ravel(hist.to_numpy()), index=series.index, name="MACD_hist"))

def stoch(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(d_period).mean()
    return (pd.Series(np.ravel(k.to_numpy()), index=close.index, name="STOCH_K"),
            pd.Series(np.ravel(d.to_numpy()), index=close.index, name="STOCH_D"))

def ema(series, span):
    val = series.ewm(span=span, adjust=False).mean()
    return pd.Series(np.ravel(val.to_numpy()), index=series.index, name=f"EMA{span}")

def williams_r(high, low, close, period=14):
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
    return pd.Series(np.ravel(wr.to_numpy()), index=close.index, name="WilliamsR")

def adx(high, low, close, period=14):
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    plus_dm = np.ravel(plus_dm)
    minus_dm = np.ravel(minus_dm)

    tr = pd.concat([(high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).sum() / (atr + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).sum() / (atr + 1e-10))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx_val = dx.rolling(period).mean()
    return pd.Series(np.ravel(adx_val.to_numpy()), index=high.index, name="ADX")

def bollinger_bands(series, period=20, std=2):
    sma = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    percent = (series - lower) / (upper - lower + 1e-10)
    return (pd.Series(np.ravel(upper.to_numpy()), index=series.index, name="BB_upper"),
            pd.Series(np.ravel(lower.to_numpy()), index=series.index, name="BB_lower"),
            pd.Series(np.ravel(percent.to_numpy()), index=series.index, name="BB_percent"))

def atr(high, low, close, period=14):
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    val = tr.rolling(period).mean()
    return pd.Series(np.ravel(val.to_numpy()), index=close.index, name="ATR")

def obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    obv_val = (direction * volume).cumsum()
    return pd.Series(np.ravel(obv_val.to_numpy()), index=close.index, name="OBV")

def cmf(high, low, close, volume, period=20):
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
    mf_volume = mf_multiplier * volume
    cmf_val = mf_volume.rolling(period).sum() / (volume.rolling(period).sum() + 1e-10)
    return pd.Series(np.ravel(cmf_val.to_numpy()), index=close.index, name="CMF")

def add_features(df):
    # Indicators
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    df["STOCH_K"], df["STOCH_D"] = stoch(df["High"], df["Low"], df["Close"])
    df["ROC"] = (df["Close"] / df["Close"].shift(10) - 1) * 100
    df["WilliamsR"] = williams_r(df["High"], df["Low"], df["Close"])

    df["EMA10"] = ema(df["Close"], 10)
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["ADX"] = adx(df["High"], df["Low"], df["Close"])

    df["BB_upper"], df["BB_lower"], df["BB_percent"] = bollinger_bands(df["Close"])
    df["ATR"] = atr(df["High"], df["Low"], df["Close"])

    df["OBV"] = obv(df["Close"], df["Volume"])
    df["CMF"] = cmf(df["High"], df["Low"], df["Close"], df["Volume"])

    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    df["Zscore_Close"] = (df["Close"] - df["Close"].rolling(20).mean()) / (df["Close"].rolling(20).std() + 1e-10)

    df["Buy_Signal"]  = (df["RSI"] < 40) & (df["EMA20"] > df["EMA50"])
    df["Sell_Signal"] = (df["RSI"] > 60) & (df["EMA20"] < df["EMA50"])
    df["Buy_Signal_Future"]  = df["Buy_Signal"].shift(-1).fillna(0).astype(int)
    df["Sell_Signal_Future"] = df["Sell_Signal"].shift(-1).fillna(0).astype(int)

    return df

def tune_and_train_model(X, y, scoring="f1", n_splits=5):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "min_child_weight": [1, 3, 5]
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=tscv,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"\nBest Params: {best_params}")
    print(f"Best CV Score ({scoring}): {best_score:.4f}")

    best_model.fit(X, y)

    return best_model, best_params, best_score

# ml prediction
def predict_movement(df, target_column):
    features = [
        "RSI", "MACD", "MACD_signal", "MACD_hist",
        "STOCH_K", "STOCH_D", "ROC", "WilliamsR",
        "EMA10", "EMA20", "EMA50", "ADX",
        "BB_upper", "BB_lower", "BB_percent", "ATR",
        "OBV", "CMF",
        "Return_1d", "Return_5d", "Volatility_10d", "Zscore_Close"
    ]

    missing = [c for c in features + [target_column] if c not in df.columns]
    if missing:
        logging.warning(f"predict_movement: missing columns {missing}")
        return {"train": {}, "test": {}}, 0  
    
    data = df[features + [target_column]].dropna()

    empty_metrics = {
    "train": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
    "test": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    }

    if data.shape[0] < 30:
        logging.info(f"predict_movement: insufficient rows ({data.shape[0]}) for target {target_column}")
        return empty_metrics, 0 
    
    X = data[features]
    y = data[target_column].astype(int)

    if len(np.unique(y)) < 2:
        logging.info(f"predict_movement: only one class found for {target_column}, relaxing for training")
        if 'Buy_Signal_Future' in target_column:
            y = (df['RSI'] < 35).astype(int).iloc[-len(X):]  # relaxed RSI for training only
        elif 'Sell_Signal_Future' in target_column:
            y = (df['RSI'] > 65).astype(int).iloc[-len(X):]
        if len(np.unique(y)) < 2:
            return empty_metrics, 0 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    try:
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.5,
            colsample_bytree= 0.8, 
            random_state=42,
            eval_metric='logloss',
            objective='binary:logistic',
            scale_pos_weight = len(y[y==0]) / max(1, len(y[y==1]))
        )

        model.fit(X_train, y_train)

    except Exception as e:
        logging.exception(f"predict_movement: model training failed: {e}")
        return empty_metrics, 0  

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # metrics dictionary
    metrics = {
        "train": {
            "accuracy": round(accuracy_score(y_train, y_pred_train), 4),
            "precision": round(precision_score(y_train, y_pred_train, zero_division=0), 4),
            "recall": round(recall_score(y_train, y_pred_train, zero_division=0), 4),
            "f1": round(f1_score(y_train, y_pred_train, zero_division=0), 4)
        },
        "test": {
            "accuracy": round(accuracy_score(y_test, y_pred_test), 4) if len(y_test) > 0 else 0.0,
            "precision": round(precision_score(y_test, y_pred_test, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_test, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_test, zero_division=0), 4)
        }
    }

    try:
        tomorrow_pred = int(model.predict(X.iloc[[-1]])[0])
    except Exception:
        tomorrow_pred = 0

    return metrics, tomorrow_pred


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
    creds_dict = json.loads(os.environ["GOOGLE_CREDS_JSON"])
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name)

def update_sheet(sheet, tab, df):
    try:
        worksheet = sheet.worksheet(tab)
        existing_records = worksheet.get_all_records()
    except Exception:
        worksheet = sheet.add_worksheet(title=tab, rows="1000", cols="20")
        existing_records = worksheet.get_all_records()
    if existing_records:
        existing_df = pd.DataFrame(existing_records)
        df = pd.concat([existing_df, df], ignore_index=True)
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
    metrics_log_list = []  

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
            df = add_features(current_df.copy())
            if df.empty:
                logging.warning(f"Strategy returned empty DataFrame for {ticker}")
                previous_data[ticker] = last_close
                continue

            buy_metrics, tomorrow_buy = predict_movement(df, 'Buy_Signal_Future')
            sell_metrics, tomorrow_sell = predict_movement(df, 'Sell_Signal_Future')

            if tomorrow_buy > 0:
                trade_log_list.append({
                    "Date": current_df.index[-1].strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Signal": "BUY",
                    "Close": last_close,
                    "Accuracy": buy_metrics.get("test", {}).get("accuracy", 0.0)
                })
            elif tomorrow_sell > 0:
                trade_log_list.append({
                    "Date": current_df.index[-1].strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Signal": "SELL",
                    "Close": last_close,
                    "Accuracy": sell_metrics.get("test", {}).get("accuracy", 0.0)
                })

            # record model metrics for Google Sheet
            if buy_metrics:
                metrics_log_list.append({
                    "Date": current_df.index[-1].strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Model": "Buy",
                    "Train_Accuracy": buy_metrics["train"]["accuracy"],
                    "Train_Precision": buy_metrics["train"]["precision"],
                    "Train_Recall": buy_metrics["train"]["recall"],
                    "Train_F1": buy_metrics["train"]["f1"],
                    "Test_Accuracy": buy_metrics["test"]["accuracy"],
                    "Test_Precision": buy_metrics["test"]["precision"],
                    "Test_Recall": buy_metrics["test"]["recall"],
                    "Test_F1": buy_metrics["test"]["f1"]
                })
            if sell_metrics:
                metrics_log_list.append({
                    "Date": current_df.index[-1].strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Model": "Sell",
                    "Train_Accuracy": sell_metrics["train"]["accuracy"],
                    "Train_Precision": sell_metrics["train"]["precision"],
                    "Train_Recall": sell_metrics["train"]["recall"],
                    "Train_F1": sell_metrics["train"]["f1"],
                    "Test_Accuracy": sell_metrics["test"]["accuracy"],
                    "Test_Precision": sell_metrics["test"]["precision"],
                    "Test_Recall": sell_metrics["test"]["recall"],
                    "Test_F1": sell_metrics["test"]["f1"]
                })

            previous_data[ticker] = last_close

            if tomorrow_buy > 0 and buy_metrics["test"]["accuracy"] > 0.5:
                alert_message = f"""
                {ticker}
                Close: {last_close}
                Tomorrow Buy: {tomorrow_buy} (Test Accuracy: {buy_metrics['test']['accuracy']:.2f})
                """
                send_telegram_alert(alert_message)
            elif tomorrow_sell > 0 and sell_metrics["test"]["accuracy"] > 0.5:
                alert_message = f"""
                {ticker}
                Close: {last_close}
                Tomorrow Sell: {tomorrow_sell} (Test Accuracy: {sell_metrics['test']['accuracy']:.2f})
                """
                send_telegram_alert(alert_message)

        except Exception as e:
            logging.exception(f"Error processing {ticker}: {e}")

    if trade_log_list or metrics_log_list:
        try:
            if trade_log_list:
                trade_log_df = pd.DataFrame(trade_log_list)
                summary_df, win_ratio_df = calculate_performance(trade_log_df)
                update_sheet(sheet, trade_log_tab, trade_log_df)
                update_sheet(sheet, summary_tab, summary_df)
                update_sheet(sheet, win_ratio_tab, win_ratio_df)

            if metrics_log_list:
                metrics_df = pd.DataFrame(metrics_log_list)
                update_sheet(sheet, "Model_Metrics", metrics_df)

            logging.info("Google Sheet updated successfully")
        except Exception as e:
            logging.exception(f"Failed to update Google Sheet: {e}")
    else:
        logging.info("No new data to update in Google Sheet")


# entry point
if __name__ == "__main__":
    send_telegram_alert("Hello from GitHub Actions!")
    sheet = connect_to_sheet()
    ws = sheet.sheet1
    ws.update("A1", [["Updated from Actions"]])
    run()













