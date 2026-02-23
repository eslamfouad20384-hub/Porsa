import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import json, os
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(layout="wide")
st.title("🔥 EGX100 AI Flexible Scanner")

# -----------------------------
EGX100 = [
    "COMI.CA","MFPC.CA","PHDC.CA","ACRI.CA","ORAS.CA","EFGH.CA",
    "HRHO.CA","TMGH.CA","FWRY.CA","SWDY.CA","ETEL.CA",
    "AMOC.CA","HELI.CA","SODIC.CA","EGCH.CA"
]

CACHE_FILE = "market_cache.json"
MODEL_FILE = "ai_model.pkl"

# -----------------------------
@st.cache_data(ttl=86400)
def load_market_data(symbols, interval, period):
    data = yf.download(symbols, period=period, interval=interval, group_by="ticker", threads=True)
    return data

# -----------------------------
def train_ai(X, y):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

def load_ai():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

# -----------------------------
def analyze(symbol, daily_data, weekly_data, monthly_data, model=None):
    try:
        daily = daily_data[symbol]
        weekly = weekly_data[symbol]
        monthly = monthly_data[symbol]

        daily['RSI'] = ta.momentum.RSIIndicator(daily['Close']).rsi()
        daily['MACD'] = ta.trend.MACD(daily['Close']).macd()
        daily['VOL_AVG'] = daily['Volume'].rolling(20).mean()
        weekly['EMA20'] = ta.trend.EMAIndicator(weekly['Close'], 20).ema_indicator()
        weekly['EMA50'] = ta.trend.EMAIndicator(weekly['Close'], 50).ema_indicator()
        weekly['ADX'] = ta.trend.ADXIndicator(weekly['High'], weekly['Low'], weekly['Close']).adx()
        monthly['EMA200'] = ta.trend.EMAIndicator(monthly['Close'], 200).ema_indicator()

        score = 0
        # --------- Flex monthly check
        if monthly['Close'].iloc[-1] > monthly['EMA200'].iloc[-1]:
            score += 3
        else:
            score += 1  # Allow some flexibility

        # --------- Flex weekly
        if weekly['EMA20'].iloc[-1] > weekly['EMA50'].iloc[-1]:
            score += 2
        if weekly['ADX'].iloc[-1] > 15:  # Lower ADX threshold
            score += 1

        last = daily.iloc[-1]
        # --------- Flex daily
        if last['RSI'] < 55 and last['MACD'] > 0 and last['Volume'] > last['VOL_AVG']:
            score += 4

        if score < 4:  # Lower minimum score for flexible display
            return None

        price = round(last['Close'], 2)
        support = round(daily['Low'].rolling(20).min().iloc[-1], 2)
        resistance = round(daily['High'].rolling(20).max().iloc[-1], 2)
        last_month_high = monthly['High'].rolling(12).max().iloc[-1]
        last_month_low = monthly['Low'].rolling(12).min().iloc[-1]
        measured_move = last_month_high - last_month_low
        far_target = round(price + measured_move, 2)

        # AI features
        features = np.array([[
            last['RSI'], last['MACD'],
            weekly['EMA20'].iloc[-1] - weekly['EMA50'].iloc[-1],
            weekly['ADX'].iloc[-1],
            daily['Close'].pct_change(5).iloc[-1],
            daily['Close'].pct_change(20).iloc[-1],
            last['Volume']/last['VOL_AVG'],
            (price - support)/price
        ]])

        if model:
            prob_up = model.predict_proba(features)[0][1]
            prob_target = prob_up
        else:
            prob_up = 0.5
            prob_target = 0.5

        return {
            "symbol": symbol.replace(".CA",""),
            "price": price,
            "support": support,
            "near_target": resistance,
            "far_target": far_target,
            "score": score,
            "prob_up": round(prob_up*100,1),
            "prob_target": round(prob_target*100,1)
        }

    except:
        return None

# -----------------------------
def scan_market(force=False):
    daily_data = load_market_data(EGX100, "1d", "6mo")
    weekly_data = load_market_data(EGX100, "1wk", "1y")
    monthly_data = load_market_data(EGX100, "1mo", "5y")

    model = load_ai()
    results = []
    for s in EGX100:
        res = analyze(s, daily_data, weekly_data, monthly_data, model)
        if res:
            results.append(res)

    results = sorted(results, key=lambda x:x['score'], reverse=True)[:3]
    return results

# -----------------------------
def load_or_scan(force=False):
    if os.path.exists(CACHE_FILE) and not force:
        with open(CACHE_FILE,"r") as f:
            data = json.load(f)
        last_update = datetime.strptime(data["date"],"%Y-%m-%d")
        if datetime.now() - last_update < timedelta(days=30):
            return data["results"]
    results = scan_market(force)
    with open(CACHE_FILE,"w") as f:
        json.dump({"date":datetime.now().strftime("%Y-%m-%d"),"results":results},f)
    return results

# -----------------------------
if st.button("🔄 تحديث السوق الآن"):
    results = load_or_scan(force=True)
else:
    results = load_or_scan()

if results:
    df = pd.DataFrame(results)
    st.dataframe(df)
else:
    st.warning("لا توجد فرص حالياً، حاول التحديث لاحقاً")
