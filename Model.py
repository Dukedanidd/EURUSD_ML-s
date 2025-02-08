import pandas as pd
import numpy as np
import yfinance as yf
import ta as ta
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Estilo de gráficos
plt.style.use('fivethirtyeight')

symbol = "EURUSD=X"
df = yf.download(symbol, start="2010-01-01", end="2025-02-07", interval="1d")

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)
df.dropna(inplace=True)


# Indicadores técnicos
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['RSI'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
macd = ta.trend.MACD(df['Close'].squeeze())
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()

# Señales de trading
df['Signal'] = 0
df.loc[(df['SMA_50'] > df['SMA_200']) & (df['RSI'] < 30), 'Signal'] = 1
df.loc[(df['SMA_50'] < df['SMA_200']) & (df['RSI'] > 70), 'Signal'] = -1

df.dropna(inplace=True)