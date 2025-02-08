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

# Preparar datos para el escalado
features = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_signal', 'Signal']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Crear secuencias para LSTM
window_size = 50
X, y = [], []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_data[i, 0])  # Índice 0 corresponde al precio de cierre

X, y = np.array(X), np.array(y)

# División de datos
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))