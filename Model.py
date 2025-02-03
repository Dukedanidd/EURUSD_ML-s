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

# Descargar datos
symbol = "EURUSD=X"
df = yf.download(symbol, start="2010-01-01", end="2024-01-01", interval="1d")

# Preparar datos
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
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

# Eliminar filas con NaN después de crear indicadores
df.dropna(inplace=True)

# Preparar datos para el escalado
features = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_signal', 'Signal']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Crear secuencias para LSTM
window_size = 60
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
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Predicciones
predictions = model.predict(X_test)

# Preparar datos para inverse transform
pred_scale = np.zeros((len(predictions), len(features)))
pred_scale[:, 0] = predictions.reshape(-1)  # Colocar predicciones en la primera columna

actual_scale = np.zeros((len(y_test), len(features)))
actual_scale[:, 0] = y_test  # Colocar valores reales en la primera columna

# Convertir predicciones a la escala original
predictions = scaler.inverse_transform(pred_scale)[:, 0]
actual_values = scaler.inverse_transform(actual_scale)[:, 0]

# Filtrar valores no válidos
mask = ~np.isnan(predictions) & ~np.isnan(actual_values) & ~np.isinf(predictions) & ~np.isinf(actual_values)
predictions = predictions[mask]
actual_values = actual_values[mask]

print(f'Forma de predictions antes de filtrar: {predictions.shape}')
print(f'Forma de actual_values antes de filtrar: {actual_values.shape}')

# Calcular métricas si hay datos válidos
if len(predictions) > 0 and len(actual_values) > 0:
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    mae = mean_absolute_error(actual_values, predictions)
    mape = np.mean(np.abs((actual_values - predictions) / np.where(actual_values == 0, 1, actual_values))) * 100
    effectiveness = 100 - mape

    print('\nResultados:')
    print(f'Número de predicciones válidas: {len(predictions)}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'Efectividad: {effectiveness:.2f}%')

    # Visualización
    plt.figure(figsize=(15,7))
    plt.plot(actual_values, label='Valores Reales', color='blue')
    plt.plot(predictions, label='Predicciones', color='red')
    plt.title('Predicciones vs Valores Reales - EUR/USD')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No hay suficientes datos válidos para calcular las métricas.")
    print("Verifique los datos de entrada y el proceso de escalado.")