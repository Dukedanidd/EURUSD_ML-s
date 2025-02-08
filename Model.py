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

# Predicciones
predictions = model.predict(X_test)

# Preparar datos para inverse transform
pred_scale = np.zeros((len(predictions), len(features)))
pred_scale[:, 0] = predictions.reshape(-1)

actual_scale = np.zeros((len(y_test), len(features)))
actual_scale[:, 0] = y_test

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
    
def predict_future(model, last_window, scaler, future_steps, features_count):
    """
    Realiza predicciones futuras utilizando el modelo LSTM
    """
    future_predictions = []
    current_batch = last_window.reshape(1, window_size, features_count)

    for _ in range(future_steps):
        # Hacer predicción
        pred = model.predict(current_batch, verbose=0)[0]

        # Preparar para inverse transform
        temp_scale = np.zeros((1, features_count))
        temp_scale[0, 0] = pred

        # Invertir escalado
        pred_unscaled = scaler.inverse_transform(temp_scale)[0, 0]
        future_predictions.append(pred_unscaled)

        # Actualizar el batch para la siguiente predicción
        current_batch = np.roll(current_batch, -1, axis=1)
        current_batch[0, -1] = temp_scale[0]

    return np.array(future_predictions)

# Obtener la última ventana de datos
last_window = X_test[-1:]

# Definir número de días para predecir
future_steps = 7

# Realizar predicciones futuras
future_predictions = predict_future(model, last_window, scaler, future_steps, len(features))

# Generar fechas futuras
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_steps + 1, freq='B')[1:]

# Mostrar predicciones
print("\nPredicciones para los próximos 7 días:")
for i, (value, date) in enumerate(zip(future_predictions, future_dates), 1):
    print(f"Día {i} ({date.date()}): {value:.4f} USD")

# Visualización
plt.figure(figsize=(15, 7))

# Plotear datos históricos (últimos 100 días)
plt.plot(df.index[-100:], df['Close'][-100:], label='Datos Históricos', color='blue')

# Plotear predicciones
plt.plot(future_dates, future_predictions, label='Predicciones Futuras',
         linestyle='--', color='red', linewidth=2)

plt.title('EUR/USD - Histórico y Predicciones Futuras', fontsize=14)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Precio (USD)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Rotar etiquetas del eje x para mejor legibilidad
plt.xticks(rotation=45)

# Ajustar layout
plt.tight_layout()

# Mostrar gráfico
plt.show()

# Calcular estadísticas de las predicciones
print("\nEstadísticas de las predicciones:")
print(f"Precio inicial: {future_predictions[0]:.4f}")
print(f"Precio final: {future_predictions[-1]:.4f}")
print(f"Cambio total: {((future_predictions[-1] - future_predictions[0]) / future_predictions[0] * 100):.2f}%")
print(f"Precio máximo: {np.max(future_predictions):.4f}")
print(f"Precio mínimo: {np.min(future_predictions):.4f}")