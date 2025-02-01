# Importar librerías
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
# Estilo de gráficos
plt.style.use('fivethirtyeight')

# Descargar datos de EUR/USD
ticker = 'EURUSD=X'
df = yf.download(ticker, start='2012-01-01', end='2025-01-31', interval='1d')

# Seleccionar columnas relevantes
data = df[['Close', 'Open', 'High', 'Low']]

# Convertir a numpy array
dataset = data.values

# Definir tamaño de ventana
window_size = 180

# Dividir en conjuntos de entrenamiento y prueba
training_data_len = math.ceil(len(dataset) * 0.8)

# Verificar que training_data_len sea menor que el tamaño total de los datos
if training_data_len >= len(dataset):
    raise ValueError("El tamaño de los datos de entrenamiento es mayor o igual al tamaño total de los datos. Ajusta el porcentaje de división.")

train_data = dataset[:training_data_len]
test_data = dataset[training_data_len:]

# Verificar que test_data no esté vacío
if len(test_data) == 0:
    raise ValueError("El conjunto de prueba está vacío. Revisa la división de los datos.")

# Escalar datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)


# Crear ventanas de datos
def create_dataset(data, window_size):
    x, y = [], []
    if len(data) > window_size:
        for i in range(window_size, len(data)):
            x.append(data[i-window_size:i, :])  # Usar todas las características
            y.append(data[i, 0])  # Predecir 'Close'
    else:
        raise ValueError(f"El tamaño de los datos ({len(data)}) es menor que el window_size ({window_size}). Ajusta el window_size o usa más datos.")
    return np.array(x), np.array(y)

x_train, y_train = create_dataset(scaled_train_data, window_size)
x_test, y_test = create_dataset(scaled_test_data, window_size)

# Verificar la forma de x_test y y_test
print("Forma de x_test:", x_test.shape)
print("Forma de y_test:", y_test.shape)

# Verificar si x_test tiene datos
if len(x_test) == 0:
    raise ValueError("x_test está vacío. Revisa la creación del conjunto de prueba.")

# Construir el modelo CNN + LSTM
model = Sequential([
    # Capa convolucional
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Capa LSTM
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),

    # Otra capa LSTM
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    BatchNormalization(),

    # Capas densas
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

# Resumen del modelo
model.summary()

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=100,
    callbacks=[early_stop]
)

# Graficar la pérdida durante el entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Verificar si x_test tiene datos
if len(x_test) == 0:
    raise ValueError("x_test está vacío. Revisa la creación del conjunto de prueba.")

# Hacer predicciones
predictions = model.predict(x_test)

# Invertir la escala de las predicciones
predictions = scaler.inverse_transform(np.concatenate((x_test[:, -1, :3], predictions), axis=1))[:, 3]

# Verificar la forma de y_test y predictions
print("Forma de y_test:", y_test.shape)
print("Forma de predictions:", predictions.shape)

# Asegurarse de que y_test y predictions sean 1D
if len(y_test.shape) > 1:
    y_test = y_test[:, 0]  # Seleccionar solo la primera columna si es 2D

if len(predictions.shape) > 1:
    predictions = predictions[:, 0]  # Seleccionar solo la primera columna si es 2D

# Calcular métricas
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
effectiveness = 100 - mape

print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')
print(f'Efectividad: {effectiveness:.2f}%')


# Crear un DataFrame para las predicciones
predictions_df = pd.DataFrame(predictions, index=data.index[training_data_len + window_size:], columns=['Predictions'])

# Combinar con los datos originales
valid = data[training_data_len + window_size:].copy()
valid['Predictions'] = predictions_df['Predictions']

# Graficar resultados
plt.figure(figsize=(15, 6))
plt.title('Modelo CNN + LSTM con múltiples características')
plt.xlabel('Fecha', fontsize=10)
plt.ylabel('Precio de Cierre (USD)', fontsize=10)
plt.plot(data['Close'][:training_data_len], label='Train')
plt.plot(valid['Close'], label='Validación')
plt.plot(valid['Predictions'], label='Predicciones')
plt.legend(loc='lower right')
plt.show()

# Predicciones futuras
def predict_future(model, last_window, scaler, future_steps):
    future_predictions = []
    current_input = last_window.reshape(1, window_size, x_test.shape[2])

    for _ in range(future_steps):
        # Predecir el siguiente valor
        prediction = model.predict(current_input)[0, 0]
        future_predictions.append(prediction)

        # Actualizar la ventana de entrada
        current_input = np.roll(current_input, shift=-1, axis=1)
        current_input[0, -1, -1] = prediction  # Agregar la nueva predicción en la última posición

    # Invertir la escala de las predicciones
    future_predictions_scaled = np.concatenate(
        (np.repeat(current_input[:, -1, :3], len(future_predictions), axis=0),  # Mantener otras características
         np.array(future_predictions).reshape(-1, 1)),  # Agregar las predicciones
        axis=1
    )
    future_predictions = scaler.inverse_transform(future_predictions_scaled)[:, -1]

    return future_predictions

# Obtener la última ventana de datos
last_window = scaled_test_data[-window_size:]

# Predecir los próximos 30 días
future_steps = 30
future_predictions = predict_future(model, last_window, scaler, future_steps)

# Generar fechas futuras
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_steps + 1, freq='B')[1:]

# Mostrar predicciones
print("Predicciones para los próximos 30 días:")
for i, (value, date) in enumerate(zip(future_predictions, future_dates), 1):
    print(f"Día {i} ({date.date()}): {value:.6f} USD")

# Graficar predicciones futuras
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Datos Reales', color='blue')
plt.plot(future_dates, future_predictions, label='Predicciones Futuras', linestyle='--', color='red')
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Precio de Cierre (USD)', fontsize=12)
plt.title('Predicciones para los Próximos 30 Días', fontsize=14)
plt.legend(loc='best')
plt.grid(True)
plt.show()