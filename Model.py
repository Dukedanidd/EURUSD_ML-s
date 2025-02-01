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