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