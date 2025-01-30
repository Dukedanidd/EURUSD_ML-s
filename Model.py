# Import libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')
import yfinance as yf

# Get stock data
ticker = 'EURUSD=X'
df = yf.download(ticker, start='2016-01-01', end='2025-01-29', interval='1d')
df.head(21)

# Create graph
plt.figure(figsize=(16,8))
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price USD ($)', fontsize=16)
plt.title('Close Price History', fontsize=16)
plt.show()

# Create a new DataFrame with only the 'Close' column
data = df[['Close']]

# Convert the DataFrame to a numpy array
dataset = data.values

# Define the training data length (80% of the dataset)
training_data_len = math.ceil(len(dataset) * 0.8)

# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the training dataset
train_data = scaled_data[:training_data_len, :]

# Split data into x_train and y_train
x_train = []
y_train = []
window_size = 100
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])
    
# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape data for LSTM input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))