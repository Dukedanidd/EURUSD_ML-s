# Import libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')
import yfinance as yf

# Get the stock quote
df = yf.download('EURUSD = X', start='2012-1-1', end='2025-1-1')
df.head(10)

# Create graph
plt.figure(figsize=(16,8))
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price USD ($)', fontsize=16)
plt.title('Close Price History', fontsize=16)
plt.show()

# Create a new df with only the 'Close' column, ensuring the column exists
data = df[['Close']]

# Convert the df to a numpy array
dataset = data.values

# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

training_data_len

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

# Create the training dataset
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(500, len(train_data)):
  x_train.append(train_data[i-500:i, 0])
  y_train.append(train_data[i, 0])
  if i <= 500:
    print(x_train)
    print(y_train)
    print()
