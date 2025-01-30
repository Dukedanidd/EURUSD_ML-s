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
