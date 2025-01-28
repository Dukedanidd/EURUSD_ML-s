# Import libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import yfinance as yf
import ta
import optuna
plt.style.use('fivethirtyeight')

# Get the stock quote
df = yf.download('EURUSD=X', start='2012-1-1', end='2025-1-1')

# Add technical indicators
def add_technical_features(df):
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Width'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / bollinger.bollinger_mavg()
    
    # Previous day features
    df['PreviousClose'] = df['Close'].shift(1)
    df['DailyRange'] = df['High'] - df['Low']
    df['PriceChange'] = df['Close'] - df['Open']
    
    # Multiple timeframe moving averages
    for window in [5, 20, 50]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    return df

df = add_technical_features(df)
df = df.dropna()

# Separate price-based and indicator features for different scaling
price_features = ['Close', 'Open', 'High', 'Low', 'PreviousClose']
indicator_features = [col for col in df.columns if col not in price_features and col not in ['Volume', 'Adj Close']]

# Create scalers
price_scaler = MinMaxScaler(feature_range=(0, 1))
indicator_scaler = RobustScaler()

# Scale price features and indicators separately
scaled_prices = price_scaler.fit_transform(df[price_features])
scaled_indicators = indicator_scaler.fit_transform(df[indicator_features])

# Combine scaled data
scaled_data = np.hstack((scaled_prices, scaled_indicators))

# Optimize hyperparameters using Optuna
def create_model(trial):
    model = Sequential([
        LSTM(trial.suggest_int('lstm1_units', 64, 256),
             return_sequences=True,
             input_shape=(window_size, scaled_data.shape[1])),
        Dropout(trial.suggest_float('dropout1', 0.1, 0.5)),
        LSTM(trial.suggest_int('lstm2_units', 32, 128),
             return_sequences=True),
        Dropout(trial.suggest_float('dropout2', 0.1, 0.5)),
        LSTM(trial.suggest_int('lstm3_units', 16, 64),
             return_sequences=False),
        Dropout(trial.suggest_float('dropout3', 0.1, 0.5)),
        Dense(trial.suggest_int('dense_units', 16, 64), activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)),
        loss='huber'
    )
    return model

# TimeSeriesSplit for temporal validation
tscv = TimeSeriesSplit(n_splits=5)

# Find optimal window size using cross-validation
window_sizes = [30, 60, 90, 120]
window_scores = []

for window_size in window_sizes:
    scores = []
    for train_idx, val_idx in tscv.split(scaled_data):
        X_train, y_train = [], []
        for i in range(window_size, len(train_idx)):
            X_train.append(scaled_data[train_idx[i-window_size:i]])
            y_train.append(scaled_data[train_idx[i], 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        # Simple model for window size selection
        temp_model = Sequential([
            LSTM(64, input_shape=(window_size, scaled_data.shape[1])),
            Dense(1)
        ])
        temp_model.compile(optimizer='adam', loss='mse')
        temp_model.fit(X_train, y_train, epochs=5, verbose=0)
        scores.append(temp_model.evaluate(X_train, y_train, verbose=0))
    
    window_scores.append(np.mean(scores))

window_size = window_sizes[np.argmin(window_scores)]
print(f"Selected optimal window size: {window_size}")

# Prepare training data with optimal window size
X, y = [], []
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Optimize model hyperparameters
def objective(trial):
    model = create_model(trial)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    val_scores = []
    for train_idx, val_idx in tscv.split(X):
        history = model.fit(
            X[train_idx], y[train_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=50,
            batch_size=trial.suggest_int('batch_size', 16, 64),
            callbacks=[early_stop],
            verbose=0
        )
        val_scores.append(min(history.history['val_loss']))
    
    return np.mean(val_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Train final model with best hyperparameters
best_params = study.best_params
final_model = create_model(study.best_trial)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Use the last fold for final evaluation
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

history = final_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=best_params['batch_size'],
    callbacks=[early_stop]
)

# Make predictions
predictions = final_model.predict(X_test)

# Inverse transform predictions for price only
pred_prices = np.zeros((len(predictions), len(price_features)))
pred_prices[:, 0] = predictions.flatten()
actual_prices = np.zeros((len(y_test), len(price_features)))
actual_prices[:, 0] = y_test

predictions = price_scaler.inverse_transform(pred_prices)[:, 0]
y_test_orig = price_scaler.inverse_transform(actual_prices)[:, 0]

# Calculate comprehensive metrics
def calculate_metrics(y_true, y_pred):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'R²': r2_score(y_true, y_pred)
    }
    
    # Add residual analysis
    residuals = y_true - y_pred
    metrics['Residual_Mean'] = np.mean(residuals)
    metrics['Residual_Std'] = np.std(residuals)
    _, metrics['Residual_Normality'] = stats.normaltest(residuals)
    
    return metrics

metrics = calculate_metrics(y_test_orig, predictions)
print("\nMétricas de Evaluación:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predictions vs actual
plt.figure(figsize=(15, 6))
plt.plot(y_test_orig, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('EURUSD Predictions vs Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot residuals
plt.figure(figsize=(12, 6))
plt.scatter(predictions, y_test_orig - predictions)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Function for future predictions
def make_future_predictions(model, last_sequence, n_steps=20):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_steps):
        next_pred = model.predict(current_sequence)
        future_predictions.append(next_pred[0, 0])
        
        # Update sequence
        new_sequence = np.zeros_like(current_sequence)
        new_sequence[0, :-1, :] = current_sequence[0, 1:, :]
        new_sequence[0, -1, 0] = next_pred[0, 0]
        current_sequence = new_sequence
    
    return np.array(future_predictions)

# Make future predictions
last_sequence = X_test[-1:].copy()
future_preds = make_future_predictions(final_model, last_sequence)

# Convert to original scale
future_prices = np.zeros((len(future_preds), len(price_features)))
future_prices[:, 0] = future_preds
future_predictions = price_scaler.inverse_transform(future_prices)[:, 0]

# Generate future dates and print predictions
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=len(future_predictions)+1, freq='B')[1:]

print("\nPredicciones para los próximos días:")
for date, pred in zip(future_dates, future_predictions):
    print(f"{date.date()}: {pred:.6f} USD")

# Plot future predictions
plt.figure(figsize=(12, 6))
plt.plot(df.index[-100:], df['Close'].tail(100), label='Historical Data', color='blue')
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--', color='red')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($USD)', fontsize=12)
plt.title('Future Price Predictions', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()