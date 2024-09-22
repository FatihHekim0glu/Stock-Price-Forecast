import numpy as np
import pandas as pd
import yfinance as yf
import ta  # Technical Analysis library
import matplotlib.pyplot as plt
import datetime


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Attention, Dropout, Flatten

# For automation
from apscheduler.schedulers.background import BackgroundScheduler

# Ignore TensorFlow warni9ngs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Data Collection and Preprocessing

def get_stock_data(ticker):
    # Download historical stock data
    data = yf.download(ticker, start='2010-01-01', end=datetime.date.today())
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    return data

def add_technical_indicators(data):
    # Add RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    
    # Add MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_diff'] = macd.macd_diff()
    data['MACD_signal'] = macd.macd_signal()
    
    # Add Bollinger Bands
    bb = ta.volatility.BollingerBands(data['Close'], window=20)
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()
    
    # Add Moving Averages
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Lag features
    data['Close_lag1'] = data['Close'].shift(1)
    data['Close_lag2'] = data['Close'].shift(2)
    
    # Rolling statistics
    data['Close_roll_mean'] = data['Close'].rolling(window=5).mean()
    data['Close_roll_std'] = data['Close'].rolling(window=5).std()
    
    data.dropna(inplace=True)
    return data

def prepare_data(ticker):
    data = get_stock_data(ticker)
    data = add_technical_indicators(data)
    
    features = ['Close', 'RSI', 'MACD', 'MACD_diff', 'MACD_signal', 'BB_high', 'BB_low',
                'MA10', 'MA50', 'MA200', 'Close_lag1', 'Close_lag2', 'Close_roll_mean', 'Close_roll_std']
    target = 'Close'
    
    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[features])
    return data, data_scaled, scaler, features

def create_sequences(dataset, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i])
        Y.append(dataset[i, 0])  # Assuming 'Close' is the first column
    return np.array(X), np.array(Y)

# 2. Model Building

def build_model(input_shape):
    # Input layer
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    
    # LSTM layers
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    lstm_out = LSTM(64, return_sequences=True)(lstm_out)
    
    # Attention layer
    attention_data = Attention()([lstm_out, lstm_out])
    attention_data = Flatten()(attention_data)
    
    # Fully connected layers
    x = Dense(64, activation='relu')(attention_data)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(1)(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 3. Model Training and Evaluation

def train_and_evaluate_model(X, Y):
    tscv = TimeSeriesSplit(n_splits=5)
    
    mae_list = []
    mape_list = []
    r2_list = []
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"\nFold {fold}")
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        Y_train_cv, Y_test_cv = Y[train_index], Y[test_index]
        
        model = build_model(X_train_cv.shape)
        history = model.fit(X_train_cv, Y_train_cv, epochs=10, batch_size=32, validation_data=(X_test_cv, Y_test_cv))
        
        # Predictions
        predictions = model.predict(X_test_cv)
        
        # Inverse transform predictions and actual values
        predictions_inverse = scaler.inverse_transform(
            np.concatenate((predictions, X_test_cv[:, -1, 1:]), axis=1)
        )[:, 0]
        Y_test_cv_inverse = scaler.inverse_transform(
            np.concatenate((Y_test_cv.reshape(-1, 1), X_test_cv[:, -1, 1:]), axis=1)
        )[:, 0]
        
        # Evaluation metrics
        mae = mean_absolute_error(Y_test_cv_inverse, predictions_inverse)
        mape = mean_absolute_percentage_error(Y_test_cv_inverse, predictions_inverse)
        r2 = r2_score(Y_test_cv_inverse, predictions_inverse)
        
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2)
        
        print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}, R2: {r2:.2f}")
        
        fold += 1
    
    print(f"\nAverage MAE: {np.mean(mae_list):.2f}")
    print(f"Average MAPE: {np.mean(mape_list):.2f}")
    print(f"Average R2: {np.mean(r2_list):.2f}")
    
    return model

# 4. Automation Enhancements

def retrain_model():
    print("Retraining model...")
    global data, data_scaled, scaler, features, X, Y, model
    data, data_scaled, scaler, features = prepare_data(ticker)
    X, Y = create_sequences(data_scaled, look_back)
    model = train_and_evaluate_model(X, Y)
    model.save('stock_model.h5')
    print("Model retrained and saved.")

# Schedule retraining every day at midnight
scheduler = BackgroundScheduler()
scheduler.add_job(retrain_model, 'cron', hour=0)
scheduler.start()

# 5. Main Execution

if __name__ == "__main__":
    ticker = 'AAPL'  # You can change this to any stock ticker
    
    # Prepare data
    data, data_scaled, scaler, features = prepare_data(ticker)
    look_back = 60
    X, Y = create_sequences(data_scaled, look_back)
    
    # Train and evaluate model
    model = train_and_evaluate_model(X, Y)
    
    # Save the model
    model.save('stock_model.h5')
    print("Model saved successfully.")
    
    # Make future predictions
    def make_future_predictions(data, forecast_days):
        # Prepare data
        data_scaled = scaler.transform(data[features])
        last_sequence = data_scaled[-look_back:]
        last_sequence = np.expand_dims(last_sequence, axis=0)
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_days):
            next_pred = model.predict(current_sequence)
            future_predictions.append(next_pred[0])
            next_pred_scaled = np.concatenate((next_pred, current_sequence[:, -1, 1:]), axis=1)
            current_sequence = np.append(current_sequence[:,1:,:], [next_pred_scaled], axis=1)
        
        future_predictions = np.array(future_predictions)
        future_predictions_inverse = scaler.inverse_transform(
            np.concatenate((future_predictions, np.zeros((future_predictions.shape[0], data_scaled.shape[1] - 1))), axis=1)
        )[:, 0]
        
        return future_predictions_inverse
    
    # Predict next week's prices
    forecast_days = 7  # You can change this to forecast more days
    predictions = make_future_predictions(data, forecast_days)
    last_date = data['Date'].iloc[-1]
    prediction_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    # Display predictions
    print(f"\nPredicted prices for the next {forecast_days} days:")
    for date, price in zip(prediction_dates, predictions):
        print(f"{date.strftime('%Y-%m-%d')}: {price:.2f} USD")
    
    # Plot predictions
    plt.figure(figsize=(14, 5))
    plt.plot(prediction_dates, predictions, marker='o', linestyle='-', label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction for Next {forecast_days} Days')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Keep the script running to allow the scheduler to work
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()