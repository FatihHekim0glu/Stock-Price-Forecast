import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Attention, Flatten
import ta  # Technical Analysis library

# Load the saved model
model = tf.keras.models.load_model('stock_model.h5', custom_objects={'Attention': Attention})

# Function to get and process data
def get_stock_data(ticker):
    data = yf.download(ticker, start='2010-01-01', end=datetime.date.today())
    data.reset_index(inplace=True)
    data = add_technical_indicators(data)
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

def make_future_predictions(data, forecast_days, scaler, features):
    # Prepare data
    data_scaled = scaler.transform(data[features])
    look_back = 60
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
        np.concatenate((future_predictions, np.zeros((future_predictions.shape[0], len(features) - 1))), axis=1)
    )[:, 0]
    
    return future_predictions_inverse

# Streamlit app
st.title('Stock Market Forecasting Bot')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    ticker_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
    forecast_days = st.sidebar.slider('Days of forecast', 1, 30, 7)
    return ticker_input.upper(), forecast_days

ticker_input, forecast_days = user_input_features()

# Display stock data
st.subheader(f'Displaying Data for {ticker_input}')
data_load_state = st.text('Loading data...')
data = get_stock_data(ticker_input)
data_load_state.text('Loading data... done!')

st.write(data.tail())

# Prepare features and scaler
features = ['Close', 'RSI', 'MACD', 'MACD_diff', 'MACD_signal', 'BB_high', 'BB_low',
            'MA10', 'MA50', 'MA200', 'Close_lag1', 'Close_lag2', 'Close_roll_mean', 'Close_roll_std']
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data[features])

# Plot closing price
def plot_close():
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], label='Closing Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

plot_close()

# Make predictions
st.subheader(f'Forecasting {forecast_days} Days Ahead')
predictions = make_future_predictions(data, forecast_days, scaler, features)

# Create a date range for future predictions
last_date = data['Date'].iloc[-1]
prediction_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

# Plot predictions
def plot_future(predictions, prediction_dates):
    fig, ax = plt.subplots()
    ax.plot(prediction_dates, predictions, marker='o', linestyle='-', label='Predicted Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

plot_future(predictions, prediction_dates)

# Display predictions in a table
prediction_df = pd.DataFrame({'Date': prediction_dates.strftime('%Y-%m-%d'), 'Predicted Close': predictions})
st.write(prediction_df)
