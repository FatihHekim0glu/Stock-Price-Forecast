"""Streamlit dashboard for the stock forecaster.

Importing this module is side-effect free. The dashboard, including the model
load and the data download, runs inside ``main`` and only when Streamlit
executes the script (or you call ``main`` yourself).
"""

from __future__ import annotations

import pandas as pd

from stockforecast.data import get_stock_data
from stockforecast.data_shaping import FEATURE_COLUMNS
from stockforecast.features import add_technical_indicators
from stockforecast.model import MODEL_PATH, load_model, make_future_predictions


def main() -> None:
    """Render the Streamlit forecasting dashboard."""
    import matplotlib.pyplot as plt
    import streamlit as st
    from sklearn.preprocessing import MinMaxScaler

    model = load_model(MODEL_PATH)

    st.title("Stock Market Forecasting Bot")
    st.sidebar.header("User Input Parameters")
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
    forecast_days = st.sidebar.slider("Days of forecast", 1, 30, 7)

    st.subheader(f"Displaying Data for {ticker}")
    load_state = st.text("Loading data...")
    data = get_stock_data(ticker)
    data = add_technical_indicators(data)
    load_state.text("Loading data... done!")
    st.write(data.tail())

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[FEATURE_COLUMNS])

    fig, ax = plt.subplots()
    ax.plot(data["Date"], data["Close"], label="Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.subheader(f"Forecasting {forecast_days} Days Ahead")
    predictions = make_future_predictions(model, data, scaler, FEATURE_COLUMNS, forecast_days)
    last_date = data["Date"].iloc[-1]
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

    fig2, ax2 = plt.subplots()
    ax2.plot(dates, predictions, marker="o", linestyle="-", label="Predicted Price")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    table = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Predicted Close": predictions})
    st.write(table)


if __name__ == "__main__":
    main()
