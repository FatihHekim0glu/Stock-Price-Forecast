"""Command-line entry point: train the forecaster and print a short forecast.

Importing this module does nothing. All work, including the data download,
TensorFlow load, the optional retraining scheduler and the run loop, happens
inside ``main`` behind the ``__main__`` guard.
"""

from __future__ import annotations

import pandas as pd

from stockforecast.data_shaping import create_sequences
from stockforecast.model import (
    DEFAULT_LOOK_BACK,
    MODEL_PATH,
    make_future_predictions,
    prepare_data,
    run_training,
    train_and_evaluate,
)


def forecast(ticker: str, forecast_days: int, look_back: int = DEFAULT_LOOK_BACK) -> None:
    """Train on ``ticker`` and print a forecast for the next few days."""
    data, scaled, scaler, features = prepare_data(ticker)
    x, y = create_sequences(scaled, look_back)
    model = train_and_evaluate(x, y, scaler)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    predictions = make_future_predictions(model, data, scaler, features, forecast_days, look_back)
    last_date = data["Date"].iloc[-1]
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

    print(f"Predicted prices for the next {forecast_days} days:")
    for date, price in zip(dates, predictions, strict=True):
        print(f"{date.strftime('%Y-%m-%d')}: {price:.2f} USD")


def start_scheduler(ticker: str) -> object:
    """Start a background job that retrains the model daily at midnight."""
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: run_training(ticker), "cron", hour=0)
    scheduler.start()
    return scheduler


def main() -> None:
    """Train, forecast, then keep a daily retraining scheduler alive."""
    ticker = "AAPL"
    forecast(ticker, forecast_days=7)

    scheduler = start_scheduler(ticker)
    try:
        import time

        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    main()
