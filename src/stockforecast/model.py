"""Model build, train, load and forecast.

TensorFlow is heavy and slow to import, so every function in this module
imports it lazily at call time. Importing this module by itself does not load
TensorFlow, download data, or train anything.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from stockforecast.data import get_stock_data
from stockforecast.data_shaping import (
    FEATURE_COLUMNS,
    create_sequences,
    scale_features,
)
from stockforecast.features import add_technical_indicators

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    import pandas as pd

DEFAULT_LOOK_BACK = 60
MODEL_PATH = "stock_model.h5"

# Quieten TensorFlow's C++ logging before it is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def build_model(input_shape: tuple[int, int, int]) -> Any:
    """Build the LSTM-with-attention model for sequences of ``input_shape``."""
    import tensorflow as tf
    from tensorflow.keras.layers import (  # type: ignore[import-untyped]
        LSTM,
        Attention,
        Dense,
        Dropout,
        Flatten,
        Input,
    )
    from tensorflow.keras.models import Model  # type: ignore[import-untyped]

    inputs = Input(shape=(input_shape[1], input_shape[2]))
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    lstm_out = LSTM(64, return_sequences=True)(lstm_out)

    attention = Attention()([lstm_out, lstm_out])
    attention = Flatten()(attention)

    dense = Dense(64, activation="relu")(attention)
    dense = Dropout(0.2)(dense)
    dense = Dense(32, activation="relu")(dense)
    dense = Dropout(0.2)(dense)
    outputs = Dense(1)(dense)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    del tf  # keep the lazy import local
    return model


def prepare_data(
    ticker: str,
) -> tuple[pd.DataFrame, np.ndarray, Any, list[str]]:
    """Download, engineer features for, and scale data for ``ticker``."""
    data = get_stock_data(ticker)
    data = add_technical_indicators(data)
    scaled, scaler = scale_features(data, FEATURE_COLUMNS)
    return data, scaled, scaler, FEATURE_COLUMNS


def load_model(path: str = MODEL_PATH) -> Any:
    """Load a saved Keras model, importing TensorFlow lazily."""
    import tensorflow as tf
    from tensorflow.keras.layers import Attention  # type: ignore[import-untyped]

    return tf.keras.models.load_model(path, custom_objects={"Attention": Attention})


def make_future_predictions(
    model: Any,
    data: pd.DataFrame,
    scaler: Any,
    features: list[str],
    forecast_days: int,
    look_back: int = DEFAULT_LOOK_BACK,
) -> np.ndarray:
    """Roll the model forward ``forecast_days`` steps from the last window."""
    data_scaled = scaler.transform(data[features])
    current = np.expand_dims(data_scaled[-look_back:], axis=0)

    preds: list[np.ndarray] = []
    for _ in range(forecast_days):
        next_pred = model.predict(current)
        preds.append(next_pred[0])
        next_scaled = np.concatenate((next_pred, current[:, -1, 1:]), axis=1)
        current = np.append(current[:, 1:, :], [next_scaled], axis=1)

    preds_arr = np.array(preds)
    padded = np.concatenate(
        (preds_arr, np.zeros((preds_arr.shape[0], len(features) - 1))),
        axis=1,
    )
    inverted: np.ndarray = np.asarray(scaler.inverse_transform(padded))
    return inverted[:, 0]


def train_and_evaluate(x: np.ndarray, y: np.ndarray, scaler: Any) -> Any:
    """Walk-forward train and evaluate the model, returning the last fit."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        r2_score,
    )
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5)
    mae_list, mape_list, r2_list = [], [], []
    model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(x), start=1):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model(x_train.shape)
        model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=32,
            validation_data=(x_test, y_test),
        )

        predictions = model.predict(x_test)
        pred_inv = scaler.inverse_transform(
            np.concatenate((predictions, x_test[:, -1, 1:]), axis=1)
        )[:, 0]
        y_inv = scaler.inverse_transform(
            np.concatenate((y_test.reshape(-1, 1), x_test[:, -1, 1:]), axis=1)
        )[:, 0]

        mae_list.append(mean_absolute_error(y_inv, pred_inv))
        mape_list.append(mean_absolute_percentage_error(y_inv, pred_inv))
        r2_list.append(r2_score(y_inv, pred_inv))
        print(
            f"Fold {fold}: MAE {mae_list[-1]:.2f}, MAPE {mape_list[-1]:.2f}, R2 {r2_list[-1]:.2f}"
        )

    print(f"Average MAE: {np.mean(mae_list):.2f}")
    print(f"Average MAPE: {np.mean(mape_list):.2f}")
    print(f"Average R2: {np.mean(r2_list):.2f}")
    return model


def run_training(
    ticker: str = "AAPL",
    look_back: int = DEFAULT_LOOK_BACK,
    model_path: str = MODEL_PATH,
) -> Any:
    """End-to-end: fetch data, build sequences, train, and save the model."""
    _data, scaled, scaler, _features = prepare_data(ticker)
    x, y = create_sequences(scaled, look_back)
    model = train_and_evaluate(x, y, scaler)
    if model is not None:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    return model
