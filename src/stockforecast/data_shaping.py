"""Data shaping for the LSTM forecaster.

Pure helpers that turn a feature DataFrame into the scaled, windowed arrays
the model expects. No network, no model, no global state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# The feature columns the model is trained on, in order. ``Close`` is first
# because the sequence target and the inverse-scaling step both assume the
# close price occupies column zero.
FEATURE_COLUMNS: list[str] = [
    "Close",
    "RSI",
    "MACD",
    "MACD_diff",
    "MACD_signal",
    "BB_high",
    "BB_low",
    "MA10",
    "MA50",
    "MA200",
    "Close_lag1",
    "Close_lag2",
    "Close_roll_mean",
    "Close_roll_std",
]


def scale_features(
    data: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Scale the selected feature columns to ``[0, 1]``.

    Returns the scaled array and the fitted scaler so callers can invert the
    transform later. Raises ``KeyError`` if a requested column is missing.
    """
    cols = FEATURE_COLUMNS if features is None else features
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data[cols])
    return scaled, scaler


def create_sequences(
    dataset: np.ndarray,
    look_back: int = 60,
    target_col: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences for a recurrent model.

    For each index ``i >= look_back`` the input is the ``look_back`` rows that
    precede ``i`` and the target is ``dataset[i, target_col]``. Returns
    ``(X, y)`` with shapes ``(n, look_back, n_features)`` and ``(n,)``.

    Raises ``ValueError`` if ``look_back`` is not positive or the dataset is
    not 2-D.
    """
    if look_back < 1:
        raise ValueError("look_back must be a positive integer")
    array = np.asarray(dataset)
    if array.ndim != 2:
        raise ValueError("dataset must be 2-D (rows x features)")

    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    for i in range(look_back, len(array)):
        x_list.append(array[i - look_back : i])
        y_list.append(array[i, target_col])

    if not x_list:
        n_features = array.shape[1]
        return (
            np.empty((0, look_back, n_features), dtype=array.dtype),
            np.empty((0,), dtype=array.dtype),
        )
    return np.asarray(x_list), np.asarray(y_list)
