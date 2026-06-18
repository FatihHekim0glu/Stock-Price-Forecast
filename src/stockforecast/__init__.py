"""Stock price forecasting toolkit.

The package is split so that importing any module has no side effects:
feature engineering and data shaping are pure functions over pandas
DataFrames and numpy arrays, and the heavy pieces (model build, training,
TensorFlow load) live behind functions that you call explicitly.
"""

from __future__ import annotations

from stockforecast.data_shaping import (
    FEATURE_COLUMNS,
    create_sequences,
    scale_features,
)
from stockforecast.features import (
    add_lag_features,
    add_moving_averages,
    add_rolling_stats,
    add_technical_indicators,
)

__all__ = [
    "FEATURE_COLUMNS",
    "add_lag_features",
    "add_moving_averages",
    "add_rolling_stats",
    "add_technical_indicators",
    "create_sequences",
    "scale_features",
]

__version__ = "0.1.0"
