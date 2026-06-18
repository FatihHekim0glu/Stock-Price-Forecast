"""Tests for the pure data-shaping helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockforecast.data_shaping import (
    FEATURE_COLUMNS,
    create_sequences,
    scale_features,
)
from stockforecast.features import add_technical_indicators


class TestCreateSequences:
    def test_shapes(self) -> None:
        dataset = np.arange(100, dtype=float).reshape(50, 2)
        x, y = create_sequences(dataset, look_back=10)
        assert x.shape == (40, 10, 2)
        assert y.shape == (40,)

    def test_target_is_close_column(self) -> None:
        dataset = np.arange(60, dtype=float).reshape(30, 2)
        _x, y = create_sequences(dataset, look_back=5, target_col=0)
        # Target for window ending at row i is dataset[i, 0].
        assert np.allclose(y, dataset[5:, 0])

    def test_window_contents(self) -> None:
        dataset = np.arange(40, dtype=float).reshape(20, 2)
        x, _ = create_sequences(dataset, look_back=3)
        # First window is rows 0,1,2; first target row is index 3.
        assert np.allclose(x[0], dataset[0:3])
        assert np.allclose(x[-1], dataset[-4:-1])

    def test_empty_when_look_back_exceeds_length(self) -> None:
        dataset = np.arange(20, dtype=float).reshape(10, 2)
        x, y = create_sequences(dataset, look_back=10)
        assert x.shape == (0, 10, 2)
        assert y.shape == (0,)

    def test_rejects_non_positive_look_back(self) -> None:
        dataset = np.arange(20, dtype=float).reshape(10, 2)
        with pytest.raises(ValueError, match="positive"):
            create_sequences(dataset, look_back=0)

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            create_sequences(np.arange(10), look_back=3)


class TestScaleFeatures:
    def test_output_in_unit_range(self, price_frame: pd.DataFrame) -> None:
        feats = add_technical_indicators(price_frame)
        scaled, _scaler = scale_features(feats)
        assert scaled.min() >= 0.0 - 1e-9
        assert scaled.max() <= 1.0 + 1e-9

    def test_inverse_round_trip(self, price_frame: pd.DataFrame) -> None:
        feats = add_technical_indicators(price_frame)
        scaled, scaler = scale_features(feats)
        restored = scaler.inverse_transform(scaled)
        assert np.allclose(restored, feats[FEATURE_COLUMNS].to_numpy())

    def test_column_order_preserved(self, price_frame: pd.DataFrame) -> None:
        feats = add_technical_indicators(price_frame)
        scaled, _ = scale_features(feats)
        assert scaled.shape[1] == len(FEATURE_COLUMNS)

    def test_missing_column_raises(self) -> None:
        frame = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        with pytest.raises(KeyError, match="missing feature columns"):
            scale_features(frame)

    def test_custom_feature_subset(self, price_frame: pd.DataFrame) -> None:
        feats = add_technical_indicators(price_frame)
        scaled, _ = scale_features(feats, features=["Close", "RSI"])
        assert scaled.shape[1] == 2


def test_features_feed_sequences_end_to_end(price_frame: pd.DataFrame) -> None:
    """The full pure pipeline produces well-formed model inputs."""
    feats = add_technical_indicators(price_frame)
    scaled, _ = scale_features(feats)
    x, y = create_sequences(scaled, look_back=10)
    assert x.ndim == 3
    assert x.shape[2] == len(FEATURE_COLUMNS)
    assert len(x) == len(y)
    assert not np.isnan(x).any()
    assert not np.isnan(y).any()
