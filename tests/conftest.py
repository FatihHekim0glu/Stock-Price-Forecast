"""Shared fixtures for the pure-logic test suite.

Everything here is offline and headless: small synthetic price frames and
seeded random walks. No network, no TensorFlow, no plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def random_walk_close() -> pd.Series:
    """A 400-step seeded random-walk close-price series, strictly positive."""
    rng = np.random.default_rng(1)
    prices = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 400))
    return pd.Series(prices, name="Close")


@pytest.fixture
def price_frame(random_walk_close: pd.Series) -> pd.DataFrame:
    """A DataFrame with ``Date`` and ``Close`` columns for the random walk."""
    n = len(random_walk_close)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2015-01-01", periods=n, freq="D"),
            "Close": random_walk_close.to_numpy(),
        }
    )


@pytest.fixture
def monotonic_close() -> pd.Series:
    """A strictly increasing close series (RSI should saturate near 100)."""
    return pd.Series(np.arange(1.0, 61.0), name="Close")
