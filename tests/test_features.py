"""Tests for the pure technical-feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockforecast.features import (
    add_lag_features,
    add_moving_averages,
    add_rolling_stats,
    add_technical_indicators,
    bollinger_bands,
    macd,
    rsi,
)


class TestRSI:
    def test_bounded_between_0_and_100(self, random_walk_close: pd.Series) -> None:
        values = rsi(random_walk_close).dropna()
        assert (values >= 0.0).all()
        assert (values <= 100.0).all()

    def test_saturates_high_when_only_gains(self, monotonic_close: pd.Series) -> None:
        # A strictly rising series has no losses, so RSI should sit at 100.
        values = rsi(monotonic_close).dropna()
        assert np.allclose(values, 100.0)

    def test_saturates_low_when_only_losses(self) -> None:
        falling = pd.Series(np.arange(60.0, 0.0, -1.0), name="Close")
        values = rsi(falling).dropna()
        assert np.allclose(values, 0.0)

    def test_warmup_is_nan(self, random_walk_close: pd.Series) -> None:
        values = rsi(random_walk_close, window=14)
        # The first ``window`` rows cannot be computed.
        assert values.iloc[:14].isna().all()
        assert values.iloc[14:].notna().any()


class TestMACD:
    def test_columns_present(self, random_walk_close: pd.Series) -> None:
        frame = macd(random_walk_close)
        assert list(frame.columns) == ["MACD", "MACD_signal", "MACD_diff"]

    def test_histogram_is_line_minus_signal(self, random_walk_close: pd.Series) -> None:
        frame = macd(random_walk_close)
        expected = frame["MACD"] - frame["MACD_signal"]
        assert np.allclose(frame["MACD_diff"], expected)

    def test_line_is_fast_minus_slow_ema(self, random_walk_close: pd.Series) -> None:
        frame = macd(random_walk_close, window_fast=12, window_slow=26)
        fast = random_walk_close.ewm(span=12, adjust=False).mean()
        slow = random_walk_close.ewm(span=26, adjust=False).mean()
        assert np.allclose(frame["MACD"], fast - slow)


class TestBollinger:
    def test_high_above_low(self, random_walk_close: pd.Series) -> None:
        frame = bollinger_bands(random_walk_close).dropna()
        assert (frame["BB_high"] >= frame["BB_low"]).all()

    def test_bands_straddle_moving_average(self, random_walk_close: pd.Series) -> None:
        frame = bollinger_bands(random_walk_close, window=20)
        mid = random_walk_close.rolling(20).mean()
        valid = frame.dropna().index
        assert (frame.loc[valid, "BB_high"] >= mid.loc[valid]).all()
        assert (frame.loc[valid, "BB_low"] <= mid.loc[valid]).all()

    def test_band_width_scales_with_window_dev(self, random_walk_close: pd.Series) -> None:
        narrow = bollinger_bands(random_walk_close, window_dev=1.0).dropna()
        wide = bollinger_bands(random_walk_close, window_dev=2.0).dropna()
        narrow_width = (narrow["BB_high"] - narrow["BB_low"]).to_numpy()
        wide_width = (wide["BB_high"] - wide["BB_low"]).to_numpy()
        assert np.allclose(wide_width, 2.0 * narrow_width)


class TestMovingAverages:
    def test_adds_expected_columns(self, price_frame: pd.DataFrame) -> None:
        out = add_moving_averages(price_frame, windows=(10, 50, 200))
        assert {"MA10", "MA50", "MA200"} <= set(out.columns)

    def test_matches_rolling_mean(self, price_frame: pd.DataFrame) -> None:
        out = add_moving_averages(price_frame, windows=(10,))
        expected = price_frame["Close"].rolling(10).mean()
        assert np.allclose(out["MA10"].dropna(), expected.dropna())

    def test_does_not_mutate_input(self, price_frame: pd.DataFrame) -> None:
        before = set(price_frame.columns)
        add_moving_averages(price_frame)
        assert set(price_frame.columns) == before


class TestLagFeatures:
    def test_lag_values_are_shifted(self, price_frame: pd.DataFrame) -> None:
        out = add_lag_features(price_frame, lags=(1, 2))
        assert np.allclose(
            out["Close_lag1"].iloc[1:], price_frame["Close"].iloc[:-1], equal_nan=False
        )
        assert pd.isna(out["Close_lag1"].iloc[0])
        assert pd.isna(out["Close_lag2"].iloc[1])


class TestRollingStats:
    def test_mean_and_std_columns(self, price_frame: pd.DataFrame) -> None:
        out = add_rolling_stats(price_frame, window=5)
        assert "Close_roll_mean" in out.columns
        assert "Close_roll_std" in out.columns

    def test_std_non_negative(self, price_frame: pd.DataFrame) -> None:
        out = add_rolling_stats(price_frame, window=5)
        assert (out["Close_roll_std"].dropna() >= 0.0).all()


class TestAddTechnicalIndicators:
    def test_returns_all_feature_columns(self, price_frame: pd.DataFrame) -> None:
        out = add_technical_indicators(price_frame)
        expected = {
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
        }
        assert expected <= set(out.columns)

    def test_no_nans_after_dropna(self, price_frame: pd.DataFrame) -> None:
        out = add_technical_indicators(price_frame)
        assert not out.isna().any().any()

    def test_index_is_reset(self, price_frame: pd.DataFrame) -> None:
        out = add_technical_indicators(price_frame)
        assert list(out.index) == list(range(len(out)))

    def test_does_not_mutate_input(self, price_frame: pd.DataFrame) -> None:
        before = price_frame.copy()
        add_technical_indicators(price_frame)
        pd.testing.assert_frame_equal(price_frame, before)

    def test_keep_warmup_when_dropna_false(self, price_frame: pd.DataFrame) -> None:
        out = add_technical_indicators(price_frame, dropna=False)
        assert len(out) == len(price_frame)
        # The 200-window moving average leaves NaNs in the early rows.
        assert out["MA200"].isna().any()
