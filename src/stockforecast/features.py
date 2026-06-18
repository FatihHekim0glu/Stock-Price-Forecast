"""Technical-feature engineering for stock price data.

Every function here is pure: it takes a DataFrame with a ``Close`` column and
returns a new DataFrame with extra columns. Nothing downloads data, loads a
model, or schedules a job, so importing this module is free of side effects.

The indicator formulas (RSI, MACD, Bollinger Bands) follow the standard
definitions used by the ``ta`` library so the features match the original
pipeline. They are written in plain pandas so the maths is easy to test and
does not pull in a heavyweight dependency.
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index over ``window`` periods.

    Uses Wilder smoothing (an exponential moving average with
    ``alpha = 1 / window``), which matches the ``ta`` default.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # When average loss is zero the series is all gains: RSI saturates at 100.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, out.where(avg_loss == 0.0, 0.0))
    return out.rename("RSI")


def macd(
    close: pd.Series,
    window_fast: int = 12,
    window_slow: int = 26,
    window_sign: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram (difference).

    Returns a frame with columns ``MACD``, ``MACD_signal`` and ``MACD_diff``.
    """
    ema_fast = close.ewm(span=window_fast, adjust=False).mean()
    ema_slow = close.ewm(span=window_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=window_sign, adjust=False).mean()
    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_signal": signal,
            "MACD_diff": macd_line - signal,
        }
    )


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    window_dev: float = 2.0,
) -> pd.DataFrame:
    """Upper and lower Bollinger Bands.

    Returns a frame with columns ``BB_high`` and ``BB_low``. The rolling
    standard deviation uses the population estimator (``ddof=0``) to match
    the ``ta`` default.
    """
    rolling = close.rolling(window=window)
    mid = rolling.mean()
    std = rolling.std(ddof=0)
    return pd.DataFrame(
        {
            "BB_high": mid + window_dev * std,
            "BB_low": mid - window_dev * std,
        }
    )


def add_moving_averages(
    data: pd.DataFrame,
    windows: tuple[int, ...] = (10, 50, 200),
    column: str = "Close",
) -> pd.DataFrame:
    """Add simple moving averages ``MA{w}`` for each window in ``windows``."""
    out = data.copy()
    for window in windows:
        out[f"MA{window}"] = out[column].rolling(window=window).mean()
    return out


def add_lag_features(
    data: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2),
    column: str = "Close",
) -> pd.DataFrame:
    """Add lagged copies of ``column`` as ``{column}_lag{n}``."""
    out = data.copy()
    for lag in lags:
        out[f"{column}_lag{lag}"] = out[column].shift(lag)
    return out


def add_rolling_stats(
    data: pd.DataFrame,
    window: int = 5,
    column: str = "Close",
) -> pd.DataFrame:
    """Add rolling mean and standard deviation of ``column``."""
    out = data.copy()
    roll = out[column].rolling(window=window)
    out[f"{column}_roll_mean"] = roll.mean()
    out[f"{column}_roll_std"] = roll.std()
    return out


def add_technical_indicators(data: pd.DataFrame, *, dropna: bool = True) -> pd.DataFrame:
    """Add the full technical feature set used by the forecaster.

    Adds RSI, MACD (line, signal, histogram), Bollinger Bands, moving
    averages, lag features and rolling statistics. Returns a new DataFrame;
    the input is not modified. With ``dropna=True`` the warm-up rows that
    contain NaNs from the rolling windows are removed.
    """
    out = data.copy()
    close = out["Close"]

    out["RSI"] = rsi(close, window=14)

    macd_frame = macd(close)
    out["MACD"] = macd_frame["MACD"]
    out["MACD_diff"] = macd_frame["MACD_diff"]
    out["MACD_signal"] = macd_frame["MACD_signal"]

    bb = bollinger_bands(close, window=20)
    out["BB_high"] = bb["BB_high"]
    out["BB_low"] = bb["BB_low"]

    out = add_moving_averages(out)
    out = add_lag_features(out)
    out = add_rolling_stats(out)

    if dropna:
        out = out.dropna().reset_index(drop=True)
    return out
