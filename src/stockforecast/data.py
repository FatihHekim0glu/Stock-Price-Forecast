"""Live data access.

This module touches the network, so it is kept apart from the pure feature
and shaping code. Importing it does nothing; you must call ``get_stock_data``.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd


def get_stock_data(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Download historical daily data for ``ticker`` from Yahoo Finance.

    The import of ``yfinance`` is deferred to call time so that importing this
    module never reaches out to the network.
    """
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=dt.date.today())
    data: pd.DataFrame = pd.DataFrame(raw).dropna().reset_index()
    return data
