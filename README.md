# Stock Market Forecasting Bot

> Now live as an interactive web tool at https://fatihhekimoglu-platform.vercel.app/tools/stock-price-forecast, part of the fatihhekimoglu.com quantitative-tools platform. The Streamlit app here remains usable for local development. The hosted version uses the same compute library wrapped in a FastAPI backend.

A stock price forecaster built around an LSTM network with an attention layer. It fits a small set of technical features (RSI, MACD, Bollinger Bands, moving averages, lag and rolling statistics) and rolls the model forward a few days. A Streamlit app lets you type in a ticker and read off the forecast.

This is a learning project, not a trading signal. An LSTM on a handful of indicators does not predict prices in any reliable way, and the walk-forward scores reflect that. Treat the output as an illustration of the pipeline, not as advice.

## Table of contents

- [How it is laid out](#how-it-is-laid-out)
- [Install](#install)
- [Usage](#usage)
- [Tests](#tests)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [Licence](#licence)

## How it is laid out

The code is split so that importing any module does no work. The download, the
TensorFlow load and the optional retraining job all sit behind functions you
call yourself.

```text
src/stockforecast/
  features.py       # pure technical-feature engineering
  data_shaping.py   # scaling and sliding-window helpers
  data.py           # Yahoo Finance download (network)
  model.py          # build, train, load, forecast (TensorFlow, lazy)
main.py             # train and print a forecast
app.py              # Streamlit dashboard
tests/              # offline tests for the pure logic
```

The feature and shaping code in `features.py` and `data_shaping.py` is pure:
it takes a DataFrame in and returns arrays out, with no network or model. The
test suite covers that logic with small synthetic frames, so it runs fully
offline and headless.

## Install

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/FatihHekim0glu/Stock-Price-Forecast.git
cd Stock-Price-Forecast
uv sync --all-extras
```

The extras are grouped so you can install only what you need:

- `data`: yfinance, for the live download
- `models`: TensorFlow, for training and inference
- `dashboard`: Streamlit and matplotlib, for the app
- `automation`: APScheduler, for the daily retraining job
- `dev`: pytest, ruff, mypy

## Usage

1. Train the model and print a short forecast:

   ```bash
   uv run python main.py
   ```

   This downloads data, fits the model, saves `stock_model.h5`, prints the next
   week of predictions, then keeps a daily retraining job alive. Stop it with
   Ctrl+C.

2. Run the Streamlit app once a model has been saved:

   ```bash
   uv run streamlit run app.py
   ```

   Open the URL Streamlit prints (usually http://localhost:8501). Enter a
   ticker, pick how many days to forecast, and read the chart and table.

## Tests

The tests cover the pure feature and shaping logic only. They do not touch the
network or run TensorFlow.

```bash
uv run pytest
```

## Dependencies

Python 3.10 or newer. Runtime packages: numpy, pandas, scikit-learn, plus the
optional extras above (yfinance, TensorFlow, Streamlit, matplotlib,
APScheduler). See `pyproject.toml` for the pinned ranges.

## Contributing

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Make your change and run `uv run ruff check .`, `uv run mypy src` and
   `uv run pytest`.
4. Commit with a clear message and open a pull request that explains the
   change.

## Licence

MIT.
