# Stock Market Forecasting Bot

An advanced stock market forecasting bot that predicts future stock prices using an LSTM neural network with an attention mechanism. The bot incorporates technical indicators to enhance prediction accuracy and provides an interactive user interface using Streamlit.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- **Data Collection**: Fetches historical stock data using Yahoo Finance.
- **Data Preprocessing**: Adds technical indicators like RSI, MACD, Bollinger Bands, and Moving Averages.
- **Model Architecture**: Uses an LSTM neural network with an attention mechanism for better performance.
- **Interactive Dashboard**: Provides an interactive web app using Streamlit for users to input stock tickers and view predictions.
- **Visualization**: Displays historical data and future predictions in charts and tables.
- **Automation**: Includes an optional scheduler to retrain the model automatically.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/stock-market-forecasting-bot.git
    cd stock-market-forecasting-bot
    ```

2. **Create a Virtual Environment** (Recommended):

    - **For Windows**:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

    - **For macOS/Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Train the Model**

    Run `main.py` to train the model and generate `stock_model.h5`.
    ```bash
    python main.py
    ```
    *Note: The model will be saved in the same directory as `main.py`.*

2. **Run the Streamlit App**

    Start the Streamlit app to interact with the forecasting bot.
    ```bash
    streamlit run app.py
    ```
    Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. **Using the App**

    - **Enter Stock Ticker**: Use the sidebar to input the stock ticker symbol (e.g., `AAPL` for Apple Inc.).
    - **Select Forecast Days**: Choose the number of days ahead you want to forecast.
    - **View Results**: The app will display the latest stock data, historical closing prices, and forecasted prices in both chart and table formats.

## Project Structure
```bash
stock-market-forecasting-bot/
├── main.py          # Script for data processing, model training, and evaluation
├── app.py           # Streamlit application for interactive forecasting
├── requirements.txt # List of required Python packages
├── stock_model.h5   # Saved trained model (generated after running main.py)
└── README.md        # Project documentation

## Dependencies
Python: Version 3.6 or higher
Required Packages (also listed in requirements.txt):
numpy
pandas
matplotlib
yfinance
ta
scikit-learn
tensorflow
streamlit
apscheduler
Contributing
Contributions are welcome! If you'd like to improve this project, please follow these steps:

Fork the Repository: Click on the 'Fork' button at the top right of the repository page.

Create a New Branch:

bash
Copy code
git checkout -b feature/YourFeatureName
Commit Your Changes: Make your changes and commit them with clear messages.

bash
Copy code
git commit -am 'Add new feature'
Push to Your Fork:

bash
Copy code
git push origin feature/YourFeatureName
Submit a Pull Request: Go to the original repository and submit a pull request explaining your changes.

License
This project is licensed under the MIT License.

Acknowledgments
TensorFlow: https://www.tensorflow.org/
Streamlit: https://www.streamlit.io/
TA-Lib (Technical Analysis Library): https://github.com/bukosabino/ta
APScheduler: https://apscheduler.readthedocs.io/
