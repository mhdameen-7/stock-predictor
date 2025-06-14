import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import ta

# Set Streamlit config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Advanced Stock Price Predictor with LSTM")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY):", "AAPL")

# Load stock data with indicators
@st.cache_data(show_spinner=True)
def load_data(ticker):
    df = yf.download(ticker, start='2015-01-01', end='2024-12-31')

    # Ensure 'Close' is a 1D pandas Series (not reshaped)
    close = df['Close']
    close = pd.Series(close.values.flatten(), index=close.index)

    # Technical indicators with 1D Series input
    df['rsi'] = ta.momentum.RSIIndicator(close=close).rsi()
    df['macd'] = ta.trend.MACD(close=close).macd()
    df['ema'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df['sma'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()

    df.dropna(inplace=True)
    return df


# Function to prepare sequences
def create_sequences(data, seq_len):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i - seq_len:i])
        y.append(data[i, 0])  # Predicting Close
    return np.array(x), np.array(y)

# Prediction logic
if st.button("🔮 Predict"):
    st.info("Fetching and processing data...")
    df = load_data(ticker)
    
    features = ['Close', 'rsi', 'macd', 'ema', 'sma']
    data = df[features].values

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Sequence length
    seq_len = 60
    x, y = create_sequences(scaled_data, seq_len)

    # Train/Test split
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    st.info("Training LSTM model...")
    # Build the model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predict
    predicted = model.predict(x_test)
    
    # Inverse scale predictions and actual values
    predicted_combined = np.concatenate((predicted, x_test[:, -1, 1:]), axis=1)
    y_test_combined = np.concatenate((y_test.reshape(-1, 1), x_test[:, -1, 1:]), axis=1)

    predicted_prices = scaler.inverse_transform(predicted_combined)[:, 0]
    actual_prices = scaler.inverse_transform(y_test_combined)[:, 0]

    # Evaluation
    rmse = sqrt(mean_squared_error(actual_prices, predicted_prices))
    st.subheader(f"📉 Model RMSE: {rmse:.2f}")

    # Plot results
    st.subheader("📊 Predicted vs Actual Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_prices, label="Actual Price")
    ax.plot(predicted_prices, label="Predicted Price")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # Latest predicted value
    st.subheader("📌 Latest Prediction")
    st.success(f"Next Predicted Close Price: **${predicted_prices[-1]:.2f}**")

    # Optional: Export prediction results
    results_df = pd.DataFrame({'Actual': actual_prices, 'Predicted': predicted_prices})
    csv = results_df.to_csv(index=False).encode()
    st.download_button("📥 Download Predictions as CSV", csv, f"{ticker}_predictions.csv", "text/csv")

