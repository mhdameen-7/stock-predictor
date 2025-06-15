# 📈 Advanced Stock Price Predictor with LSTM

This project is an **AI-powered stock price prediction web application** built using **Streamlit**. It utilizes **LSTM (Long Short-Term Memory)** neural networks to predict the future closing price of a stock based on historical data and technical indicators.

## 🚀 Features

- 📥 **Stock Data Download** using Yahoo Finance (via `yfinance`)
- 🧮 **Technical Indicators**: RSI, MACD, EMA, SMA (via `ta` library)
- 🔧 **Data Preprocessing** and **Normalization**
- 🤖 **LSTM Model Training** using TensorFlow/Keras
- 📊 **Prediction Visualization** with Matplotlib
- ✅ **Evaluation Metrics (RMSE)**
- 📎 **Export Results to CSV**
- 💻 **Streamlit UI** for interactive experience

---

## 📚 How It Works

### 1️⃣ Data Loading
- Uses **Yahoo Finance** to fetch historical stock prices from 2015 to 2024.
- Adds technical indicators:
  - **RSI (Relative Strength Index)**
  - **MACD (Moving Average Convergence Divergence)**
  - **EMA (Exponential Moving Average)**
  - **SMA (Simple Moving Average)**

### 2️⃣ Data Preprocessing
- Selects features:
  - `['Close', 'rsi', 'macd', 'ema', 'sma']`
- Normalizes the dataset using **MinMaxScaler** (range [0,1])
- Creates **time series sequences** of length 60 for LSTM input.

### 3️⃣ Model Architecture
- **Two stacked LSTM layers (100 units each)** for capturing temporal dependencies.
- **Dropout layers (20%)** for regularization to prevent overfitting.
- **Dense layer (1 unit)** as output for predicting the stock’s next closing price.

### 4️⃣ Training & Testing
- **80% of data** used for training, **20% for testing**.
- Model trained for **10 epochs** with a batch size of 32.

### 5️⃣ Evaluation
- Uses **Root Mean Squared Error (RMSE)** to evaluate model accuracy.
- Displays predicted vs actual prices using Matplotlib plot.
- Provides **Next Predicted Close Price** for the stock.

### 6️⃣ Output
- Downloadable CSV file of actual vs predicted prices.

---

## 🏗️ Installation

### ⚙️ Requirements
Install the required libraries using pip:

```bash
pip install streamlit numpy pandas yfinance matplotlib scikit-learn tensorflow ta
