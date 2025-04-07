import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from statsmodels.tsa.arima.model import ARIMA

# App title
st.title("📈 Stock Price Forecast - ARIMA & LSTM")

# Sidebar options
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

if st.sidebar.button("Run Forecast"):
    with st.spinner("Downloading data..."):
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        df_original = df.copy()

    st.subheader("Raw Data")
    st.line_chart(df["Close"])

    # --- Preprocessing ---
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    train_size = int(len(df_scaled) * 0.8)
    train_data = df_scaled[:train_size]
    test_data = df_scaled[train_size:]

    # --- Sequence Creation for LSTM ---
    def create_sequences(data, time_steps=60):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i - time_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    time_steps = 60
    X_train, y_train = create_sequences(train_data, time_steps)
    X_test, y_test = create_sequences(test_data, time_steps)
    X_train = X_train.reshape((-1, time_steps, 1))
    X_test = X_test.reshape((-1, time_steps, 1))

    # --- LSTM Model ---
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # --- LSTM Predictions ---
    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], df.shape[1]-1)))))[:, 0]
    real_prices = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], df.shape[1]-1)))))[:, 0]

    # Plot LSTM
    st.subheader("📉 LSTM Predictions vs Actual")
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(real_prices, label="Actual Price")
    ax1.plot(predicted_prices, label="Predicted Price", linestyle='--')
    ax1.legend()
    ax1.set_title("LSTM Forecast")
    st.pyplot(fig1)

    # --- ARIMA Forecast ---
    st.subheader("📊 ARIMA Forecast")
    arima_model = ARIMA(df["Close"], order=(5, 1, 0))
    arima_fit = arima_model.fit()
    forecast_steps = 30
    forecast = arima_fit.forecast(steps=forecast_steps)
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(df.index[-100:], df["Close"].values[-100:], label="Actual Close Price")
    ax2.plot(future_dates, forecast.values, label="ARIMA Forecast", linestyle='--', color='red')
    ax2.legend()
    ax2.set_title("ARIMA Forecast - Next 30 Days")
    st.pyplot(fig2)
