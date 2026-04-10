import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Title
st.title("Stock Price Prediction using ARIMA")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("nsestockindia.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df

df = load_data()

# Show available stocks
stocks = df["stock"].unique()

# Dropdown for stock selection
selected_stock = st.selectbox(
    "Select Stock Name",
    stocks
)

# Filter selected stock
st_data = df[df["stock"] == selected_stock]

# Show data
st.subheader("Stock Data Preview")
st.write(st_data.tail())

# Stationarity Check Function
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())

    st.write("ADF Statistic:", result[0])
    st.write("P-value:", result[1])

    if result[1] < 0.05:
        st.success("Data is Stationary")
    else:
        st.warning("Data is NOT Stationary")

# Prepare Close Data
data = st_data[["Close"]].copy()

data["Returns"] = data["Close"].pct_change()
data.dropna(inplace=True)

# Differencing
data["Close_Diff"] = data["Close"].diff()

# Stationarity Check
st.subheader("Stationarity Test")

check_stationarity(data["Close"])

# ARIMA Model
st.subheader("Model Training")

model = ARIMA(data["Close"], order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast_steps = st.slider(
    "Select Forecast Days",
    1,
    30,
    10
)

forecast = model_fit.forecast(steps=forecast_steps)

# Future Dates
dates = pd.date_range(
    start=data.index[-1],
    periods=forecast_steps + 1,
    freq="B"
)[1:]

# Plot
st.subheader("Stock Price Prediction")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(data["Close"], label="Actual Prices")
ax.plot(dates, forecast,
        label="Predicted Prices",
        linestyle="dashed")

plt.xticks(rotation=90)
plt.legend()

st.pyplot(fig)
