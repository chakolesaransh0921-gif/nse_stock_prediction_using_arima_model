import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="NSE Stock Prediction Dashboard",
    layout="wide"
)

st.title("📈 NSE Stock Prediction Using ARIMA Model")

# -----------------------------
# Load Data
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("nsestockindia.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df

df = load_data()

# -----------------------------
# Sidebar Controls
# -----------------------------

st.sidebar.header("⚙️ Settings")

stocks = df["stock"].unique()

selected_stock = st.sidebar.selectbox(
    "Select Stock",
    stocks
)

forecast_steps = st.sidebar.slider(
    "Forecast Days",
    1,
    30,
    10
)

# Filter Data
st_data = df[df["stock"] == selected_stock]

# -----------------------------
# Top Metrics Row
# -----------------------------

latest_price = st_data["Close"].iloc[-1]
mean_price = st_data["Close"].mean()
max_price = st_data["Close"].max()
min_price = st_data["Close"].min()

col1, col2, col3, col4 = st.columns(4)

col1.metric("📌 Latest Price", f"{latest_price:.2f}")
col2.metric("📊 Average Price", f"{mean_price:.2f}")
col3.metric("🔼 Max Price", f"{max_price:.2f}")
col4.metric("🔽 Min Price", f"{min_price:.2f}")

# -----------------------------
# Data Preview Section
# -----------------------------

st.subheader("📄 Stock Data Preview")

st.dataframe(st_data.tail(10))

# -----------------------------
# Prepare Data
# -----------------------------

data = st_data[["Close"]].copy()

data["Returns"] = data["Close"].pct_change()
data.dropna(inplace=True)

data["Close_Diff"] = data["Close"].diff()

# -----------------------------
# Stationarity Test
# -----------------------------

st.subheader("📉 Stationarity Test (ADF Test)")

def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())

    col1, col2 = st.columns(2)

    col1.metric("ADF Statistic", round(result[0], 4))
    col2.metric("P-value", round(result[1], 6))

    if result[1] < 0.05:
        st.success("✅ Data is Stationary")
    else:
        st.warning("⚠️ Data is NOT Stationary")

check_stationarity(data["Close"])

# -----------------------------
# Price Trend Graph
# -----------------------------

st.subheader("📊 Historical Price Trend")

fig1, ax1 = plt.subplots(figsize=(10,5))

ax1.plot(data["Close"])
ax1.set_title("Historical Closing Prices")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")

st.pyplot(fig1)

# -----------------------------
# ARIMA Model
# -----------------------------

st.subheader("🤖 ARIMA Model Training")

model = ARIMA(data["Close"], order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=forecast_steps)

# Future Dates
dates = pd.date_range(
    start=data.index[-1],
    periods=forecast_steps + 1,
    freq="B"
)[1:]

# -----------------------------
# Forecast Data Table
# -----------------------------

forecast_df = pd.DataFrame({
    "Date": dates,
    "Predicted Price": forecast
})

forecast_df.set_index("Date", inplace=True)

st.subheader("📅 Forecast Table")

st.dataframe(forecast_df)

# Download Button
csv = forecast_df.to_csv().encode('utf-8')

st.download_button(
    "⬇️ Download Forecast CSV",
    csv,
    "forecast.csv",
    "text/csv"
)

# -----------------------------
# Forecast Graph
# -----------------------------

st.subheader("📈 Stock Price Prediction")

fig2, ax2 = plt.subplots(figsize=(12,6))

ax2.plot(data["Close"], label="Actual Prices")

ax2.plot(
    dates,
    forecast,
    label="Predicted Prices",
    linestyle="dashed"
)

ax2.set_title("Actual vs Predicted Prices")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")

plt.xticks(rotation=45)

plt.legend()

st.pyplot(fig2)

# -----------------------------
# Returns Visualization
# -----------------------------

st.subheader("📉 Daily Returns Analysis")

fig3, ax3 = plt.subplots(figsize=(10,5))

ax3.plot(data["Returns"])
ax3.set_title("Daily Returns")

st.pyplot(fig3)

# -----------------------------
# Summary Statistics
# -----------------------------

st.subheader("📊 Summary Statistics")

stats = data["Close"].describe()

st.write(stats)

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.markdown(
    """
    **Project:** NSE Stock Prediction Using ARIMA  
    **Model Used:** ARIMA (AutoRegressive Integrated Moving Average)  
    **Visualization:** Matplotlib  
    **Interface:** Streamlit
    """
)
