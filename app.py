import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error

from streamlit_extras.metric_cards import style_metric_cards

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

# Filter stock safely
st_data = df[df["stock"] == selected_stock].copy()

# -----------------------------
# Moving Averages
# -----------------------------

st_data["MA20"] = st_data["Close"].rolling(20).mean()
st_data["MA50"] = st_data["Close"].rolling(50).mean()

# -----------------------------
# Top Metrics
# -----------------------------

latest_price = st_data["Close"].iloc[-1]
mean_price = np.mean(st_data["Close"])
volatility = np.std(st_data["Close"])

col1, col2, col3 = st.columns(3)

col1.metric("📌 Latest Price", f"{latest_price:.2f}")
col2.metric("📊 Average Price", f"{mean_price:.2f}")
col3.metric("📉 Volatility", f"{volatility:.2f}")

style_metric_cards()

# -----------------------------
# Tabs Layout
# -----------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price Charts",
    "📉 Stationarity",
    "🤖 Forecast",
    "📊 Statistics"
])

# =============================
# TAB 1 — PRICE CHARTS
# =============================

with tab1:

    st.subheader("📊 Price with Moving Averages")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=st_data.index,
        y=st_data["Close"],
        name="Close Price"
    ))

    fig.add_trace(go.Scatter(
        x=st_data.index,
        y=st_data["MA20"],
        name="MA20"
    ))

    fig.add_trace(go.Scatter(
        x=st_data.index,
        y=st_data["MA50"],
        name="MA50"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Candlestick Chart (if OHLC exists)

    if {"Open","High","Low","Close"}.issubset(st_data.columns):

        st.subheader("🕯️ Candlestick Chart")

        candle = go.Figure(data=[go.Candlestick(
            x=st_data.index,
            open=st_data["Open"],
            high=st_data["High"],
            low=st_data["Low"],
            close=st_data["Close"]
        )])

        st.plotly_chart(candle, use_container_width=True)

# =============================
# TAB 2 — STATIONARITY
# =============================

with tab2:

    st.subheader("📉 Stationarity Test")

    data = st_data[["Close"]].copy()

    data["Returns"] = data["Close"].pct_change()

    def check_stationarity(timeseries):

        result = adfuller(timeseries.dropna())

        col1, col2 = st.columns(2)

        col1.metric(
            "ADF Statistic",
            round(result[0], 4)
        )

        col2.metric(
            "P-value",
            round(result[1], 6)
        )

        if result[1] < 0.05:
            st.success("✅ Data is Stationary")
        else:
            st.warning("⚠️ Data is NOT Stationary")

    check_stationarity(data["Close"])

    # Returns Chart

    st.subheader("📉 Daily Returns")

    fig_returns = px.line(
        data,
        y="Returns",
        title="Daily Returns Trend"
    )

    st.plotly_chart(fig_returns,
                    use_container_width=True)

# =============================
# TAB 3 — FORECAST
# =============================

with tab3:

    st.subheader("🤖 ARIMA Forecast")

    model = ARIMA(
        data["Close"].dropna(),
        order=(5,1,0)
    )

    model_fit = model.fit()

    forecast = model_fit.forecast(
        steps=forecast_steps
    )

    # Future Dates

    dates = pd.date_range(
        start=data.index[-1],
        periods=forecast_steps + 1,
        freq="B"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": dates,
        "Predicted Price": forecast
    })

    forecast_df.set_index(
        "Date",
        inplace=True
    )

    # Forecast Plot

    fig_forecast = go.Figure()

    fig_forecast.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Actual"
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Predicted Price"],
            name="Prediction"
        )
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

    # Accuracy Metrics

    train = data["Close"][:-forecast_steps]
    test = data["Close"][-forecast_steps:]

    model2 = ARIMA(train, order=(5,1,0))
    model2_fit = model2.fit()

    pred = model2_fit.forecast(
        steps=len(test)
    )

    rmse = np.sqrt(
        mean_squared_error(test, pred)
    )

    mae = mean_absolute_error(
        test,
        pred
    )

    col1, col2 = st.columns(2)

    col1.metric("RMSE", round(rmse,2))
    col2.metric("MAE", round(mae,2))

    # Forecast Table

    st.subheader("📅 Forecast Table")

    st.dataframe(forecast_df)

    # Download

    csv = forecast_df.to_csv().encode('utf-8')

    st.download_button(
        "⬇️ Download Forecast CSV",
        csv,
        "forecast.csv",
        "text/csv"
    )

# =============================
# TAB 4 — STATISTICS
# =============================

with tab4:

    st.subheader("📊 Summary Statistics")

    st.write(data.describe())

    st.subheader("📄 Raw Data")

    st.dataframe(st_data.tail(20))

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.markdown(
"""
**Project:** NSE Stock Prediction Using ARIMA  
**Model:** ARIMA (5,1,0)  
**Visualization:** Plotly  
**Interface:** Streamlit  

**Libraries Used:**  
Streamlit, Pandas, NumPy, Plotly, Statsmodels, Scikit-learn, Streamlit-Extras
"""
)
