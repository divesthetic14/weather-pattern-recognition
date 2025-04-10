import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import gdown

import os

def download_models():
    os.makedirs("models", exist_ok=True)

    files = {
       "models/forecasting_lstm_model.h5": "1ywjAxavyxVoagoHitQL9HtHKzMgRUKXQ",
        "models/max_temp_scaler.pkl": "1_Cc0JVKZRcYf4rnOZi3zExdCg1XYfwoT",
        "models/classifier_rain_tomorrow.pkl": "1fzdhjugfo9l8I9MfKx850lXN2bWPuFk1",
        "models/classifier_scaler.pkl": "1lGmJZS8XkUXJeIhP35jSu7JAJ-UkZeZh",
        "models/kmeans_weather.pkl": "1NkRFfoexm6GxAwdUDZv-CaZQRGi1hN2w",
        "models/cluster_scaler.pkl": "1HhyVWMQPEbcxg4bk1Gk88epUYg395iUp",
    }

    for out_path, file_id in files.items():
        if not os.path.exists(out_path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", out_path, quiet=False)

download_models()

# Load models
forecast_model = load_model("models/forecasting_lstm_model.h5")
forecast_scaler = joblib.load("models/max_temp_scaler.pkl")

clf_model = joblib.load("models/classifier_rain_tomorrow.pkl")
clf_scaler = joblib.load("models/classifier_scaler.pkl")

cluster_model = joblib.load("models/kmeans_weather.pkl")
cluster_scaler = joblib.load("models/cluster_scaler.pkl")
cluster_df = pd.read_csv("data/clustered_locations.csv")

# Load data
weather_data = pd.read_csv("data/weatherAUS.csv")

st.title("🌦️ Weather Pattern Recognition Dashboard")

tab1, tab2, tab3 = st.tabs(["🔮 Forecasting", "🌧️ Classification", "🧩 Clustering"])

# -------------------
# 🔮 Forecasting Tab
# -------------------
with tab1:
    st.header("Predict Tomorrow's Max Temperature (LSTM)")
    city = st.selectbox("Select a city", weather_data["Location"].unique())

    df_city = weather_data[weather_data["Location"] == city].copy()
    df_city['Date'] = pd.to_datetime(df_city['Date'])
    df_city = df_city.sort_values("Date")
    df_city['MaxTemp'] = df_city['MaxTemp'].fillna(method='ffill')
    df_city = df_city.dropna(subset=['MaxTemp'])

    scaled = forecast_scaler.transform(df_city[['MaxTemp']])
    last_30 = scaled[-30:].reshape((1, 30, 1))

    pred_scaled = forecast_model.predict(last_30)[0][0]
    pred_temp = forecast_scaler.inverse_transform([[pred_scaled]])[0][0]

    st.success(f"📈 Predicted Max Temperature in {city} for Tomorrow: **{pred_temp:.2f} °C**")

# -------------------
# 🌧️ Classification Tab
# -------------------
with tab2:
    st.header("Will it Rain Tomorrow? 🌧️")

    st.write("Enter today's weather data:")

    MinTemp = st.number_input("MinTemp (°C)", value=10.0)
    MaxTemp = st.number_input("MaxTemp (°C)", value=20.0)
    Rainfall = st.number_input("Rainfall (mm)", value=0.0)
    Humidity3pm = st.number_input("Humidity at 3pm (%)", value=50.0)
    Pressure9am = st.number_input("Pressure at 9am (hPa)", value=1015.0)
    WindGustSpeed = st.number_input("Wind Gust Speed (km/h)", value=35.0)

    if st.button("Predict Rain Tomorrow"):
        user_input = np.array([[MinTemp, MaxTemp, Rainfall, Humidity3pm, Pressure9am, WindGustSpeed]])
        scaled_input = clf_scaler.transform(user_input)
        prediction = clf_model.predict(scaled_input)[0]
        result = "Yes 🌧️" if prediction == 1 else "No ☀️"
        st.info(f"Prediction: **Rain Tomorrow?** → {result}")

# -------------------
# 🧩 Clustering Tab
# -------------------
with tab3:
    st.header("Clustering Cities by Weather Patterns 🧩")
    st.write("Cities are grouped based on average weather using KMeans clustering.")

    st.dataframe(cluster_df.style.background_gradient(cmap="coolwarm", subset=["Cluster"]))

    selected_cluster = st.selectbox("Select a Cluster", sorted(cluster_df["Cluster"].unique()))
    st.write(f"Cities in Cluster {selected_cluster}:")
    st.write(cluster_df[cluster_df["Cluster"] == selected_cluster]["Location"].tolist())
