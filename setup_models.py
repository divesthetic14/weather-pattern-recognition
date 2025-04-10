import gdown
import os

os.makedirs("models", exist_ok=True)

files = {
    "models/forecasting_lstm_model.h5": "1DEKkdz6qVZb5JKM777hCrAsy0F29ZlCt",
    "models/max_temp_scaler.pkl": "1_Cc0JVKZRcYf4rnOZi3zExdCg1XYfwoT",
    "models/classifier_rain_tomorrow.pkl": "1fzdhjugfo9l8I9MfKx850lXN2bWPuFk1",
    "models/classifier_scaler.pkl": "1lGmJZS8XkUXJeIhP35jSu7JAJ-UkZeZh",
    "models/kmeans_weather.pkl": "1NkRFfoexm6GxAwdUDZv-CaZQRGi1hN2w",
    "models/cluster_scaler.pkl": "1HhyVWMQPEbcxg4bk1Gk88epUYg395iUp",
}

for out_path, file_id in files.items():
    if not os.path.exists(out_path):
        print(f"Downloading {out_path}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", out_path, quiet=False)
    else:
        print(f"{out_path} already exists.")
