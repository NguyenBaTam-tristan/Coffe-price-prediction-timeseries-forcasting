"""Chuẩn bị dữ liệu cho bài toán dự đoán giá cà phê theo ngày.

Chạy từ thư mục gốc project:
    python src/pipeline.py

Raw files chỉ được đọc, không bị sửa. Các file kết quả nằm trong data/processed
và data/features để có thể kiểm tra từng bước riêng biệt.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Dùng dataset coffee đã được bổ sung đủ từng ngày trong khoảng nghiên cứu.
COFFEE_RAW = PROJECT_ROOT / "data" / "raw" / "coffee" / "coffee_daklak_3y_complete.csv"
WEATHER_RAW = PROJECT_ROOT / "data" /"raw"/ "weather" /"weather.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

WEATHER_COLUMNS = {
    "time": "date",
    "temperature_2m_max (°C)": "temperature_2m_max",
    "temperature_2m_min (°C)": "temperature_2m_min",
    "temperature_2m_mean (°C)": "temperature_2m_mean",
    "precipitation_sum (mm)": "precipitation_sum",
    "rain_sum (mm)": "rain_sum",
    "precipitation_hours (h)": "precipitation_hours",
    "relative_humidity_2m_mean (%)": "relative_humidity_2m_mean",
    "soil_moisture_0_to_7cm_mean (m³/m³)": "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm_mean (m³/m³)": "soil_moisture_7_to_28cm",
    "shortwave_radiation_sum (MJ/m²)": "shortwave_radiation_sum",
}


def clean_coffee():
    """Làm sạch giá, giữ một dòng cho mỗi ngày nguồn đã công bố."""
    coffee = pd.read_csv(COFFEE_RAW)
    coffee["date"] = pd.to_datetime(coffee["date"], errors="coerce")
    coffee["price"] = pd.to_numeric(coffee["price"], errors="coerce")
    coffee = coffee.dropna(subset=["date", "price"]).copy()
    coffee = coffee.drop_duplicates(subset="date", keep="last").sort_values("date")
    return coffee[["date", "province", "product", "price", "unit", "source", "source_url"]]

def clean_coffee_remake():
    """Làm sạch giá, giữ một dòng cho mỗi ngày nguồn đã công bố."""
    coffee = pd.read_csv(COFFEE_RAW)
    coffee["date"] = pd.to_datetime(coffee["date"], errors="coerce")
    coffee["price"] = pd.to_numeric(coffee["price"], errors="coerce")
    coffee = coffee.dropna(subset=["date", "price"]).copy()
    coffee = coffee.drop_duplicates(subset="date", keep="last").sort_values("date")
    return coffee[["date", "price"]]

def clean_weather():
    """Đọc daily weather.csv, bỏ metadata và chuẩn hóa tên cột/kiểu dữ liệu."""
    # File Open-Meteo này có 2 dòng metadata trước dòng header daily.
    weather = pd.read_csv(WEATHER_RAW, skiprows=2)
    missing_columns = set(WEATHER_COLUMNS) - set(weather.columns)
    if missing_columns:
        raise ValueError(f"Weather thiếu các cột bắt buộc: {sorted(missing_columns)}")

    weather = weather.rename(columns=WEATHER_COLUMNS)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    numeric_columns = [column for column in WEATHER_COLUMNS.values() if column != "date"]
    weather[numeric_columns] = weather[numeric_columns].apply(pd.to_numeric, errors="coerce")
    weather = weather.dropna(subset=["date"])
    weather = weather.drop_duplicates(subset="date", keep="last").sort_values("date")
    return weather[["date", *numeric_columns]]


def build_features(merged):
    """Tạo feature chỉ từ hiện tại/quá khứ; không dùng dữ liệu tương lai."""
    features = merged.sort_values("date").copy()
    price = features["coffee_price"]

    # shift dương chỉ lấy giá của các bản ghi trước đó, nên không leakage.
    for lag in (1, 2, 3, 7, 14, 30):
        features[f"price_lag_{lag}"] = price.shift(lag)
    for window in (7, 14, 30):
        history = price.shift(1)
        features[f"price_mean_{window}"] = history.rolling(window).mean()
        if window in (7, 30):
            features[f"price_std_{window}"] = history.rolling(window).std()

    for column, name in (
        ("rain_sum", "rain"),
        ("temperature_2m_mean", "temperature_mean"),
        ("relative_humidity_2m_mean", "humidity_mean"),
    ):
        features[f"{name}_7d"] = features[column].shift(1).rolling(7).mean()
        features[f"{name}_30d"] = features[column].shift(1).rolling(30).mean()

    features["day_of_week"] = features["date"].dt.dayofweek
    features["day_of_month"] = features["date"].dt.day
    features["month"] = features["date"].dt.month
    features["quarter"] = features["date"].dt.quarter
    #features["sin_month"] = (2 * 3.141592653589793 * features["month"] / 12).map(__import__("math").sin)
    #features["cos_month"] = (2 * 3.141592653589793 * features["month"] / 12).map(__import__("math").cos)

    # Dataset coffee đầy đủ theo ngày nên target là giá của ngày kế tiếp.
    # date_gap_days vẫn được lưu để kiểm tra dữ liệu có bị đứt quãng hay không.
    features["next_day_price"] = price.shift(-1)
    features["date_gap_days"] = features["date"].shift(-1).sub(features["date"]).dt.days
    return features.dropna().reset_index(drop=True)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    coffee = clean_coffee_remake()
    weather = clean_weather()
    merged = coffee.merge(weather, on="date", how="inner").rename(columns={"price": "coffee_price"}).sort_values("date")
    features = build_features(merged)

    coffee.to_csv(PROCESSED_DIR / "coffee_clean.csv", index=False, date_format="%Y-%m-%d")
    weather.to_csv(PROCESSED_DIR / "weather_clean.csv", index=False, date_format="%Y-%m-%d")
    merged.to_csv(PROCESSED_DIR / "merged_daily.csv", index=False, date_format="%Y-%m-%d")
    features.to_csv(FEATURES_DIR / "training_dataset.csv", index=False, date_format="%Y-%m-%d")

    print(f"coffee_clean: {len(coffee)} dòng")
    print(f"weather_clean: {len(weather)} dòng")
    print(f"merged_daily: {len(merged)} dòng")
    print(f"training_dataset: {len(features)} dòng")


if __name__ == "__main__":
    main()
