# Dự đoán giá cà phê Đắk Lắk

Project Machine Learning dự đoán giá cà phê tại tỉnh Đắk Lắk theo ngày bằng dữ liệu lịch sử giá và dữ liệu thời tiết từ Open-Meteo.

## Mục tiêu

- Làm sạch và kiểm tra dữ liệu giá cà phê.
- Kết hợp giá cà phê với thời tiết theo từng ngày.
- Tạo lag features, rolling features và calendar features.
- So sánh Linear Regression, Random Forest và XGBoost.
- Đánh giá model bằng MAE, RMSE và R².
- Tuân thủ chronological split, không shuffle dữ liệu time series và không dùng dữ liệu tương lai.

Project hiện chỉ dùng file CSV và model artifacts. Không có frontend, backend, database, Docker hoặc API riêng.

## Dữ liệu

### Coffee price

- File: `data/raw/coffee/coffee_daklak_3y_complete.csv`
- Nguồn: [Chợ Giá](https://chogia.vn/gia-ca-phe-dak-lak-hom-nay/)
- Phạm vi hiện tại: `2024-01-13` đến `2026-09-20`
- Đơn vị: VND/kg

### Weather
- File: `data/raw/weather/weather.csv`
- Nguồn: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- Tần suất: daily
- Khu vực: tọa độ khoảng `12.688928, 108.01631`
- Timezone: `Asia/Bangkok` / GMT+7 trong file nguồn

Raw data được giữ nguyên. Các bước làm sạch và biến đổi tạo ra file mới trong `data/processed` và `data/features`.

## Kiến trúc

```text
data/raw/coffee/coffee_daklak_3y_complete.csv
                         |
                         v
                 src/pipeline.py <---- data/raw/weather/weather.csv
                         |
       +-----------------+------------------+
       |                 |                  |
       v                 v                  v
coffee_clean.csv  weather_clean.csv  merged_daily.csv
                                             |
                                             v
                                  training_dataset.csv
                                             |
                         +-----------------+------------------+
                         |                 |                  |
                         v                 v                  v
                           Linear Regression   Tree models
                                        |             Random Forest
                                        |               XGBoost
                         +-----------------+------------------+
                                             |
                                             v
                                  models/model_comparison.csv
```

### Các thư mục chính

```text
.
├── data/
│   ├── raw/coffee/                  # Dữ liệu giá gốc
│   ├── raw/weather/                 # Dữ liệu thời tiết gốc
│   ├── processed/                   # Dữ liệu sau cleaning và merge
│   ├── features/                    # Dataset dùng để train
│   └── raw/weather/weather.csv      # Daily weather input
├── models/                          # Metrics và model artifacts
├── src/
│   ├── pipeline.py                  # Cleaning, merge, feature engineering
│   └── train_models.py              # Huấn luyện và so sánh model
├── requirements.txt
└── README.md
```

## Cài đặt

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate script, chạy lệnh này trong phiên PowerShell hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Cách chạy

Luôn chạy từ thư mục gốc project.

### 1. Tạo dữ liệu processed và features

```powershell
python .\src\pipeline.py
```

Kết quả:

- `data/processed/coffee_clean.csv`
- `data/processed/weather_clean.csv`
- `data/processed/merged_daily.csv`
- `data/features/training_dataset.csv`

Pipeline merge bằng cột `date`. Feature rolling dùng `.shift(1)` trước khi tính trung bình để không lấy thông tin tương lai. Target `next_day_price` là giá của ngày kế tiếp.

### 2. Chạy EDA notebook

Mở `notebooks/02_eda.ipynb` trong VS Code, chọn kernel `.venv`, rồi chạy toàn bộ notebook. Notebook tạo bảng và biểu đồ inline cho trend, phân phối, outlier, seasonality, volatility, missingness, correlation và weather-price. Phần cuối giải thích lý do chọn/không chọn feature và leakage.

Có thể tạo báo cáo EDA dạng Markdown và file hình bằng script:

```powershell
python .\src\eda.py
```

Kết quả nằm trong `reports/eda_summary.md` và `reports/figures/`.

### 3. Huấn luyện và so sánh các model

```powershell
python .\src\train_models.py
```

Kết quả được lưu tại:

- `models/model_comparison.csv`
- `models/linear_regression/`
- `models/random_forest/`
- `models/xgboost/`

Mỗi model gồm `model.pkl`, `metrics.json` và `feature_columns.json`.

## Feature engineering

### Price features

- `price_lag_1`, `price_lag_2`, `price_lag_3`
- `price_lag_7`, `price_lag_14`, `price_lag_30`
- `price_mean_7`, `price_mean_14`, `price_mean_30`
- `price_std_7`, `price_std_30`

### Weather features

- Nhiệt độ max, min, mean
- Lượng mưa và số giờ có mưa
- Độ ẩm trung bình
- Độ ẩm đất ở lớp 0-7 cm và 7-28 cm
- Bức xạ mặt trời
- Rolling weather 7 ngày và 30 ngày

### Calendar features

- Day of week
- Day of month
- Month
- Quarter
- Sin/cos của month

## Đánh giá hiện tại

Kết quả trên test set chronological 20% cuối của dataset:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | 677.91 | 1034.49 | 0.9437 |
| Linear Regression | 734.35 | 1092.30 | 0.9372 |
| Random Forest | 1383.55 | 1883.44 | 0.8134 |

Đơn vị MAE và RMSE là VND/kg. XGBoost dự đoán mức thay đổi giá giữa ngày hiện tại và ngày kế tiếp, sau đó cộng vào giá hiện tại. Tham số được chọn trên validation theo thứ tự thời gian; test chỉ dùng để đánh giá cuối.

## Quy tắc chống leakage

- Không dùng `train_test_split(shuffle=True)`.
- Train/test được chia theo thứ tự thời gian.
- Lag lấy từ giá quá khứ.
- Rolling features được tính sau `shift(1)`.
- Target được tạo bằng giá ngày kế tiếp.
- Raw files không bị ghi đè.

## Giới hạn hiện tại

- Dữ liệu giá và thời tiết mới được merge trong phần giao nhau của hai khoảng ngày.
- Một số ngày cuối của coffee dataset chưa có weather tương ứng nên chưa được đưa vào merged dataset.
- Model hiện mới dự đoán one-step-ahead. Forecast 3 ngày và 7 ngày chưa được triển khai.
- SHAP explanation và walk-forward validation là các phần có thể bổ sung tiếp theo.

## Nguồn tham khảo

- [Chợ Giá - Giá cà phê Đắk Lắk](https://chogia.vn/gia-ca-phe-dak-lak-hom-nay/)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
