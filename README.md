# 🏠 California House Price Prediction

A machine learning project that predicts California house prices using a **Random Forest Regressor** trained on the California Housing dataset from scikit-learn. It ships with a **FastAPI** backend and a **zero-dependency browser frontend**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Model Details](#model-details)
- [Frontend UI](#frontend-ui)
- [API Endpoints](#api-endpoints)
- [Input Features](#input-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Model](#training-the-model)
  - [Running the API](#running-the-api)
  - [Opening the Frontend](#opening-the-frontend)
- [Usage Examples](#usage-examples)
  - [Single Prediction](#single-prediction)
  - [Batch Prediction (CSV Upload)](#batch-prediction-csv-upload)
- [Sample Data](#sample-data)
- [Logging](#logging)

---

## 📌 Overview

This project exposes a trained machine learning model as a REST API, paired with a modern browser-based frontend. Given house attributes for a California neighborhood, it returns the predicted house price. It supports:

- **Single prediction** via the web UI form or a JSON payload
- **Batch prediction** via drag-and-drop CSV upload (UI) or API file upload
- **CORS enabled** — the frontend can run directly from `file://` without a separate web server

---

## 📁 Project Structure

```
house-price-prediction/
│
├── main.py                  # FastAPI app — endpoints + CORS middleware
├── index.html               # Browser frontend (no build step needed)
│
├── src/
│   ├── train.py             # Model training script
│   └── explore.py           # Dataset exploration script
│
├── data/
│   ├── house_model.joblib   # Serialized trained Random Forest model
│   ├── house_features.joblib# Serialized feature names list
│   ├── test_houses.csv      # Sample input CSV for batch prediction testing
│   └── predictions.csv      # Sample output CSV with predicted prices
│
├── logs/
│   └── app.log              # Training logs (MAE, R² score, etc.)
│
├── pyproject.toml           # Project metadata and dependencies (uv)
├── .python-version          # Pinned Python version (3.12)
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

## 🛠️ Tech Stack

| Layer           | Technology                                          |
|-----------------|-----------------------------------------------------|
| Language        | Python 3.12                                         |
| API Framework   | FastAPI 0.138+                                      |
| ASGI Server     | Uvicorn 0.49+                                       |
| ML Library      | scikit-learn 1.9+                                   |
| Data            | pandas 3.0+                                         |
| Serialization   | joblib 1.5+                                         |
| File Upload     | python-multipart 0.0.32+                            |
| CORS            | FastAPI `CORSMiddleware`                             |
| Frontend        | Vanilla HTML / CSS / JavaScript (no frameworks)     |
| Package Manager | [uv](https://github.com/astral-sh/uv)               |

---

## 🤖 Model Details

| Property          | Value                         |
|-------------------|-------------------------------|
| Algorithm         | Random Forest Regressor       |
| Dataset           | California Housing (scikit-learn) |
| Total Records     | 20,640                        |
| Train / Test Split| 80% / 20% (random_state=42)   |
| Number of Trees   | 100 estimators                |
| Mean Absolute Error | **$32,773**                 |
| R² Score          | **0.805**                     |

The model is trained once via `src/train.py` and serialized to `data/house_model.joblib`. It is loaded into memory once when the API server starts.

---

## 🖥️ Frontend UI

`index.html` is a self-contained, single-file web application that talks directly to the FastAPI backend. No build step, no npm — just open the file in a browser.

| Feature | Description |
|---|---|
| **API Status Indicator** | Auto-pings `/health` on load — shows 🟢 Online or 🔴 Offline |
| **Model Stats Bar** | Displays algorithm, MAE, R² score and dataset size |
| **Single Prediction Tab** | 8 labeled input fields grouped by Location / Neighborhood / House Details |
| **📋 Load Demo** | Pre-fills the real sample from `test_houses.csv` in one click |
| **Predicted Price Card** | Animated result showing price and confidence range |
| **Batch CSV Tab** | Drag-and-drop or browse to upload a `.csv` file |
| **Results Table** | Displays all predictions in a scrollable table (handles quoted prices correctly) |
| **⬇ Download CSV** | One-click download of the batch prediction output |
| **Toast Notifications** | Friendly pop-ups for validation errors and success messages |

> **CSV Parsing:** Prices like `"$430,217"` are quoted in the API's CSV response (they contain a comma). The frontend uses a proper quoted-field parser so the value is never split across two columns.

---

## 🚀 API Endpoints

### `GET /`
Returns a welcome message confirming the API is running.

**Response:**
```json
{
  "message": "California house price prediction api",
  "status": "running",
  "endpoint": "send post request to /predict"
}
```

---

### `GET /health`
Health check endpoint. Returns model information and performance metrics.

**Response:**
```json
{
  "status": "running",
  "model": "RandomForestRegressor",
  "features": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
  "avg_error": "$32,773",
  "R2 Score": "0.805"
}
```

---

### `POST /predict`
Predicts the price of a **single house** from a JSON body.

**Request Body:**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.984,
  "AveBedrms": 1.023,
  "Population": 322,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Response:**
```json
{
  "predicted_price": "$430,217",
  "predicted_price_short": "4.30 hundred thousands",
  "fidence range": "$397,444 to $462,990"
}
```

---

### `POST /predictfile`
Predicts prices for **multiple houses** from an uploaded CSV file.

- **Input**: A `.csv` file with the required feature columns
- **Output**: The same CSV file returned as a download with an additional `predicted_columns_usd` column

**Required CSV columns:**
```
MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
```

**Error Handling:**
| Scenario | HTTP Status | Message |
|---|---|---|
| Non-CSV file uploaded | 400 | `please upload csv file only` |
| Required columns missing | 400 | Lists the missing columns |
| Empty CSV (no data rows) | 400 | `The uploaded file has no data rows` |
| Internal model error | 500 | Error detail message |

---

## 📊 Input Features

All 8 features are derived from the **California Housing dataset** and represent neighborhood-level statistics:

| Feature      | Type  | Constraints              | Description                              |
|--------------|-------|--------------------------|------------------------------------------|
| `MedInc`     | float | > 0                      | Median income in the block group (tens of thousands USD) |
| `HouseAge`   | float | > 0                      | Median age of houses in the block group  |
| `AveRooms`   | float | > 0                      | Average number of rooms per household    |
| `AveBedrms`  | float | > 0                      | Average number of bedrooms per household |
| `Population` | float | > 0                      | Total population of the block group      |
| `AveOccup`   | float | > 0                      | Average number of household members      |
| `Latitude`   | float | 32 ≤ value ≤ 42          | Latitude of the block group (California) |
| `Longitude`  | float | -125 ≤ value ≤ -114      | Longitude of the block group (California)|

---

## ⚙️ Getting Started

### Prerequisites

- Python **3.12+**
- [uv](https://github.com/astral-sh/uv) package manager

> Install `uv` if you don't have it:
> ```bash
> pip install uv
> ```

---

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChandraCherupally/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create and activate the virtual environment, then install dependencies:**
   ```bash
   uv sync
   ```

---

### Training the Model

Before running the API, you need to train the model and generate the serialized `.joblib` files:

```bash
uv run python src/train.py
```

This will:
- Download the California Housing dataset from scikit-learn
- Train a `RandomForestRegressor` (100 trees)
- Save `data/house_model.joblib` and `data/house_features.joblib`
- Log MAE and R² score to `logs/app.log`

> **Note:** `data/` is listed in `.gitignore`, so the model files are not committed to the repository. You must train the model locally before starting the API.

---

### Running the API

```bash
uv run uvicorn main:app --reload
```

The API will be available at: **http://127.0.0.1:8000**

Interactive API docs (Swagger UI): **http://127.0.0.1:8000/docs**

ReDoc API docs: **http://127.0.0.1:8000/redoc**

---

### Opening the Frontend

With the API running, simply open `index.html` in your browser — no additional server required:

```bash
# Windows — double-click index.html, or run:
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

The frontend auto-detects whether the API is online and shows the status in the header.

> **CORS** is configured in `main.py` via `CORSMiddleware` with `allow_origins=["*"]`, which allows the page to call the API from any origin including `file://`.

---

## 💡 Usage Examples

### Single Prediction

**Using `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

**Using Python `requests`:**
```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
}

response = requests.post(url, json=payload)
print(response.json())
```

---

### Batch Prediction (CSV Upload)

**Using `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/predictfile" \
  -F "file=@data/test_houses.csv" \
  --output predictions.csv
```

**Using Python `requests`:**
```python
import requests

url = "http://127.0.0.1:8000/predictfile"

with open("data/test_houses.csv", "rb") as f:
    response = requests.post(url, files={"file": ("test_houses.csv", f, "text/csv")})

with open("predictions.csv", "wb") as out:
    out.write(response.content)

print("Predictions saved to predictions.csv")
```

---

## 📂 Sample Data

The `data/test_houses.csv` file contains 5 sample California neighborhoods:

| MedInc | HouseAge | AveRooms | AveBedrms | Population | AveOccup | Latitude | Longitude | Predicted Price |
|--------|----------|----------|-----------|------------|----------|----------|-----------|-----------------|
| 8.3252 | 41 | 6.984 | 1.023 | 322 | 2.555 | 37.88 | -122.23 | $430,217 |
| 2.5 | 30 | 4.5 | 1.2 | 1200 | 3.5 | 34.05 | -118.25 | $173,764 |
| 5.0 | 15 | 7.0 | 1.5 | 800 | 2.8 | 36.5 | -120.0 | $191,591 |
| 3.8 | 20 | 5.5 | 1.1 | 950 | 3.0 | 35.0 | -119.5 | $96,020 |
| 9.5 | 10 | 8.0 | 2.0 | 400 | 2.2 | 37.78 | -122.41 | $464,527 |

---

## 📝 Logging

Training logs are written to `logs/app.log` in the following format:

```
YYYY-MM-DD HH:MM:SS,ms | LEVEL | house-price-prediction | message
```

**Example log output:**
```
2026-06-28 00:13:23,967 | INFO | house-price-prediction | loading datasets
2026-06-28 00:13:23,979 | INFO | house-price-prediction | total records: 20640
2026-06-28 00:13:23,979 | INFO | house-price-prediction | train and test splitting of data started....
2026-06-28 00:13:23,985 | INFO | house-price-prediction | train and test splitting of data completed....
2026-06-28 00:13:29,579 | INFO | house-price-prediction | average error: $32,773
2026-06-28 00:13:29,579 | INFO | house-price-prediction | R2 score: 0.805
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
