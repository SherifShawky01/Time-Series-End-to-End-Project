# Time-Series-End-to-End-Project


# 🧠 Time Series Forecasting with Classical, ML, DL, Prophet & FastAPI Deployment

A complete pipeline for time series forecasting using traditional statistical methods, machine learning regressors, deep learning models, Facebook Prophet, and the N-BEATS architecture — finalized with FastAPI deployment.

---

## 📚 Table of Contents

- [📌 Overview](#-overview)
- [📁 Dataset](#-dataset)
- [📊 Exploratory Data Analysis](#-exploratory-data-analysis)
- [⚙️ Models Implemented](#️-models-implemented)
  - [Classical Time Series Models](#classical-time-series-models)
  - [Machine Learning Models](#machine-learning-models)
  - [Deep Learning Models](#deep-learning-models)
  - [Prophet Forecasting](#prophet-forecasting)
  - [Innovative Models (N-BEATS)](#innovative-models-n-beats)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [🚀 Deployment (FastAPI)](#-deployment-fastapi)
- [📦 Installation](#-installation)
- [🧪 Running the Project](#-running-the-project)
- [📌 Future Improvements](#-future-improvements)

---

## 📌 Overview

This project demonstrates the design and evaluation of multiple forecasting techniques on financial time series data. It emphasizes performance comparisons and real-world deployment using FastAPI.

---

## 📁 Dataset

- Historical price data with features like `open`, `close`, `volume`, etc.
- Data is preprocessed, resampled, and split into training and test sets.

---

## 📊 Exploratory Data Analysis

- ADF test for stationarity
- Seasonal decomposition
- Autocorrelation (ACF) and Partial Autocorrelation (PACF)

---

## ⚙️ Models Implemented

### Classical Time Series Models
| Model                   | MAE   | RMSE  | MAPE   |
|------------------------|-------|-------|--------|
| Simple Exp Smoothing   | 8.77  | 10.67 | 10.88% |
| Holt’s Linear Trend    | 12.55 | 14.66 | 15.82% |
| Holt-Winters           | 12.14 | 14.24 | 15.27% |
| ARIMA/SARIMA           | 12.41 | 14.52 | 15.65% |
| **SARIMAX**            | **0.84** | **1.17** | **0.99%** ✅ |

---

### Machine Learning Models
| Model           | MAE   | RMSE  | R²       |
|----------------|--------|--------|----------|
| Random Forest   | 5.23   | 5.87   | -2.76    |
| XGBoost         | 6.17   | 6.83   | -4.08    |
| SVM             | 16.95  | 17.83  | -33.69   |

---

### Deep Learning Models
| Model           | MAE     | RMSE    | R²     |
|----------------|---------|---------|--------|
| ANN             | **0.26** | **0.36** | **0.98** ✅ |
| CNN             | 0.39    | 0.61    | 0.96   |
| LSTM            | 66.29   | 66.36   | -479   |
| GRU             | 61.06   | 61.14   | -406   |

---

### Prophet Forecasting
- RMSE: **0.498**
- Trend, seasonal components, external regressors
- Cross-validation using Prophet's built-in tools

---

### Innovative Models (N-BEATS)
- Neural basis expansion model (N-BEATS)
- Competitive results on forecasting horizon
- Suitable for real-time stream forecasting

---

## 📈 Evaluation Metrics

Models are evaluated using:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- R² Score (for regressors)

---

## 🚀 Deployment (FastAPI)

The best model (CNN) is deployed using FastAPI:

- `GET /` — Model health check
- `POST /predict` — Accepts 13-feature JSON array and returns prediction

Example input:
```json
{ "features": [1.2, 2.3, ..., 0.8] }
````

Example output:

```json
{ "prediction": 20453.21 }
```

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt
```

---

## 🧪 Running the Project

### Run Notebook:

```bash
jupyter notebook Full\ Project.ipynb
```

### Run FastAPI App:

```bash
uvicorn app_cnn_fastapi:app --reload
```

---

## 📌 Future Improvements

* Ensemble of top models (e.g., CNN + Prophet)
* Add real-time data pipeline with Apache Kafka or Airflow
* Visual dashboard using Streamlit
* Hyperparameter tuning using Optuna
* Exportable forecasting reports (PDF/Excel)

---

## 🙌 Acknowledgments

* Facebook Prophet
* PyTorch & scikit-learn
* Statsmodels, pmdarima
* N-BEATS from Oreshkin et al.

