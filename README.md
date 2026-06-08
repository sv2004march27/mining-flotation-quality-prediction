# 🏭 Quality Prediction in a Mining Process using XGBoost

![XGBoost](https://img.shields.io/badge/XGBoost-GPU_Accelerated-orange)
![Python](https://img.shields.io/badge/Python-3.10-green)
![Data Science](https://img.shields.io/badge/Data_Science-Time_Series_Forecasting-blue)

## 📌 Overview
This project implements a custom machine learning pipeline to predict the percentage of silica (impurity) in iron ore concentrate. The primary aim of this system is to enable smart industrial monitoring by forecasting impurity levels accurately, allowing engineers to take proactive corrective actions in the flotation plant. This reduces waste, optimizes the extraction process, and lowers environmental impact.

## 📊 Dataset 
The custom dataset consists of real-world industrial telemetry from a mining flotation plant.
* **Original Capture:** Highly granular sensor data containing mismatched timestamps (ranging from 20-second intervals to hourly lab measurements).
* **Constraints:** The target variable (`% Silica Concentrate`) must be predicted without using the highly correlated `% Iron Concentrate` column.
* **Preprocessing:** Images were resampled to a uniform 1-minute frequency. Strict Forward-Fill (`ffill`) interpolation was applied to prevent data leakage (peeking into the future).
* **Feature Engineering:** Lag intervals (60 and 120 minutes) and 60-minute rolling averages were engineered to capture the delayed physical reactions of the plant's machinery.

## 🧠 Model Architecture & Training
The system utilizes **XGBoost Regressor**, chosen for its industry-leading performance on tabular data and its ability to seamlessly utilize GPU acceleration for heavy time-series tasks.

* **Estimators (Trees):** 500 (with Early Stopping at 20 rounds)
* **Learning Rate:** 0.05
* **Tree Method:** `hist` (Optimized for GPU)
* **Hardware:** Trained natively on NVIDIA GPU (`device='cuda'`)
* **Validation Split:** 70% Training / 15% Validation / 15% Test

### 📈 Performance Results
The model achieved highly accurate trend tracking on the unseen test dataset:
* **Root Mean Squared Error (RMSE):** Consistently minimized to track volatile plant shifts.
* **Mean Absolute Error (MAE):** Validated against real-world allowable margins of error.
* **Visual Evaluation:** The 4-panel evaluation dashboard confirms high feature importance reliance on physical plant controls (like air flow) and normal error distribution.

![XGBoost Model Evaluation Dashboard](model_dashboard.png)

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sv2004march27/mining-flotation-quality-prediction.git](https://github.com/sv2004march27/mining-flotation-quality-prediction.git)
