# 🏭 Quality Prediction in a Mining Process

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU_Accelerated-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Time_Series-success.svg)

## 📌 Overview
This repository contains a complete, GPU-accelerated machine learning pipeline designed to predict the percentage of silica (impurity) in iron ore concentrate. Using real-world industrial telemetry from a flotation plant, the model forecasts impurity levels to help plant engineers optimize the extraction process, minimize waste, and take proactive corrective actions.

## ⚙️ Methodology & Pipeline
The pipeline is entirely contained within the provided Jupyter Notebook and executes the following core steps:

### 1. Data Preprocessing & Alignment
* **Handling Mixed Frequencies:** The raw dataset contains mismatched timestamps ranging from 20-second sensor intervals to hourly lab measurements. The data was strictly resampled to a uniform **1-minute frequency**.
* **Leakage Prevention:** Forward-Fill (`ffill`) interpolation was applied to perfectly align the timestamps without looking ahead, preventing any data leakage into the future.
* **Feature Constraints:** The target variable (`% Silica Concentrate`) is predicted *without* relying on the highly correlated `% Iron Concentrate` column.

### 2. Advanced Feature Engineering
Because physical plant machinery does not react instantly to control changes, "memory" features were engineered:
* **Time Lags:** 60-minute and 120-minute historical snapshots for all sensor columns.
* **Rolling Trends:** 60-minute rolling averages to capture the momentum of the flotation columns.

### 3. Model Architecture (GPU Accelerated)
* **Algorithm:** XGBoost Regressor (`XGBRegressor`)
* **Hardware Optimization:** Trained natively on an NVIDIA GPU (`device='cuda'`, `tree_method='hist'`) to process the heavy time-series data rapidly.
* **Validation:** A 3-way time-series split (70% Train / 15% Validation / 15% Test) utilizing **Early Stopping** (20 rounds) to prevent overfitting and capture optimal tree iterations.

## 📊 Evaluation & Visual Dashboard
The model is evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). 

The code automatically generates a comprehensive **4-Panel Evaluation Dashboard** upon completion, which includes:
1. **Feature Importances:** Identifies which physical plant sensors drive the predictions.
2. **12-Hour Prediction Window:** A zoomed-in look at how well the model tracks sudden silica spikes.
3. **Prediction Accuracy Spread:** Scatter plot verifying overestimation vs. underestimation.
4. **Residual (Error) Distribution:** Histogram ensuring a normal distribution of model errors.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sv2004march27/mining-flotation-quality-prediction.git](https://github.com/sv2004march27/mining-flotation-quality-prediction.git)
