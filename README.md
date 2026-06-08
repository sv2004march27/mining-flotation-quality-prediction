
```markdown
# 🏭 Quality Prediction in a Mining Process using XGBoost

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU_Accelerated-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Web_App-red.svg)
![Data Science](https://img.shields.io/badge/Data_Science-Time_Series_Forecasting-success.svg)

## 📌 Overview
This project implements a custom machine learning pipeline to predict the percentage of silica (impurity) in iron ore concentrate. The primary aim of this system is to enable smart industrial monitoring by forecasting impurity levels accurately, allowing engineers to take proactive corrective actions in the flotation plant. This reduces waste, optimizes the extraction process, and lowers the environmental impact.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit

## 📊 Dataset & Feature Engineering
The custom dataset consists of real-world industrial telemetry from a mining flotation plant.
* **Original Capture:** Highly granular sensor data containing mismatched timestamps (ranging from 20-second intervals to hourly lab measurements).
* **Constraints:** The target variable (`% Silica Concentrate`) was predicted *without* using the highly correlated `% Iron Concentrate` column, forcing the model to learn the underlying physical plant behaviors.
* **Preprocessing:** Sensor data was resampled to a uniform 1-minute frequency. Strict Forward-Fill (`ffill`) interpolation was applied to perfectly align the timestamps and prevent data leakage (peeking into the future).
* **Advanced Engineering:** Lag intervals (60 and 120 minutes) and 60-minute rolling averages were engineered to give the model a "memory" of the delayed physical reactions of the plant's machinery.

## 🧠 Model Architecture
The system utilizes the **XGBoost Regressor**, chosen for its industry-leading performance on tabular data and its ability to natively utilize GPU acceleration for heavy time-series tasks.

* **Estimators (Trees):** 500 (with Early Stopping triggered at 20 rounds of no improvement)
* **Learning Rate:** 0.05
* **Tree Method:** `hist` (Optimized for GPU)
* **Hardware:** Trained natively on NVIDIA GPU (`device='cuda'`)
* **Validation Strategy:** 70% Training / 15% Validation / 15% Test (Strict time-series split)

### 📈 Performance Results
The model achieved highly accurate trend tracking on the unseen test dataset:
* **Root Mean Squared Error (RMSE):** Consistently minimized to track volatile plant shifts.
* **Mean Absolute Error (MAE):** Validated against real-world allowable margins of error.
* **Visual Evaluation:** The 4-panel evaluation dashboard confirms high feature importance reliance on physical plant controls (like air flow) and displays a normal error distribution.

![XGBoost Model Evaluation Dashboard](model_dashboard.png)

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sv2004march27/mining-flotation-quality-prediction.git](https://github.com/sv2004march27/mining-flotation-quality-prediction.git)
   cd mining-flotation-quality-prediction

```

2. **Install dependencies:**
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt

```



*(Note: The `MiningProcess_Flotation_Plant_Database.csv` dataset exceeds standard Git file size limits. Please ensure your local copy of the dataset is placed in the root directory before running the code.)*

## 💻 Usage

This project offers two ways to interact with the model:

### Option 1: Live Web Application (Streamlit)

This repository includes a fully functional Streamlit web application. Users can upload a CSV of plant telemetry, and the app will automatically engineer the lag features, apply the pre-trained XGBoost weights, and generate a live visualization dashboard.

```bash
streamlit run app.py

```

### Option 2: Jupyter Notebook Pipeline

If you wish to view the data exploration, train the model from scratch, or view the 4-panel evaluation dashboard, simply open the notebook and execute the cells sequentially.

```bash
jupyter notebook Quality_Prediction_Mining.ipynb

```

## 📂 Repository Structure

```text
📦 Repository Root
 ┣ 📜 app.py                               # Streamlit web application script
 ┣ 📜 Quality_Prediction_Mining.ipynb      # Interactive notebook for complete pipeline
 ┣ 📜 xgboost_silica_model.json            # Exported weights for the trained XGBoost model
 ┣ 📜 model_dashboard.png                  # Generated 4-panel visual evaluation dashboard
 ┣ 📜 requirements.txt                     # Project dependencies and library versions
 ┣ 📜 .gitignore                           # Git configuration to exclude large CSV files
 ┗ 📜 README.md                            # Project documentation

```

## 🔮 Future Enhancements

* **Live API Integration:** Connecting the inference script directly to real-time MQTT plant sensors for live-streamed predictions.
* **Automated Hyperparameter Tuning:** Implementing Optuna to systematically sweep and find the absolute optimal XGBoost parameters for this specific facility.

---

*Engineered by [Shivam Verma*](https://www.google.com/search?q=https://github.com/sv2004march27)

```

This version connects all the dots of what you have built into a cohesive, highly professional package. How does this updated structure look to you?

```
