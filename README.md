# Traffic Speed Prediction using Deep Learning (LSTM)

This repository contains the implementation and evaluation of a deep learning model designed to forecast average traffic speeds in urban road networks. It was developed as the final assignment for the "Data Science Topics" course.

**Author:** Antonios Bardanis (Registration Number: P21110)

## 🛠️ Technology Stack
* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras (Sequential, LSTM, Dense)
* **Data Manipulation:** Pandas, NumPy, Scikit-learn (MinMaxScaler, Metrics)
* **Time-Series Analysis & Visualization:** Matplotlib, Seaborn, Statsmodels (ACF/PACF, Seasonal Decompose)

## 📊 Dataset
The project utilizes the open **PEMS-BAY** dataset, which includes traffic speed measurements from 325 loop detectors in the California Bay Area. The extracted subset spans the first half of 2017 (Jan 1, 2017 - Jun 30, 2017), with data recorded at 5-minute intervals (288 steps per day).

## ✨ Methodology & Key Features
* **Preprocessing & Feature Engineering:** Handled missing values (via time interpolation and backward filling), clipped outliers, and scaled data using `MinMaxScaler` (strictly fit on the training set to prevent data leakage). Additional temporal features were generated, including hour, day, weekend flags, and US holidays.
* **Sliding Window Approach:** Transformed the time-series forecasting task into a supervised learning problem using sliding windows.
* **LSTM Architecture:** The neural network features stacked LSTM layers (64 and 32 units) followed by a fully connected Dense output layer. The model was compiled with the Adam optimizer, MAE loss, and utilized EarlyStopping.

## 📈 Prediction Scenarios & Evaluation
The model's performance was evaluated using MAE, MAPE, RMSE, and R² across two main scenarios:
1. **Short-Term Prediction:** Predicting the next 1 hour (at 5-minute steps) based on a 1-hour input history. The model showed high accuracy, maintaining a low MAE (~2.8 mph) for immediate horizons.
2. **Medium-Term Prediction:** Predicting the next 12 hours (at 1-hour steps) based on a 2-hour input history. As expected, errors increased for distant horizons (MAE reached 4.43 mph at the 12th hour), highlighting the stochastic nature of long-term traffic flow.
