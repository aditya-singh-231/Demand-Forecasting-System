# Demand Forecasting using LSTM and Transformers

## Overview

This project implements a deep learning-based demand forecasting system using **LSTM** and **Transformer** architectures in **PyTorch**.

The models are trained on real-world retail sales data from Kaggle's **Store Sales - Time Series Forecasting** dataset. The project uses approximately **400,000 observations** and performs time-series feature engineering, chronological data splitting, sequence generation, model training, and comparative evaluation.

The main objective is to compare the forecasting performance of:

* Long Short-Term Memory (LSTM)
* Transformer Encoder

---

## Project Objectives

* Build deep learning models for retail demand forecasting.
* Use approximately 400,000 real-world sales observations.
* Engineer temporal, lag, rolling, and external features.
* Create sequential time-series inputs for deep learning models.
* Implement LSTM and Transformer models using PyTorch.
* Compare both models using MAE, RMSE, and MAPE.
* Visualize training performance and demand predictions.

---

## Dataset

The project uses the Kaggle **Store Sales - Time Series Forecasting** dataset.

The dataset contains historical sales information from multiple stores and product families.

### Main Data Sources

* `train.csv` - Historical sales data
* `stores.csv` - Store information
* `oil.csv` - Daily oil prices
* `holidays_events.csv` - Holiday and event information
* `transactions.csv` - Store transaction data

### Main Columns

| Column         | Description                         |
| -------------- | ----------------------------------- |
| `date`         | Date of the observation             |
| `store_nbr`    | Store identifier                    |
| `family`       | Product family                      |
| `sales`        | Target variable representing demand |
| `onpromotion`  | Number of promoted products         |
| `transactions` | Number of transactions              |
| `dcoilwtico`   | Oil price                           |
| `is_holiday`   | Holiday indicator                   |

Approximately **400,000 observations** were used for model development.

---

## Feature Engineering

More than 20 features were engineered from the raw data.

### 1. Lag Features

Historical demand was captured using:

* Lag 1
* Lag 7
* Lag 14
* Lag 28

Example:

```text
Sales at Day t-7 → Helps predict Sales at Day t
```

---

### 2. Rolling Statistics

Rolling historical demand statistics were created using:

* 7-day rolling mean
* 14-day rolling mean
* 28-day rolling mean
* 7-day rolling standard deviation
* 14-day rolling standard deviation

To prevent data leakage, rolling calculations were shifted before calculating statistics.

---

### 3. Calendar Features

The following temporal features were extracted:

* Month
* Day
* Day of week
* Weekend indicator
* Quarter
* Day of year
* Week of year

---

### 4. Cyclical Seasonal Features

Cyclical encoding was used for temporal variables:

* Month sine
* Month cosine
* Day-of-week sine
* Day-of-week cosine

This helps represent cyclical relationships such as December being close to January.

---

### 5. External Features

Additional demand-related features included:

* Promotion information
* Oil prices
* Store transactions
* Holiday indicators
* Store information
* Product family information
* City
* State
* Store type
* Store cluster

---

## Data Preprocessing

The preprocessing pipeline includes:

1. Loading the Kaggle datasets.
2. Converting date columns to datetime format.
3. Merging store, oil, transaction, and holiday information.
4. Handling missing oil prices using forward and backward filling.
5. Encoding categorical variables.
6. Creating lag and rolling features.
7. Removing rows with missing values created during feature engineering.
8. Selecting approximately 400,000 observations.
9. Scaling numerical features using `StandardScaler`.

---

## Train, Validation, and Test Split

A chronological split was used to preserve the temporal structure of the dataset.

```text
Past ----------------------------------------------> Future

|--------------- Train 70% ---------------|
                         |--- Validation 15% ---|
                                               |--- Test 15% ---|
```

The model was never trained on future observations.

---

## Sequence Generation

A sequence length of **30 days** was used.

The models receive:

```text
Day 1
Day 2
Day 3
...
Day 30
   ↓
Predict demand for the next observation
```

Each sequence contains 30 historical time steps and multiple engineered features.

---

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Google Colab

---

## Models

### LSTM

The LSTM model consists of:

* Input features
* 2 LSTM layers
* Hidden size of 128
* Dropout
* Fully connected prediction layers

The LSTM architecture learns sequential dependencies from historical demand data.

---

### Transformer

The Transformer model consists of:

* Input projection layer
* Positional encoding
* Transformer Encoder layers
* Multi-head attention
* Feed-forward layers
* Fully connected prediction layer

The Transformer architecture uses self-attention to capture relationships between different time steps in the input sequence.

---

## Training

Both models were trained using:

* PyTorch
* MSE Loss
* AdamW optimizer
* Learning rate scheduling
* Gradient clipping
* Early stopping

The best model based on validation loss was saved during training.

---

## Evaluation Metrics

The models were evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual demand.

[
MAE = \frac{1}{n}\sum |y-\hat{y}|
]

---

### Root Mean Squared Error (RMSE)

Penalizes larger forecasting errors more heavily.

[
RMSE = \sqrt{\frac{1}{n}\sum(y-\hat{y})^2}
]

---

### Mean Absolute Percentage Error (MAPE)

Measures forecasting error as a percentage.

[
MAPE =
\frac{1}{n}
\sum
\left|
\frac{y-\hat{y}}{y}
\right|
\times 100
]

> **Note:** The Store Sales dataset contains zero and very small sales values. Therefore, MAPE can become unstable and produce very large values when actual demand is close to zero. Zero-valued targets were excluded during MAPE calculation to avoid division by zero.

---

## Model Comparison

Current experimental results:

| Model       |   MAE |   RMSE |                                     MAPE |
| ----------- | ----: | -----: | ---------------------------------------: |
| LSTM        | 73.61 | 289.82 | Very high due to near-zero demand values |
| Transformer | 84.49 | 261.86 | Very high due to near-zero demand values |

### Observations

* The **LSTM achieved lower MAE**, indicating better average absolute forecasting accuracy.
* The **Transformer achieved lower RMSE**, indicating comparatively better handling of some larger errors.
* MAPE values were unstable because percentage-based error metrics are highly sensitive to near-zero actual sales.

---

## Prediction Visualization

The project also visualizes:

* Actual demand
* LSTM predictions
* Transformer predictions

This allows direct visual comparison between predicted and actual demand over time.

---

## Project Workflow

```text
Raw Kaggle Data
       │
       ▼
Data Cleaning and Merging
       │
       ▼
Feature Engineering
       │
       ├── Lag Features
       ├── Rolling Statistics
       ├── Calendar Features
       ├── Seasonal Features
       └── External Features
       │
       ▼
Chronological Data Split
       │
       ▼
Feature Scaling
       │
       ▼
30-Day Sequence Generation
       │
       ├───────────────┐
       ▼               ▼
      LSTM        Transformer
       │               │
       └───────┬───────┘
               ▼
        Model Evaluation
               │
               ▼
      MAE / RMSE / MAPE
               │
               ▼
       Model Comparison
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the project in Google Colab or Jupyter Notebook.

### 3. Download the Store Sales dataset from Kaggle.

Place the dataset files in the appropriate data directory.

### 4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tqdm joblib torch
```

### 5. Run the notebook sequentially

The notebook performs:

* Data loading
* Data preprocessing
* Feature engineering
* Sequence generation
* LSTM training
* Transformer training
* Model evaluation
* Model comparison
* Prediction visualization

---

## Future Improvements

Possible future improvements include:

* Using WMAPE or SMAPE for more robust percentage-based evaluation.
* Improving handling of zero-demand observations.
* Hyperparameter tuning.
* Testing different sequence lengths.
* Using product/store embeddings.
* Adding attention visualization.
* Implementing multi-step demand forecasting.
* Adding FastAPI deployment.
* Creating an automated model retraining pipeline.

---

## Key Learnings

This project demonstrates:

* Practical time-series preprocessing.
* Demand forecasting with real-world retail data.
* Feature engineering for time-series models.
* Building custom PyTorch datasets.
* Sequential data modeling with LSTMs.
* Attention-based forecasting with Transformers.
* Chronological validation for time-series data.
* Comparative evaluation of deep learning architectures.

---

## Author

**Aditya Singh**

Deep Learning | Machine Learning | Time Series Forecasting | PyTorch
