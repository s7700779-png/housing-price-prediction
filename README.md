# 🏠 Housing Price Prediction & Future Price Increment Prediction

A beginner-friendly, college-level **Machine Learning project built with Python** that predicts the current selling price of a house and estimates how its price may change over the next 1–5 years.

The project contains two separate regression problems:

1. **Current House Price Prediction** — predicts the estimated current value of a house using property and location features.
2. **Future Price Growth Prediction** — estimates the expected annual price growth rate and uses it to calculate a scenario-based future price for 1–5 years.

> **Important:** The future-price result is a scenario estimate, not a guaranteed market forecast.

The project also includes data preprocessing, exploratory data analysis, model comparison, evaluation metrics, saved models, visualizations, and an optional Streamlit interface.

**Tested with:** Python 3.12, pandas 3.0, scikit-learn 1.8

---

## 📌 Project Objective

The goal of this project is to build a machine learning system that can:

* Predict the current price of a house.
* Estimate annual housing-price growth.
* Calculate an estimated price after 1–5 years.
* Compare multiple regression algorithms.
* Handle missing values, duplicates, inconsistent data, and outliers.
* Evaluate models using multiple performance metrics.
* Save trained models for later predictions.
* Visualize model performance and prediction results.

The project intentionally treats **current-price prediction** and **future-growth estimation** as two different ML tasks.

A normal house-price regression model predicts the value of a property based on available features. It does not directly forecast the future because the dataset is primarily a snapshot of houses rather than a true time series.

Therefore, a second model is trained to estimate how the annual growth rate varies according to factors such as economic growth, historical price trends, location quality, accessibility, and property characteristics.

---

# 📁 Project Structure

```text
housing-price-prediction/
│
├── data/
│   ├── housing_data.csv
│   └── historical_price_index.csv
│
├── models/
│   ├── housing_price_model.pkl
│   ├── future_growth_model.pkl
│   ├── model_metadata.json
│   ├── price_model_comparison.csv
│   └── growth_model_comparison.csv
│
├── notebooks/
│   └── housing_analysis.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   └── prediction.py
│
├── app.py
├── requirements.txt
└── README.md
```

### Folder and File Description

| File / Folder        | Purpose                                           |
| -------------------- | ------------------------------------------------- |
| `data/`              | Contains the datasets                             |
| `models/`            | Contains trained ML models and evaluation results |
| `notebooks/`         | Jupyter notebook for analysis                     |
| `reports/figures/`   | Generated EDA and evaluation charts               |
| `data_generation.py` | Generates the synthetic dataset                   |
| `preprocessing.py`   | Cleans and transforms the data                    |
| `eda.py`             | Performs exploratory data analysis                |
| `train_model.py`     | Trains and compares regression models             |
| `prediction.py`      | Loads models and performs predictions             |
| `app.py`             | Optional Streamlit interface                      |
| `requirements.txt`   | Required Python packages                          |

---

# ⚙️ Installation and Setup

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd housing-price-prediction
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

All commands should be executed from the **project root directory**.

## Step 1 — Generate the Dataset

```bash
python src/data_generation.py
```

This generates:

```text
data/housing_data.csv
```

The project uses a synthetic dataset containing approximately **10,100 rows**, including intentionally duplicated and inconsistent records for demonstrating data cleaning.

---

## Step 2 — Train and Compare Models

```bash
python src/train_model.py
```

This step:

* Loads the dataset.
* Cleans the data.
* Performs feature engineering.
* Splits the data into training and testing sets.
* Trains four regression algorithms.
* Evaluates their performance.
* Performs cross-validation.
* Selects the models according to the defined selection rule.
* Saves the trained models.
* Generates evaluation charts.

The trained models are saved inside:

```text
models/
```

---

## Step 3 — Make Predictions

```bash
python src/prediction.py
```

The prediction script loads the saved models and demonstrates:

* Current house-price prediction.
* Future-price estimation.
* Expected price increase.
* Expected percentage growth.

---

## Step 4 — Optional Streamlit Interface

The Streamlit interface is optional.

```bash
streamlit run app.py
```

It provides an interactive interface for entering house details and viewing predictions and charts.

---

# 📊 Dataset

The project uses a **synthetic housing dataset** created specifically for demonstrating a complete ML workflow.

The dataset intentionally contains common data-quality problems such as:

* Missing values.
* Duplicate records.
* Invalid values.
* Inconsistent category names.
* Extreme outliers.
* Different numerical scales.

This makes the project useful for demonstrating real-world preprocessing techniques.

---

# 🏠 Features

The model uses property, location, accessibility, and economic features.

| Feature                      | Description                                  |
| ---------------------------- | -------------------------------------------- |
| `area_sqft`                  | House area in square feet                    |
| `bedrooms`                   | Number of bedrooms                           |
| `bathrooms`                  | Number of bathrooms                          |
| `age_years`                  | Age of the property                          |
| `location_score`             | Location desirability score from 1–10        |
| `distance_to_city_km`        | Distance from city centre                    |
| `distance_to_metro_km`       | Distance from nearest metro/public transport |
| `school_rating`              | Nearby school quality rating                 |
| `hospital_distance_km`       | Distance from nearest hospital               |
| `crime_rate`                 | Crime incidents per 1,000 residents          |
| `parking_spaces`             | Number of parking spaces                     |
| `economic_growth_rate`       | Regional economic growth percentage          |
| `historical_price_trend_pct` | Recent local housing-price trend             |
| `property_type`              | Apartment, Independent House, or Villa       |

---

# 🎯 Target Variables

The project works with two main targets.

### Model 1 — Current Price

```text
current_price
```

This model estimates the current value of the property.

### Model 2 — Annual Growth Rate

The dataset contains:

```text
years_ahead
future_price
price_growth_percent
```

The training process converts these values into an annualized growth rate:

```text
annual_growth_rate =
((future_price / current_price) ^ (1 / years) - 1) × 100
```

This allows different prediction horizons to be compared using the same annual growth metric.

---

# 🧹 Data Preprocessing

The project demonstrates several important preprocessing techniques.

### 1. Duplicate Removal

Duplicate records are removed so that repeated houses do not receive additional influence during training.

### 2. Missing Values

Numerical features use **median imputation**, while categorical features use the **most frequent value**.

### 3. Invalid Values

Impossible values such as:

* Negative age.
* Zero bedrooms.
* Invalid location scores.
* Unrealistic crime rates.

are converted into missing values and subsequently handled during preprocessing.

### 4. Outlier Handling

The project uses the **Interquartile Range (IQR)** method to identify unusual values.

Extreme feature values are capped, while severe price-per-square-foot anomalies can be removed.

### 5. Categorical Encoding

Categorical features such as `property_type` and `age_group` are converted using **One-Hot Encoding**.

### 6. Feature Scaling

`StandardScaler` is used to standardize numerical features.

### 7. Feature Engineering

Additional features are created to improve the representation of the property:

```text
total_rooms
area_per_bedroom
accessibility_score
neighbourhood_quality
age_group
```

---

# 🔄 Machine Learning Pipeline

The preprocessing operations are implemented inside a **scikit-learn Pipeline**.

The general workflow is:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Missing Value Handling
     ↓
Outlier Treatment
     ↓
Encoding
     ↓
Scaling
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Saved Model
```

Keeping preprocessing inside the pipeline helps ensure that transformations are fitted using the training data and then consistently applied to unseen data.

---

# ✂️ Train-Test Split

The dataset uses an **80/20 train-test split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Approximately:

* **80%** → training data
* **20%** → testing data

The testing data is kept separate so that model performance can be evaluated on previously unseen houses.

---

# 🎲 Why `random_state=42`?

`random_state=42` makes the random train-test split reproducible.

This means that when the code is executed again using the same dataset and configuration, the same records are assigned to the training and testing sets.

The value `42` itself has no special mathematical significance. Any fixed integer can be used.

---

# 🤖 Regression Models

Four regression algorithms are compared for both prediction tasks.

## 1. Linear Regression

Linear Regression models the relationship between input features and the target using a linear equation:

```text
price = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

It is fast and relatively easy to interpret, but it may not capture complex non-linear relationships.

---

## 2. Decision Tree Regressor

A Decision Tree repeatedly divides the data using feature-based conditions such as:

```text
area_sqft > 1500
```

It can model non-linear relationships but may overfit if the tree becomes too complex.

---

## 3. Random Forest Regressor

Random Forest combines predictions from multiple decision trees.

This generally makes it more stable than a single decision tree and allows it to capture non-linear relationships and feature interactions.

---

## 4. Gradient Boosting Regressor

Gradient Boosting builds trees sequentially.

Each new tree attempts to reduce the errors made by the previous trees.

It can perform well on structured/tabular datasets but requires more computation than simple linear models.

---

# 📈 Model Evaluation

The project evaluates models using several metrics.

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual values.

For example:

```text
MAE = ₹5,59,551
```

means the average absolute prediction error is approximately ₹5.6 lakh.

---

### MSE — Mean Squared Error

Squares each prediction error before calculating the average.

Large errors therefore have a greater effect on the metric.

---

### RMSE — Root Mean Squared Error

RMSE is the square root of MSE.

Because it is expressed in the same units as the target, it is easier to interpret for house prices.

---

### R² — Coefficient of Determination

R² measures how much of the variation in the target is explained by the model.

A value closer to 1 indicates a stronger fit to the evaluated data.

However, R² alone does not reveal whether a model is overfitting.

Therefore, this project also considers:

* Train R².
* Test R².
* Cross-validation R².
* RMSE.
* MAE.

---

# 🏆 Model Selection Rule

The project does **not select a model using R² alone**.

The selection rule is:

> Select the model with the lowest test RMSE among models whose train R² − test R² gap is no greater than 0.10.

This provides a simple check against excessive overfitting.

Other metrics such as MAE, cross-validation R², training time, and interpretability are also reported.

---

# 📊 Current Price Model Results

Test set: approximately **20% of the cleaned dataset**.

| Model                 |      MAE (₹) |         MSE |      RMSE (₹) |        R² | Train R² | 5-Fold CV R² |
| --------------------- | -----------: | ----------: | ------------: | --------: | -------: | -----------: |
| Linear Regression     |    11,32,769 |     3.48e12 |     18,65,569 |     0.848 |    0.848 |        0.847 |
| Decision Tree         |     9,93,648 |     2.54e12 |     15,95,091 |     0.889 |    0.941 |        0.884 |
| Random Forest         |     6,50,210 |     1.33e12 |     11,52,833 |     0.942 |    0.986 |        0.946 |
| **Gradient Boosting** | **5,59,551** | **1.06e12** | **10,28,686** | **0.954** |    0.981 |        0.966 |

Based on the project's selection rule, **Gradient Boosting** is selected for the current-price task.

---

# 📈 Future Growth Model Results

| Model                 |       MAE |       MSE |      RMSE |        R² | Train R² | 5-Fold CV R² |
| --------------------- | --------: | --------: | --------: | --------: | -------: | -----------: |
| **Linear Regression** | **0.580** | **0.535** | **0.731** | **0.815** |    0.820 |        0.819 |
| Decision Tree         |     0.735 |     0.867 |     0.931 |     0.700 |    0.795 |        0.690 |
| Random Forest         |     0.611 |     0.590 |     0.768 |     0.796 |    0.944 |        0.795 |
| Gradient Boosting     |     0.591 |     0.552 |     0.743 |     0.809 |    0.849 |        0.809 |

Based on the project's selection rule, **Linear Regression** is selected for the annual-growth task.

The Random Forest model shows a larger difference between training and test performance, which is consistent with an overfitting concern under the project's predefined threshold.

---

# 🔮 Future Price Calculation

After predicting the annual growth rate, the future price is calculated using compound growth.

```text
Future Price =
Current Price × (1 + Annual Growth Rate / 100) ^ Years
```

Then:

```text
Expected Increase =
Future Price − Current Price
```

and:

```text
Growth Percentage =
((Future Price − Current Price) / Current Price) × 100
```

For example, if the estimated annual growth rate is 5%:

```text
Current Price = ₹50,00,000
Years = 5

Future Price =
₹50,00,000 × (1.05)^5
```

The resulting value represents a **scenario estimate** based on the model's predicted growth rate.

---

# 🧮 Example Prediction

An example output from the prediction script may look like:

```text
Current Price:                 ₹54,15,875
Predicted Price After 5 Years: ₹73,37,237
Expected Increase:             ₹19,21,362
Expected Growth:               35.5%
```

The values are generated by the trained models and are not hard-coded.

---

# 📊 Visualizations

The project generates several charts inside:

```text
reports/figures/
```

These include:

* Actual vs Predicted Price.
* Residual plots.
* Model comparison charts.
* Feature importance.
* Growth-model evaluation.
* Prediction comparisons.

The Streamlit interface also provides:

* Year 0 → Year 5 line chart.
* Current vs predicted future price bar chart.

---

# 🌐 Optional Streamlit Interface

The project includes an optional Streamlit interface.

Run:

```bash
streamlit run app.py
```

The interface provides:

### 1. House Details

Users can enter property and location information.

### 2. Current Price Prediction

The trained price model estimates the current house value.

### 3. Future Price Prediction

Users can select a future horizon from:

```text
1 year → 5 years
```

The system displays:

* Current estimated price.
* Future estimated price.
* Expected increase.
* Expected growth percentage.

Users can also enter a known current house price instead of using the model's current-price estimate.

### 4. Visualization

The interface displays the estimated price trajectory and a current-versus-future comparison.

---

# ⚠️ Limitations

## 1. Synthetic Dataset

The dataset is artificially generated.

The relationships between features and prices were designed for demonstration purposes, so performance on real-world housing data may be substantially different.

---

## 2. Not a True Time-Series Forecast

The dataset represents houses rather than a continuous historical record of market prices.

Therefore, the future-growth model estimates differences in expected growth between properties based on their characteristics.

It does not directly predict future events such as:

* Interest-rate changes.
* Government policies.
* Economic shocks.
* Market crashes.
* Unexpected infrastructure development.
* Changes in local demand.

---

## 3. Constant Growth Assumption

The future-price calculation assumes that the predicted annual growth rate remains constant over the selected period.

Real housing markets do not necessarily grow at a constant rate.

---

## 4. Prediction Range

Predictions should generally be interpreted within the feature ranges represented in the training dataset.

For example, if the training data contains properties primarily between approximately 350 and 6,000 square feet, predictions for properties far outside that range may be less reliable.

---

## 5. Not for Financial Decisions

This project is intended for **educational and demonstration purposes**.

The predictions should not be treated as guaranteed property valuations or financial advice.

---

# 🕐 Future Time-Series Extension

A more realistic future-price forecasting system would require actual historical housing data.

For example:

```text
Monthly locality price index
        ↓
Historical time series
        ↓
Time-series forecasting model
        ↓
Future locality price index
        ↓
House-level price adjustment
```

Possible forecasting approaches include:

* ARIMA.
* SARIMA.
* Prophet.
* LSTM.

A hybrid approach could estimate the future locality index first and then adjust the individual property's current value:

```text
Future Price =
Current Price × Future Locality Index / Current Locality Index
```

For time-series forecasting, the model should use **time-based train/test splits** rather than randomly mixing historical and future observations.

---

# 🚀 Possible Future Improvements

Some possible extensions include:

* Use real-world housing datasets.
* Predict `log(price)` instead of raw price.
* Perform hyperparameter tuning with `GridSearchCV`.
* Add SHAP-based model explanations.
* Add more location-level features.
* Incorporate actual historical housing prices.
* Build a locality-level time-series forecasting model.
* Combine house-level regression with locality-level forecasting.
* Deploy the model using Streamlit or another deployment platform.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Joblib

### Optional Interface

* Streamlit

### Development

* Jupyter Notebook
* VS Code / PyCharm
* Git & GitHub

---

# 📦 Requirements

The main dependencies are:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

For the optional Streamlit interface:

```text
streamlit
```

---

# 🎓 Learning Outcomes

This project demonstrates a complete beginner-to-intermediate machine learning workflow:

```text
Data Generation
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Model Comparison
      ↓
Model Evaluation
      ↓
Model Selection
      ↓
Model Persistence
      ↓
Prediction
      ↓
Future Scenario Estimation
```

Through this project, the following ML concepts are demonstrated:

* Regression.
* Train-test splitting.
* Data preprocessing.
* Missing-value handling.
* Outlier treatment.
* Feature engineering.
* One-hot encoding.
* Feature scaling.
* Cross-validation.
* Model comparison.
* Overfitting detection.
* Model persistence.
* Prediction pipelines.
* Compound growth calculations.

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to estimate a property's current price and construct a scenario for its potential future value.

The two prediction tasks are deliberately separated:

```text
House Features
     │
     ├──────────────► Current Price Model
     │                       │
     │                       ▼
     │                Current Price
     │
     └──────────────► Growth Model
                             │
                             ▼
                    Annual Growth Rate
                             │
                             ▼
                  Compound Growth Formula
                             │
                             ▼
                    Future Price Estimate
```

The project is primarily designed to demonstrate **data preprocessing, regression modeling, evaluation, model selection, and prediction workflows** in Python.
