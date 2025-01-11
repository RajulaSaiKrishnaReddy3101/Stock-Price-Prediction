# Stock--Price-Prediction


This project demonstrates the use of machine learning techniques to predict stock prices. It involves data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation to identify the best-performing model for stock price prediction.

---

## Key Features
- **Data Preprocessing**: Handled missing values using median imputation and created time-based features from the date column.
- **Feature Selection**: Identified key features correlated to the target variable using a correlation matrix and `SelectKBest`.
- **Models Used**:
  - K-Nearest Neighbors (KNN) [Best Model: R² = 0.986, RMSE = 1.43]
  - Random Forest Regressor
  - XGBoost Regressor
  - Linear Regression
- **Hyperparameter Tuning**: Optimized model parameters using `GridSearchCV`.
- **Model Deployment**: Saved the best-performing model (KNN) using `joblib` for future predictions.

---

## Dataset
The dataset contains time-series stock price data with features such as:
- `Open`, `High`, `Low`, `Close` prices
- `Volume`, `Day`, `Month`, and `Day of the Week`

### Data Source
The dataset is sourced from [Kaggle](https://kaggle.com), specifically the "Stock Price Prediction" dataset.

---

## Project Workflow
1. **Data Preprocessing**:
   - Converted date column to datetime format.
   - Extracted additional features like `month`, `day`, and `day_of_week`.
   - Filled missing values with the median.

2. **Exploratory Data Analysis**:
   - Visualized correlations between features using a heatmap.
   - Selected features with high correlation to the target variable.

3. **Model Training and Evaluation**:
   - Trained and tested models using `train_test_split`.
   - Evaluated models with metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score.

4. **Hyperparameter Tuning**:
   - Performed grid search for KNN, Random Forest, and XGBoost models.

5. **Deployment**:
   - Saved the best model (KNN with optimized hyperparameters) as `best_model.pkl`.

---

## Results
- **Best Model**: K-Nearest Neighbors (KNN)
  - **Parameters**: `n_neighbors=7`, `p=1`, `weights='distance'`
  - **Performance**:
    - R²: 0.986
    - RMSE: 1.43
    - MAE: 1.09

---

## Dependencies
The following libraries are required to run this project:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `joblib`

Install dependencies using:
```bash
pip install -r requirements.txt
