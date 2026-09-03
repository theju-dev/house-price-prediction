# House Price Prediction

## Project Overview

This project builds a machine learning model to predict property price using property-related features from an Indian real estate dataset.

The project includes data cleaning, exploratory analysis, feature engineering, preprocessing, model training, evaluation, error analysis, and model saving.

## Dataset

The dataset contains Indian property listings with information such as:

- Location
- Carpet Area
- Super Area
- Bathrooms
- Balconies
- Furnishing
- Transaction Type
- Car Parking

The target variable is:

- Price (in rupees)

## Project Workflow

1. Load and inspect the dataset
2. Handle missing target values and invalid prices
3. Clean and transform property features
4. Analyze price distribution and locations
5. Create numeric and categorical features
6. Split data into training and testing sets
7. Build preprocessing pipelines
8. Train Linear Regression models
9. Evaluate using MAE, RMSE, and R²
10. Perform prediction error analysis
11. Save the trained model using pickle

## Model

The project currently uses Linear Regression with a scikit-learn preprocessing pipeline.

The preprocessing pipeline handles:

- Missing numeric values
- Missing categorical values
- Categorical encoding

## Model Evaluation

The final model was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Location was added as an important feature to improve the model.

## Project Structure

house_price_prediction/
│
├── data/
├── outputs/
├── house_price_analysis.py
├── house_price_model.pkl
├── README.md
├── requirements.txt
└── .gitignore

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

## Author

House Price Prediction - Machine Learning Practice Project
## Final Model Performance

The final Linear Regression model includes location as an additional categorical feature.

- MAE: approximately 2258.64
- RMSE: approximately 3341.72
- R² Score: approximately 0.463

Adding location significantly improved model performance compared with the earlier baseline models.