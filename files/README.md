# Used Car Price Estimator

Predicts the resale price of a used car from brand, age, fuel, transmission, ownership,
engine size and kilometres driven. Trains three regression models (Linear, Random Forest,
Gradient Boosting), keeps the best one, and serves it through a Streamlit app.

## Setup
```bash
pip install -r requirements.txt
python generate_sample_data.py     # synthetic data -> data/used_cars.csv
python train_model.py              # compares models, saves models/car_price_model.joblib
streamlit run app.py
```

## Use real data
Use a dataset such as Kaggle's CarDekho / "Vehicle dataset". Rename columns to:
`brand, year, fuel, transmission, owner, engine_cc, km_driven, selling_price`
then run `python train_model.py path/to/cars.csv`.

## Structure
- `estimator.py` – feature engineering (car age, km per year), prediction, price formatting
- `train_model.py` – model comparison (R², MAE, RMSE), log-price target, saving
- `app.py` – Streamlit UI with price estimate, range and depreciation curve
- `generate_sample_data.py` – synthetic starter data

## Ideas to extend
- Add car model/variant, location and service history as features
- Hyperparameter tuning with GridSearchCV; try XGBoost/LightGBM
- Explain predictions with SHAP; deploy on Azure App Service
