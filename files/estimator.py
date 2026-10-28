"""Shared helpers: feature engineering, model loading, prediction."""
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/car_price_model.joblib"
CURRENT_YEAR = 2026
CATEGORICAL = ["brand", "fuel", "transmission", "owner"]
NUMERIC = ["car_age", "km_driven", "engine_cc", "km_per_year"]
TARGET = "selling_price"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create car_age and km_per_year from the raw columns."""
    df = df.copy()
    df["car_age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)
    df["km_per_year"] = df["km_driven"] / df["car_age"].clip(lower=1)
    return df


def load_bundle(path: str = MODEL_PATH) -> dict:
    """Bundle = {'model', 'mae', 'metrics', 'categories'}"""
    return joblib.load(path)


def estimate_price(bundle: dict, **car) -> dict:
    """Predict the price for a single car and return a +/- typical-error range (MAPE)."""
    row = add_features(pd.DataFrame([car]))
    price = float(bundle["model"].predict(row[CATEGORICAL + NUMERIC])[0])
    err = price * bundle["mape"]  # error band scales with the price
    return {"price": price, "low": max(price - err, 0), "high": price + err}


def format_inr(value: float) -> str:
    """Format as lakhs / crores, e.g. 4.25 Lakh."""
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    return f"₹{value / 1e5:.2f} Lakh"


class LogPriceModel:
    """Wraps a pipeline trained on log(price) so predict() returns rupees.
    Lives here (not in train_model.py) so joblib can load it from the app."""
    def __init__(self, pipe):
        self.pipe = pipe

    def predict(self, X):
        return np.expm1(self.pipe.predict(X))
