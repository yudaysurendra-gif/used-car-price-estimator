"""Train, compare and save the price model.
Usage: python train_model.py [path/to/cars.csv]
CSV columns: brand, year, fuel, transmission, owner, engine_cc, km_driven, selling_price"""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from estimator import CATEGORICAL, MODEL_PATH, NUMERIC, TARGET, LogPriceModel, add_features

csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/used_cars.csv"
df = pd.read_csv(csv_path).dropna()
df = add_features(df)
df = df[df[TARGET] > 0]

X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
# Train on log(price): prices are skewed and errors are proportional.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])

candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                                                   max_depth=4, random_state=42),
}

results, fitted = [], {}
for name, reg in candidates.items():
    pipe = Pipeline([("pre", pre), ("reg", reg)])
    pipe.fit(X_train, np.log1p(y_train))
    pred = np.expm1(pipe.predict(X_test))
    results.append({
        "Model": name,
        "R2": round(r2_score(y_test, pred), 4),
        "MAE": round(mean_absolute_error(y_test, pred)),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred)))),
        "MAPE_%": round(float(np.mean(np.abs(pred - y_test) / y_test) * 100), 1),
    })
    fitted[name] = pipe

res = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
print(res.to_string(index=False))
best = res.loc[0, "Model"]
print(f"\nBest model: {best}")


bundle = {
    "model": LogPriceModel(fitted[best]),
    "mae": float(res.loc[0, "MAE"]),
    "mape": float(res.loc[0, "MAPE_%"]) / 100,
    "metrics": res,
    "best_name": best,
    "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
    "year_range": (int(df["year"].min()), int(df["year"].max())),
}
joblib.dump(bundle, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
