"""Generates a synthetic used-car dataset (prices in INR) so the project runs
out of the box. Replace with a real dataset (e.g. Kaggle 'Vehicle dataset' /
'CarDekho') for realistic results."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000
CURRENT_YEAR = 2026

# brand: (base new price in INR for a typical model, brand premium)
brands = {
    "Maruti": 700000, "Hyundai": 900000, "Tata": 850000, "Honda": 1200000,
    "Toyota": 1800000, "Mahindra": 1400000, "Kia": 1300000, "BMW": 5500000,
}
brand_list = list(brands)
brand_probs = [0.26, 0.20, 0.12, 0.10, 0.09, 0.10, 0.08, 0.05]

df = pd.DataFrame({
    "brand": rng.choice(brand_list, N, p=brand_probs),
    "year": rng.integers(2008, CURRENT_YEAR, N),
    "fuel": rng.choice(["Petrol", "Diesel", "CNG", "Electric"], N, p=[0.50, 0.38, 0.08, 0.04]),
    "transmission": rng.choice(["Manual", "Automatic"], N, p=[0.68, 0.32]),
    "owner": rng.choice(["First", "Second", "Third+"], N, p=[0.62, 0.28, 0.10]),
    "engine_cc": rng.choice([800, 1000, 1200, 1500, 1800, 2000, 2500, 3000], N,
                            p=[0.05, 0.15, 0.30, 0.22, 0.10, 0.10, 0.05, 0.03]),
})
age = CURRENT_YEAR - df["year"]
df["km_driven"] = (age * rng.normal(11000, 3500, N)).clip(1000, 300000).round(-2).astype(int)

base = df["brand"].map(brands) * (1 + (df["engine_cc"] - 1200) / 4000)
price = base * np.exp(-0.095 * age)                      # yearly depreciation
price *= np.exp(-df["km_driven"] / 600000)              # usage wear
price *= df["fuel"].map({"Petrol": 1.0, "Diesel": 1.06, "CNG": 0.9, "Electric": 1.25})
price *= df["transmission"].map({"Manual": 1.0, "Automatic": 1.12})
price *= df["owner"].map({"First": 1.0, "Second": 0.90, "Third+": 0.80})
price *= rng.normal(1.0, 0.07, N)                       # market noise
df["selling_price"] = price.clip(40000).round(-3).astype(int)

df.to_csv("data/used_cars.csv", index=False)
print(f"Saved {len(df)} cars to data/used_cars.csv")
print(df.head())
