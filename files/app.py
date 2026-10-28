"""Streamlit app for the Used Car Price Estimator.
Run with: streamlit run app.py"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from estimator import CURRENT_YEAR, MODEL_PATH, estimate_price, format_inr, load_bundle

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="wide")
st.title("🚗 Used Car Price Estimator")
st.caption("Enter the car's details to get an estimated resale price.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()
cats = bundle["categories"]
y_min, y_max = bundle["year_range"]

tab1, tab2 = st.tabs(["Estimate Price", "Model Performance"])

with tab1:
    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Brand", cats["brand"])
    year = c2.slider("Manufacturing year", y_min, y_max, min(2018, y_max))
    fuel = c3.selectbox("Fuel type", cats["fuel"])

    c4, c5, c6 = st.columns(3)
    transmission = c4.selectbox("Transmission", cats["transmission"])
    owner = c5.selectbox("Ownership", cats["owner"])
    engine_cc = c6.select_slider("Engine (cc)", [800, 1000, 1200, 1500, 1800, 2000, 2500, 3000], value=1200)

    km = st.number_input("Kilometres driven", 0, 500000, 40000, step=1000)

    car = dict(brand=brand, year=year, fuel=fuel, transmission=transmission,
               owner=owner, engine_cc=engine_cc, km_driven=km)

    if st.button("Estimate Price", type="primary"):
        res = estimate_price(bundle, **car)
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated price", format_inr(res["price"]))
        m2.metric("Likely range (low)", format_inr(res["low"]))
        m3.metric("Likely range (high)", format_inr(res["high"]))

        # Depreciation curve: same car, different manufacturing years
        years = list(range(y_min, y_max + 1))
        prices = []
        for y in years:
            c = dict(car, year=y, km_driven=int((CURRENT_YEAR - y) * (km / max(CURRENT_YEAR - year, 1))))
            prices.append(estimate_price(bundle, **c)["price"] / 1e5)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(years, prices, marker="o")
        ax.axvline(year, color="red", linestyle="--", label="Your car")
        ax.set_xlabel("Manufacturing year")
        ax.set_ylabel("Price (₹ Lakh)")
        ax.set_title("How price changes with age (same car, same yearly usage)")
        ax.legend()
        st.pyplot(fig)

with tab2:
    st.write(f"Best model: **{bundle['best_name']}**")
    st.dataframe(bundle["metrics"], use_container_width=True)
    st.caption("R² closer to 1 is better. MAE/RMSE = error in rupees; MAPE = average % error (used for the price range).")
