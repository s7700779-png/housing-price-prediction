"""
app.py - Streamlit web app for house price + future price estimation.

Run from the project root:
    streamlit run app.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Make the "src" folder importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.prediction import (  # noqa: E402
    estimate_future_price, format_inr, format_short_inr, load_models, models_available,
    predict_current_price, price_trajectory,
)

st.set_page_config(page_title="House Price & Future Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 House Price & Future Price Predictor")
st.caption("College ML project - trained on a synthetic dataset. For learning purposes, not financial advice.")

if not models_available():
    st.error("Trained models were not found. Run `python src/train_model.py` in the project folder, "
             "then restart this app.")
    st.stop()

_, _, meta = load_models()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About the models")
    st.write(f"**Price model:** {meta['price_model']}  \n"
             f"R² = {meta['price_metrics']['R2']:.3f}, MAE = {format_inr(meta['price_metrics']['MAE'])}")
    st.write(f"**Growth model:** {meta['growth_model']}  \n"
             f"R² = {meta['growth_metrics']['R2']:.3f}, RMSE = {meta['growth_metrics']['RMSE']:.2f} "
             "percentage points / year")
    st.info(
        "**Two different things are predicted:**\n\n"
        "1. *Current price* - what the house is worth today, from its features.\n"
        "2. *Future price* - today's price grown at an expected annual rate that depends on the "
        "economy, location, metro access and recent price trend.\n\n"
        "The future figure is a scenario estimate, not a guarantee."
    )

# ---------------------------------------------------------------- Section 1
st.header("1. House details")
col1, col2, col3 = st.columns(3)

with col1:
    area_sqft = st.number_input("Area (sq ft)", min_value=300, max_value=8000, value=1200, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2, step=1)
    age_years = st.number_input("Property age (years)", min_value=0, max_value=60, value=8, step=1)
    property_type = st.selectbox("Property type", meta["property_types"], index=0)

with col2:
    location_score = st.slider("Location score (1-10)", 1.0, 10.0, 7.0, 0.1)
    distance_to_city_km = st.number_input("Distance to city centre (km)", min_value=0.5, max_value=45.0,
                                          value=9.0, step=0.5)
    distance_to_metro_km = st.number_input("Distance to metro / public transport (km)", min_value=0.1,
                                           max_value=25.0, value=1.5, step=0.1)
    hospital_distance_km = st.number_input("Distance to hospital (km)", min_value=0.3, max_value=25.0,
                                           value=2.0, step=0.1)
    parking_spaces = st.number_input("Parking spaces", min_value=0, max_value=4, value=1, step=1)

with col3:
    school_rating = st.slider("Nearby school rating (1-10)", 1.0, 10.0, 7.0, 0.1)
    crime_rate = st.slider("Crime rate (per 1,000 residents, 0-100)", 0.0, 100.0, 25.0, 1.0)
    economic_growth_rate = st.slider("Regional economic growth (% per year)", 0.5, 10.0, 6.5, 0.1)
    historical_price_trend_pct = st.slider("Recent local price trend (% per year)", -3.0, 14.0, 5.0, 0.1)

house = {
    "area_sqft": float(area_sqft), "bedrooms": float(bedrooms), "bathrooms": float(bathrooms),
    "age_years": float(age_years), "location_score": float(location_score),
    "distance_to_city_km": float(distance_to_city_km), "distance_to_metro_km": float(distance_to_metro_km),
    "school_rating": float(school_rating), "hospital_distance_km": float(hospital_distance_km),
    "crime_rate": float(crime_rate), "parking_spaces": float(parking_spaces),
    "economic_growth_rate": float(economic_growth_rate),
    "historical_price_trend_pct": float(historical_price_trend_pct),
    "property_type": property_type,
}

# ---------------------------------------------------------------- Section 2
st.header("2. Current price prediction")
estimated_price = predict_current_price(house)
st.success(f"Estimated Current House Price: **{format_inr(estimated_price)}**")

# ---------------------------------------------------------------- Section 3
st.header("3. Future price prediction")
left, right = st.columns(2)
with left:
    years = st.selectbox("Prediction period:", [1, 2, 3, 4, 5], index=4,
                         format_func=lambda y: f"{y} year" + ("s" if y > 1 else ""))
with right:
    use_own_price = st.checkbox("I know the actual current price - use it instead of the estimate")
    if use_own_price:
        current_price = st.number_input("Current price (₹)", min_value=100000.0,
                                        value=float(round(estimated_price, -3)), step=50000.0)
    else:
        current_price = estimated_price

result = estimate_future_price(house, current_price=current_price, years=years)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Estimated Price", format_inr(result["current_price"]))
m2.metric("Future Predicted Price", format_inr(result["future_price"]),
          delta=f"{result['growth_percentage']:.1f}%")
m3.metric("Expected Price Increase", format_inr(result["price_increase"]))
m4.metric("Expected Growth %", f"{result['growth_percentage']:.1f}%")

st.write(
    f"Expected yearly appreciation for this house: **{result['annual_growth_rate_pct']:.2f}% per year**. "
    f"A typical-error range for the price after {years} year(s) is "
    f"{format_inr(result['future_price_low'])} to {format_inr(result['future_price_high'])}."
)

# ---------------------------------------------------------------- Section 4
st.header("4. Visualization")
trajectory = price_trajectory(house, current_price=current_price, max_years=5)
chart_left, chart_right = st.columns(2)

with chart_left:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(trajectory["Year"], trajectory["Low"] / 1e5, trajectory["High"] / 1e5,
                    alpha=0.2, color="#3b7dd8", label="Typical-error range")
    ax.plot(trajectory["Year"], trajectory["Price"] / 1e5, marker="o", lw=2.5, color="#3b7dd8",
            label="Predicted price")
    selected = trajectory[trajectory["Year"] == years].iloc[0]
    ax.scatter([years], [selected["Price"] / 1e5], s=140, color="crimson", zorder=5,
               label=f"Selected: year {years}")
    for _, row in trajectory.iterrows():
        ax.annotate(format_short_inr(row["Price"]), (row["Year"], row["Price"] / 1e5),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xticks(trajectory["Year"])
    ax.set_xticklabels(["Year 0\n(now)"] + [f"Year {y}" for y in range(1, 6)])
    ax.set_ylabel("Price (₹ lakh)")
    ax.set_title("Predicted price trend over 5 years")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

with chart_right:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Current price", f"Predicted price\n(after {years} yr)"]
    values = [result["current_price"], result["future_price"]]
    bars = ax.bar(labels, [v / 1e5 for v in values], color=["#8d99ae", "#2a9d8f"], width=0.5)
    for bar, value in zip(bars, values):
        ax.annotate(format_short_inr(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Price (₹ lakh)")
    ax.set_title(f"Current vs predicted price (+{result['growth_percentage']:.1f}%)")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with st.expander("How is the future price calculated?"):
    st.markdown(
        """
        1. A **growth model** predicts the expected *annual* price growth (%) from the house's features -
           especially economic growth, recent price trend, location score and distance to metro.
        2. The price is **compounded**: `future = current × (1 + rate/100) ^ years`.
        3. `Increase = future − current` and `Growth % = (future − current) / current × 100`.

        **Limits:** the models are trained on a synthetic dataset with no real time-series. The result is
        a scenario estimate - real prices also depend on interest rates, policy and events that no model here sees.
        """
    )
