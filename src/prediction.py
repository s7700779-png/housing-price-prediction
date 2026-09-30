"""
prediction.py
-------------
Functions used by the Streamlit app (and usable from a notebook / terminal).

HOW THE FUTURE PRICE IS CALCULATED
    1. price model   : house features            -> estimated CURRENT price
    2. growth model  : house features (incl.
                       economic growth rate,
                       historical price trend,
                       location, metro access...) -> expected ANNUAL growth rate (%)
    3. compounding   : future = current * (1 + rate/100) ** years

    price_increase    = future_price - current_price
    growth_percentage = (future_price - current_price) / current_price * 100

IMPORTANT: this is a scenario estimate learned from (synthetic) cross-sectional
data. It is not a market forecast. See README.md, "Limitations".

Try it:
    python src/prediction.py
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import MODEL_INPUT_COLUMNS  # noqa: E402  (needed so joblib can unpickle)

MODELS_DIR = PROJECT_ROOT / "models"
PRICE_MODEL_PATH = MODELS_DIR / "housing_price_model.pkl"
GROWTH_MODEL_PATH = MODELS_DIR / "future_growth_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Safety limits for the predicted yearly growth rate (% per year)
MIN_RATE, MAX_RATE = -5.0, 20.0


def models_available() -> bool:
    """True if the trained model files exist."""
    return PRICE_MODEL_PATH.exists() and GROWTH_MODEL_PATH.exists() and METADATA_PATH.exists()


@lru_cache(maxsize=1)
def load_models():
    """Load (and cache) the two trained pipelines and the metadata."""
    if not models_available():
        raise FileNotFoundError(
            "Trained models not found. Run `python src/train_model.py` from the project root first."
        )
    price_model = joblib.load(PRICE_MODEL_PATH)
    growth_model = joblib.load(GROWTH_MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return price_model, growth_model, metadata


# ----------------------------------------------------------------- formatting
def format_inr(amount: float) -> str:
    """Indian digit grouping:  6000000 -> '₹60,00,000'."""
    sign = "-" if amount < 0 else ""
    digits = str(int(round(abs(amount))))
    if len(digits) <= 3:
        grouped = digits
    else:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        grouped = ",".join(parts + [tail])
    return f"{sign}₹{grouped}"


def format_short_inr(amount: float) -> str:
    """Compact form for chart labels: '₹78.0 L' (lakh) or '₹1.25 Cr' (crore)."""
    if abs(amount) >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr"
    return f"₹{amount / 1e5:.1f} L"


# ----------------------------------------------------------------- predictions
def build_input_frame(house: dict) -> pd.DataFrame:
    """Turn a dict of house details into the one-row DataFrame the models expect."""
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in house]
    if missing:
        raise ValueError(f"Missing house details: {missing}")
    return pd.DataFrame([{c: house[c] for c in MODEL_INPUT_COLUMNS}])


def predict_current_price(house: dict) -> float:
    """Estimated CURRENT price (Rs.) of a house."""
    price_model, _, _ = load_models()
    return float(price_model.predict(build_input_frame(house))[0])


def predict_annual_growth(house: dict) -> float:
    """Expected annual price growth (% per year) for a house."""
    _, growth_model, _ = load_models()
    rate = float(growth_model.predict(build_input_frame(house))[0])
    return min(max(rate, MIN_RATE), MAX_RATE)


def estimate_future_price(house: dict, current_price: float = None, years: int = 5) -> dict:
    """
    Estimate the price after `years` (1-5).

    current_price : the price the user believes the house has today. If None,
                    the price model's estimate is used.
    Returns a dict with the future price, increase, growth % and a +/- 1 RMSE
    uncertainty range (the typical error of the growth model).
    """
    if not 1 <= int(years) <= 5:
        raise ValueError("years must be between 1 and 5")
    years = int(years)

    if current_price is None:
        current_price = predict_current_price(house)
    if current_price <= 0:
        raise ValueError("current_price must be positive")

    _, _, meta = load_models()
    rate = predict_annual_growth(house)
    band = meta["growth_rmse_pct_points"]

    def compound(r):
        return current_price * (1 + r / 100) ** years

    future_price = compound(rate)
    price_increase = future_price - current_price
    growth_percentage = (future_price - current_price) / current_price * 100

    return {
        "years": years,
        "current_price": current_price,
        "annual_growth_rate_pct": rate,
        "future_price": future_price,
        "price_increase": price_increase,
        "growth_percentage": growth_percentage,
        "future_price_low": compound(rate - band),
        "future_price_high": compound(rate + band),
    }


def price_trajectory(house: dict, current_price: float = None, max_years: int = 5) -> pd.DataFrame:
    """Year 0 ... Year `max_years` prices (with low/high range) - used for the line chart."""
    if current_price is None:
        current_price = predict_current_price(house)
    rows = [{"Year": 0, "Price": current_price, "Low": current_price, "High": current_price}]
    for y in range(1, max_years + 1):
        est = estimate_future_price(house, current_price, y)
        rows.append({"Year": y, "Price": est["future_price"],
                     "Low": est["future_price_low"], "High": est["future_price_high"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- demo
if __name__ == "__main__":
    example_house = {
        "area_sqft": 1200, "bedrooms": 3, "bathrooms": 2, "age_years": 8,
        "location_score": 7.5, "distance_to_city_km": 9, "distance_to_metro_km": 1.5,
        "school_rating": 7, "hospital_distance_km": 2, "crime_rate": 25,
        "parking_spaces": 1, "economic_growth_rate": 6.5,
        "historical_price_trend_pct": 5.0, "property_type": "Apartment",
    }
    est = estimate_future_price(example_house, years=5)
    print(f"Current Price:                {format_inr(est['current_price'])}")
    print(f"Predicted Price After 5 Years: {format_inr(est['future_price'])}")
    print(f"Expected Increase:            {format_inr(est['price_increase'])}")
    print(f"Expected Growth:              {est['growth_percentage']:.1f}%")
    print(f"(annual rate {est['annual_growth_rate_pct']:.2f}% ; range "
          f"{format_inr(est['future_price_low'])} - {format_inr(est['future_price_high'])})")
