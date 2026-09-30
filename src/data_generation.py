"""
data_generation.py
------------------
Creates a realistic SYNTHETIC housing dataset (about 10,000 records).

Why synthetic?
    Real housing data with all the columns we need (crime rate, school rating,
    future prices ...) is hard to find. So we invent it, but we make the
    relationships between columns follow real-world logic:

        larger area          -> higher price
        better location      -> higher price
        older property       -> lower price
        higher crime rate    -> lower price
        far from metro/city  -> lower price
        strong economy       -> faster future price growth ... and so on.

    The relationships are NOT perfectly linear: we add random noise and a few
    interaction effects (e.g. good location AND good school = extra premium).

    After the "clean" data is generated we deliberately break it a little
    (missing values, outliers, duplicates, inconsistent labels) so that the
    preprocessing step has real work to do.

Run:
    python src/data_generation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_RECORDS = 10_000  # before duplicates are added

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

NUMERIC_FEATURES = [
    "area_sqft", "bedrooms", "bathrooms", "age_years", "location_score",
    "distance_to_city_km", "distance_to_metro_km", "school_rating",
    "hospital_distance_km", "crime_rate", "parking_spaces",
    "economic_growth_rate", "historical_price_trend_pct",
]


def generate_clean_data(n_records: int = N_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a clean synthetic dataset (no errors yet)."""
    rng = np.random.default_rng(seed)
    n = n_records

    # ---------------------------------------------------------------- features
    # Property type decides typical size, parking and price premium.
    property_type = rng.choice(
        ["Apartment", "Independent House", "Villa"], size=n, p=[0.60, 0.30, 0.10]
    )
    is_house = property_type == "Independent House"
    is_villa = property_type == "Villa"

    # Location desirability (1-10). Most areas are average, few are top/bottom.
    location_score = np.clip(rng.normal(6.0, 2.0, n), 1, 10).round(1)

    # Good locations are closer to the city centre and to public transport.
    distance_to_city_km = np.clip(rng.normal(22 - 1.8 * location_score, 5, n), 0.5, 45).round(1)
    distance_to_metro_km = np.clip(
        rng.gamma(2.0, 0.6 + 0.12 * distance_to_city_km, n), 0.1, 25
    ).round(1)

    # Area in square feet (log-normal => a few very large homes, like real life).
    median_area = np.where(is_villa, 3200, np.where(is_house, 1800, 1100))
    sigma = np.where(is_villa, 0.25, 0.30)
    area_sqft = np.clip(np.exp(rng.normal(np.log(median_area), sigma)), 350, 6000).round(0)

    # Bigger homes have more rooms.
    bedrooms = np.clip(np.round(area_sqft / 550 + rng.normal(0, 0.6, n)), 1, 6)
    bathrooms = np.clip(np.round(bedrooms * 0.75 + rng.normal(0.5, 0.5, n)), 1, 6)

    age_years = np.clip(rng.gamma(2.2, 6.0, n), 0, 50).round(0)

    school_rating = np.clip(rng.normal(5.5 + 0.3 * (location_score - 6), 1.5, n), 1, 10).round(1)
    hospital_distance_km = np.clip(
        rng.gamma(2.0, 0.4 + 0.08 * distance_to_city_km, n) + 0.3, 0.3, 25
    ).round(1)

    # Crime indicator: incidents per 1,000 residents (0-100). Safer in good areas.
    crime_rate = np.clip(rng.normal(30 - 2.5 * (location_score - 6), 9, n), 2, 90).round(1)

    parking_low = np.where(is_villa, 2, np.where(is_house, 1, 0))
    parking_spaces = rng.integers(parking_low, parking_low + 3).astype(float)

    # Regional economy (annual GDP growth %) and the recent price trend (% per year).
    economic_growth_rate = np.clip(
        rng.normal(6.0 + 0.15 * (location_score - 6), 1.3, n), 0.5, 10
    ).round(2)
    historical_price_trend_pct = np.clip(
        rng.normal(4 + 0.5 * (economic_growth_rate - 6) + 0.35 * (location_score - 6), 1.8, n),
        -3, 14,
    ).round(2)

    # ------------------------------------------------------- current price (Rs.)
    # We model the LOG of price-per-sqft, which makes effects multiplicative
    # ("+10% for a better location") - just like the real market.
    age_penalty = np.where(is_villa, 0.005, 0.010)  # villas keep their land value
    log_price_per_sqft = (
        np.log(4500)                                         # base Rs. 4,500 / sqft
        + 0.10 * (location_score - 6)                        # better location
        + 0.05 * (school_rating - 5.5)                       # better schools
        - 0.006 * (crime_rate - 30)                          # crime hurts
        - 0.012 * distance_to_city_km                        # far from city
        - 0.030 * distance_to_metro_km                       # far from metro
        - 0.010 * hospital_distance_km                       # far from hospital
        - age_penalty * age_years                            # older = cheaper
        + 0.030 * parking_spaces                             # parking premium
        + 0.030 * (bedrooms - 3) + 0.020 * (bathrooms - 2)   # rooms
        + 0.020 * (economic_growth_rate - 6)                 # strong regional economy
        + np.where(is_villa, 0.20, np.where(is_house, 0.08, 0.0))  # property type
        + 0.008 * (location_score - 6) * (school_rating - 5.5)     # INTERACTION
        - 0.05 * np.log(area_sqft / 1200)                    # bulk discount for big homes
        + rng.normal(0, 0.08, n)                             # random market noise
    )
    current_price = (area_sqft * np.exp(log_price_per_sqft) / 1000).round(0) * 1000

    # ---------------------------------------------------- future price (Rs.)
    # Hidden "true" annual appreciation rate (% per year).
    annual_growth = (
        1.5
        + 0.55 * economic_growth_rate                        # economy
        + 0.25 * historical_price_trend_pct                  # momentum / history
        + 0.20 * (location_score - 6)                        # location appreciation
        - 0.10 * distance_to_metro_km                        # infrastructure access
        - 0.03 * distance_to_city_km
        + 0.10 * (school_rating - 5.5)
        - 0.02 * (crime_rate - 30)
        - 0.03 * age_years                                   # property characteristics
        + 0.04 * (economic_growth_rate - 6) * (location_score - 6)  # INTERACTION
        + rng.normal(0, 0.7, n)                              # unpredictable market shocks
    )
    annual_growth = np.clip(annual_growth, -3, 18)

    years_ahead = rng.integers(1, 6, n)  # 1-5 years
    future_price = (current_price * (1 + annual_growth / 100) ** years_ahead / 1000).round(0) * 1000
    price_growth_percent = ((future_price - current_price) / current_price * 100).round(2)

    return pd.DataFrame({
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "location_score": location_score,
        "distance_to_city_km": distance_to_city_km,
        "distance_to_metro_km": distance_to_metro_km,
        "school_rating": school_rating,
        "hospital_distance_km": hospital_distance_km,
        "crime_rate": crime_rate,
        "parking_spaces": parking_spaces,
        "economic_growth_rate": economic_growth_rate,
        "historical_price_trend_pct": historical_price_trend_pct,
        "property_type": property_type,
        "current_price": current_price,
        "years_ahead": years_ahead,
        "future_price": future_price,
        "price_growth_percent": price_growth_percent,
    })


def inject_data_quality_issues(df: pd.DataFrame, seed: int = RANDOM_SEED + 1) -> pd.DataFrame:
    """Add realistic (but not excessive) errors to a clean dataset."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)

    # 1) OUTLIERS ------------------------------------------------------------
    # a) area typed with extra digits (e.g. 1,200 -> 6,000)
    idx = rng.choice(n, size=int(0.004 * n), replace=False)
    df.loc[idx, "area_sqft"] = (df.loc[idx, "area_sqft"].to_numpy() * rng.uniform(4, 6, len(idx))).round(0)
    # b) price entry errors (both prices multiplied so growth stays consistent)
    idx = rng.choice(n, size=int(0.003 * n), replace=False)
    factor = rng.uniform(4, 8, len(idx))
    for col in ["current_price", "future_price"]:
        df.loc[idx, col] = (df.loc[idx, col].to_numpy() * factor / 1000).round(0) * 1000
    # c) impossible crime-rate readings (scale is 0-100)
    idx = rng.choice(n, size=int(0.003 * n), replace=False)
    df.loc[idx, "crime_rate"] = rng.uniform(120, 250, len(idx)).round(1)

    # 2) INCONSISTENT VALUES -------------------------------------------------
    # a) the same category written in different ways
    variants = {
        "Apartment": ["apartment", "APARTMENT", "Apt", "Flat"],
        "Independent House": ["independent house", "House", "INDEPENDENT HOUSE"],
        "Villa": ["villa", "Villa ", "VILLA"],
    }
    idx = rng.choice(n, size=int(0.06 * n), replace=False)
    df["property_type"] = df["property_type"].astype(object)
    col_pos = df.columns.get_loc("property_type")
    for i in idx:
        original = df.iat[i, col_pos]
        df.iat[i, col_pos] = str(rng.choice(variants[original]))
    # b) negative ages, c) location scores outside 1-10, d) zero bedrooms
    idx = rng.choice(n, size=int(0.003 * n), replace=False)
    df.loc[idx, "age_years"] = -df.loc[idx, "age_years"] - 1
    idx = rng.choice(n, size=int(0.003 * n), replace=False)
    df.loc[idx, "location_score"] = rng.choice([0.0, 11.0, 12.0], len(idx))
    idx = rng.choice(n, size=int(0.002 * n), replace=False)
    df.loc[idx, "bedrooms"] = 0.0

    # 3) MISSING VALUES (different rate per column) -------------------------
    missing_rates = {
        "area_sqft": 0.01, "bedrooms": 0.02, "bathrooms": 0.02, "age_years": 0.03,
        "location_score": 0.01, "distance_to_city_km": 0.01, "distance_to_metro_km": 0.02,
        "school_rating": 0.04, "hospital_distance_km": 0.03, "crime_rate": 0.04,
        "parking_spaces": 0.02, "economic_growth_rate": 0.015,
        "historical_price_trend_pct": 0.03, "property_type": 0.01,
    }
    for col, rate in missing_rates.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    # 4) DUPLICATE ROWS (1%) -------------------------------------------------
    duplicates = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, duplicates]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_historical_index(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    A small SYNTHETIC yearly price index (average Rs./sqft) for one city.
    Used only to draw the 'historical trend' chart and to explain how a
    time-series model could be used if real historical data were available.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2015, 2027)
    yoy_growth = np.array([6.5, 7.0, 5.5, 4.8, 5.2, -2.0, 3.5, 7.5, 8.0, 6.0, 5.5, 5.0])
    yoy_growth = yoy_growth + rng.normal(0, 0.4, len(years))
    price = 3200 * np.cumprod(1 + yoy_growth / 100)
    return pd.DataFrame({
        "year": years,
        "avg_price_per_sqft": price.round(0),
        "yoy_growth_pct": yoy_growth.round(2),
    })


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = inject_data_quality_issues(generate_clean_data())
    df.to_csv(DATA_DIR / "housing_data.csv", index=False)
    generate_historical_index().to_csv(DATA_DIR / "historical_price_index.csv", index=False)
    print(f"Saved {len(df):,} rows to {DATA_DIR / 'housing_data.csv'}")
    print(df.head())
    missing = df.isna().sum()
    print("\nMissing values per column:\n", missing[missing > 0])


if __name__ == "__main__":
    main()
