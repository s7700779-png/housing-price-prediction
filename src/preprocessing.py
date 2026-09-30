"""
preprocessing.py
----------------
Everything needed to turn the raw, messy CSV into model-ready data.

Two stages:

1. clean_data()        - plain pandas cleaning (duplicates, wrong labels,
                         impossible values, broken target rows).
2. build_preprocessor() - a scikit-learn Pipeline that is FITTED ONLY ON THE
                         TRAINING DATA and then re-used for test data and for
                         the Streamlit app:
                             impute missing values
                             -> engineer new features
                             -> cap extreme outliers (IQR)
                             -> scale numbers (StandardScaler)
                             -> one-hot encode categories

Keeping imputation / scaling inside the pipeline avoids "data leakage"
(the test set must never influence anything learned during training).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Column definitions (single source of truth for the whole project)
# --------------------------------------------------------------------------
BASE_NUMERIC_FEATURES = [
    "area_sqft", "bedrooms", "bathrooms", "age_years", "location_score",
    "distance_to_city_km", "distance_to_metro_km", "school_rating",
    "hospital_distance_km", "crime_rate", "parking_spaces",
    "economic_growth_rate", "historical_price_trend_pct",
]
BASE_CATEGORICAL_FEATURES = ["property_type"]

# Columns created by feature engineering
ENGINEERED_NUMERIC = ["total_rooms", "area_per_bedroom", "accessibility_score", "neighbourhood_quality"]
ENGINEERED_CATEGORICAL = ["age_group"]

ALL_NUMERIC = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC
ALL_CATEGORICAL = BASE_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL

# The columns a user has to supply for a prediction
MODEL_INPUT_COLUMNS = BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES

TARGET_PRICE = "current_price"
TARGET_GROWTH = "annual_growth_rate_pct"   # derived from future_price, see add_growth_target()

# Values outside these ranges are impossible -> treated as data-entry errors.
VALID_RANGES = {
    "area_sqft": (100, 50_000), "bedrooms": (1, 10), "bathrooms": (1, 10),
    "age_years": (0, 150), "location_score": (1, 10),
    "distance_to_city_km": (0, 200), "distance_to_metro_km": (0, 100),
    "school_rating": (1, 10), "hospital_distance_km": (0, 100),
    "crime_rate": (0, 100), "parking_spaces": (0, 10),
    "economic_growth_rate": (-10, 20), "historical_price_trend_pct": (-30, 50),
}

# Different spellings of the same category -> one standard label
PROPERTY_TYPE_MAP = {
    "apartment": "Apartment", "apt": "Apartment", "flat": "Apartment",
    "independent house": "Independent House", "house": "Independent House",
    "villa": "Villa",
}


# --------------------------------------------------------------------------
# Stage 1: pandas cleaning
# --------------------------------------------------------------------------
def load_data(path) -> pd.DataFrame:
    """Load the raw CSV file."""
    return pd.read_csv(path)


def detect_outliers(df: pd.DataFrame, columns, factor: float = 1.5) -> pd.DataFrame:
    """
    IQR rule: a value is an outlier if it lies more than `factor` x IQR
    below Q1 or above Q3.  (IQR = Q3 - Q1, the spread of the middle 50%.)
    Returns a small summary table.
    """
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - factor * iqr, q3 + factor * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append({"column": col, "lower_fence": round(low, 2), "upper_fence": round(high, 2),
                     "n_outliers": n_out, "pct_outliers": round(100 * n_out / len(s), 2)})
    return pd.DataFrame(rows).set_index("column")


def clean_data(df: pd.DataFrame):
    """
    Clean the raw dataframe. Returns (clean_df, report_dict).

    Steps
      1. remove exact duplicate rows
      2. standardise inconsistent category labels
      3. turn impossible values into NaN (they are imputed later)
      4. detect missing values
      5. drop rows whose TARGET values are unusable
      6. remove price outliers (implausible price per sqft)
    """
    report = {"rows_raw": len(df)}
    df = df.copy()

    # 1. Duplicates
    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Inconsistent labels: ' villa', 'VILLA', 'Apt' ...
    raw_labels = df["property_type"]
    cleaned = raw_labels.astype(str).str.strip().str.lower().map(PROPERTY_TYPE_MAP)
    cleaned = cleaned.astype(object).where(cleaned.notna(), np.nan)
    report["labels_standardised"] = int(
        (raw_labels.notna() & cleaned.notna() & (raw_labels != cleaned)).sum()
    )
    df["property_type"] = cleaned

    # 3. Impossible values -> NaN
    invalid_total = 0
    for col, (low, high) in VALID_RANGES.items():
        bad = df[col].notna() & ((df[col] < low) | (df[col] > high))
        invalid_total += int(bad.sum())
        df.loc[bad, col] = np.nan
    report["invalid_values_set_to_nan"] = invalid_total

    # 4. Missing-value detection (after step 3, so invalid values are included)
    missing = df[MODEL_INPUT_COLUMNS].isna().sum()
    report["missing_per_column"] = missing[missing > 0].to_dict()
    report["missing_total"] = int(missing.sum())

    # 5. Unusable targets: we can NOT impute the thing we want to predict
    good_target = (
        df["current_price"].notna() & (df["current_price"] > 0)
        & df["future_price"].notna() & (df["future_price"] > 0)
        & df["years_ahead"].between(1, 5)
    )
    report["rows_dropped_bad_target"] = int((~good_target).sum())
    df = df[good_target].reset_index(drop=True)

    # 6. Price outliers: judge price per sqft (a 6 crore flat of 900 sqft is a typo).
    #    We use the strict 3 x IQR rule so genuine luxury homes are kept.
    ppsf = df["current_price"] / df["area_sqft"]
    q1, q3 = ppsf.quantile(0.25), ppsf.quantile(0.75)
    iqr = q3 - q1
    keep = ppsf.isna() | ((ppsf >= q1 - 3 * iqr) & (ppsf <= q3 + 3 * iqr))
    report["price_outliers_removed"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)

    report["rows_clean"] = len(df)
    return df, report


def add_growth_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert (current_price, future_price, years_ahead) into ONE comparable
    number: the compound ANNUAL growth rate in %.

        annual_rate = ((future / current) ** (1 / years) - 1) * 100

    Why? A 20% rise over 4 years and a 20% rise over 1 year are very different.
    Predicting the yearly rate lets us compute the future price for ANY horizon
    (1-5 years) with: future = current * (1 + rate/100) ** years.
    """
    df = df.copy()
    df[TARGET_GROWTH] = ((df["future_price"] / df["current_price"]) ** (1 / df["years_ahead"]) - 1) * 100
    return df


# --------------------------------------------------------------------------
# Stage 2: scikit-learn pipeline pieces
# --------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates new columns from existing ones (feature engineering).
    It runs AFTER imputation, so it never sees missing values.

      total_rooms          bedrooms + bathrooms
      area_per_bedroom     spaciousness of the layout
      accessibility_score  0-10; higher = closer to metro, city and hospital
      neighbourhood_quality 0-10; mixes location, school rating and safety
      age_group            New (<=5 yrs) / Mid (5-20) / Old (>20)   [categorical]
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
        X["area_per_bedroom"] = X["area_sqft"] / X["bedrooms"].clip(lower=1)
        X["accessibility_score"] = 10 * (
            1 / (1 + X["distance_to_metro_km"])
            + 1 / (1 + X["distance_to_city_km"])
            + 1 / (1 + X["hospital_distance_km"])
        ) / 3
        X["neighbourhood_quality"] = (
            X["location_score"] + X["school_rating"] + (100 - X["crime_rate"]) / 10
        ) / 3
        X["age_group"] = pd.cut(
            X["age_years"], bins=[-np.inf, 5, 20, np.inf], labels=["New", "Mid", "Old"]
        ).astype(str)
        return X


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Outlier TREATMENT by capping (a.k.a. winsorising).

    Learns the fences  Q1 - factor*IQR  and  Q3 + factor*IQR  from the
    TRAINING data and clips every value to lie inside them.  We use factor=3
    (only extreme values are touched). Columns with IQR = 0 (e.g. constant
    columns) are left alone.
    """

    def __init__(self, factor: float = 3.0):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = np.where(iqr > 0, q1 - self.factor * iqr, -np.inf)
        self.upper_ = np.where(iqr > 0, q3 + self.factor * iqr, np.inf)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessor() -> Pipeline:
    """Build the full (unfitted) preprocessing pipeline."""
    # Step A - fill missing values.
    #   numbers     -> MEDIAN (robust: not pulled around by outliers)
    #   categories  -> MOST FREQUENT value (mode)
    imputer = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), BASE_NUMERIC_FEATURES),
            ("cat", SimpleImputer(strategy="most_frequent"), BASE_CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # Step C - treat outliers + scale numbers, one-hot encode categories.
    #   Scaling puts 'area_sqft' (hundreds) and 'school_rating' (1-10) on the same
    #   footing. Trees do not need it, but Linear Regression does.
    encode_scale = ColumnTransformer(
        [
            ("num", Pipeline([("cap", IQRCapper(factor=3.0)), ("scale", StandardScaler())]), ALL_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ALL_CATEGORICAL),
        ]
    )

    return Pipeline([
        ("impute", imputer),
        ("engineer", FeatureEngineer()),
        ("encode_scale", encode_scale),
    ])


def get_feature_names(fitted_pipeline: Pipeline) -> list:
    """Column names of the matrix that the final model receives."""
    encode_scale = fitted_pipeline.named_steps["preprocess"].named_steps["encode_scale"]
    cat_names = encode_scale.named_transformers_["cat"].get_feature_names_out(ALL_CATEGORICAL)
    return list(ALL_NUMERIC) + list(cat_names)
