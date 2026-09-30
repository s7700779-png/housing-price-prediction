"""
train_model.py
--------------
Trains and compares four regression models for TWO separate tasks:

  Task 1 - HOUSE PRICE model      : features -> current_price (Rs.)
  Task 2 - PRICE GROWTH model     : features -> annual price growth rate (%)

They are kept separate on purpose. A house-price model says what a house is
worth TODAY given its features. It cannot see the future. The growth model
learns how fast prices tend to rise for houses with certain characteristics
(strong economy, good location, near metro, ...). Future price is then
calculated by compounding the predicted growth rate (see prediction.py).

Run from the project root:
    python src/train_model.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# Make "src" importable when running `python src/train_model.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import eda  # noqa: E402
from src.data_generation import main as generate_dataset  # noqa: E402
from src.preprocessing import (  # noqa: E402
    ALL_NUMERIC, BASE_NUMERIC_FEATURES, MODEL_INPUT_COLUMNS, TARGET_GROWTH, TARGET_PRICE,
    add_growth_target, build_preprocessor, clean_data, detect_outliers, get_feature_names, load_data,
)

DATA_PATH = PROJECT_ROOT / "data" / "housing_data.csv"
HIST_PATH = PROJECT_ROOT / "data" / "historical_price_index.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
# Rule for picking the "best" model: lowest test RMSE among the models that do
# not overfit (train R2 minus test R2 must be <= this gap).
MAX_OVERFIT_GAP = 0.10


def get_models() -> dict:
    """Fresh (untrained) copies of the four algorithms."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.08, max_depth=3, subsample=0.8, random_state=RANDOM_STATE),
    }


def evaluate(y_true, y_pred) -> dict:
    """The four regression metrics used in this project."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_compare(X_train, X_test, y_train, y_test, task_name):
    """Train every model, return (comparison_table, dict_of_fitted_pipelines)."""
    print(f"\n{'=' * 70}\nTraining models for: {task_name}\n{'=' * 70}")
    rows, fitted = [], {}
    for name, estimator in get_models().items():
        # Pipeline = preprocessing + model, so raw input can go straight in.
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])

        start = time.time()
        pipe.fit(X_train, y_train)
        fit_seconds = time.time() - start

        metrics = evaluate(y_test, pipe.predict(X_test))
        train_r2 = r2_score(y_train, pipe.predict(X_train))
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()

        rows.append({
            "Model": name, **metrics,
            "Train_R2": train_r2, "R2_gap": train_r2 - metrics["R2"],
            "CV_R2_mean": cv_r2, "Train_time_s": fit_seconds,
        })
        fitted[name] = pipe
        print(f"  {name:<18} RMSE={metrics['RMSE']:>12,.3f}  R2={metrics['R2']:.4f}  ({fit_seconds:.1f}s)")

    return pd.DataFrame(rows), fitted


def select_best_model(results: pd.DataFrame) -> str:
    """Lowest test RMSE among models that generalise well (small train/test R2 gap)."""
    ok = results[results["R2_gap"] <= MAX_OVERFIT_GAP]
    if ok.empty:
        ok = results
    return ok.sort_values("RMSE").iloc[0]["Model"]


def print_table(results: pd.DataFrame, title: str) -> None:
    fmt = lambda v: f"{v:,.4f}" if abs(v) < 1e4 else f"{v:,.0f}"  # noqa: E731
    print(f"\n{title}")
    print(results.to_string(index=False, float_format=fmt))


def main() -> None:
    matplotlib.use("Agg")  # when run as a script, draw figures to files (no window needed)
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data (generate it first if the CSV does not exist) ----------
    if not DATA_PATH.exists():
        print("Dataset not found - generating it ...")
        generate_dataset()
    raw = load_data(DATA_PATH)
    print(f"Loaded {len(raw):,} raw rows, {raw.shape[1]} columns")

    # ---- 2. Clean --------------------------------------------------------------
    df, report = clean_data(raw)
    df = add_growth_target(df)
    print("\nCleaning report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("\nOutlier check (IQR rule, factor 1.5) on the cleaned data:")
    print(detect_outliers(df, ["area_sqft", "current_price", "crime_rate", "distance_to_metro_km"]))

    # ---- 3. EDA figures ------------------------------------------------------------
    hist_df = pd.read_csv(HIST_PATH) if HIST_PATH.exists() else None
    eda.run_eda(df, hist_df, save_dir=FIGURES_DIR)
    print(f"\nEDA figures saved to {FIGURES_DIR}")

    # ---- 4. Feature / target separation ------------------------------------------------
    X = df[MODEL_INPUT_COLUMNS]
    y_price = df[TARGET_PRICE]
    y_growth = df[TARGET_GROWTH]

    # ---- 5. Train / test split ------------------------------------------------------------
    # 80% of the rows train the model, 20% are held back to test it on unseen data.
    # random_state=42 fixes the random shuffle so anyone re-running this code gets
    # exactly the same split (and therefore the same results). 42 is only a
    # convention - any fixed number works, but it must stay the same.
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(X, y_growth, test_size=0.2, random_state=42)
    print(f"\nTrain rows: {len(Xp_train):,}   Test rows: {len(Xp_test):,}")

    # ---- 6. Train + compare: PRICE model ------------------------------------------------------
    price_results, price_models = train_and_compare(Xp_train, Xp_test, yp_train, yp_test, "HOUSE PRICE (Rs.)")
    best_price = select_best_model(price_results)
    print_table(price_results, "PRICE MODEL COMPARISON")
    print(f"--> Selected price model: {best_price}")

    # ---- 7. Train + compare: GROWTH model ----------------------------------------------------------
    growth_results, growth_models = train_and_compare(
        Xg_train, Xg_test, yg_train, yg_test, "ANNUAL PRICE GROWTH (% per year)")
    best_growth = select_best_model(growth_results)
    print_table(growth_results, "GROWTH MODEL COMPARISON")
    print(f"--> Selected growth model: {best_growth}")

    # ---- 8. Evaluation figures for the selected models -------------------------------------------------
    p_pipe, g_pipe = price_models[best_price], growth_models[best_growth]
    p_pred, g_pred = p_pipe.predict(Xp_test), g_pipe.predict(Xg_test)
    eda.plot_actual_vs_predicted(yp_test, p_pred, f"Price model: actual vs predicted ({best_price})",
                                 "price", "09_price_actual_vs_predicted.png", FIGURES_DIR)
    eda.plot_residuals(yp_test, p_pred, f"Price model residuals ({best_price})",
                       "price", "10_price_residuals.png", FIGURES_DIR)
    eda.plot_actual_vs_predicted(yg_test, g_pred, f"Growth model: actual vs predicted ({best_growth})",
                                 "growth", "11_growth_actual_vs_predicted.png", FIGURES_DIR)
    eda.plot_residuals(yg_test, g_pred, f"Growth model residuals ({best_growth})",
                       "growth", "12_growth_residuals.png", FIGURES_DIR)
    eda.plot_model_comparison(price_results, "Price model comparison", "13_price_model_comparison.png", FIGURES_DIR)
    eda.plot_model_comparison(growth_results, "Growth model comparison", "14_growth_model_comparison.png", FIGURES_DIR)
    eda.plot_feature_importance(p_pipe, get_feature_names(p_pipe), f"Top features - price ({best_price})",
                                "15_price_feature_importance.png", save_dir=FIGURES_DIR)
    eda.plot_feature_importance(g_pipe, get_feature_names(g_pipe), f"Top features - growth ({best_growth})",
                                "16_growth_feature_importance.png", save_dir=FIGURES_DIR)

    # ---- 9. Save models + metadata ----------------------------------------------------------------------------
    joblib.dump(p_pipe, MODELS_DIR / "housing_price_model.pkl", compress=3)
    joblib.dump(g_pipe, MODELS_DIR / "future_growth_model.pkl", compress=3)
    price_results.to_csv(MODELS_DIR / "price_model_comparison.csv", index=False)
    growth_results.to_csv(MODELS_DIR / "growth_model_comparison.csv", index=False)

    best_price_row = price_results.set_index("Model").loc[best_price]
    best_growth_row = growth_results.set_index("Model").loc[best_growth]
    metadata = {
        "price_model": best_price,
        "growth_model": best_growth,
        "price_metrics": {k: float(best_price_row[k]) for k in ["MAE", "MSE", "RMSE", "R2"]},
        "growth_metrics": {k: float(best_growth_row[k]) for k in ["MAE", "MSE", "RMSE", "R2"]},
        "growth_rmse_pct_points": float(best_growth_row["RMSE"]),
        "n_rows_after_cleaning": int(len(df)),
        "random_state": RANDOM_STATE,
        "input_columns": MODEL_INPUT_COLUMNS,
        "property_types": sorted(df["property_type"].dropna().unique().tolist()),
        "feature_stats": {
            col: {"min": float(df[col].min()), "max": float(df[col].max()), "median": float(df[col].median())}
            for col in BASE_NUMERIC_FEATURES
        },
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved models to {MODELS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
