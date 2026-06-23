"""
train_final_model.py

End-to-end pipeline for the House Price Prediction project:
load -> clean -> feature engineer -> split -> scale -> train champion model -> evaluate -> save.

Champion model: Random Forest Regressor (n_estimators=300, max_features='sqrt')
Selected in 04_model_tuning.ipynb (V6: Feature Randomness) — best cross-validated
RMSE among 8 tuned variants, confirmed via a single final test-set evaluation.

Usage:
    python train_final_model.py
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "housing.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2

CHAMPION_PARAMS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows identified in 01_eda.ipynb as data artifacts:
    - AveOccup > 10: institutional/group housing, occupancy values not
      representative of real households.
    - AveRooms > 20: tiny-household-count artifact inflating the per-household
      ratio; also resolves AveBedrms outliers as a side effect.
    """
    mask = (df["AveOccup"] <= 10) & (df["AveRooms"] <= 20)
    removed = len(df) - mask.sum()
    df_clean = df[mask].copy()
    print(f"[remove_outliers] Removed {removed} rows ({removed/len(df)*100:.2f}%). "
          f"Remaining: {len(df_clean)}")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - is_max_house_age: flags the censored/capped HouseAge value (52),
      so the model can treat "52-or-older, unknown" separately from a
      literal precise age of 52.
    - bedroom_ratio: replaces AveBedrms (0.85 correlated with AveRooms).
      Keeps the size signal in AveRooms while capturing layout/type
      separately via the ratio.
    - log_MedInc / log_Population: added as EXTRA columns (originals kept)
      for models that benefit from reduced skew. Not used by the champion
      (tree-based), but kept here for pipeline parity with 02_data_preprocessing.ipynb.
    """
    df = df.copy()

    df["is_max_house_age"] = (df["HouseAge"] == 52).astype(int)

    df["bedroom_ratio"] = df["AveBedrms"] / df["AveRooms"]
    df = df.drop(columns=["AveBedrms"])

    df["log_MedInc"] = np.log1p(df["MedInc"])
    df["log_Population"] = np.log1p(df["Population"])

    print(f"[engineer_features] Final columns: {list(df.columns)}")
    return df


def split_data(df: pd.DataFrame):
    """
    Stratified 80/20 split on binned MedInc (income_cat), since MedInc is
    the strongest predictor (0.69 correlation with target) — stratifying
    keeps the income distribution representative across train and test.
    """
    df = df.copy()
    df["income_cat"] = pd.cut(
        df["MedInc"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    X = df.drop(columns=["MedHouseVal", "income_cat"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["income_cat"],
    )

    print(f"[split_data] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def fit_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit StandardScaler on training data only (no leakage). Produced for
    pipeline parity / reuse by linear-model variants; the champion model
    (Random Forest) does not require scaled input.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    print("[fit_scaler] Scaler fit on training data only.")
    return scaler, X_train_scaled, X_test_scaled


def get_tree_feature_set(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Champion model uses the unscaled, raw feature set — drop the log_*
    columns (redundant for tree-based splitting; kept only for linear models).
    """
    log_cols = ["log_MedInc", "log_Population"]
    return X_train.drop(columns=log_cols), X_test.drop(columns=log_cols)


def train_champion(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(**CHAMPION_PARAMS)
    model.fit(X_train, y_train)
    print(f"[train_champion] Trained RandomForestRegressor with params: {CHAMPION_PARAMS}")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[evaluate_model] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    # Known limitation check: error on capped target rows (MedHouseVal >= 5.0)
    is_capped = (y_test >= 5.0)
    capped_rmse = np.sqrt(mean_squared_error(y_test[is_capped], y_pred[is_capped]))
    uncapped_rmse = np.sqrt(mean_squared_error(y_test[~is_capped], y_pred[~is_capped]))
    print(f"[evaluate_model] Capped-row RMSE: {capped_rmse:.4f} (n={is_capped.sum()}) "
          f"| Non-capped RMSE: {uncapped_rmse:.4f} (n={(~is_capped).sum()})")

    return {
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "Capped_RMSE": capped_rmse, "NonCapped_RMSE": uncapped_rmse,
    }


def save_artifacts(model, scaler, X_train, X_test, X_train_scaled, X_test_scaled,
                    y_train, y_test, metrics: dict, output_dir: str = OUTPUT_DIR):
    import os
    os.makedirs(output_dir, exist_ok=True)

    artifacts = {
        "champion_model_final.pkl": model,
        "scaler.pkl": scaler,
        "X_train.pkl": X_train,
        "X_test.pkl": X_test,
        "X_train_scaled.pkl": X_train_scaled,
        "X_test_scaled.pkl": X_test_scaled,
        "y_train.pkl": y_train,
        "y_test.pkl": y_test,
    }

    for filename, obj in artifacts.items():
        with open(f"{output_dir}/{filename}", "wb") as f:
            pickle.dump(obj, f)

    pd.DataFrame([metrics]).to_csv(f"{output_dir}/final_metrics.csv", index=False)

    print(f"[save_artifacts] Saved {len(artifacts)} artifacts + final_metrics.csv to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data(DATA_PATH)
    df = remove_outliers(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)
    scaler, X_train_scaled, X_test_scaled = fit_scaler(X_train, X_test)

    X_train_tree, X_test_tree = get_tree_feature_set(X_train, X_test)

    model = train_champion(X_train_tree, y_train)
    metrics = evaluate_model(model, X_test_tree, y_test)

    save_artifacts(
        model, scaler,
        X_train_tree, X_test_tree, X_train_scaled, X_test_scaled,
        y_train, y_test, metrics,
    )

    print("\nDone. Final model saved to outputs/champion_model_final.pkl")


if __name__ == "__main__":
    main()
