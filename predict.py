"""
Predictive modelling on beatmaps.

For each target below, we ask: "How well can the OTHER beatmap settings
predict this one?" using cross-validation, and compare against a naive
baseline that always guesses the mean.

Targets:
    star_rating <- BPM, AR, CS, OD
    bpm         <- AR, CS, OD
    ar          <- BPM, CS, OD
    accuracy_pct <- BPM, AR, CS, OD, star_rating

Abbreviations:
    BPM = beats per minute
    AR = approach rate
    CS = circle size
    OD = overall difficulty
    R2 = coefficient of determination
    MAE = mean absolute error
    CV = cross-validation

Run with: python predict.py
"""

import os
import sqlite3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db_setup import DB_PATH

CHART_DIR = "charts"

# If True, accuracy modelling only uses plays with no mods. Mods like DT
# (Double Time) change the effective BPM and AR you actually experience,
# but the database stores the unmodded map values
NOMOD_ONLY = False

MIN_ROWS = 20 # Refuse to fit anything on fewer rows than this

# (target column, feature columns, one-row-per-score?)
TASKS = [
    ("star_rating", ["bpm", "ar", "cs", "overall_difficulty"], False),
    ("bpm", ["ar", "cs", "overall_difficulty"], False),
    ("ar", ["bpm", "cs", "overall_difficulty"], False),
    ("accuracy_pct", ["bpm", "ar", "cs", "overall_difficulty", "star_rating"], True),
]

def load_scores() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM scores", conn)
    conn.close()
    df["accuracy_pct"] = df["accuracy"] * 100
    return df


def build_models() -> dict:
    return {
        "Baseline (always the mean)": DummyRegressor(strategy="mean"),
        "Ridge (linear)": make_pipeline(
            StandardScaler(), RidgeCV(alphas = np.logspace(-2, 3, 20))
        ),
        "Random forest": RandomForestRegressor(
            n_estimators = 300, min_samples_leaf = 3, random_state = 0
        ),
    }


def evaluate_task(df: pd.DataFrame, target: str, features: list, per_score: bool):
    data = df if per_score else df.drop_duplicates("beatmap_id")
    if per_score and NOMOD_ONLY:
        data = data[data["mods"].fillna("") == ""]
    data = data.dropna(subset = features + [target]).reset_index(drop = True)

    n_maps = data["beatmap_id"].nunique()
    print(f"--- Target: {target} ---")
    print(f"Rows: {len(data)} | distinct beatmaps: {n_maps}")
    print(f"Features: {', '.join(features)}")

    if len(data) < MIN_ROWS or n_maps < 5:
        print("Not enough data to model this target reliably. Skipping.\n")
        return None

    X, y, groups = data[features], data[target], data["beatmap_id"]
    cv = GroupKFold(n_splits = min(5, n_maps))
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

    print(f"{'Model':<30}{'R2 (mean +/- sd)':<22}{'MAE':<10}")
    for name, model in build_models().items():
        res = cross_validate(model, X, y, groups = groups, cv = cv, scoring = scoring)
        r2, mae = res["test_r2"], -res["test_mae"].mean()
        print(f"{name:<30}{r2.mean():>6.3f} +/- {r2.std():<8.3f}{mae:<10.3f}")

    # Out-of-fold predictions from the random forest: every row is predicted
    # by a model that never saw its beatmap during training
    rf = build_models()["Random forest"]
    oof = cross_val_predict(rf, X, y, groups = groups, cv = cv)
    data["predicted"] = oof

    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index = features)
    print("Random forest feature importance (impurity-based, rough guide only):")
    print(importances.sort_values(ascending = False).round(3).to_string())
    print()

    return data, y.values, oof


def print_surprising_maps(data: pd.DataFrame):
    """
    Maps whose real star rating is far above what BPM/AR/CS/CD would suggest
    """
    d = data.copy()
    d["residual"] = d["star_rating"] - d["predicted"]
    cols = ["title", "difficulty_name", "star_rating", "predicted", "residual"]
    print("Maps harder than their raw settings suggest (patterning/density, not numbers):")
    print(d.sort_values("residual", ascending = False).head(5)[cols].round(2).to_string(index = False))
    print()
    print("Maps easier than their raw settings suggest:")
    print(d.sort_values("residual").head(5)[cols].round(2).to_string(index = False))
    print()


def plot_predictions(results: dict):
    fig, axes = plt.subplots(2, 2, figsize = (11, 9))
    for ax, (target, (actual, pred)) in zip(axes.ravel(), results.items()):
        ax.scatter(actual, pred, alpha = 0.6)
        lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], linestyle = "--", color = "gray")
        ss_res = ((actual - pred) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        ax.set_title(f"{target} (out-of-fold R2 = {1 - ss_res / ss_tot:.2f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    for ax in axes.ravel()[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "predictions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    warnings.filterwarnings("ignore", category = UserWarning)
    os.makedirs(CHART_DIR, exist_ok = True)
    df = load_scores()
    if df.empty:
        print("No scores found. Run fetch_scores.py first.")
        return

    plot_data = {}
    for target, features, per_score in TASKS:
        out = evaluate_task(df, target, features, per_score)
        if out is None:
            continue
        data, actual, oof = out
        plot_data[target] = (actual, oof)
        if target == "star_rating":
            print_surprising_maps(data)

    if plot_data:
        plot_predictions(plot_data)


if __name__ == "__main__":
    main()