"""
Do session features (fatigue, warm-up, time of day, recent form) help predict
your accuracy beyond what the beatmap settings already tell us?

How the test is kept honest:
  * Sessions are built from ALL attempts (passes and fails), then the model is
    trained on passed plays only (a failed play's accuracy is cut short).
  * Cross-validation (CV) is grouped by session: the model is always scored on
    whole sessions it never trained on. Without this, plays from the same
    session would sit in both train and test, and a good or bad night would
    leak across the split.
  * Every model is compared against a baseline that always guesses the mean.

Usage:
    python predict_accuracy.py                # uses only plays logged by collect_plays.py
    python predict_accuracy.py --since 2026-10-10
    python predict_accuracy.py --all          # dry run on everything; sessions are
                                              # unreliable in pre-collector data

Abbreviations: R2 = coefficient of determination, MAE = mean absolute error,
RF = random forest, CV = cross-validation.
"""

import argparse
import os
import sqlite3
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db_setup import DB_PATH
from features import (
    HISTORY,
    SESSION_FEATURES,
    SESSION_GAP_MIN,
    add_mod_stars,
    add_session_features,
    describe_sessions,
    get_collector_start,
)
from mods import MOD_FLAG_COLS, add_mod_columns

CHART_DIR = "charts"
# Mod-aware: BPM/AR/CS/OD are the EFFECTIVE values you played at (e.g. AR 10.33
# for AR 9 + DT), plus flags for which mods were on. star_rating is the unmodded
# rating unless --mod-stars is used (then it is the rating under your mods).
MAP_FEATURES = ["eff_bpm", "eff_ar", "eff_cs", "eff_od", "star_rating"] + MOD_FLAG_COLS

MIN_PASSED = 60  # fewer passed plays than this and the results would be noise
MIN_SESSIONS = 10


def make_rf():
    # the imputer fills missing values (e.g. "previous accuracy" on the first
    # play of a session) using only the training fold's data
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0),
    )


def make_ridge():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 3, 20)),
    )


def load_attempts(args):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM scores", conn)
    started = get_collector_start(conn)
    conn.close()
    df["played_at"] = pd.to_datetime(df["played_at"], utc=True)

    if args.all:
        print("NOTE: --all uses every stored play. Plays from before collect_plays.py "
              "started have no fails and big gaps, so sessions here are unreliable. "
              "Treat the output as a code check, not a result.\n")
        return df

    cutoff = pd.Timestamp(args.since, tz="UTC") if args.since else started
    if cutoff is None:
        print("No play log yet. Session features need every attempt, fails included,\n"
              "and that has to be logged as it happens. Start the collector:\n\n"
              "    python collect_plays.py --watch 20\n\n"
              "Leave it running while you play (a terminal tab is fine), then come back.\n"
              "Add --all to this script to dry-run the code on your existing data.")
        sys.exit(0)
    return df[df["played_at"] >= cutoff]


def evaluate(models, m, groups, cv):
    y = m["accuracy_pct"]
    r2s = {}
    print(f"{'Model':<30}{'R2 (mean +/- sd)':<22}{'MAE (pts)':<10}")
    for name, (model, cols) in models.items():
        res = cross_validate(
            model, m[cols], y, groups=groups, cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
        )
        r2 = res["test_r2"]
        r2s[name] = r2
        print(f"{name:<30}{r2.mean():>6.3f} +/- {r2.std():<8.3f}{-res['test_mae'].mean():<10.3f}")
    print()
    return r2s


def print_gains(r2s):
    """Compare like with like: the same model class on the same folds, with and
    without session features. Comparing a ridge model to a random forest would
    mix 'session features help' with 'this model class fits better'."""
    print("Gain from adding session features (same model class, same folds):")
    diffs = []
    for label, base, plus in (
        ("RF", "Map only (RF)", "Map + session (RF)"),
        ("Ridge", "Map only (Ridge)", "Map + session (Ridge)"),
    ):
        diff = r2s[plus] - r2s[base]
        diffs.append(diff)
        print(f"  {label:<6} change in R2 = {diff.mean():+.3f} +/- {diff.std():.3f} "
              f"| better in {(diff > 0).sum()} of {len(diff)} folds")

    # Heuristic, not a formal test: with this little data one model class can show
    # a spurious gain on its own (in testing, a random forest did on pure noise),
    # so BOTH must clearly improve, in nearly every fold.
    detected = all(
        d.mean() > 0.05 and (d > 0).sum() >= len(d) - 1 and d.mean() > d.std()
        for d in diffs
    )
    if detected:
        print("Verdict: both model types improve consistently with session features.\n"
              "         That is evidence the effect is real, not a certainty.\n")
    else:
        print("Verdict: no reliable improvement from session features at this sample size\n"
              "         (the two model types disagree, or gains are within fold-to-fold noise).\n"
              "         That is 'not detected yet', not 'proven absent'.\n")


def permutation_table(m, cols, groups, cv):
    """Held-out permutation importance: how much worse do predictions get when one
    feature is shuffled? Measured on sessions the model never trained on."""
    y = m["accuracy_pct"]
    scores = []
    for tr, te in cv.split(m[cols], y, groups):
        mdl = make_rf().fit(m.iloc[tr][cols], y.iloc[tr])
        r = permutation_importance(
            mdl, m.iloc[te][cols], y.iloc[te], n_repeats=10, random_state=0,
            scoring="neg_mean_absolute_error",
        )
        scores.append(r.importances_mean)
    imp = pd.Series(np.mean(scores, axis=0), index=cols).sort_values(ascending=False)
    print("Permutation importance (increase in MAE, in accuracy points, when shuffled).")
    print("Near zero or negative = the model gains nothing from that feature.")
    print(imp.round(3).to_string())
    print()


def hour_block(h: int) -> str:
    if 5 <= h < 12:
        return "morning (5-11)"
    if 12 <= h < 18:
        return "afternoon (12-17)"
    if 18 <= h < 23:
        return "evening (18-22)"
    return "night (23-4)"


def effects(m):
    """Accuracy relative to what the map alone predicts, by position in session
    and by time of day. Residual = actual minus map-only out-of-fold prediction,
    so harder maps late in a session don't masquerade as fatigue."""
    m = m.copy()
    m["play_bucket"] = pd.cut(
        m["play_index"], [0, 1, 3, 6, 10, np.inf], labels=["1", "2-3", "4-6", "7-10", "11+"]
    )
    m["hour_block"] = m["local_hour"].map(hour_block)

    tables = {}
    for col, resid_col, label in (
        ("play_bucket", "resid", "play number in session, vs what the map alone predicts"),
        ("hour_block", "resid_pos", "time of day (local), after adjusting for map and position in session"),
    ):
        # Average within each session first, then across sessions. Plays in the same
        # session share a good or bad night, so treating each play as independent
        # would make the error bars look far tighter than they really are.
        per_session = (
            m.groupby([col, "session_id"], observed=True)[resid_col].mean().rename("s").reset_index()
        )
        t = per_session.groupby(col, observed=True)["s"].agg(sessions="count", mean="mean", std="std")
        t["se"] = t["std"] / np.sqrt(t["sessions"])
        t.insert(0, "plays", m.groupby(col, observed=True).size())
        tables[col] = t
        print(f"Accuracy by {label} (points; + = better than expected; se is across sessions):")
        print(t[["plays", "sessions", "mean", "se"]].round(2).to_string())
        print()
    return tables


def plot_effects(tables):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (col, t) in zip(axes, tables.items()):
        ax.bar(t.index.astype(str), t["mean"], yerr=t["se"].fillna(0), capsize=4, alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_ylabel("Accuracy vs map-only expectation (pts)")
        ax.set_title("By play number in session" if col == "play_bucket" else "By time of day")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "session_effects.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="dry run on all stored plays")
    parser.add_argument("--since", help="only use plays on/after this date (YYYY-MM-DD)")
    parser.add_argument("--gap", type=int, default=SESSION_GAP_MIN,
                        help=f"minutes of inactivity that end a session (default {SESSION_GAP_MIN})")
    parser.add_argument("--history", type=int, default=HISTORY,
                        help=f"how many previous attempts rolling features use (default {HISTORY})")
    parser.add_argument("--mod-stars", action="store_true",
                        help="use each map's star rating UNDER the mods played (downloads .osu "
                             "files once; needs rosu-pp-py). Recommended if you play with mods.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(CHART_DIR, exist_ok=True)

    attempts = load_attempts(args)
    if attempts.empty:
        print("No plays in the selected window yet.")
        return

    attempts = add_mod_columns(attempts)
    modded = int((attempts["mods_key"] != "NM").sum())
    print(f"Plays with difficulty-affecting mods: {modded} of {len(attempts)}")
    if args.mod_stars:
        attempts = add_mod_stars(attempts)
        # use the modded star rating; for unmodded plays fall back to the stored one
        fallback = attempts["star_rating"].where(attempts["mods_key"] == "NM")
        attempts["star_rating"] = attempts["star_mod"].fillna(fallback)
    elif modded:
        print("Tip: --mod-stars would use each map's star rating under your mods "
              "(currently the unmodded rating).")
    print()

    d = add_session_features(attempts, gap_minutes=args.gap, history=args.history)
    describe_sessions(d)

    m = d[d["passed"] == 1].dropna(subset=MAP_FEATURES + ["accuracy_pct"]).reset_index(drop=True)
    n_sessions = m["session_id"].nunique()
    if not args.all and (len(m) < MIN_PASSED or n_sessions < MIN_SESSIONS):
        print(f"Not enough data yet: {len(m)} passed plays across {n_sessions} sessions "
              f"(need at least {MIN_PASSED} and {MIN_SESSIONS}).\n"
              "Keep collect_plays.py running and try again after more play sessions.")
        return
    if n_sessions < 3:
        print("Fewer than 3 sessions: cannot cross-validate by session. Need more data.")
        return

    groups = m["session_id"]
    cv = GroupKFold(n_splits=min(5, n_sessions))
    all_cols = MAP_FEATURES + SESSION_FEATURES

    # rel_star is built from the CURRENT map's star rating, so it is excluded from
    # the session-only model (otherwise that model could see map information).
    session_only = [c for c in SESSION_FEATURES if c != "rel_star"]

    models = {
        "Baseline (always the mean)": (DummyRegressor(strategy="mean"), MAP_FEATURES),
        "Map only (RF)": (make_rf(), MAP_FEATURES),
        "Map + session (RF)": (make_rf(), all_cols),
        "Map only (Ridge)": (make_ridge(), MAP_FEATURES),
        "Map + session (Ridge)": (make_ridge(), all_cols),
        "Session only (RF)": (make_rf(), session_only),
    }
    print(f"Target: accuracy_pct | rows: {len(m)} passed plays | sessions: {n_sessions} "
          f"| CV folds: {cv.n_splits} (grouped by session)")
    r2s = evaluate(models, m, groups, cv)
    print_gains(r2s)

    permutation_table(m, all_cols, groups, cv)

    # Residuals for the descriptive tables. Position-in-session effects use the
    # map-only expectation. Time-of-day effects must ALSO adjust for position in
    # session, because early plays happen at the start hour and late plays at the
    # end hour, so a raw time-of-day table would mostly re-measure fatigue.
    y = m["accuracy_pct"]
    m["resid"] = y - cross_val_predict(make_rf(), m[MAP_FEATURES], y, groups=groups, cv=cv)
    pos_cols = MAP_FEATURES + ["play_index", "session_minutes"]
    m["resid_pos"] = y - cross_val_predict(make_rf(), m[pos_cols], y, groups=groups, cv=cv)
    plot_effects(effects(m))


if __name__ == "__main__":
    main()