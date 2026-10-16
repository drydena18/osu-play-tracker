"""
Beatmap recommender: which maps, with which mods, and what to expect.

For each candidate map (from fetch_candidates.py) and each mod option:
  1. Star rating and settings UNDER those mods (mods.py + pp_engine.py).
  2. Expected accuracy: a model trained on your own passed plays, using the
     effective BPM/AR/CS/OD, star rating under mods, and mod flags. Its typical
     error comes from held-out predictions on plays with a similar predicted
     accuracy (maps it never trained on), so the "safe" accuracy is a measured
     lower-quartile estimate for plays like this one, not a guess.
  3. pp at that accuracy, assuming you pass, scaled by how much of the
     full-combo pp you actually realise (calibrated from your own plays).
  4. Marginal gain: how much your weighted top-100 pp total would rise, using
     osu!'s weighting (each score counts 0.95 times the one above it). Replacing
     an existing score on the same map is handled; a score that doesn't beat your
     current best on that map gains nothing.

Goals (--goal):
  pp       biggest gain in total pp, at your "safe" accuracy           (default)
  comfort  biggest gain among maps where even a worse-than-usual day should
           still match your usual accuracy (override with --comfort-acc)
  stretch  the top end of your range: maps at or above the 90th percentile of
           the difficulty you play (up to --stretch stars beyond your hardest),
           biggest gain

Usage:
    python recommend.py
    python recommend.py --goal comfort --top 20
    python recommend.py --mods NM,DT,HDDT --risk 0.4 --exclude-played

Mods: separate OPTIONS with commas and write a combination without commas
(NM,DT,HDDT means three options: no mod, DT, and HD+DT).

Abbreviations: pp = performance points, AR = approach rate, CS = circle size,
OD = overall difficulty, BPM = beats per minute, MAE = mean absolute error.
"""

import argparse
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db_setup import DB_PATH
from mods import MOD_FLAG_COLS, add_mod_columns, effective_settings, mods_key
from pp_engine import MapUnavailable, PPEngine, osu_path

FEATURES = ["eff_bpm", "eff_ar", "eff_cs", "eff_od", "star_mod"] + MOD_FLAG_COLS
TAIL_FEATURES = ["star_mod"] + MOD_FLAG_COLS  # few, uncorrelated inputs extrapolate more safely
NUMERIC = ["eff_bpm", "eff_ar", "eff_cs", "eff_od", "star_mod"]
TOP_N = 100
WEIGHT = 0.95
MIN_TRAIN = 30
MIN_MOD_PLAYS = 8
MIN_GAIN = 0.1  # pp; for the comfort goal, ignore maps that add essentially nothing


# --------------------------------------------------------------------------- pp weighting
def weighted_total(pps) -> float:
    pps = np.sort(np.asarray(pps, dtype=float))[::-1][:TOP_N]
    return float((pps * WEIGHT ** np.arange(len(pps))).sum())


def marginal_gain(top_desc, existing, new_pp: float) -> float:
    """Rise in weighted top-100 pp if you set `new_pp` on a map where your current
    best is `existing` (None if unplayed). Only your best score per map counts."""
    top = list(np.sort(np.asarray(top_desc, dtype=float))[::-1][:TOP_N])
    if existing is not None and not np.isnan(existing):
        if new_pp <= existing:
            return 0.0
        if existing in top:
            top.remove(existing)
    top.append(new_pp)
    return weighted_total(top) - weighted_total(top_desc)


# --------------------------------------------------------------------------- data
def count_uncached(beatmap_ids) -> int:
    return sum(not os.path.exists(osu_path(b)) for b in set(beatmap_ids))


def load_history(engine: PPEngine):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM scores", conn)
    best_pp = pd.read_sql(
        "SELECT beatmap_id, MAX(pp) AS pp FROM scores WHERE pp IS NOT NULL GROUP BY beatmap_id", conn
    ).set_index("beatmap_id")["pp"]
    conn.close()

    df = df[(df["passed"] == 1) & df["accuracy"].notna()].copy()
    df = add_mod_columns(df)
    n_dl = 0 if not engine.download else count_uncached(df["beatmap_id"])
    if n_dl:
        print(f"Fetching {n_dl} of your beatmap files, one time only (about {n_dl * 0.8:.0f} s)...")
    stars = []
    for bid, key in zip(df["beatmap_id"], df["mods_key"]):
        try:
            stars.append(engine.attrs(bid, key)["stars"])
        except MapUnavailable:
            stars.append(np.nan)
    df["star_mod"] = stars
    dropped = int(df["star_mod"].isna().sum())
    df = df.dropna(subset=NUMERIC)
    return df, best_pp, dropped


def pp_ratio(hist: pd.DataFrame, engine: PPEngine):
    """Median of (real pp / full-combo pp at the same accuracy): how much of the
    ideal pp you actually realise once misses and combo breaks are counted."""
    ratios = []
    for r in hist[hist["pp"].notna() & (hist["pp"] > 0)].itertuples():
        try:
            calc = engine.pp(r.beatmap_id, r.mods_key, r.accuracy * 100)
        except MapUnavailable:
            continue
        if calc > 0:
            ratios.append(r.pp / calc)
    arr = np.array(ratios)
    arr = arr[(arr > 0.3) & (arr < 1.7)]
    if len(arr) < 5:
        return 1.0, len(arr)
    return float(np.median(arr)), len(arr)


# --------------------------------------------------------------------------- accuracy model
class HardRegionBlend(BaseEstimator):
    """Random forest everywhere, except on hard maps, where a straight-line (ridge)
    model trained ONLY on your harder plays takes over.

    Why: a random forest can only output values it has seen, so past the edge of
    your data it goes flat and stays optimistic. A linear model keeps following the
    downward trend of accuracy with difficulty. `threshold` is the star rating (under
    mods) at which the hand-over happens."""

    def __init__(self, threshold):
        self.threshold = threshold

    def _ridge(self):
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))

    def fit(self, X, y):
        X, y = pd.DataFrame(X), np.asarray(y, dtype=float)
        self.rf_ = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0).fit(X, y)
        hard = X["star_mod"].to_numpy() >= self.threshold
        self.ridge_ = self._ridge().fit(X.loc[hard, TAIL_FEATURES], y[hard]) if hard.sum() >= 12 else None
        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        pred = self.rf_.predict(X)
        if self.ridge_ is not None:
            hard = X["star_mod"].to_numpy() >= self.threshold
            if hard.any():
                pred[hard] = self.ridge_.predict(X.loc[hard, TAIL_FEATURES])
        return pred


def hard_threshold(stars, acc, ceiling: float = 97.5, window: int = 15, min_hard: int = 25) -> float:
    """Star rating where the 'hard-map' model takes over: at least the 67th percentile
    of your plays, and past the zone where your accuracy sits against the ceiling
    (there accuracy barely responds to difficulty, which would flatten a straight
    line fitted to it). Always leaves at least `min_hard` plays above the line."""
    stars, acc = np.asarray(stars, float), np.asarray(acc, float)
    thr = float(np.percentile(stars, 67))
    order = np.argsort(stars)
    s_sorted = stars[order]
    med = pd.Series(acc[order]).rolling(window, center=True, min_periods=window // 2).median().to_numpy()
    below = np.where(med < ceiling)[0]
    if len(below):
        thr = max(thr, float(s_sorted[below[0]]))
    if len(stars) > min_hard:
        thr = min(thr, float(s_sorted[-min_hard]))
    return thr


def fit_accuracy_model(hist: pd.DataFrame) -> dict:
    X, y, groups = hist[FEATURES], hist["accuracy"] * 100, hist["beatmap_id"]
    yv = y.to_numpy()
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    stars = hist["star_mod"].to_numpy()
    thr = hard_threshold(stars, yv)
    hard = stars >= thr

    # Recommendations concentrate on the hard end of what you play, where there is
    # the least data. So models are compared on the hardest third of your plays,
    # not on everything (easy maps are plentiful and would decide the winner).
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0)
    oof_ridge = cross_val_predict(ridge, X, yv, groups=groups, cv=cv)
    oof_rf = cross_val_predict(rf, X, yv, groups=groups, cv=cv)

    # Blend out-of-fold predictions: ridge trained on hard plays only, evaluated
    # on held-out hard plays (folds split by beatmap within the hard subset).
    oof_blend = oof_rf.copy()
    idx = np.where(hard)[0]
    hg = groups.iloc[idx]
    if len(idx) >= 20 and hg.nunique() >= 3:
        hcv = GroupKFold(n_splits=min(5, hg.nunique()))
        hard_ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
        oof_blend[idx] = cross_val_predict(hard_ridge, X.iloc[idx][TAIL_FEATURES], yv[idx], groups=hg, cv=hcv)

    options = {
        "Ridge": (ridge, oof_ridge),
        "Random forest": (rf, oof_rf),
        "Forest + hard-map ridge": (HardRegionBlend(thr), oof_blend),
    }
    # Interior accuracy does not show how a model behaves PAST the edge of your data,
    # which is where stretch recommendations live. So also run an extrapolation test:
    # train on everything below the middle of the hard region, predict its top 40%
    # (the hand-over line stays where it is).
    cut = float(np.percentile(stars[hard], 60))
    easy, top = stars < cut, stars >= cut
    ext = {}
    if hard.sum() >= 20 and (easy & hard).sum() >= 12 and top.sum() >= 5:
        for name, (model, _) in options.items():
            fitted = clone(model)
            if isinstance(fitted, HardRegionBlend):
                fitted.threshold = thr
            fitted.fit(X[easy], yv[easy])
            err = yv[top] - fitted.predict(X[top])
            ext[name] = (float(np.mean(np.abs(err))), float(-err.mean()))
    # Choice rule. Trees cannot extrapolate (they go flat past the edge of the data and
    # stay optimistic), so the hand-over model is preferred whenever it exists, unless
    # it is clearly worse (>25%) than the forest on the hard plays or on the
    # extrapolation test. Held-out error on a few dozen hard plays is too noisy to
    # pick a winner by tiny differences.
    def hard_mae(name):
        return float(np.mean(np.abs(yv - options[name][1])[hard]))

    blend = "Forest + hard-map ridge"
    use_blend = hard.sum() >= 20
    if use_blend and hard_mae(blend) > 1.25 * hard_mae("Random forest"):
        use_blend = False
    if use_blend and blend in ext and "Random forest" in ext and ext[blend][0] > 1.25 * ext["Random forest"][0]:
        use_blend = False
    if use_blend:
        name = blend
    else:
        name = min(("Ridge", "Random forest"), key=hard_mae)
    model, oof = options[name]
    best = {"name": name, "model": model, "oof": oof, "mae_hard": hard_mae(name),
            "ext_mae": ext.get(name, (None, None))[0], "ext_bias": ext.get(name, (None, None))[1]}
    best["ext_all"] = ext
    resid = yv - best["oof"]
    best.update(
        resid=resid,
        stars=stars,
        mae=float(np.mean(np.abs(resid))),
        r2=float(1 - (resid ** 2).sum() / ((yv - yv.mean()) ** 2).sum()),
        base_mae=float(np.mean(np.abs(yv - yv.mean()))),
        bias_hard=float(-resid[hard].mean()),  # + = model too optimistic on your harder plays
    )
    best["model"].fit(X, yv)
    return best


def local_quantiles(model: dict, cand_stars, qs, k_frac: float = 0.2, k_min: int = 30) -> np.ndarray:
    """Error quantiles for maps of SIMILAR DIFFICULTY. For each candidate, take the
    held-out errors of the training plays closest in star rating (nearest 20%, at
    least 30) and read the quantiles off those.

    This both measures the spread and corrects the bias: where your data is thin
    (your hardest maps) the model tends to be too optimistic, and those plays'
    errors carry exactly that. One global error distribution would be wrong in both
    directions: easy maps near the 100% ceiling are predicted tightly, hard maps
    much less so."""
    stars, resid = model["stars"], model["resid"]
    k = int(min(len(stars), max(k_min, k_frac * len(stars))))
    out = np.empty((len(cand_stars), len(qs)))
    for i, s in enumerate(np.asarray(cand_stars, dtype=float)):
        nearest = np.argsort(np.abs(stars - s))[:k]
        out[i] = np.quantile(resid[nearest], qs)
    return out


# --------------------------------------------------------------------------- candidates
def choose_mods(hist: pd.DataFrame, requested) -> list:
    if requested:
        keys = ["NM"] + [mods_key(m) for m in requested.split(",") if m.strip()]
    else:
        counts = hist["mods_key"].value_counts()
        keys = ["NM"] + [k for k, n in counts.items() if k != "NM" and n >= MIN_MOD_PLAYS]
    return list(dict.fromkeys(keys))


def build_candidates(cand: pd.DataFrame, mod_keys: list, engine: PPEngine):
    rows, skipped = [], 0
    for c in cand.itertuples():
        for key in mod_keys:
            try:
                stars = engine.attrs(c.beatmap_id, key)["stars"]
            except MapUnavailable:
                skipped += 1
                break
            eff = effective_settings(c.bpm, c.ar, c.cs, c.od, key)
            row = {
                "beatmap_id": c.beatmap_id, "title": c.title, "artist": c.artist,
                "version": c.version, "mods": key, "star_mod": stars,
                "eff_bpm": float(eff["bpm"]), "eff_ar": float(eff["ar"]),
                "eff_cs": float(eff["cs"]), "eff_od": float(eff["od"]),
                "length_s": c.length_s,
            }
            for flag in MOD_FLAG_COLS:
                row[flag] = int(flag.split("_")[1] in key)
            rows.append(row)
    return pd.DataFrame(rows), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", choices=["pp", "comfort", "stretch"], default="pp")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--mods", help="options to consider, e.g. NM,DT,HDDT (default: no mod plus the mods you actually use)")
    ap.add_argument("--risk", type=float, default=0.25,
                    help="accuracy quantile to plan around: 0.25 = a 'worse than usual' day (default), 0.5 = typical")
    ap.add_argument("--stretch", type=float, default=0.4,
                    help="how many stars above your hardest play to still consider (default 0.4)")
    ap.add_argument("--comfort-acc", type=float,
                    help="accuracy %% the comfort goal must still reach on a bad day (default: your median)")
    ap.add_argument("--max-candidates", type=int, default=150,
                    help="score at most this many candidate maps, most popular first (default 150)")
    ap.add_argument("--exclude-played", action="store_true", help="hide maps you have already played")
    ap.add_argument("--allow-extrapolation", action="store_true",
                    help="include maps with settings outside what you have played (marked *)")
    ap.add_argument("--no-download", action="store_true", help="only use already-cached .osu files")
    ap.add_argument("--csv", default="recommendations.csv")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    engine = PPEngine(download=not args.no_download)
    hist, best_pp, dropped = load_history(engine)
    if len(hist) < MIN_TRAIN:
        print(f"Only {len(hist)} usable passed plays (need {MIN_TRAIN}). "
              "Play and run fetch_scores.py / collect_plays.py to build up history first.")
        return
    if dropped:
        print(f"({dropped} of your plays skipped: beatmap file unavailable.)")

    conn = sqlite3.connect(DB_PATH)
    try:
        cand = pd.read_sql("SELECT * FROM candidates", conn)
    except Exception:
        cand = pd.DataFrame()
    conn.close()
    if cand.empty:
        print("No candidate maps yet. Run:  python fetch_candidates.py")
        return

    # --- model + calibration ---------------------------------------------------
    model = fit_accuracy_model(hist)
    ratio, n_ratio = pp_ratio(hist, engine)
    print(f"Accuracy model: {model['name']}, trained on {len(hist)} passed plays.")
    print(f"  Held-out error: {model['mae']:.2f} accuracy points (guessing your average: {model['base_mae']:.2f}); "
          f"R2 = {model['r2']:.2f}")
    print(f"  On your hardest third of plays: error {model['mae_hard']:.2f} points, "
          f"bias {model['bias_hard']:+.2f} (+ = tends to be optimistic; corrected for below).")
    if model["r2"] < 0.1:
        print("  Heads-up: map settings explain very little of your accuracy, so expected accuracy is\n"
              "  close to your average. Rankings will mostly reflect pp potential, not accuracy.")
    print(f"  pp realization ratio: {ratio:.2f} (from {n_ratio} plays; scales pp estimates).")

    mod_keys = choose_mods(hist, args.mods)
    print(f"Mod options considered: {', '.join(mod_keys)}")

    # --- score candidates --------------------------------------------------------
    cand = cand.sort_values("playcount", ascending=False).head(args.max_candidates)
    n_dl = 0 if args.no_download else count_uncached(cand["beatmap_id"])
    if n_dl:
        print(f"Fetching {n_dl} candidate beatmap files, one time only (about {n_dl * 0.8:.0f} s)...")
    cdf, skipped = build_candidates(cand, mod_keys, engine)
    if skipped:
        print(f"({skipped} candidate map(s) skipped: beatmap file unavailable.)")
    if cdf.empty:
        print("No candidates could be scored.")
        return

    # --- range gating: don't extrapolate far beyond what you have played ----------------
    lo, hi = hist[NUMERIC].min(), hist[NUMERIC].max()
    pad = 0.05 * (hi - lo)
    cdf["extrapolating"] = ((cdf[NUMERIC] < lo - pad) | (cdf[NUMERIC] > hi + pad)).any(axis=1)

    star_lo = float(np.percentile(hist["star_mod"], 5))
    star_hi = float(hist["star_mod"].max()) + args.stretch
    p90 = float(np.percentile(hist["star_mod"], 90))
    in_band = (cdf["star_mod"] >= star_lo) & (cdf["star_mod"] <= star_hi)
    if args.goal == "stretch":
        in_band &= cdf["star_mod"] >= p90
    n_before = len(cdf)
    cdf = cdf[in_band]
    n_extrap = int(cdf["extrapolating"].sum())
    if not args.allow_extrapolation:
        cdf = cdf[~cdf["extrapolating"]]
    print(f"Candidates in range for goal '{args.goal}': {len(cdf)} of {n_before} map/mod combinations "
          f"(star rating {star_lo:.2f} to {star_hi:.2f}"
          + (f"; {n_extrap} excluded for settings outside your experience" if n_extrap and not args.allow_extrapolation else "")
          + ").")
    if cdf.empty:
        print("Nothing left after range filtering. Try --stretch, --allow-extrapolation, or fetch more candidates.")
        return
    cdf = cdf.copy()

    # --- predictions -----------------------------------------------------------------
    pred = model["model"].predict(cdf[FEATURES])
    q = local_quantiles(model, cdf["star_mod"].to_numpy(), [0.5, args.risk])
    cdf["exp_acc"] = np.clip(pred + q[:, 0], 50, 100)
    cdf["safe_acc"] = np.clip(pred + q[:, 1], 50, 100)

    pp_mid, pp_safe, gain, played = [], [], [], []
    top = np.sort(best_pp.to_numpy())[::-1][:TOP_N]
    for r in cdf.itertuples():
        pm = engine.pp(r.beatmap_id, r.mods, r.exp_acc) * ratio
        ps = engine.pp(r.beatmap_id, r.mods, r.safe_acc) * ratio
        existing = best_pp.get(r.beatmap_id)
        pp_mid.append(pm)
        pp_safe.append(ps)
        gain.append(marginal_gain(top, existing, ps))
        played.append(existing if existing is not None else np.nan)
    cdf["pp_exp"], cdf["pp_safe"], cdf["gain"], cdf["your_best_pp"] = pp_mid, pp_safe, gain, played

    # --- rank ----------------------------------------------------------------------------
    if args.goal == "comfort":
        comfort_acc = args.comfort_acc if args.comfort_acc is not None else float(np.median(hist["accuracy"] * 100))
        ok = (cdf["safe_acc"] >= comfort_acc) & (cdf["gain"] >= MIN_GAIN)
        print(f"Comfort filter: safe accuracy at least {comfort_acc:.1f}% -> {int(ok.sum())} combinations qualify.")
        if not ok.any():
            print("  Nothing reaches that bar. Lower it with --comfort-acc (for example --comfort-acc 96),\n"
                  "  or widen the pool with --stretch or more fetch_candidates.py pages.")
        ranked = cdf[ok].sort_values("gain", ascending=False)
    else:
        ranked = cdf.sort_values("gain", ascending=False)
    if args.exclude_played:
        ranked = ranked[ranked["your_best_pp"].isna()]
    ranked = ranked.drop_duplicates("beatmap_id")  # best mod option per map

    cdf.sort_values("gain", ascending=False).to_csv(args.csv, index=False)

    # --- print -----------------------------------------------------------------------------
    show = ranked.head(args.top).copy()
    if show.empty:
        print("No maps met the goal's conditions.")
        return

    def label(r):
        t = f"{r.title} [{r.version}]"
        return (t[:42] + "...") if len(t) > 45 else t

    show["map"] = [label(r) for r in show.itertuples()]
    show["note"] = [
        ("*extrapolating " if r.extrapolating else "")
        + (f"your best {r.your_best_pp:.0f}pp" if pd.notna(r.your_best_pp) else "new")
        for r in show.itertuples()
    ]
    out = show[["map", "mods", "star_mod", "eff_bpm", "eff_ar", "exp_acc", "safe_acc", "pp_safe", "gain", "note"]].rename(
        columns={"star_mod": "stars", "eff_bpm": "bpm", "eff_ar": "AR", "exp_acc": "acc%", "safe_acc": "safe%",
                 "pp_safe": "pp", "gain": "+total"})
    print(f"\nTop {len(out)} for goal '{args.goal}' (planning around the {int(args.risk * 100)}th-percentile accuracy):\n")
    print(out.round({"stars": 2, "bpm": 0, "AR": 1, "acc%": 1, "safe%": 1, "pp": 0, "+total": 1}).to_string(index=False))
    print(f"\nFull list saved to {args.csv}.")
    print("\nHow to read this:\n"
          "  acc% = typical expected accuracy; safe% = the lower-quartile case the pp and +total are based on.\n"
          "  +total = rise in your weighted top-100 pp if you set that score (0 = wouldn't beat your best there).\n"
          "  Everything assumes you PASS the map: failed attempts aren't in your history yet, so pass\n"
          "  probability isn't modelled (collect_plays.py is gathering that data).\n"
          "  Your history is mostly top plays, which flatters accuracy estimates; that's why safe% exists.")


if __name__ == "__main__":
    main()