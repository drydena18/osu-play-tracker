"""
Session feature engineering.

A "session" is a run of attempts where no two consecutive plays are more than
SESSION_GAP_MIN minutes apart. Everything below is computed from information
that was already known BEFORE the play in question, so the model is never
handed the answer. That is the most important property of these features:
for example, prev_acc_mean uses .shift(1), which excludes the current play.

Features (per attempt):
  play_index        which attempt this is within its session (1 = first)
  session_minutes   minutes since the session's first attempt
  gap_prev_min      minutes since the previous attempt (capped at the session
                    gap; the first attempt of a session gets the cap)
  plays_last_24h    attempts in the previous 24 hours (accumulated fatigue)
  prev_acc_mean     mean accuracy of the last HISTORY passed plays in the session
  prev_fail_rate    share of the last HISTORY attempts that were fails
  fails_in_session  fails so far in this session
  rel_star          star rating minus the mean star rating of recent attempts
                    (positive = stepping up in difficulty)
  hour_sin/hour_cos time of day, local time, encoded as a circle so that
                    23:00 and 01:00 are treated as close together
  is_weekend        Saturday or Sunday (local time)

Abbreviation: HISTORY = how many previous attempts the rolling features look at.
"""

import sqlite3

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Toronto"  # stored timestamps are UTC; hour-of-day needs local time
SESSION_GAP_MIN = 30
HISTORY = 5

SESSION_FEATURES = [
    "play_index",
    "session_minutes",
    "gap_prev_min",
    "plays_last_24h",
    "prev_acc_mean",
    "prev_fail_rate",
    "fails_in_session",
    "rel_star",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]


def add_mod_stars(df: pd.DataFrame, engine=None) -> pd.DataFrame:
    """Adds star_mod: the star rating of the map UNDER the mods that were used.

    Needs each beatmap's .osu file (downloaded once, then cached). Rows whose file
    can't be fetched get NaN. Requires a 'mods_key' column (see mods.add_mod_columns).
    """
    import os

    from pp_engine import MapUnavailable, PPEngine, osu_path

    engine = engine or PPEngine()
    todo = [b for b in df["beatmap_id"].unique() if not os.path.exists(osu_path(b))]
    if todo:
        print(f"Downloading {len(todo)} beatmap file(s), one time only "
              f"(about {len(todo) * (engine.delay + 0.3):.0f} seconds)...")
    stars, missing = [], 0
    for bid, key in zip(df["beatmap_id"], df["mods_key"]):
        try:
            stars.append(engine.attrs(bid, key)["stars"])
        except MapUnavailable:
            stars.append(np.nan)
            missing += 1
    if missing:
        print(f"Could not get star ratings for {missing} play(s); those use NaN.")
    d = df.copy()
    d["star_mod"] = stars
    return d


def get_collector_start(conn: sqlite3.Connection):
    """When collect_plays.py first ran. Only plays after this are a complete log."""
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'collector_started_at'"
        ).fetchone()
    except sqlite3.OperationalError:  # meta table doesn't exist yet
        return None
    return pd.Timestamp(row[0]) if row else None


def add_session_features(
    df: pd.DataFrame,
    gap_minutes: int = SESSION_GAP_MIN,
    history: int = HISTORY,
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    d = df.copy()
    d["played_at"] = pd.to_datetime(d["played_at"], utc=True)
    d = d.sort_values(["played_at", "score_id"]).reset_index(drop=True)
    d["accuracy_pct"] = d["accuracy"] * 100

    passed = d["passed"].fillna(0).astype(int)
    failed = 1 - passed

    # --- session boundaries -------------------------------------------------
    gap = d["played_at"].diff().dt.total_seconds() / 60
    d["session_id"] = (gap.isna() | (gap > gap_minutes)).cumsum()
    sid = d["session_id"]
    g = d.groupby("session_id")

    d["play_index"] = g.cumcount() + 1
    d["session_minutes"] = (
        d["played_at"] - g["played_at"].transform("min")
    ).dt.total_seconds() / 60
    d["gap_prev_min"] = gap.where(d["play_index"] > 1, gap_minutes).clip(upper=gap_minutes)

    # --- history features (shift(1) keeps the current play out of its own features)
    def prev_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(history, min_periods=1).mean()

    acc_if_passed = d["accuracy_pct"].where(passed == 1)
    d["prev_acc_mean"] = acc_if_passed.groupby(sid).transform(prev_mean)
    d["prev_fail_rate"] = failed.groupby(sid).transform(prev_mean)
    d["fails_in_session"] = failed.groupby(sid).cumsum() - failed
    d["prev_star_mean"] = d["star_rating"].groupby(sid).transform(prev_mean)
    d["rel_star"] = d["star_rating"] - d["prev_star_mean"]

    # attempts in the previous 24 hours, excluding the current one
    d["plays_last_24h"] = (
        d.set_index("played_at")["score_id"]
        .rolling("24h", closed="left")
        .count()
        .fillna(0)
        .to_numpy()
    )

    # --- time of day (local) -------------------------------------------------
    local = d["played_at"].dt.tz_convert(tz)
    hour = local.dt.hour + local.dt.minute / 60
    d["local_hour"] = local.dt.hour
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)

    return d


def describe_sessions(d: pd.DataFrame):
    sizes = d.groupby("session_id").size()
    spans = d.groupby("session_id")["session_minutes"].max()
    n_pass = int((d["passed"] == 1).sum())
    print("=== Session overview ===")
    print(f"Attempts: {len(d)} ({n_pass} passes, {len(d) - n_pass} fails)")
    print(f"Date range: {d['played_at'].min().date()} to {d['played_at'].max().date()}")
    print(f"Sessions: {len(sizes)} | median attempts/session: {sizes.median():.0f} "
          f"| sessions with 3+ attempts: {(sizes >= 3).sum()}")
    print(f"Longest session: {spans.max():.0f} minutes")
    print()