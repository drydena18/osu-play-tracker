"""
Trust check: does the local pp engine agree with osu! on YOUR plays?

The recommender estimates pp and star ratings with rosu-pp. Before relying on
those numbers, compare them with what the osu! API reported for plays you have
already made:

  1. Star rating: for unmodded plays, rosu's star rating vs the API's
     difficulty_rating (the API only gives the unmodded value).
  2. pp: for plays with a recorded pp, rosu's full-combo pp at your accuracy vs
     the real pp. For plays that really were full combos the two should agree
     closely. For the rest, the ratio (real / full-combo) shows how much misses
     and combo breaks typically cost you; the recommender uses this to scale
     its pp estimates.

Run:  python calibrate.py            (downloads each map's .osu file once)
      python calibrate.py --limit 40 (quick check on 40 maps)
"""

import argparse
import sqlite3

import numpy as np
import pandas as pd

from db_setup import DB_PATH
from mods import add_mod_columns
from pp_engine import MapUnavailable, PPEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only check this many distinct beatmaps")
    parser.add_argument("--no-download", action="store_true", help="use cached .osu files only")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM scores WHERE passed = 1 AND accuracy IS NOT NULL", conn)
    conn.close()
    if df.empty:
        print("No passed plays in the database. Run fetch_scores.py first.")
        return
    df = add_mod_columns(df)

    if args.limit:
        keep = df["beatmap_id"].drop_duplicates().head(args.limit)
        df = df[df["beatmap_id"].isin(keep)]

    n_maps = df["beatmap_id"].nunique()
    print(f"Checking {len(df)} plays on {n_maps} beatmaps. First run downloads missing .osu "
          f"files (about 1 second each); later runs are instant.\n")

    engine = PPEngine(download=not args.no_download)
    star_rows, pp_rows, skipped = [], [], 0
    for r in df.itertuples():
        try:
            stars = engine.attrs(r.beatmap_id, r.mods_key)["stars"]
            calc_pp = engine.pp(r.beatmap_id, r.mods_key, r.accuracy * 100)
        except MapUnavailable:
            skipped += 1
            continue
        if r.mods_key == "NM" and pd.notna(r.star_rating):
            star_rows.append((r.beatmap_id, r.title, r.difficulty_name, r.star_rating, stars))
        if pd.notna(r.pp) and r.pp > 0 and calc_pp > 0:
            pp_rows.append((r.title, r.difficulty_name, r.mods_key, int(r.is_perfect_combo or 0),
                            r.accuracy * 100, r.pp, calc_pp, r.pp / calc_pp))

    if skipped:
        print(f"Skipped {skipped} play(s): .osu file unavailable.\n")

    # ---- 1. star ratings ----------------------------------------------------
    print("=== Star rating: rosu vs osu! API (unmodded plays) ===")
    if star_rows:
        s = pd.DataFrame(star_rows, columns=["beatmap_id", "title", "diff", "api", "rosu"]).drop_duplicates("beatmap_id")
        s["delta"] = s["rosu"] - s["api"]
        print(f"Maps compared: {len(s)}")
        print(f"Mean absolute difference: {s['delta'].abs().mean():.3f} stars "
              f"| max: {s['delta'].abs().max():.3f} | within 0.05: {(s['delta'].abs() < 0.05).mean():.0%}")
        verdict = "good" if s["delta"].abs().mean() < 0.08 else "LARGE - treat star ratings with caution"
        print(f"Verdict: {verdict}")
        worst = s.reindex(s["delta"].abs().sort_values(ascending=False).index).head(3)
        if s["delta"].abs().max() >= 0.1:
            print("Largest differences:")
            print(worst[["title", "diff", "api", "rosu"]].round(2).to_string(index=False))
    else:
        print("No unmodded plays with API star ratings to compare.")
    print()

    # ---- 2. pp ---------------------------------------------------------------
    print("=== pp: real pp vs rosu full-combo pp at your accuracy ===")
    if not pp_rows:
        print("No plays with pp recorded.")
        return
    p = pd.DataFrame(pp_rows, columns=["title", "diff", "mods", "fc", "acc", "real", "calc", "ratio"])
    p = p[(p["ratio"] > 0.3) & (p["ratio"] < 1.7)]  # drop absurd values from bad data

    def line(label, x):
        if len(x) == 0:
            print(f"{label:<28} n=0")
            return
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        print(f"{label:<28} n={len(x):<4} real/calc median {med:.3f}  (middle half {q1:.3f} to {q3:.3f})")

    line("Perfect-combo plays", p.loc[p["fc"] == 1, "ratio"])
    line("All other plays", p.loc[p["fc"] == 0, "ratio"])
    line("Everything", p["ratio"])
    fc = p.loc[p["fc"] == 1, "ratio"]
    print()
    if len(fc) >= 5:
        med, spread = fc.median(), np.percentile(fc, 75) - np.percentile(fc, 25)
        if 0.97 <= med <= 1.03 and spread < 0.06:
            print("Verdict: the engine agrees with osu! on your full-combo plays. Safe to use for recommendations.")
        else:
            print("Verdict: the engine disagrees with osu! on full-combo plays (median not near 1.00, or wide spread).\n"
                  "Likely causes: lazer scoring differences, custom mod speeds (only acronyms are stored),\n"
                  "or mods like Difficulty Adjust. Recommendations are still usable as a RANKING, but\n"
                  "treat absolute pp numbers as approximate. Worst mismatches:")
            bad = p.loc[p["fc"] == 1].reindex((p.loc[p["fc"] == 1, "ratio"] - 1).abs().sort_values(ascending=False).index).head(5)
            print(bad[["title", "diff", "mods", "acc", "real", "calc"]].round(1).to_string(index=False))
    else:
        print("Too few perfect-combo plays to judge the engine; the 'Everything' line is the guide.")
    print("\nThe 'Everything' median is the pp realization ratio the recommender applies to its pp estimates.")


if __name__ == "__main__":
    main()