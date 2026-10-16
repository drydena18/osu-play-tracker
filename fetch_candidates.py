"""
Builds a pool of candidate beatmaps for the recommender.

Searches ranked osu!standard maps through the osu! API (the same search as the
beatmap listing page) and stores each difficulty in a `candidates` table with
its star rating, BPM, AR, CS, OD, length and popularity.

By default the star range comes from your own history (a bit below your usual
range up to a bit above your hardest play). The search is filtered by star
range on the server AND again here, so a pool outside your range can't sneak in.

Usage:
    python fetch_candidates.py                       # sensible defaults
    python fetch_candidates.py --pages 10            # a bigger pool (50 sets per page)
    python fetch_candidates.py --stars-min 5 --stars-max 7 --sort ranked_desc

Sort options (as the website names them): plays_desc (default, popular maps),
ranked_desc (newest), favourites_desc, rating_desc, difficulty_asc ...

Abbreviations: AR = approach rate, CS = circle size, OD = overall difficulty.
"""

import argparse
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from ossapi.enums import BeatmapsetSearchCategory, BeatmapsetSearchMode, BeatmapsetSearchSort

from db_setup import DB_PATH, init_db
from osu_client import get_client

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    beatmap_id      INTEGER PRIMARY KEY,
    beatmapset_id   INTEGER,
    title           TEXT,
    artist          TEXT,
    version         TEXT,
    star_rating     REAL,
    bpm             REAL,
    ar              REAL,
    cs              REAL,
    od              REAL,
    drain           REAL,
    length_s        INTEGER,
    max_combo       INTEGER,
    playcount       INTEGER,
    set_play_count  INTEGER,
    favourites      INTEGER,
    fetched_at      TEXT
);
"""

UPSERT = """
INSERT OR REPLACE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def default_star_range(conn) -> tuple:
    """A bit below your usual range, up to a bit above your hardest play."""
    stars = pd.read_sql(
        "SELECT star_rating FROM scores WHERE star_rating IS NOT NULL", conn
    )["star_rating"]
    if stars.empty:
        raise SystemExit("No scores in the database to infer a star range from. "
                         "Pass --stars-min and --stars-max, or run fetch_scores.py first.")
    # Modded plays (e.g. DT) correspond to LOWER unmodded stars, so reach down further
    return max(1.0, float(np.percentile(stars, 5)) - 1.0), float(stars.max()) + 1.0


def beatmap_row(bs, b, now: str) -> tuple:
    return (
        b.id, bs.id, bs.title, bs.artist, b.version, b.difficulty_rating,
        b.bpm, b.ar, b.cs, b.accuracy, b.drain,
        b.total_length, b.max_combo, b.playcount, bs.play_count, bs.favourite_count, now,
    )


def collect(api, lo: float, hi: float, pages: int, sort: str, delay: float = 1.0) -> list:
    query = f"stars>={lo:.2f} stars<={hi:.2f}"
    now = datetime.now(timezone.utc).isoformat()
    rows, cursor, scanned = [], None, 0

    for page in range(pages):
        result = api.search_beatmapsets(
            query=query,
            mode=BeatmapsetSearchMode.OSU,
            category=BeatmapsetSearchCategory.RANKED,
            sort=sort,
            cursor=cursor,
        )
        sets = result.beatmapsets or []
        scanned += len(sets)
        for bs in sets:
            for b in bs.beatmaps or []:
                if b.mode.value != "osu":
                    continue
                if not (lo <= b.difficulty_rating <= hi):  # filter again client-side
                    continue
                rows.append(beatmap_row(bs, b, now))
        print(f"  page {page + 1}: {len(sets)} beatmap sets, {len(rows)} difficulties kept so far")
        cursor = result.cursor
        if not sets or cursor is None:
            break
        time.sleep(delay)

    print(f"Scanned {scanned} beatmap sets.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stars-min", type=float)
    parser.add_argument("--stars-max", type=float)
    parser.add_argument("--pages", type=int, default=6, help="pages of 50 beatmap sets (default 6)")
    parser.add_argument("--sort", default="plays_desc",
                        choices=[s.value for s in BeatmapsetSearchSort],
                        help="how the website sorts results (default plays_desc)")
    parser.add_argument("--clear", action="store_true", help="empty the candidate pool first")
    args = parser.parse_args()

    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    if args.clear:
        conn.execute("DELETE FROM candidates")

    lo, hi = args.stars_min, args.stars_max
    if lo is None or hi is None:
        dlo, dhi = default_star_range(conn)
        lo = dlo if lo is None else lo
        hi = dhi if hi is None else hi
    print(f"Searching ranked osu!standard maps, {lo:.2f} to {hi:.2f} stars, sorted by {args.sort}...")

    rows = collect(get_client(), lo, hi, args.pages, args.sort)
    conn.executemany(UPSERT, rows)
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    conn.close()
    print(f"Stored {len(rows)} difficulties. Candidate pool now holds {total}.")
    print("Next: python recommend.py")


if __name__ == "__main__":
    main()