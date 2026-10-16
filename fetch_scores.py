"""
Pulls your best scores and recent scores from the osu! API and stores
them in SQLite. Safe to re-run any time -- INSERT OR IGNORE means
scores you've already stored won't be duplicated or overwritten.

Run this with: python fetch_scores.py
"""

import sqlite3
from datetime import datetime, timezone

from osu_client import get_client, get_username
from db_setup import DB_PATH, init_db


def mods_to_string(mods) -> str:
    """Convert a list of Mod objects into a comma-separated string like 'HD,DT'."""
    if not mods:
        return ""
    parts = []
    for mod in mods:
        # Mods can come back as objects with an .acronym attribute, or
        # occasionally as plain strings depending on the endpoint -- handle both.
        acronym = getattr(mod, "acronym", None)
        parts.append(acronym if acronym else str(mod))
    return ",".join(parts)


def score_to_row(score, score_type: str) -> tuple:
    beatmap = score.beatmap
    beatmapset = score.beatmapset

    weight_pp = score.weight.pp if score.weight else None

    return (
        score.id,
        score.beatmap_id,
        beatmap.beatmapset_id if beatmap else None,
        beatmapset.title if beatmapset else None,
        beatmapset.artist if beatmapset else None,
        beatmap.version if beatmap else None,
        beatmap.difficulty_rating if beatmap else None,
        beatmap.mode.value if beatmap else None,
        mods_to_string(score.mods),
        score.accuracy,
        score.pp,
        weight_pp,
        score.rank.value if score.rank else None,
        score.max_combo,
        score.total_score,
        int(bool(score.is_perfect_combo)),
        int(bool(score.passed)),
        score.ended_at.isoformat() if score.ended_at else None,
        beatmap.bpm if beatmap else None,
        beatmap.ar if beatmap else None,
        beatmap.cs if beatmap else None,
        beatmap.accuracy if beatmap else None,  # this is Overall Difficulty, not play accuracy
        beatmap.drain if beatmap else None,
        score_type,
        datetime.now(timezone.utc).isoformat(),
    )


INSERT_SQL = """
INSERT OR IGNORE INTO scores (
    score_id, beatmap_id, beatmapset_id, title, artist, difficulty_name,
    star_rating, mode, mods, accuracy, pp, weight_pp, rank, max_combo,
    total_score, is_perfect_combo, passed, played_at, bpm, ar, cs,
    overall_difficulty, drain, score_type, fetched_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def fetch_and_store():
    init_db()  # safe no-op if the table already exists

    api = get_client()
    username = get_username()
    me = api.user(username)

    conn = sqlite3.connect(DB_PATH)
    total_inserted = 0

    for score_type in ("best", "recent"):
        limit = 100 if score_type == "best" else 50
        scores = api.user_scores(me.id, score_type, limit=limit)
        print(f"Fetched {len(scores)} '{score_type}' scores.")

        rows = [score_to_row(s, score_type) for s in scores]
        cursor = conn.executemany(INSERT_SQL, rows)
        conn.commit()
        total_inserted += cursor.rowcount if cursor.rowcount != -1 else 0

    conn.close()
    print(f"Done. New rows inserted this run (approx): {total_inserted}")
    print("Note: re-running this script is safe -- duplicates are skipped automatically.")


if __name__ == "__main__":
    fetch_and_store()