"""
Creates an SQLite database and scores table, if they don't already exist.

One row per score. score_id is the primary key, which is what allows us to
re-run fetch_scores.py without creating duplicate rows.
"""

import sqlite3

DB_PATH = "osu_scores.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    score_id INTEGER PRIMARY KEY,
    beatmap_id INTEGER,
    beatmapset_id INTEGER,
    title TEXT,
    artist TEXT,
    difficulty_name TEXT,
    star_rating REAL,
    mode TEXT,
    mods TEXT,
    accuracy REAL,
    pp REAL,
    weight_pp REAL,
    rank TEXT,
    max_combo INTEGER,
    total_score INTEGER,
    is_perfect_combo INTEGER,
    passed INTEGER,
    played_at TEXT,
    bpm REAL,
    ar REAL,
    cs REAL,
    overall_difficulty REAL,
    drain REAL,
    score_type TEXT,
    fetched_at TEXT
    );
"""

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Database ready at {DB_PATH}.")

if __name__ == "__main__":
    init_db()