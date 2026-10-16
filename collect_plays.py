"""
Logs EVERY recent attempt (passes AND fails) into the scores table.

Why this exists: session features (how many plays into a session, how long
since the last play, how recent plays went) need a complete play history.
The "best" scores are only the top-100 pp plays, so the gaps between them
are invisible. The only way to get the full picture is to log plays as they
happen.

Usage:
    python collect_plays.py             # one pass, then exit
    python collect_plays.py --watch 20  # poll every 20 minutes until Ctrl+C

Safe to re-run: score_id is the primary key and INSERT OR IGNORE skips
anything already stored.
"""

import argparse
import sqlite3
import time
from datetime import datetime, timezone

from db_setup import DB_PATH, init_db
from fetch_scores import INSERT_SQL, score_to_row
from osu_client import get_client, get_username

PAGE_SIZE = 50
MAX_PAGES = 6 # safety cap per poll: 300 attempts


def mark_collector_start():
    """
    Remember when logging began. Plays before this moment were NOT captured
    with their fails, so session analysis only trusts plays after it.
    """
    conn =sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('collector_started_at', ?)",
        (datetime.now(timezone.utc).isoformat(),)
    )
    conn.commit()
    conn.close()


def collect_once(api, user_id: int) -> int:
    """
    Fetch recent attempts (newest first), page back until we hit known ones
    """
    conn = sqlite3.connect(DB_PATH)
    new_total = 0

    for page in range(MAX_PAGES):
        score = api.user_scores(
            user_id,
            "recent",
            include_fails = True,
            limit = PAGE_SIZE,
            offset = page * PAGE_SIZE,
        )
        if not score:
            break

            ids = [s.id for s in scores]
            placeholders = ",".join("?" * len(ids))
            known = {
                r[0]
                for r in conn.execute(
                    f"SELECT score_id FROM scores WHERE score_id IN ({placeholders})", ids
                )
            }

            rows = [score_to_row(s, "recent") for s in scores if s.id not in known]
            if rows:
                conn.executemany(INSERT_SQL, rows)
                conn.commit()
            new_total += len(rows)

            # Newest-first ordering: once a page contains something we already
            # have, everything older is already stored too. Also stop on a short page.
            if known or len(scores) < PAGE_SIZE:
                break

        conn.close()
        return new_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watch",
        type = int,
        metavar = "MINUTES",
        help = "keep running and poll every N minutes (Ctrl+C to stop)"
    )
    args = parser.parse_args()

    init_db()
    mark_collector_start()
    api = get_client()
    user_id = api.user(get_username()).id

    while True:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        try:
            n = collect_once(api, user_id)
            print(f"[{stamp}] stored {n} new attempt(s).")
        except Exception as exc: # keep a long-running watcher alive through network parse
            if not args.watch:
                raise
            print(f"[{stamp}] poll failed ({exc!r}); will retry next cycle.")

        if not args.watch:
            break
        time.sleep(args.watch * 60)



if __name__ == "__main__":
    main()