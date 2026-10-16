import sqlite3
from db_setup import DB_PATH
from fetch_scores import score_to_row, INSERT_SQL
from osu_client import get_client, get_username

api = get_client()
uid = api.user(get_username()).id
print("user id:", uid, "| database file:", DB_PATH)

scores = api.user_scores(uid, "recent", include_fails=True, limit=50, offset=0)
print("API returned:", len(scores), "scores; newest:", scores[0].ended_at)

conn = sqlite3.connect(DB_PATH)
ids = [s.id for s in scores]
print("first 5 score ids:", ids[:5])
known = {r[0] for r in conn.execute(
    f"SELECT score_id FROM scores WHERE score_id IN ({','.join('?' * len(ids))})", ids)}
print("already in database:", len(known), "of", len(ids))

rows = [score_to_row(s, "recent") for s in scores if s.id not in known]
print("rows to insert:", len(rows))
if rows:
    before = conn.execute("select count(*) from scores").fetchone()[0]
    conn.executemany(INSERT_SQL, rows)
    conn.commit()
    after = conn.execute("select count(*) from scores").fetchone()[0]
    print("table rows before/after:", before, after)
print("newest in table:", conn.execute("select max(played_at) from scores").fetchone())