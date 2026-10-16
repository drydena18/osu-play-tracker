"""
One-off inspection script.

Before we design the SQLite schema, let's see exactly what fields osu!
actually gives us back for a single score. This avoids guessing at
attribute names and getting AttributeErrors later.

Run this with: python inspect_score.py
"""

import os
from dotenv import load_dotenv
from ossapi import Ossapi

load_dotenv()

CLIENT_ID = int(os.environ["OSU_CLIENT_ID"])
CLIENT_SECRET = os.environ["OSU_CLIENT_SECRET"]
REDIRECT_URI = os.environ["OSU_REDIRECT_URI"]
USERNAME = os.environ["OSU_USERNAME"]

TOKEN_DIR = "./tokens"
os.makedirs(TOKEN_DIR, exist_ok=True)

api = Ossapi(
    CLIENT_ID,
    CLIENT_SECRET,
    REDIRECT_URI,
    scopes=["public", "identify"],
    token_directory=TOKEN_DIR,
)

me = api.user(USERNAME)

# Pull just your single best score to inspect its structure
best_scores = api.user_scores(me.id, "best", limit=1)

print(f"Got {len(best_scores)} score(s) back.\n")

if not best_scores:
    print("No best scores found -- you may need at least one ranked play.")
else:
    score = best_scores[0]

    print("=== repr() of the score object ===")
    print(repr(score))
    print()

    print("=== Top-level attribute names (via __dict__ or __slots__) ===")
    if hasattr(score, "__dict__"):
        for key, value in vars(score).items():
            print(f"{key}: {value!r}")
    elif hasattr(score, "__slots__"):
        for key in score.__slots__:
            print(f"{key}: {getattr(score, key, '<missing>')!r}")
    else:
        print("Neither __dict__ nor __slots__ found. Falling back to dir().")
        for attr in dir(score):
            if not attr.startswith("_"):
                print(attr)

    print()
    print("=== Nested beatmap object, if present ===")
    beatmap = getattr(score, "beatmap", None)
    if beatmap is not None:
        if hasattr(beatmap, "__dict__"):
            for key, value in vars(beatmap).items():
                print(f"{key}: {value!r}")
        else:
            for attr in dir(beatmap):
                if not attr.startswith("_"):
                    print(attr)
    else:
        print("No 'beatmap' attribute found on score object.")