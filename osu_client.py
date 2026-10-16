"""
Shared helper for created an authenticated Ossapi client.

Every script that needs to talk to the osu! API imports get_client()
from here instead of repeating the same six lines everywhere.
"""

import os
from dotenv import load_dotenv
from ossapi import Ossapi

def get_client() -> Ossapi:
    load_dotenv()

    CLIENT_ID = int(os.environ["OSU_CLIENT_ID"])
    CLIENT_SECRET = os.environ["OSU_CLIENT_SECRET"]
    REDIRECT_URI = os.environ["OSU_REDIRECT_URI"]

    TOKEN_DIR = "./tokens"
    os.makedirs(TOKEN_DIR, exist_ok=True)

    return Ossapi(
        CLIENT_ID,
        CLIENT_SECRET,
        REDIRECT_URI,
        scopes=["public", "identify"],
        token_directory=TOKEN_DIR,
    )

def get_username() -> str:
    load_dotenv()
    return os.environ["OSU_USERNAME"]