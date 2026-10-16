"""
First-run OAuth test.
"""

import os
from dotenv import load_dotenv
from ossapi import Ossapi

load_dotenv()

CLIENT_ID = int(os.environ["OSU_CLIENT_ID"])
CLIENT_SECRET = os.environ["OSU_CLIENT_SECRET"]
REDIRECT_URI = os.environ["OSU_REDIRECT_URI"]
USERNAME = os.environ["OSU_USERNAME"]

# token_directory tells ossapi where to cache your access/refresh tokens
# after each successful login, so you're not re-authenticating every run

TOKEN_DIR = "./tokens"
os.makedirs(TOKEN_DIR, exist_ok = True)

api = Ossapi(
    CLIENT_ID,
    CLIENT_SECRET,
    REDIRECT_URI,
    scopes = ["public", "identify"],
    token_directory = TOKEN_DIR,
)

print("Authenticated. Fetching your profile as a sanity check...")

me = api.user(USERNAME)

print(f"Success. Logged in as: {me.username} (user id: {me.id})")
print(f"Global rank: {me.statistics.global_rank}")
print(f"Total PP: {me.statistics.pp}")