"""
Star rating and performance points (pp) under any mod combination.

Uses rosu-pp-py, a Python binding for the library that implements osu!'s
official difficulty and pp formulas. It needs each beatmap's .osu file, which
is downloaded once from osu.ppy.sh and cached in ./osu_files/ (so each map is
fetched at most one time, with a short delay between downloads to be polite).

    engine = PPEngine()
    engine.attrs(1988753, "HD,DT")
    engine.pp(1988753, "DT", accuracy_pct = 98.0)

pp here assumed a full combo (no misses) unless you pass misses/combo. Scores
are calculated with rosu's defaults (lazer-style); small differences from the
website's number are possible, which is exactly what calibrate.py measures.
"""

import os
import time

import requests
import rosu_pp_py as rosu

from mods import mods_key

OSU_DIR = "osu_files"
DOWNLOAD_URL = "https://osu.ppy.sh/osu/{}"
USER_AGENT = "osu-play-tracker (personal analytics script)"

class MapUnavailable(Exception):
    """The .osu file could not be found or downloaded."""


def osu_path(beatmap_id: int) -> str:
    return os.path.join(OSU_DIR, f"{int(beatmap_id)}.osu")


def ensure_map_file(beatmap_id: int, download: bool = True, delay: float = 0.5) -> str:
    path = osu_path(beatmap_id)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if not download:
        raise MapUnavailable(f"Map {beatmap_id} not found in {OSU_DIR}.")

    os.makedirs(OSU_DIR, exist_ok = True)
    try:
        resp = requests.get(
            DOWNLOAD_URL.format(int(beatmap_id)),
            headers = {"User-Agent": USER_AGENT},
            timeout = 30,
        )
    except requests.RequestException as exc:
        raise MapUnavailable(f"Error downloading map {beatmap_id}: {exc}") from exc

    if resp.status_code != 200 or b"osu file format" not in resp.content[:64]:
        raise MapUnavailable(f"Map {beatmap_id} could not be downloaded (HTTP {resp.status_code}).")
    with open(path, "wb") as f:
        f.write(resp.content)
    time.sleep(delay)
    return path


def _mod_kwargs(mods) -> dict:
    key = mods_key(mods)
    return {} if key == "NM" else {"mods": key}


class PPEngine:
    def __init__(self, download: bool = True, delay: float = 0.5):
        self.download = download
        self.delay = delay
        self._maps = {}
        self._attrs = {}

    def beatmap(self, beatmap_id: int):
        bid = int(beatmap_id)
        if bid not in self._maps:
            path = ensure_map_file(bid, self.downlaod, self.delay)
            try:
                self._maps[bid] = rosu.Beatmap(path = path)
            except Exception as exc:
                raise MapUnavailable(f"Error parsing map {bid}: {exc}") from exc
        return self._maps[bid]

    def attrs(self, beatmap_id: int, mods = None) -> dict:
        """
        Star rating and base settings for this map under these mods
        """
        key = (int(beatmap_id), mods_key(mods))
        if key not in self._attrs:
            bm = self.beatmap(beatmap_id)
            d = rosu.Difficulty(**_mod_kwargs(mods)).calculate(bm)
            self._attrs[key] = {
                "stars": d.stars,
                "max_combo": d.max_combo,
                "n_objects": bm.n_objects,
                "base_bpm": bm.bpm,
                "base_ar": bm.ar,
                "base_cs": bm.cs,
                "base_od": bm.od,
            }
        return self._attrs[key]

    def pp(self, beatmap_id: int, mods = None, accuracy_pct: float = 100.0, misses: int = 0, combo: int = None) -> float:
        bm = self.beatmap(beatmap_id)
        kw = dict(_mod_kwargs(mods), accuracy = float(accuracy_pct), misses = int(misses))
        if combo is not None:
            kw["combo"] = int(combo)
        return rosu.Performance(**kw).calculate(bm).pp