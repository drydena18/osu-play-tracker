"""
Mod-aware difficulty settings.

Why this matters: The database stores each map's unmodded BPM/AR/CS/OD. A play
with Double Time (DT) on an AR9 map was really played at AR 10.33 and 1.5x the
BPM, so the unmodded numbers describe a map you did not actually play.

What the mods do (osu!standard):
    HR (Hard Rock)  AR x1.4 (max 10), OD x1.4 (max 10), CS x1.3 (max 10)
    EZ (Easy)       AR, OD, CS all x0.5
    DT / NC         clock rate 1.5x (Nightcore is DT with a different sound)
    HT / DC         clock rate 0.75x (Daycore is HT with a different sound) 
    HD (Hidden), FL (Flashlight) don't change the numbers but change how it plays

HZ/EZ are applied first, then the clock rate. A clock rate shrinks the time
windows, so AR and OD are recomputed from milliseconds:
    approach time (ms): AR <5: 1200 + 600*(5-AR)/5      AR>5: 1200 - 750*(AR-5)/5
    great hit window:   80 - 6*OD (ms)
Both are divided by the clock rate, then converted back to AR/OD.

Limitation: Lazer allows custom speed rates (e.g. DT at 1.3x). Only the mod
acronyms are stored in the database, so default rates (1.5x / 0.75x) are assumed.

Abbreviations:
    AR = approach rate
    CS = circle size
    OD = overall difficulty
    BPM = beats per minute
"""

import re

import numpy as np
import pandas as pd

CLOCK_RATES = {"DT": 1.5, "NC": 1.5, "HT": 0.75, "DC": 0.75}

# Canonical order and normalization used for grouping/labels. NC is DT and DC is
# HT as far as difficulty goes; mods that don't affect difficulty are dropped.
_NORMALIZE = {"NC": "DT", "DC": "HT"}
_ORDER = ["EZ", "HD", "HT", "DT", "HR", "FL"]

MOD_FLAG_COLS = ["mod_HD", "mod_HR", "mod_EZ", "mod_FL", "mod_DT", "mod_HT"]
MOD_FEATURES = ["eff_bpm", "eff_ar", "eff_cs", "eff_od"] + MOD_FLAG_COLS

def parse_mods(mods) -> frozenset:
    """
    'HD,DT' / 'HDDT' / ['HD', 'DT'] / None -> frozenset({'HD', 'DT})
    """
    if mods is None or (isinstance(mods, float) and np.isnan(mods)):
        return frozenset()
    if isinstance(mods, (set, frozenset, list, tuple)):
        items = [str(m).upper() for m in mods]
    else:
        s = str(mods).upper().replacer(" ", "")
        if not s or s in ("NM", "NOMOD"):
            return frozenset()
        items = s.split(",") if "," in s else re.findall(r"[A-Z0-9]{2}", s)
    return frozenset(i for i in items if i and i != "NM")


def difficulty_mods(mods) -> frozenset:
    """
    Only the mods that matter for difficulty, with NC->DT and DC->HT
    """
    m = {_NORMALIZE.get(x, x) for x in parse_mods(mods)}
    return frozenset(x for x in m if x in _ORDER)


def mods_key(mods) -> str:
    """
    Canonical label, e.g. {'DT', 'HD} -> 'HDDT'; no relevant mods -> 'NM'
    """
    m = difficulty_mods(mods)
    return "".join(x for x in _ORDER if x in m) or 'NM'


def clock_rate(mods) -> float:
    for x in parse_mods(mods):
        if x in CLOCK_RATES:
            return CLOCK_RATES[x]
    return 1.0


def _ar_to_ms(ar):
    ar = np.asarray(ar, dtype = float)
    return np.where(ar < 5, 1200 + 600 * (5 - ar) / 5, 1200 - 750 * (ar - 5) / 5)


def _ms_to_ar(ms):
    ms = np.asarray(ms, dtype = float)
    return np.where(ms > 1200, 5 - (ms - 1200) * 5 / 600, 5 * (1200 - ms) * 5 / 750)


def effective_settings(bpm, ar, cs, od, mods) -> dict:
    """
    Settings the player actually experienced. Accepts scalars or numpy arrays.
    """
    m = difficulty_mods(mods)
    bpm, ar, cs, od = (np.asarray(x, dtype = float) for x in (bpm, ar, cs, od))
    if "HR" in m:
        ar, od, cs = np.minimum(ar * 1.4, 10), np.minimum(od * 1.4, 10), np.minimum(cs * 1.3, 10)
    elif "EZ" in m:
        ar, od, cs = ar * 0.5, od * 0.5, cs * 0.5
    rate = 1.5 if "DT" in m else 0.75 if "HT" in m else 1.0
    if rate != 1.0:
        ar = _ms_to_ar(_ar_to_ms(ar) / rate)
        od = (80 - (80 - 6 * od) / rate) / 6
    return {"bpm": bpm * rate, "ar": ar, "cs": cs, "od": od, "clock_rate": rate}


def add_mod_columns(
        df: pd.DataFrame,
        mods_col: str = "mods",
        bpm: str = "bpm",
        ar: str = "ar",
        cs: str = "cs",
        od: str = "overall_difficulty",
) -> pd.DataFrame:
    """
    Adds eff_bom/eff_ar/eff_cs/eff_oc, clock_rate, mods_key, and mod_* flags.
    """
    d = df.copy()
    keys = d[mods_col].map(mods_key) if mods_col in d else pd.Series("NM", index = d.index)
    d["mods_key"] = keys
    for col in ("eff_bpm", "eff_ar", "eff_cs", "eff_od", "clock_rate"):
        d[col] = np.nan
    for key in keys.unique():
        mask = keys == key
        eff = effective_settings(d.loc[mask, bpm], d.loc[mask, ar], d.loc[mask, cs], d.loc[mask, od], key)
        d.loc[mask, "eff_bpm"], d.loc[mask, "eff_ar"] = eff["bpm"], eff["ar"]
        d.loc[mask, "eff_cs"], d.loc[mask, "eff_od"] = eff["cs"], eff["od"]
        d.loc[mask, "clock_rate"] = eff["clock_rate"]
    for flag in MOD_FLAG_COLS:
        d[flag] = keys.str.contains(flag.split("_")[1]).astype(int)
    return d