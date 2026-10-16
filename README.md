# osu! Analytics

A personal toolkit that pulls your osu!standard plays from the osu! API (Application Programming Interface), stores them in a local SQLite database, analyses them, predicts your accuracy, and recommends beatmaps with the mods to use, the accuracy to expect and the performance points (pp) you would gain.

Everything runs locally. Your data never leaves your machine except for requests to the osu! servers.

---

## Contents

1. [What's in the box](#1-whats-in-the-box)
2. [Installation](#2-installation)
3. [Credentials and first login](#3-credentials-and-first-login)
4. [Typical workflows](#4-typical-workflows)
5. [Script reference](#5-script-reference)
6. [Understanding the recommender](#6-understanding-the-recommender)
7. [Limitations (read this before trusting a number)](#7-limitations-read-this-before-trusting-a-number)
8. [Troubleshooting](#8-troubleshooting)
9. [Files created on disk](#9-files-created-on-disk)

---

## 1. What's in the box

| Stage | Script | What it does |
|---|---|---|
| Setup | `oauth_setup.py` | One-time browser login (OAuth, Open Authorization); caches tokens in `./tokens` |
| Setup | `inspect_score.py` | Prints one raw score object, useful for debugging the API |
| Data | `fetch_scores.py` | Downloads your best and recent scores into `osu_scores.db` |
| Data | `collect_plays.py` | Logs every attempt (passes **and** fails) as you play; `--watch` keeps it running |
| Analysis | `analyze.py` | First-pass charts saved to `./charts` |
| Modelling | `predict.py` | Predicts map-level targets (star rating, BPM, AR) and accuracy |
| Modelling | `predict_accuracy.py` | Accuracy model with session features (fatigue, streaks, time of day) and optional mod-aware difficulty |
| Recommender | `calibrate.py` | **Trust check**: does the local pp engine agree with osu! on your plays? |
| Recommender | `fetch_candidates.py` | Builds a pool of ranked maps to recommend from |
| Recommender | `recommend.py` | Ranks maps, mods, expected accuracy and pp gain |
| Library | `db_setup.py`, `osu_client.py`, `features.py`, `mods.py`, `pp_engine.py` | Shared code; not run directly |

Abbreviations used throughout: AR = Approach Rate, CS = Circle Size, OD = Overall Difficulty, BPM = beats per minute, pp = performance points, HD = Hidden, HR = Hard Rock, EZ = Easy, DT = Double Time, NC = Nightcore, HT = Half Time, FL = Flashlight, NM = No Mod, R² = coefficient of determination.

---

## 2. Installation

Requirements: Python 3.10 or newer, an osu! account, and internet access.

### 2.1 Create an isolated environment

Use a virtual environment rather than a global or Anaconda base install. Mixing package versions in a shared environment is what produces errors like `numpy.dtype size changed`.

```bash
cd osu-analytics
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)
```

Run the activate line every time you open a new terminal for this project.

### 2.2 Install dependencies

```bash
pip install -r requirements.txt
```

This installs `ossapi` (osu! API client), `python-dotenv`, `pandas`, `matplotlib`, `scikit-learn`, `rosu-pp-py` (a local pp and star-rating calculator) and `requests`.

Check the pp calculator imported correctly:

```bash
python -c "import rosu_pp_py; print('rosu-pp OK')"
```

---

## 3. Credentials and first login

### 3.1 Register an OAuth application

1. Go to <https://osu.ppy.sh/home/account/edit#oauth> and choose **New OAuth Application**.
2. Name it anything. Set **Application Callback URL** to `http://localhost:8080/callback`.
3. Copy the **Client ID** and **Client Secret**.

### 3.2 Create `.env`

Create a file called `.env` in the project folder:

```bash
OSU_CLIENT_ID=12345
OSU_CLIENT_SECRET=paste-your-secret-here
OSU_REDIRECT_URI=http://localhost:8080/callback
OSU_USERNAME=YourOsuUsername
```

- `OSU_REDIRECT_URI` must match the registered callback URL **character for character**, including any trailing slash.
- Keep `.env`, `tokens/` and `osu_scores.db` out of version control. If you use Git, add them to `.gitignore`:

```
.env
tokens/
osu_scores.db
osu_files/
charts/
.venv/
```

### 3.3 First login

```bash
python oauth_setup.py
```

A browser tab opens. Log in to osu!, approve the application, and return to the terminal. You should see `Success. Logged in as: <you>`. Tokens are cached in `./tokens`, so you will not need to log in again.

---

## 4. Typical workflows

### 4.1 First-time run, in this order

```bash
python fetch_scores.py                 # 1. load your best + recent scores into the database
python analyze.py                      # 2. charts in ./charts
python calibrate.py                    # 3. check the pp engine against your real plays (downloads map files once)
python fetch_candidates.py             # 4. build the pool of maps to recommend from
python recommend.py                    # 5. recommendations (default goal: pp)
```

**Do step 3 before trusting step 5.** See [section 5.6](#56-calibratepy).

### 4.2 Start logging every attempt (do this early)

The API only exposes your top plays and a short recent list, and neither includes most fails. To model *when* you fail or tire, the tool has to watch you play:

```bash
python collect_plays.py                # one sweep
python collect_plays.py --watch 10     # keep running, check every 10 minutes (Ctrl+C to stop)
```

Leave `--watch` running in a spare terminal while you play. Session features only become trustworthy for plays logged **after** the first collector run, because the earlier history is sparse and passes-only.

### 4.3 Accuracy modelling

```bash
python predict.py                      # map-level targets and accuracy from map settings
python predict_accuracy.py             # adds session features, using plays logged by the collector
python predict_accuracy.py --mod-stars # uses mod-adjusted star rating (needs map files; see 5.4)
python predict_accuracy.py --all       # dry run on ALL stored plays (see warning below)
```

### 4.4 Getting recommendations

```bash
python recommend.py                                  # biggest pp gain at your "safe" accuracy
python recommend.py --goal comfort --top 20          # maps you should reliably clear at your usual accuracy
python recommend.py --goal stretch                   # the top end of your range
python recommend.py --mods NM,DT,HDDT --exclude-played
python recommend.py --risk 0.5 --csv my_recs.csv     # plan around a typical day, save the full table
```

### 4.5 Keeping things fresh

Re-run `fetch_scores.py` and `fetch_candidates.py` every few weeks. `recommend.py` always uses whatever is currently in the database.

---

## 5. Script reference

### 5.1 `fetch_scores.py`

Stores your best and recent scores. Safe to re-run: scores already stored are never duplicated or overwritten.

### 5.2 `collect_plays.py`

| Option | Meaning |
|---|---|
| `--watch MINUTES` | Poll continuously every N minutes instead of once |

Pages through your recent plays (including fails) until it reaches scores it already has. On first run it records `collector_started_at` in the database, which later defines the window in which the play log is complete.

### 5.3 `analyze.py` and `predict.py`

No options. `analyze.py` writes PNG charts to `./charts`. `predict.py` reports cross-validated results per target against a "always guess the average" baseline, grouped by beatmap so a map can never appear in both training and test data.

### 5.4 `predict_accuracy.py`

| Option | Meaning |
|---|---|
| `--all` | Dry run on all stored plays, not only those after the collector started. Session reconstruction is invalid on the older, sparse history, so treat results as a pipeline check only |
| `--since YYYY-MM-DD` | Only use plays on or after this date |
| `--gap MINUTES` | Idle time that starts a new session (default 30) |
| `--history N` | How many previous attempts the rolling features look at (default 5) |
| `--mod-stars` | Use star rating **under your mods** (needs `.osu` files; downloaded once) |

It compares four models: baseline, map only, map plus session, session only. A session effect is reported as "detected" only if **both** the random forest and the ridge regression improve (mean R² gain above 0.05, positive in at least n−1 folds, and larger than its own fold-to-fold spread).

### 5.5 `fetch_candidates.py`

| Option | Meaning |
|---|---|
| `--pages N` | Pages of 50 beatmap sets to scan (default 6) |
| `--stars-min`, `--stars-max` | Star range. Default: a bit below your 5th percentile (modded plays correspond to lower unmodded stars) up to 1 star above your hardest play |
| `--sort` | Website sort order: `plays_desc` (default), `ranked_desc`, `favourites_desc`, `rating_desc`, `difficulty_asc`, … |
| `--clear` | Empty the candidate pool first |

### 5.6 `calibrate.py`

| Option | Meaning |
|---|---|
| `--limit N` | Check only N distinct beatmaps (quick test) |
| `--no-download` | Use cached `.osu` files only |

It reports two things:

1. **Star ratings**: rosu-pp's value versus the API's, for your unmodded plays.
2. **pp**: real pp versus the engine's full-combo pp at your accuracy.
   - On plays that were genuinely full combos, the median ratio should sit near **1.00**. If so, you get "the engine agrees with osu! on your full-combo plays".
   - The "Everything" median is your **pp realization ratio**: the fraction of ideal pp you actually keep after misses and combo breaks. The recommender multiplies its pp estimates by it.

If the verdict says the engine disagrees, the usual causes are lazer scoring differences, custom rate mods (only the mod acronym is stored, so a custom DT speed is lost) or Difficulty Adjust. The recommender still works as a *ranking*, but absolute pp is approximate.

### 5.7 `recommend.py`

| Option | Default | Meaning |
|---|---|---|
| `--goal pp\|comfort\|stretch` | `pp` | What to optimise (section 6) |
| `--top N` | 15 | Rows to print |
| `--mods A,B,C` | NM plus mods you use ≥ 8 times | Mod **options**, comma-separated. Write a combination without commas: `NM,DT,HDDT` means three options (none, DT, HD+DT) |
| `--risk Q` | 0.25 | Accuracy quantile to plan around. 0.25 is a worse-than-usual day; 0.5 is typical |
| `--stretch STARS` | 0.4 | How far above your hardest play to still consider |
| `--comfort-acc PCT` | your median | Accuracy the `comfort` goal must still reach on a bad day |
| `--max-candidates N` | 150 | Score at most N candidate maps, most popular first |
| `--exclude-played` | off | Hide maps you have already played |
| `--allow-extrapolation` | off | Include maps whose settings lie outside what you have played (marked `*`) |
| `--no-download` | off | Do not fetch missing `.osu` files |
| `--csv PATH` | none | Save the full ranked table |

The first run downloads one `.osu` file per beatmap it needs (about 1 second each) and caches them in `./osu_files/`.

---

## 6. Understanding the recommender

For every candidate map and mod option, `recommend.py`:

1. **Recomputes difficulty under the mods.** HR multiplies AR and OD by 1.4 (capped at 10) and CS by 1.3; EZ halves them; DT/NC speed the map by 1.5×, HT/DC slow it to 0.75×. AR and OD are converted through their millisecond timing windows so rate changes are handled correctly. Star rating comes from rosu-pp.
2. **Predicts your accuracy** from your own passed plays, using effective BPM, AR, CS, OD, mod-adjusted star rating and mod flags.
   - Below the hardest part of your history, a random forest is used.
   - On your hard maps, a straight-line model trained only on your harder plays takes over, because a random forest cannot extrapolate: past the edge of its data it goes flat and stays optimistic.
3. **Measures uncertainty from held-out errors** on plays of similar difficulty that the model never trained on. `acc%` is the typical (median) expectation; `safe%` is the lower-quartile case at `--risk`.
4. **Converts accuracy to pp** at `safe%`, scaled by your realization ratio.
5. **Computes marginal gain.** osu! sums your top 100 scores with weights 0.95ⁱ, so a new score is worth more the higher it ranks. The tool recomputes your weighted total with the new score inserted (replacing your existing score on that map if you have one). A score that does not beat your current best on a map gains 0.

### What the goals mean

- **`pp`**: largest gain in your weighted total, at your safe accuracy.
- **`comfort`**: maps where even a worse-than-usual day should still reach your usual accuracy, ranked by gain. If nothing qualifies, lower the bar with `--comfort-acc`.
- **`stretch`**: maps at or above the 90th percentile of the difficulty you actually play, up to `--stretch` stars beyond your hardest, ranked by gain.

### Reading the table

| Column | Meaning |
|---|---|
| `mods` | Recommended mod option for that map (best one per map) |
| `stars` | Star rating under those mods |
| `acc%` | Typical expected accuracy |
| `safe%` | Lower-quartile accuracy that `pp` and `+total` are based on |
| `pp` | Estimated pp at `safe%` |
| `+total` | Rise in your weighted top-100 total |
| `note` | `new`, or your existing best on that map |

---

## 7. Limitations (read this before trusting a number)

- **Assumes you pass.** Fails are not in your history until `collect_plays.py` has logged some, so pass probability is not modelled. A map with a 94% expected accuracy might still be a fail for you.
- **Selection bias.** Your stored history is dominated by top plays. Those are your good days, so accuracy estimates lean flattering. The `safe%` column exists to counter this; it does not remove it.
- **Thin hard end.** Recommendations concentrate where you have the least data. On synthetic tests the hand-over model reduced optimism on the hardest maps to roughly 0.2–0.6 accuracy points and brought "safe" coverage to roughly 0.5–0.65 against a 0.75 target. Better, not perfect. Treat the very top results with extra suspicion.
- **Weak predictors on real data.** Earlier accuracy models on your real plays had low R² (around 0.1). Wide `safe%` gaps are expected, not a bug.
- **Lazer custom rates** (for example 1.35× DT) are not stored by the API response in a recoverable form and are treated as standard rates.
- **Session features are only valid for plays logged by the collector.** `--all` is for testing the pipeline, not for conclusions.
- **Network steps were unverified in development.** The `.osu` download from osu.ppy.sh, the star-filter search syntax and the live API calls could not be exercised from the development sandbox. If one fails on your machine, see the troubleshooting section and report the exact error.
- **Not affiliated with osu!.** Be considerate with request volume. The scripts include short delays between requests; do not remove them.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `KeyError: 'OSU_CLIENT_ID'` (or similar) | `.env` is missing a value or not in the project folder. Check all four variables |
| Redirect URI mismatch | `OSU_REDIRECT_URI` must match the registered callback exactly |
| Port already in use | Pick a free port (for example 8765) in **both** `.env` and the osu! OAuth application |
| `FileNotFoundError` for a token file | Run from the project folder; `./tokens` is created on first login |
| `numpy.dtype size changed` or a scipy `ValueError` | Mixed package versions, usually from Anaconda base. Use the virtual environment in section 2 |
| `KeyError: 'Random forest'` | A typo in a local edit of a model name. Compare with the delivered file |
| `rosu_pp_py` import error | `pip install rosu-pp-py` inside the activated environment |
| Many `.osu` downloads fail | The file download is rate-limited or blocked. Wait and re-run (cached files are kept), or use `--no-download` with the maps you already have |
| `Nothing left after range filtering` | Fetch more candidates, or use `--stretch` / `--allow-extrapolation` |
| `Comfort filter … 0 combinations qualify` | Lower `--comfort-acc`, or widen the pool |
| `Too few plays` messages | The relevant model needs more data. Keep the collector running |

If the database gets into a bad state you can delete `osu_scores.db` and re-run `fetch_scores.py`; you will lose only the collector's history of fails.

---

## 9. Files created on disk

| Path | Contents | Safe to delete? |
|---|---|---|
| `.env` | Your credentials | No: keep private |
| `tokens/` | Cached OAuth tokens | Yes (you will log in again) |
| `osu_scores.db` | All plays, the `meta` table and the candidate pool | Only if you accept losing logged fails |
| `osu_files/` | Cached `.osu` map files | Yes (re-downloaded when needed) |
| `charts/` | PNGs from `analyze.py` and `predict.py` | Yes |