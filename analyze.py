"""
Loads scores from SQLite into pandas and generates a first pass of
analytics charts, saved as PNGs in ./charts
"""

import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd

from db_setup import DB_PATH

CHART_DIR = "charts"

def load_scores() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM scores", conn)
    conn.close()

    # played_at was stored as an ISO string -- parse it into an actual
    # dataframe so we can sort chronologically and plot it on a time axis
    df["played_at"] = pd.to_datetime(df["played_at"])
    df.sort_values("played_at")

    # accuracy is stored as a 0-1 fraction, convert to a 0-100 percentage
    df["accuracy_pct"] = df["accuracy"] * 100

    return df

def print_summary(df: pd.DataFrame):
    print("=== Summary ===")
    print(f"Total scores in database: {len(df)}")
    print(f"Date range: {df['played_at'].min().date()} to {df['played_at'].max().date()}")
    print(f"Average accuracy: {df['accuracy_pct'].mean():.2f}%")
    print(f"Highest single-play pp: {df['pp'].max():.1f}")
    print(f"Highest star rating played: {df['star_rating'].max():.2f}")
    print()
 
    print("=== Most common mods used ===")
    print(df["mods"].replace("", "(no mod)").value_counts().head(5))
    print()

def plot_pp_over_time(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["played_at"], df["pp"], marker="o", linestyle="-", alpha=0.6)
    ax.set_title("pp per score over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("pp")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "pp_over_time.png"))
    plt.close(fig)

def plot_star_rating_vs_accuracy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["star_rating"], df["accuracy_pct"], alpha=0.6)
    ax.set_title("Star rating vs. accuracy")
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "star_rating_vs_accuracy.png"))
    plt.close(fig)
 
 
def plot_accuracy_rolling_trend(df: pd.DataFrame):
    df_sorted = df.sort_values("played_at").reset_index(drop=True)
    rolling = df_sorted["accuracy_pct"].rolling(window=10, min_periods=3).mean()
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_sorted["played_at"], df_sorted["accuracy_pct"], alpha=0.25, label="Individual scores")
    ax.plot(df_sorted["played_at"], rolling, linewidth=2, label="Rolling avg (10 scores)")
    ax.set_title("Accuracy trend over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "accuracy_trend.png"))
    plt.close(fig)
 
 
def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    df = load_scores()
 
    if df.empty:
        print("No scores found in the database. Run fetch_scores.py first.")
        return
 
    print_summary(df)
 
    plot_pp_over_time(df)
    plot_star_rating_vs_accuracy(df)
    plot_accuracy_rolling_trend(df)
 
    print(f"Charts saved to ./{CHART_DIR}/")
 
 
if __name__ == "__main__":
    main()