#!/usr/bin/env python3
"""
Plot histograms of silence (inter-response) periods for SVF and AHC tasks.
One subplot per session.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project config
from configs.config import DATA_DIR, FIGS_DIR

SVF_DIR = DATA_DIR / "rec/svf_annotated"
AHC_DIR = DATA_DIR / "rec/ahc_sentences"
FIGS_OUT = FIGS_DIR / "behavior"
FIGS_OUT.mkdir(parents=True, exist_ok=True)


def load_svf_silences():
    """Load SVF files, return dict of session_id -> array of silence durations."""
    sessions = {}
    for fp in sorted(SVF_DIR.glob("*.csv")):
        parts = fp.stem.split("_")
        subject, session = parts[0], parts[1]
        label = f"{subject}\n{session}"

        df = pd.read_csv(fp).sort_values("start").reset_index(drop=True)
        # Exclude "next" words (category markers)
        df = df[df["transcription"].str.lower() != "next"].copy()
        silences = (df["start"] - df["end"].shift(1)).dropna()
        # Keep only positive silences (actual gaps)
        silences = silences[silences > 0].values
        sessions[label] = silences
    return sessions


def load_ahc_silences():
    """Load AHC files, return dict of session_id -> array of silence durations."""
    sessions = {}
    for fp in sorted(AHC_DIR.glob("*.xlsx")):
        parts = fp.stem.split("_")
        subject, session = parts[0], parts[1]
        label = f"{subject}\n{session}"

        df = pd.read_excel(fp)
        df["Prompt Number"] = df["Prompt Number"].ffill()
        df = df[df["Possibility Number"].notna()].copy()
        df = df.sort_values("Start Time").reset_index(drop=True)

        silences = (df["Start Time"] - df["End Time"].shift(1)).dropna()
        silences = silences[silences > 0].values
        sessions[label] = silences
    return sessions


def plot_histograms(sessions, task_name, out_path):
    n = len(sessions)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8),
                             sharey=True, sharex=True)
    axes = np.array(axes).flatten()

    all_vals = np.concatenate(list(sessions.values()))
    # Clip extreme outliers for display (99th percentile)
    xmax = np.percentile(all_vals, 99)
    bins = np.linspace(0, xmax, 30)

    for ax, (label, silences) in zip(axes, sessions.items()):
        ax.hist(silences, bins=bins, color="#4c8bbe", edgecolor="white",
                linewidth=0.4)
        ax.set_title(label, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelbottom=True)

        median_val = np.median(silences)
        ax.axvline(median_val, color="tomato", linewidth=1.2, linestyle="--",
                   label=f"med={median_val:.1f}s")
        ax.legend(fontsize=7, frameon=False)

        n5 = np.sum(silences >= 5)
        n10 = np.sum(silences >= 10)
        ax.text(1.02, 0.95, f"≥5s: {n5}\n≥10s: {n10}",
                transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                color="#444444")

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    # Shared labels
    fig.text(0.5, 0.02, "Silence duration (s)", ha="center", fontsize=11)
    fig.text(0.02, 0.5, "Count", va="center", rotation="vertical", fontsize=11)
    fig.suptitle(f"{task_name} — Silence period distribution per session",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    svf_sessions = load_svf_silences()
    ahc_sessions = load_ahc_silences()

    plot_histograms(svf_sessions, "SVF",
                    FIGS_OUT / "svf_silence_histograms.png")
    plot_histograms(ahc_sessions, "AHC",
                    FIGS_OUT / "ahc_silence_histograms.png")
