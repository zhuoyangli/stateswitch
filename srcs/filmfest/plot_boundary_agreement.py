"""
Plot filmfest event boundary agreement over time, colored by boundary type.

Movie onsets are derived from the strength-rating xlsx files: each row's
within-movie timestamp is subtracted from the corresponding boundary TR to
give a consistent per-movie onset in concatenated TR space.

Usage:
    python srcs/filmfest/plot_boundary_agreement.py [--out PATH]
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import scipy.io

import configs.config as cfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HRF_SHIFT    = 3
RUN1_LEN     = 996
RUN2_LEN     = 917
ARTIFACT_TRS = [357, 991, 1634, 1904]
TR           = 1.5  # seconds

RETRO_MAT   = cfg.DATA_DIR / "filmfest_boundarystrength" / "retrospective_boundaries" / \
              "retro_boundaries_from_concat_smoothed_only_20231027_formula2.mat"
CLOSE_MAT   = cfg.DATA_DIR / "filmfest_boundarystrength" / "results" / \
              "filmfest_bbr_close_boundaries_closerthan_6sec_20231128.mat"
RATINGS_DIR = cfg.DATA_DIR / "filmfest_boundarystrength" / "boundary_strength"
DEFAULT_OUT = cfg.FIGS_DIR / "filmfest" / "filmfest_boundary_agreement_over_time.png"

COLORS = {"weak": "#7fc4e8", "moderate": "#f4a44a", "strong": "#d63b3b"}
LABELS = {"weak": "Weak", "moderate": "Moderate", "strong": "Strong"}

MOVIE_NAMES = {
    1: "CMIYC", 2: "The Record", 3: "Coherence", 4: "Operator", 5: "Panic Attack",
    6: "Validation", 7: "Cargo", 8: "Alike", 9: "One Small Step", 10: "La Jetée",
}


def _parse_timestamp(ts):
    """Parse 'm.ss.d' timestamp to seconds."""
    parts = str(ts).split(".")
    return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 10


def load():
    m  = scipy.io.loadmat(str(RETRO_MAT))
    cr = m["concatRetro"][0, 0]
    peaks    = cr["peaks"].flatten()
    TR_field = cr["TR"][0, 0]

    def _cat(field):
        r1 = TR_field[field][0, 0]["run01"].flatten() + HRF_SHIFT
        r2 = TR_field[field][0, 0]["run02"].flatten() + HRF_SHIFT + RUN1_LEN
        return np.concatenate([r1, r2])

    weak   = _cat("weak")
    mid    = _cat("mid")
    strong = _cat("strong")
    allB   = np.sort(np.concatenate([weak, mid, strong]))   # (314,) unfiltered

    close_m  = scipy.io.loadmat(str(CLOSE_MAT))
    excl_trs = np.concatenate([close_m["closeB_TR"].flatten(), ARTIFACT_TRS])
    excl_idx = np.isin(allB, excl_trs)

    allB_filt  = allB[~excl_idx]
    peaks_filt = peaks[~excl_idx]

    labels = np.empty(len(allB_filt), dtype=object)
    labels[np.isin(allB_filt, weak)]   = "weak"
    labels[np.isin(allB_filt, mid)]    = "moderate"
    labels[np.isin(allB_filt, strong)] = "strong"

    # Derive movie onsets from rating xlsx (rows aligned with allB)
    xlsx_files = sorted(glob.glob(os.path.join(str(RATINGS_DIR), "*.xlsx")))
    df = pd.read_excel(xlsx_files[0], header=0)
    df["ts_sec"] = df.iloc[:, 1].apply(_parse_timestamp)
    df["movie"]  = df.iloc[:, 0].astype(int)

    # onset_TR = (boundary_TR - HRF_SHIFT) - timestamp_within_movie / TR
    df["onset_TR"] = (allB - HRF_SHIFT) - df["ts_sec"] / TR
    movie_onsets_sec = {
        movie: df.loc[df["movie"] == movie, "onset_TR"].mean() * TR
        for movie in range(1, 11)
    }

    return allB_filt, peaks_filt, labels, movie_onsets_sec


def main(out_path):
    allB, agreement, labels, movie_onsets_sec = load()
    time_sec = allB * TR

    fig, ax = plt.subplots(figsize=(14, 4), dpi=300, facecolor="white")

    for btype in ("weak", "moderate", "strong"):
        mask = labels == btype
        ax.scatter(
            time_sec[mask], agreement[mask],
            color=COLORS[btype], label=LABELS[btype],
            s=30, zorder=3, alpha=0.85, linewidths=0,
        )

    # Mark movie onsets
    for movie, onset_sec in movie_onsets_sec.items():
        ax.axvline(onset_sec, color="#aaaaaa", lw=0.8, ls=":", zorder=1)
        ax.text(onset_sec + 3, 0.97, f"M{movie}", fontsize=6, color="#888888", va="top")

    # Mark run boundary
    run_boundary_sec = RUN1_LEN * TR
    ax.axvline(run_boundary_sec, color="#555555", lw=1, ls="--", alpha=0.7, zorder=2)
    ax.text(run_boundary_sec + 3, 0.88, "run 2", fontsize=7, color="#555555", va="top")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Agreement", fontsize=11)
    ax.set_title("Filmfest event boundary agreement over time", fontsize=12)
    ax.set_xlim(0, (RUN1_LEN + RUN2_LEN) * TR)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mlines.Line2D([], [], color=COLORS[k], marker="o", linestyle="None",
                      markersize=6, label=LABELS[k])
        for k in ("weak", "moderate", "strong")
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output figure path (default: %(default)s)")
    args = parser.parse_args()
    main(__import__("pathlib").Path(args.out))
