"""
Export filmfest event boundary strength ratings to CSV.

For each of 131 retained event boundaries (after excluding close pairs and
4 artifact TRs), writes:
  TR         – boundary onset in concatenated TR space (HRF-shifted)
  agreement  – fraction of raters who marked this boundary
  rater1–4   – z-scored strength ratings per rater

Usage:
    python srcs/filmfest/filmfest_boundary_strength_ratings.py [--out PATH]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import zscore

import configs.config as cfg

# ---------------------------------------------------------------------------
# Constants (matching the MATLAB script)
# ---------------------------------------------------------------------------
HRF_SHIFT = 3        # TRs to shift boundary timestamps for HRF delay
RUN1_LEN  = 996      # total TRs in run 1
ARTIFACT_TRS = [357, 991, 1634, 1904]  # manually excluded TRs

RETRO_MAT  = cfg.DATA_DIR / "filmfest_boundarystrength" / "retrospective_boundaries" / \
             "retro_boundaries_from_concat_smoothed_only_20231027_formula2.mat"
CLOSE_MAT  = cfg.DATA_DIR / "filmfest_boundarystrength" / "results" / \
             "filmfest_bbr_close_boundaries_closerthan_6sec_20231128.mat"
RATINGS_DIR = cfg.DATA_DIR / "filmfest_boundarystrength" / "boundary_strength"
DEFAULT_OUT = cfg.DATA_DIR / "filmfest_boundarystrength" / \
              "filmfest_bbr_concat_strength_ratings.csv"


def load_boundaries():
    """Return (allB, peaks, exclude_idx) before any filtering."""
    m  = scipy.io.loadmat(str(RETRO_MAT))
    cr = m["concatRetro"][0, 0]

    peaks    = cr["peaks"].flatten()            # (314,) agreement fractions
    TR_field = cr["TR"][0, 0]

    def _cat(field):
        r1 = TR_field[field][0, 0]["run01"].flatten() + HRF_SHIFT
        r2 = TR_field[field][0, 0]["run02"].flatten() + HRF_SHIFT + RUN1_LEN
        return np.concatenate([r1, r2])

    weak   = _cat("weak")
    mid    = _cat("mid")
    strong = _cat("strong")
    allB   = np.sort(np.concatenate([weak, mid, strong]))   # (314,)

    # Build exclusion mask: close boundaries + 4 artifact TRs
    close_m    = scipy.io.loadmat(str(CLOSE_MAT))
    close_trs  = close_m["closeB_TR"].flatten()
    exclude_trs = np.concatenate([close_trs, ARTIFACT_TRS])
    exclude_idx = np.isin(allB, exclude_trs)

    return allB, peaks, exclude_idx


def load_ratings():
    """Return z-scored ratings array (314, n_raters), rater names."""
    files = sorted(glob.glob(os.path.join(str(RATINGS_DIR), "*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {RATINGS_DIR}")

    cols, names = [], []
    for f in files:
        df  = pd.read_excel(f, header=0)
        raw = df.iloc[:, 2].astype(float).values   # 3rd column = strength rating
        cols.append(zscore(raw, ddof=1))            # normalize within rater
        names.append(os.path.splitext(os.path.basename(f))[0])

    return np.column_stack(cols), names


def main(out_path):
    allB, peaks, exclude_idx = load_boundaries()
    norm_data, rater_names   = load_ratings()

    # Apply exclusion to boundaries and agreement
    allB_filt  = allB[~exclude_idx]
    peaks_filt = peaks[~exclude_idx]

    norm_filt = norm_data[~exclude_idx, :]

    n_raters = norm_filt.shape[1]
    rater_cols = {f"rater{i+1}": norm_filt[:, i] for i in range(n_raters)}

    df_out = pd.DataFrame({
        "TR":        allB_filt.astype(int),
        "agreement": peaks_filt,
        **rater_cols,
    })

    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    df_out.to_csv(str(out_path), index=False)
    print(f"Wrote {len(df_out)} boundaries → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output CSV path (default: %(default)s)")
    args = parser.parse_args()
    main(args.out)
