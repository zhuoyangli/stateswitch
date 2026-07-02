"""
Parcel-level rendering of the fixed-window (NOT GLM) boundary activation.

Model-free analogue of boundary_glm_parcel_map.py. Reuses the per-subject vertex
mean-window maps cached by boundary_fixedwindow_vertex.py. For each condition x
alignment it averages every subject's vertex map within each Schaefer-400
(17-network) parcel, runs a one-sample t-test across subjects per parcel, and
renders the thresholded (p<0.05, per-df t-crit) parcel map — kept directly
comparable to the fixed-window vertex maps (same window, just parcel-averaged
before the group test).

Run under the MNE/PyVista env (NOT uv):
    /home/envs/ssgl/bin/python srcs/fmrianalysis/boundary_fixedwindow_parcel_map.py
    /home/envs/ssgl/bin/python srcs/fmrianalysis/boundary_fixedwindow_parcel_map.py --alpha 0.01
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # mne_plotting
sys.path.insert(0, str(_HERE.parent))   # configs.config

import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy import stats

from configs.config import FIGS_DIR, ANALYSIS_CACHE_DIR, SUBJECT_IDS
from mne_plotting import _load_schaefer_annot, plot_parcel_surface_map

CACHE_DIR = ANALYSIS_CACHE_DIR / 'boundary_fixedwindow_vertex'
OUTPUT_DIR = FIGS_DIR / 'boundary_fixedwindow_vertex' / 'group_parcel'

CONDITIONS = ['wordGen', 'explGen', 'movieBetween', 'movieRecall']
COND_LABEL = {
    'wordGen':      'Word Generation (between-category)',
    'explGen':      'Explanation Generation (between-scenario)',
    'movieBetween': 'Movie Watching (between-movie)',
    'movieRecall':  'Movie Recall (between-movie)',
}
REGRESSORS = ['onset', 'offset']
REG_LABEL = {'onset': 'boundary onset', 'offset': 'boundary offset'}

N_PARCELS = 400
NETWORKS = 17
GROUP_VMAX = 8.0


def _p_tag(alpha):
    return 'p' + f'{alpha:g}'.split('.')[-1]


def _win_note(d):
    try:
        return f'{float(d["window_start"]):g}-{float(d["window_end"]):g} s'
    except KeyError:
        return 'fixed window'


def _vertex_to_parcels(lh_vals, rh_vals, lh_labels, rh_labels, n_lh):
    """Mean of vertex values within each parcel -> length-N_PARCELS vector."""
    vec = np.full(N_PARCELS, np.nan)
    for local_id in range(1, n_lh + 1):
        m = lh_labels == local_id
        if m.any():
            vec[local_id - 1] = lh_vals[m].mean()
    n_rh = N_PARCELS - n_lh
    for local_id in range(1, n_rh + 1):
        m = rh_labels == local_id
        if m.any():
            vec[n_lh + local_id - 1] = rh_vals[m].mean()
    return vec


def _subject_parcel_maps(condition, subjects, lh_labels, rh_labels, n_lh):
    """Return ({reg: (n_subjects, N_PARCELS)}, used_subjects, win_note)."""
    out = {reg: [] for reg in REGRESSORS}
    used = []
    win = 'fixed window'
    for subject in subjects:
        path = CACHE_DIR / f'{subject}_{condition}.npz'
        if not path.exists():
            continue
        d = np.load(path)
        win = _win_note(d)
        used.append(subject)
        for reg in REGRESSORS:
            vec = _vertex_to_parcels(
                d[f'{reg}_L'].astype(float), d[f'{reg}_R'].astype(float),
                lh_labels, rh_labels, n_lh)
            out[reg].append(vec)
    return {reg: np.array(v) for reg, v in out.items()}, used, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument('--subjects', nargs='+', default=list(SUBJECT_IDS))
    ap.add_argument('--alpha', type=float, nargs='+', default=[0.05],
                    help='two-tailed p-threshold(s) (default: 0.05)')
    ap.add_argument('--vmax', type=float, default=GROUP_VMAX,
                    help=f'colorbar limit ±vmax (default {GROUP_VMAX:g})')
    args = ap.parse_args()

    lh_labels, rh_labels, _ = _load_schaefer_annot(N_PARCELS, NETWORKS)
    n_lh = N_PARCELS // 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        subj_maps, used, win = _subject_parcel_maps(
            condition, args.subjects, lh_labels, rh_labels, n_lh)
        n_sub = len(used)
        if n_sub < 2:
            print(f'{condition}: <2 subjects, skipping')
            continue
        df = n_sub - 1
        print(f'\n{condition} (N={n_sub}, df={df})')
        for reg in REGRESSORS:
            arr = subj_maps[reg]                        # (n_sub, 400)
            t, _p = stats.ttest_1samp(arr, popmean=0.0, axis=0)
            t = np.nan_to_num(t, nan=0.0)
            for alpha in args.alpha:
                t_crit = float(stats.t.ppf(1 - alpha / 2, df))
                tag = _p_tag(alpha)
                out = OUTPUT_DIR / f'group_parcel_{condition}_{reg}_{tag}.png'
                n_sig = int((np.abs(t) > t_crit).sum())
                plot_parcel_surface_map(
                    t, out, n_parcels=N_PARCELS, networks=NETWORKS,
                    title=(f'{COND_LABEL[condition]} — {REG_LABEL[reg]} '
                           f'(fixed {win}, parcel one-sample t, N={n_sub}, '
                           f'p<{alpha:g} unc.)'),
                    threshold=t_crit, vmax=args.vmax,
                    colorbar_label=f't (df={df})')
                print(f'  saved {out.relative_to(FIGS_DIR)}  '
                      f'(t-crit={t_crit:.2f}, {n_sig}/{N_PARCELS} parcels)')


if __name__ == '__main__':
    main()
