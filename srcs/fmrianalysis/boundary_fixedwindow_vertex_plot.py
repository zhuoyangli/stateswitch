"""
Stage 2 (plot) — render the fixed-window boundary activation maps computed by
boundary_fixedwindow_vertex.py.

Model-free analogue of boundary_glm_vertex_plot.py. For each condition (wordGen,
explGen, movieBetween, movieRecall) and each alignment (onset, offset) it renders
one 4-panel (L/R x lateral/medial) vertex map. By default it renders the group
one-sample-t maps; --per-subject also renders each subject's mean-window map.

Run under the MNE/PyVista env (NOT uv):
    /home/envs/ssgl/bin/python srcs/fmrianalysis/boundary_fixedwindow_vertex_plot.py
    /home/envs/ssgl/bin/python srcs/fmrianalysis/boundary_fixedwindow_vertex_plot.py \
        --per-subject --subjects sub-003 sub-004
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # mne_plotting / plotting_config
sys.path.insert(0, str(_HERE.parent))   # configs.config

import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy import stats

from configs.config import FIGS_DIR, ANALYSIS_CACHE_DIR, SUBJECT_IDS
from mne_plotting import plot_surface_stat_map

CACHE_DIR = ANALYSIS_CACHE_DIR / 'boundary_fixedwindow_vertex'
OUTPUT_DIR = FIGS_DIR / 'boundary_fixedwindow_vertex'

CONDITIONS = ['wordGen', 'explGen', 'movieBetween', 'movieRecall']
COND_LABEL = {
    'wordGen':      'Word Generation (between-category)',
    'explGen':      'Explanation Generation (between-scenario)',
    'movieBetween': 'Movie Watching (between-movie)',
    'movieRecall':  'Movie Recall (between-movie)',
}
REGRESSORS = ['onset', 'offset']
REG_LABEL = {'onset': 'boundary onset', 'offset': 'boundary offset'}

# Per-subject mean-window map is in z-scored-BOLD units; small dynamic range.
SUBJ_THRESHOLD = 0.0
SUBJ_VMAX = 0.5

# Group one-sample t across subjects; threshold at the per-df t-crit.
GROUP_VMAX = 8.0


def _p_tag(alpha):
    return 'p' + f'{alpha:g}'.split('.')[-1]


def _win_note(d):
    try:
        return f'{float(d["window_start"]):g}-{float(d["window_end"]):g} s'
    except KeyError:
        return 'fixed window'


def _render_group(npz_path, condition, out_dir, alpha, vmax=GROUP_VMAX):
    d = np.load(npz_path)
    n_sub = int(d['n_subjects'])
    df = int(d['df'])
    win = _win_note(d)
    if alpha is None:
        t_crit, tag, pnote = 0.0, 'unthresh', 'unthresholded'
    else:
        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
        tag, pnote = _p_tag(alpha), f'p<{alpha:g} unc.'
    out_dir.mkdir(parents=True, exist_ok=True)
    for reg in REGRESSORS:
        lh = d[f'{reg}_L_t'].astype(float)
        rh = d[f'{reg}_R_t'].astype(float)
        out = out_dir / f'group_{condition}_{reg}_{tag}.png'
        plot_surface_stat_map(
            lh, rh, out,
            title=(f'{COND_LABEL[condition]} — {REG_LABEL[reg]} '
                   f'(fixed {win}, group one-sample t, N={n_sub}, {pnote})'),
            threshold=t_crit, vmax=vmax,
            colorbar_label=f't (df={df})')
        print(f'  saved {out.relative_to(FIGS_DIR)}  (t-crit={t_crit:.2f})')


def _render_subject(npz_path, condition, subject, out_dir):
    d = np.load(npz_path)
    win = _win_note(d)
    out_dir.mkdir(parents=True, exist_ok=True)
    for reg in REGRESSORS:
        lh = d[f'{reg}_L'].astype(float)
        rh = d[f'{reg}_R'].astype(float)
        out = out_dir / f'{subject}_{condition}_{reg}.png'
        plot_surface_stat_map(
            lh, rh, out,
            title=(f'{COND_LABEL[condition]} — {REG_LABEL[reg]} '
                   f'({subject}, fixed {win})'),
            threshold=SUBJ_THRESHOLD, vmax=SUBJ_VMAX,
            colorbar_label='mean z-BOLD')
        print(f'  saved {out.relative_to(FIGS_DIR)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument('--subjects', nargs='+', default=list(SUBJECT_IDS))
    ap.add_argument('--per-subject', action='store_true',
                    help='also render each subject (default: group maps only)')
    ap.add_argument('--alpha', type=float, nargs='+', default=[0.05, 0.01],
                    help='two-tailed p-threshold(s) for the group t-maps '
                         '(default: 0.05 and 0.01)')
    ap.add_argument('--unthresholded', action='store_true',
                    help='also render an unthresholded group t-map')
    ap.add_argument('--vmax', type=float, default=GROUP_VMAX,
                    help=f'colorbar limit ±vmax (default {GROUP_VMAX:g})')
    args = ap.parse_args()

    alphas = list(args.alpha)
    if args.unthresholded:
        alphas.append(None)

    for condition in args.conditions:
        gpath = CACHE_DIR / f'group_{condition}.npz'
        if gpath.exists():
            n_sub = int(np.load(gpath)['n_subjects'])
            print(f'\ngroup {condition} (N={n_sub})')
            for alpha in alphas:
                _render_group(gpath, condition, OUTPUT_DIR / 'group', alpha,
                              vmax=args.vmax)
        else:
            print(f'\ngroup {condition}: no cache ({gpath.name}), skipping')

        if args.per_subject:
            for subject in args.subjects:
                spath = CACHE_DIR / f'{subject}_{condition}.npz'
                if not spath.exists():
                    continue
                print(f'{subject} {condition}')
                _render_subject(spath, condition, subject, OUTPUT_DIR / subject)


if __name__ == '__main__':
    main()
