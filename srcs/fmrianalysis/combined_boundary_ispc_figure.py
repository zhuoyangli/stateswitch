"""
combined_univariate_ispc_figure.py

5-row × 2-column figure combining:
  Column 1: Boundary-aligned BOLD time courses per ROI
  Column 2: Inter-subject pattern correlation (boundary vs mid-event) bar charts per ROI

Rows (top to bottom): Hippocampus, PMC, mPFC, dACC, dlPFC

Usage:
    uv run python srcs/fmrianalysis/combined_univariate_ispc_figure.py --hp --onset --preset onset0 --align onset
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

from configs.config import FIGS_DIR, TR, FILMFEST_SUBJECTS, ANALYSIS_CACHE_DIR

# Import BOLD data pipeline from multitask_boundary_parcel
from fmrianalysis.multitask_boundary_parcel import (
    SUBJECT_IDS, TASK_KEYS, TRS_BEFORE, TRS_AFTER, TITLE_SCENE_OFFSET,
    discover_sessions, find_psychopy_csv,
    parse_trial_onsets, parse_trial_offsets,
    extract_roi_timeseries, extract_event_locked,
    get_movie_boundary_offsets,
)

# Import ISPC helpers from cross_task_ispc_cell_stats
from fmrianalysis.cross_task_ispc_cell_stats import (
    build_cond_indices, _comparison_cell_pairs, _per_subject_means,
    _draw_colored_segments, N_FILM, _DEFAULT_N_SVF, _DEFAULT_N_AHC,
)

RESULT_CACHE_DIR = ANALYSIS_CACHE_DIR / 'cross_task_ispc'
OUTPUT_DIR = FIGS_DIR / 'combined_univariate_ispc'

ROI_SPEC_COMBINED = [
    ('hipp',  'Hippocampus'),
    ('pmc',   'PMC'),
    ('mpfc',  'mPFC'),
    ('dacc',  'dACC'),
    ('dlpfc', 'dlPFC'),
]

TASK_COLORS = {'svf': '#e41a1c', 'ahc': '#ff7f00', 'movie': '#377eb8'}
TASK_LABELS = {'svf': 'WG', 'ahc': 'EG', 'movie': 'MW'}
LABEL_FS = 18


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bold_group_data(align):
    """Load per-subject mean BOLD epochs for the 5 ROIs."""
    roi_keys = [k for k, _ in ROI_SPEC_COMBINED]
    group_data = {t: {k: [] for k in roi_keys} for t in TASK_KEYS}

    # Movie boundary times (same for all subjects)
    movie_boundaries = {}
    for ff_task in ('filmfest1', 'filmfest2'):
        bnd = get_movie_boundary_offsets(ff_task)
        if align == 'onset':
            bnd = [t + TITLE_SCENE_OFFSET for t in bnd]
        # title_onset: use raw movie offsets (= onset of title scene), no shift
        movie_boundaries[ff_task] = bnd

    for subject in SUBJECT_IDS:
        print(f'  Loading BOLD: {subject}')
        fmri_hits = discover_sessions(subject)
        if not fmri_hits:
            continue

        sessions_with_tasks = {}
        for session, task in fmri_hits:
            sessions_with_tasks.setdefault(session, []).append(task)

        task_epochs = {t: {k: [] for k in roi_keys} for t in TASK_KEYS}

        # SVF and AHC sessions
        for session, tasks in sorted(sessions_with_tasks.items()):
            for task_key in tasks:
                csv_path = find_psychopy_csv(subject, session, task_key)
                if csv_path is None:
                    continue
                if align in ('onset', 'title_onset'):
                    trial_events = parse_trial_onsets(csv_path, task_key)
                else:
                    trial_events = parse_trial_offsets(csv_path, task_key)
                if not trial_events:
                    continue
                roi_ts = extract_roi_timeseries(subject, session, task_key)
                if roi_ts is None:
                    continue
                for k in roi_keys:
                    if k not in roi_ts:
                        continue
                    epochs = extract_event_locked(roi_ts[k], trial_events)
                    if epochs is not None:
                        task_epochs[task_key][k].append(epochs)

        # Movie boundary epochs
        if subject in FILMFEST_SUBJECTS:
            ff_session = FILMFEST_SUBJECTS[subject]
            for ff_task in ('filmfest1', 'filmfest2'):
                roi_ts = extract_roi_timeseries(subject, ff_session, ff_task)
                if roi_ts is None:
                    continue
                bnd_times = movie_boundaries[ff_task]
                for k in roi_keys:
                    if k not in roi_ts:
                        continue
                    epochs = extract_event_locked(roi_ts[k], bnd_times)
                    if epochs is not None:
                        task_epochs['movie'][k].append(epochs)

        # Per-subject means
        for task_key in TASK_KEYS:
            for k in roi_keys:
                if task_epochs[task_key][k]:
                    stacked = np.vstack(task_epochs[task_key][k])
                    group_data[task_key][k].append((subject, stacked.mean(axis=0)))

    return group_data


def load_ispc_data(hp, preset, onset, n_svf, n_ahc):
    """Load ISPC boundary and mid-event data for the 5 ROIs, FDR-correct."""
    hp_tag    = '_hp' if hp else ''
    onset_tag = '_onset' if onset else ''
    roi_keys  = [k for k, _ in ROI_SPEC_COMBINED]
    cond_indices = build_cond_indices(n_svf, n_ahc)
    within_p0, xsame_p0, xdiff_p0 = _comparison_cell_pairs(0)

    # --- Boundary (window 0) ---
    bnd_raw = {}
    raw_p_bnd = [[], [], []]
    for roi_key, _ in ROI_SPEC_COMBINED:
        cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6{hp_tag}{onset_tag}_{preset}_ispc.npz'
        if not cache.exists():
            print(f'  [ISPC boundary] cache not found: {cache.name}')
            continue
        data = np.load(cache)
        per_subj = data['per_subject'] if 'per_subject' in data else None
        inst_ispc = data['instance_ispc']

        means, sems, tps, subj_vals = [], [], [], []
        for ci, pairs in enumerate([within_p0, xsame_p0, xdiff_p0]):
            if per_subj is not None:
                sv = _per_subject_means(per_subj, cond_indices, pairs)
                sv_valid = sv[~np.isnan(sv)]
                mu  = float(np.nanmean(sv)) if len(sv_valid) > 0 else np.nan
                sem = sv_valid.std(ddof=1) / np.sqrt(len(sv_valid)) if len(sv_valid) >= 2 else np.nan
                tp  = ttest_1samp(sv_valid, 0.0).pvalue if len(sv_valid) >= 2 else np.nan
                subj_vals.append(sv)
            else:
                mu, sem, tp = np.nan, np.nan, np.nan
                subj_vals.append(None)
            means.append(mu); sems.append(sem); tps.append(tp)
            raw_p_bnd[ci].append((roi_key, tp))
        bnd_raw[roi_key] = dict(mean=means, sem=sems, sig=[False]*3, subj_vals=subj_vals)

    # FDR for boundary
    all_p_bnd = [(rk, ci, p) for ci, pl in enumerate(raw_p_bnd) for rk, p in pl]
    ps = np.array([t[2] for t in all_p_bnd], dtype=float)
    valid = ~np.isnan(ps)
    p_fdr = np.full(len(ps), np.nan)
    if valid.sum() > 0:
        _, p_corr, _, _ = multipletests(ps[valid], method='fdr_bh')
        p_fdr[valid] = p_corr
    for idx, (rk, ci, _) in enumerate(all_p_bnd):
        if rk in bnd_raw:
            bnd_raw[rk]['sig'][ci] = bool(p_fdr[idx] < 0.05)

    # --- Mid-event ---
    n_instances = n_svf + n_ahc + N_FILM
    mid_cond_indices = [
        np.arange(0, n_svf),
        np.arange(n_svf, n_svf + n_ahc),
        np.arange(n_svf + n_ahc, n_instances),
    ]
    mid_cell_pairs = [
        [(0, 0), (1, 1), (2, 2)],
        [(0, 1), (1, 0)],
        [(0, 2), (2, 0), (1, 2), (2, 1)],
    ]

    mid_raw = {}
    raw_p_mid = [[], [], []]
    for roi_key, _ in ROI_SPEC_COMBINED:
        cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6{hp_tag}_onset_mid_event_ispc.npz'
        if not cache.exists():
            print(f'  [ISPC mid] cache not found: {cache.name}')
            continue
        data = np.load(cache)
        per_subj_mid = data['per_subject']

        means, sems, tps, subj_vals = [], [], [], []
        for ci, pairs in enumerate(mid_cell_pairs):
            sv = _per_subject_means(per_subj_mid, mid_cond_indices, pairs)
            sv_valid = sv[~np.isnan(sv)]
            mu  = float(np.nanmean(sv)) if len(sv_valid) > 0 else np.nan
            sem = sv_valid.std(ddof=1) / np.sqrt(len(sv_valid)) if len(sv_valid) >= 2 else np.nan
            tp  = ttest_1samp(sv_valid, 0.0).pvalue if len(sv_valid) >= 2 else np.nan
            means.append(mu); sems.append(sem); tps.append(tp)
            subj_vals.append(sv)
            raw_p_mid[ci].append((roi_key, tp))
        mid_raw[roi_key] = dict(mean=means, sem=sems, sig=[False]*3, subj_vals=subj_vals)

    # FDR for mid-event
    all_p_mid = [(rk, ci, p) for ci, pl in enumerate(raw_p_mid) for rk, p in pl]
    ps_m = np.array([t[2] for t in all_p_mid], dtype=float)
    valid_m = ~np.isnan(ps_m)
    p_fdr_m = np.full(len(ps_m), np.nan)
    if valid_m.sum() > 0:
        _, p_corr_m, _, _ = multipletests(ps_m[valid_m], method='fdr_bh')
        p_fdr_m[valid_m] = p_corr_m
    for idx, (rk, ci, _) in enumerate(all_p_mid):
        if rk in mid_raw:
            mid_raw[rk]['sig'][ci] = bool(p_fdr_m[idx] < 0.05)

    return bnd_raw, mid_raw


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bold_timecourse(ax, group_data, roi_key, roi_title,
                          show_ylabel=True, show_legend=False,
                          show_subtask_annot=False, show_movie_annot=False,
                          title_pad=6, ymax=1.5, legend_loc='upper right',
                          movie_boundary_ref=-6, movie_annot_label='offset of previous\nmovie (MW)'):
    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR

    for task_key in TASK_KEYS:
        subj_means = group_data[task_key].get(roi_key, [])
        if not subj_means:
            continue
        data = np.array([m for _, m in subj_means])
        gmean = data.mean(axis=0)
        gsem  = data.std(axis=0) / np.sqrt(len(data))
        ax.plot(time_vec, gmean, color=TASK_COLORS[task_key], lw=2,
                label=TASK_LABELS[task_key])
        ax.fill_between(time_vec, gmean - gsem, gmean + gsem,
                        color=TASK_COLORS[task_key], alpha=0.18)

    # fraction of axes height corresponding to y=1.5 (data cap)
    _ymin, _ycap = -1.0, 1.5
    cap_frac = (_ycap - _ymin) / (ymax - _ymin)

    ax.axvspan(0, 15, ymax=cap_frac, color='lightgrey', alpha=0.5, zorder=0)
    ax.axvline(movie_boundary_ref, ymax=cap_frac, color='#377eb8', ls='--', lw=1.2)
    ax.axvline(-15, ymax=cap_frac, color='#FF5200', ls='--', lw=1.2)
    ax.axvline(0,   ymax=cap_frac, color='#555555', ls='-',  lw=1.2)
    ax.axhline(0,   color='k',     ls='-',  lw=0.5, alpha=0.3)
    ax.set_xlim(-30, 60)
    ax.set_xticks([-30, -15, 0, 15, 30, 45, 60])
    ax.set_ylim(_ymin, ymax)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['-1', '0', '1'], fontsize=LABEL_FS)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_bounds(_ymin, _ycap)
    ax.set_title(roi_title, fontsize=LABEL_FS, fontweight='bold', pad=title_pad)
    ax.set_xlabel('Time from boundary (s)', fontsize=LABEL_FS)
    ax.tick_params(axis='x', labelsize=LABEL_FS-2)

    if show_ylabel:
        ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FS)
    if show_legend:
        ax.legend(fontsize=LABEL_FS, frameon=False,
                  loc=legend_loc, bbox_to_anchor=(1.0, 0.8))
    if show_subtask_annot:
        ax.text(-15, 1.55, 'offset of previous\nsubtask (WG, EG)',
                color='#FF5200', fontsize=LABEL_FS-4, ha='center', va='bottom')
    if show_movie_annot:
        ax.text(movie_boundary_ref, 1.55, movie_annot_label,
                color='#377eb8', fontsize=LABEL_FS-4, ha='center', va='bottom')


def plot_ispc_bar(ax, bnd_data, mid_data, roi_title,
                  show_ylabel=True, show_legend=False):
    rng   = np.random.default_rng(0)
    bar_w = 0.35
    x     = np.arange(3)

    for wi, (wdata, facecolor) in enumerate([
            (bnd_data, 'white'),
            (mid_data, '#AAAAAA'),
    ]):
        offset = (wi - 0.5) * bar_w
        for ci in range(3):
            mu  = wdata['mean'][ci]
            sem = wdata['sem'][ci]
            sig = wdata['sig'][ci]
            sv  = wdata['subj_vals'][ci]

            ax.bar(x[ci] + offset, mu, bar_w,
                   color=facecolor, edgecolor='black', linewidth=0.8,
                   yerr=sem,
                   error_kw=dict(elinewidth=1.0, capsize=3, ecolor='black'))

            if sv is not None:
                sv_valid = sv[~np.isnan(sv)]
                jitter   = rng.uniform(-bar_w * 0.25, bar_w * 0.25, size=len(sv_valid))
                ax.scatter(x[ci] + offset + jitter, sv_valid,
                           s=12, color=facecolor, edgecolors='black',
                           linewidths=0.4, zorder=4, alpha=0.9)

            if sig and not np.isnan(mu):
                y_ast = max(mu, 0) + (sem if not np.isnan(sem) else 0) + 0.008
                ax.text(x[ci] + offset, y_ast, '*',
                        ha='center', va='bottom', fontsize=LABEL_FS,
                        fontweight='bold', color='black')

    ax.axhline(0, color='black', linewidth=1.0, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(which='both', bottom=False)
    ax.set_title(roi_title, fontsize=LABEL_FS, fontweight='bold')
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.set_yticks([0, 0.1])
    ax.set_yticklabels(['0', '0.1'], fontsize=LABEL_FS)

    if show_ylabel:
        ax.set_ylabel('Inter-subject\npattern correlation ($r$)', fontsize=LABEL_FS)

    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    _main_labels = ['within-task', 'cross-task', 'cross-task']
    _sub_segments = [
        None,
        [('WG', 'red'), ('\u00d7', 'black'), ('EG', 'orange')],
        [('{', 'black'), ('WG', 'red'), (', ', 'black'), ('EG', 'orange'),
         ('}\u00d7', 'black'), ('MW', 'blue')],
    ]
    for xi, main, sub_segs in zip(x, _main_labels, _sub_segments):
        ax.text(xi, -0.06, main, fontsize=LABEL_FS-2, ha='center', va='top', transform=trans)
        if sub_segs:
            _draw_colored_segments(ax, xi, -0.14, sub_segs, fontsize=LABEL_FS-4)

    if show_legend:
        leg_handles = [
            mpatches.Patch(facecolor='white',   edgecolor='black', label='boundary'),
            mpatches.Patch(facecolor='#AAAAAA', edgecolor='black', label='middle'),
        ]
        ax.legend(handles=leg_handles, loc='upper right',
                  bbox_to_anchor=(1.1, 1.0), fontsize=LABEL_FS-2, frameon=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--align',   choices=['offset', 'onset', 'title_onset'], default='onset')
    parser.add_argument('--preset',  default='onset0', choices=['hrf', 'post4', 'onset0'])
    parser.add_argument('--hp',      action='store_true')
    parser.add_argument('--onset',   action='store_true', help='Use onset-aligned ISPC cache')
    parser.add_argument('--n-svf',   type=int, default=_DEFAULT_N_SVF)
    parser.add_argument('--n-ahc',   type=int, default=_DEFAULT_N_AHC)
    parser.add_argument('--rois',    nargs='+',
                        default=[k for k, _ in ROI_SPEC_COMBINED],
                        choices=[k for k, _ in ROI_SPEC_COMBINED],
                        help='ROI keys to include (default: all 5)')
    parser.add_argument('--layout', choices=['vertical', 'horizontal'], default='vertical',
                        help='vertical = 5 rows × 2 cols; horizontal = 2 rows × 5 cols')
    args = parser.parse_args()

    roi_spec = [(k, lbl) for k, lbl in ROI_SPEC_COMBINED if k in args.rois]
    n_rois   = len(roi_spec)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine movie annotation position and label based on alignment
    if args.align == 'title_onset':
        _movie_boundary_ref   = 0
        _movie_annot_label    = 'onset of title\nscene (MW)'
    else:
        _movie_boundary_ref   = -6
        _movie_annot_label    = 'offset of previous\nmovie (MW)'

    print('Loading BOLD group data...')
    group_data = load_bold_group_data(args.align)

    print('Loading ISPC data...')
    bnd_by_roi, mid_by_roi = load_ispc_data(
        args.hp, args.preset, args.onset, args.n_svf, args.n_ahc)

    hp_label    = 'hp' if args.hp else 'no_hp'
    onset_label = 'onset' if args.onset else 'offset'
    roi_tag     = '_'.join(k for k, _ in roi_spec)

    if args.layout == 'vertical':
        fig = plt.figure(figsize=(12, 4.5 * n_rois))
        gs  = gridspec.GridSpec(n_rois, 2,
                                width_ratios=[1.0, 1.0],
                                hspace=0.55, wspace=0.20,
                                left=0.08, right=0.97, top=0.97, bottom=0.08)
        ispc_axes = []
        for i, (roi_key, roi_title) in enumerate(roi_spec):
            ax_bold = fig.add_subplot(gs[i, 0])
            ax_ispc = fig.add_subplot(gs[i, 1])
            ispc_axes.append(ax_ispc)
            plot_bold_timecourse(ax_bold, group_data, roi_key, roi_title,
                                 show_ylabel=True,
                                 show_legend=(i == 0),
                                 show_subtask_annot=(i == 0),
                                 show_movie_annot=(i == 1),
                                 movie_boundary_ref=_movie_boundary_ref,
                                 movie_annot_label=_movie_annot_label)
            bnd_data = bnd_by_roi.get(roi_key)
            mid_data = mid_by_roi.get(roi_key)
            if bnd_data is not None and mid_data is not None:
                plot_ispc_bar(ax_ispc, bnd_data, mid_data, roi_title,
                              show_ylabel=True, show_legend=(i == 0))
            else:
                ax_ispc.set_visible(False)

    else:  # horizontal: 2 rows × n_rois cols
        fig = plt.figure(figsize=(4.5 * n_rois, 9))
        gs  = gridspec.GridSpec(2, n_rois,
                                hspace=0.30, wspace=0.15,
                                left=0.06, right=0.97, top=0.97, bottom=0.08)
        ispc_axes = []
        for i, (roi_key, roi_title) in enumerate(roi_spec):
            ax_bold = fig.add_subplot(gs[0, i])
            ax_ispc = fig.add_subplot(gs[1, i])
            ispc_axes.append(ax_ispc)
            plot_bold_timecourse(ax_bold, group_data, roi_key, roi_title,
                                 show_ylabel=(i == 0),
                                 show_legend=(i == 0),
                                 show_subtask_annot=(i == 0),
                                 show_movie_annot=(i == 1),
                                 title_pad=18, ymax=1.7, legend_loc='center right',
                                 movie_boundary_ref=_movie_boundary_ref,
                                 movie_annot_label=_movie_annot_label)
            bnd_data = bnd_by_roi.get(roi_key)
            mid_data = mid_by_roi.get(roi_key)
            if bnd_data is not None and mid_data is not None:
                plot_ispc_bar(ax_ispc, bnd_data, mid_data, roi_title,
                              show_ylabel=(i == 0), show_legend=(i == 0))
            else:
                ax_ispc.set_visible(False)

    for ax in ispc_axes:
        ax.set_ylim(-0.05, 0.20)

    out_path = OUTPUT_DIR / f'combined_bold_ispc_{args.align}_{hp_label}_{args.preset}_{onset_label}_{roi_tag}_{args.layout}.png'
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
