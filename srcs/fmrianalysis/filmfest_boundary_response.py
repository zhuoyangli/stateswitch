"""
Filmfest Boundary-Locked fMRI Response

ROI time courses locked to:
  - between-movie  : transitions between consecutive movies (8 total)
  - within-movie, split by boundary_type (strong / moderate / weak)

Produces two figures:
  filmfest_boundary_response_all314.png      — all 314 within-movie boundaries
  filmfest_boundary_response_retained131.png — 131 boundaries that pass fMRI exclusion criteria

Layout follows combined_boundary_ispc_figure.py: ROIs laid out horizontally,
group mean ± SEM, gray shaded window 0–15 s.

Usage:
    python srcs/fmrianalysis/filmfest_boundary_response.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.config import FIGS_DIR, TR, FILMFEST_SUBJECTS
from configs.schaefer_rois import POSTERIOR_MEDIAL, ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL
from fmrianalysis.utils import get_parcel_data, get_movie_boundary_offsets
from fmrianalysis.svf_switch_boundary import (
    find_rated_sessions, load_multirater_events,
    extract_roi_timeseries as svf_extract_roi,
    CONTEXT_CATS, _add_context_filter,
)

ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
BS_CSV          = ANNOTATIONS_DIR / 'filmfest_boundary_strength.csv'
OUTPUT_DIR      = FIGS_DIR / 'filmfest'

# Window
TRS_BEFORE = 20   # 30 s
TRS_AFTER  = 40   # 60 s

# Style (matching combined_boundary_ispc_figure.py)
LABEL_FS = 16
COLORS = {
    'between':  '#e41a1c',
    'strong':   '#1a237e',
    'moderate': '#3498db',
    'weak':     '#90caf9',
}
LABELS = {
    'between':  'Between-movie',
    'strong':   'Within — strong',
    'moderate': 'Within — moderate',
    'weak':     'Within — weak',
}
CONDITIONS = ['between', 'strong', 'moderate', 'weak']

ROI_SPEC = [
    ('pmc',  'PMC'),
    ('hipp', 'Hippocampus'),
    ('ag',   'Angular Gyrus'),
    ('eac',  'Auditory Cortex'),
    ('evc',  'Early Visual Cortex'),
]
PMC_LABELS    = POSTERIOR_MEDIAL.get('left_labels', []) + POSTERIOR_MEDIAL.get('right_labels', [])
AG_LABELS     = ANGULAR_GYRUS.get('left_labels', [])    + ANGULAR_GYRUS.get('right_labels', [])
EAC_LABELS    = EARLY_AUDITORY.get('left_labels', [])   + EARLY_AUDITORY.get('right_labels', [])
EVC_LABELS    = EARLY_VISUAL.get('left_labels', [])     + EARLY_VISUAL.get('right_labels', [])
HIPP_KEYWORDS = ['hippocampus']


# ============================================================================
# BOUNDARY TIMES
# ============================================================================

def get_within_movie_boundaries(task, retained_only=False):
    """Return run-relative boundary times split by boundary_type.

    Returns dict: {'strong': [...], 'moderate': [...], 'weak': [...]}
    """
    bs = pd.read_csv(BS_CSV)
    movie_ids = [1, 2, 3, 4, 5] if task == 'filmfest1' else [6, 7, 8, 9, 10]
    bs_task = bs[bs['movie'].isin(movie_ids)].copy()
    if retained_only:
        bs_task = bs_task[bs_task['retained_for_fmri'] == 1]

    HRF_SHIFT, RUN1_LEN = 3, 996
    bs_task['run_rel_TR'] = bs_task['concat_TR'] - HRF_SHIFT
    if task == 'filmfest2':
        bs_task['run_rel_TR'] -= RUN1_LEN

    bs_task['movie_onset_run_TR'] = bs_task['run_rel_TR'] - bs_task['timestamp_sec'] / TR
    onset_TR = bs_task.groupby('movie')['movie_onset_run_TR'].mean()
    bs_task['run_rel_sec'] = (
        onset_TR[bs_task['movie'].values].values * TR + bs_task['timestamp_sec'].values
    )

    return {
        btype: sorted(bs_task.loc[bs_task['boundary_type'] == btype, 'run_rel_sec'].values)
        for btype in ('strong', 'moderate', 'weak')
    }


# ============================================================================
# ROI TIME SERIES
# ============================================================================

def _avg_labels(parcel_dict, label_list):
    ts = [parcel_dict[l] for l in label_list if l in parcel_dict]
    return np.column_stack(ts).mean(axis=1)


def _avg_keywords(parcel_dict, keywords):
    ts = [v for l, v in parcel_dict.items()
          if l != 'Background' and any(kw in l.lower() for kw in keywords)]
    return np.column_stack(ts).mean(axis=1)


def extract_roi_timeseries(subject, session, task):
    schaefer = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
    ho_sub   = get_parcel_data(subject, session, task, atlas='HarvardOxford_sub')
    return {
        'pmc':  _avg_labels(schaefer, PMC_LABELS),
        'ag':   _avg_labels(schaefer, AG_LABELS),
        'eac':  _avg_labels(schaefer, EAC_LABELS),
        'evc':  _avg_labels(schaefer, EVC_LABELS),
        'hipp': _avg_keywords(ho_sub, HIPP_KEYWORDS),
    }


def extract_event_locked(signal, event_times_sec):
    n = len(signal)
    if len(event_times_sec) == 0:
        return None
    centers = np.round(np.array(event_times_sec) / TR).astype(int)
    offsets = np.arange(-TRS_BEFORE, TRS_AFTER + 1)
    idx     = centers[:, None] + offsets[None, :]
    valid   = np.all((idx >= 0) & (idx < n), axis=1)
    if not valid.any():
        return None
    epochs = signal[idx[valid]]
    return epochs.reshape(-1, offsets.size) if epochs.ndim == 1 else epochs


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_subject_data(retained_only=False):
    """Return subject_data[cond][roi_key] = (n_subj, n_tp) array."""
    between_times = {t: get_movie_boundary_offsets(t)              for t in ('filmfest1', 'filmfest2')}
    within_times  = {t: get_within_movie_boundaries(t, retained_only) for t in ('filmfest1', 'filmfest2')}

    for task in ('filmfest1', 'filmfest2'):
        counts = {bt: len(within_times[task][bt]) for bt in ('strong', 'moderate', 'weak')}
        print(f"  {task} — between: {len(between_times[task])}, within: {counts}")

    roi_keys = [k for k, _ in ROI_SPEC]
    n_tp     = TRS_BEFORE + TRS_AFTER + 1
    subject_data = {cond: {k: [] for k in roi_keys} for cond in CONDITIONS}

    for subject, session in FILMFEST_SUBJECTS.items():
        subj_epochs = {cond: {k: [] for k in roi_keys} for cond in CONDITIONS}

        for task in ('filmfest1', 'filmfest2'):
            roi_ts = extract_roi_timeseries(subject, session, task)
            for k in roi_keys:
                for cond, times in ([('between', between_times[task])] +
                                    [(bt, within_times[task][bt])
                                     for bt in ('strong', 'moderate', 'weak')]):
                    epochs = extract_event_locked(roi_ts[k], times)
                    if epochs is not None:
                        subj_epochs[cond][k].append(epochs)

        for cond in CONDITIONS:
            for k in roi_keys:
                if subj_epochs[cond][k]:
                    stacked = np.vstack(subj_epochs[cond][k])
                    subject_data[cond][k].append(stacked.mean(axis=0))
                else:
                    subject_data[cond][k].append(np.full(n_tp, np.nan))

    for cond in CONDITIONS:
        for k in roi_keys:
            subject_data[cond][k] = np.array(subject_data[cond][k])

    return subject_data, between_times, within_times


# ============================================================================
# SVF CONTEXT DATA COLLECTION
# ============================================================================

def collect_svf_context():
    """Collect SVF CONTEXT_CATS (CC-Switch, CC-No-Consensus, CC-Cluster) per ROI.

    Returns (time_vec, stacks, n_subjects) where
    stacks[cat_label][roi_key] = (n_subjects, n_tp) array.
    Uses the same -30 to +60 s window as the filmfest row.
    """
    roi_keys = [k for k, _ in ROI_SPEC]
    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR
    sessions = find_rated_sessions()
    cat_labels = [l for l, _, _, _ in CONTEXT_CATS]

    from collections import defaultdict
    sub_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for subject, session in sessions:
        try:
            events_df = load_multirater_events(subject, session)
            roi_ts    = svf_extract_roi(subject, session)
        except Exception:
            continue
        events_df = _add_context_filter(events_df)
        for cat_label, cond_fn, _, _ in CONTEXT_CATS:
            mask = cond_fn(events_df)
            tp   = events_df.loc[mask, 'onset'].values
            for roi_key in roi_keys:
                if roi_key not in roi_ts:
                    continue
                epochs = extract_event_locked(roi_ts[roi_key], tp)
                if epochs is not None:
                    sub_data[subject][cat_label][roi_key].append(epochs.mean(0))
                else:
                    sub_data[subject][cat_label][roi_key].append(np.full(len(time_vec), np.nan))

    subjects = sorted(sub_data.keys())
    stacks = {}
    for cat_label in cat_labels:
        stacks[cat_label] = {}
        for roi_key in roi_keys:
            sub_means = []
            for sub in subjects:
                tcs = sub_data[sub][cat_label][roi_key]
                sub_means.append(np.nanmean(tcs, axis=0) if tcs else np.full(len(time_vec), np.nan))
            stacks[cat_label][roi_key] = np.array(sub_means)

    print(f"  SVF context: {len(subjects)} subjects")
    return stacks, len(subjects)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_roi_timecourse(ax, subject_data, roi_key, roi_title,
                        show_ylabel=True, show_legend=False):
    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR
    _ymin, _ycap = -1.0, 1.5

    for cond in CONDITIONS:
        data  = subject_data[cond][roi_key]
        color = COLORS[cond]
        gmean = np.nanmean(data, axis=0)
        gsem  = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data[:, 0])))
        ax.plot(time_vec, gmean, color=color, lw=2, label=LABELS[cond])
        ax.fill_between(time_vec, gmean - gsem, gmean + gsem, color=color, alpha=0.18)

    cap_frac = (_ycap - _ymin) / (1.8 - _ymin)   # fraction for axvspan/axvline
    ax.axvspan(0, 15, ymax=cap_frac, color='lightgrey', alpha=0.5, zorder=0)
    ax.axvline(0, ymax=cap_frac, color='#555555', ls='-',  lw=1.2)
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)

    ax.set_xlim(-30, 60)
    ax.set_ylim(_ymin, 1.8)
    ax.set_xticks([-30, -15, 0, 15, 30, 45, 60])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['-1', '0', '1'], fontsize=LABEL_FS)
    ax.tick_params(axis='x', labelsize=LABEL_FS - 2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_bounds(_ymin, _ycap)
    ax.set_title(roi_title, fontsize=LABEL_FS, fontweight='bold', pad=6)
    ax.set_xlabel('Time from boundary (s)', fontsize=LABEL_FS)
    if show_ylabel:
        ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FS)
    if show_legend:
        ax.legend(fontsize=LABEL_FS - 4, frameon=False, loc='upper right')


def make_figure(subject_data, between_times, within_times, retained_only,
                svf_stacks, svf_n):
    n_between = sum(len(v) for v in between_times.values())
    n_within  = {bt: sum(len(within_times[t][bt]) for t in ('filmfest1', 'filmfest2'))
                 for bt in ('strong', 'moderate', 'weak')}

    n_rois = len(ROI_SPEC)
    fig, axes = plt.subplots(2, n_rois, figsize=(4.5 * n_rois, 9),
                             facecolor='white')

    tag = 'retained (n=131)' if retained_only else 'all (n=314)'
    fig.suptitle(
        f"Filmfest Boundary-Locked fMRI Response  —  within-movie: {tag}\n"
        f"N={len(FILMFEST_SUBJECTS)} subjects  |  "
        f"between-movie: {n_between}  |  "
        f"strong: {n_within['strong']}  moderate: {n_within['moderate']}  weak: {n_within['weak']}",
        fontsize=LABEL_FS - 2, fontweight='bold', y=1.02,
    )

    # Row 0: filmfest boundary response
    for col, (roi_key, roi_title) in enumerate(ROI_SPEC):
        ax = axes[0, col]
        plot_roi_timecourse(ax, subject_data, roi_key, roi_title,
                            show_ylabel=(col == 0),
                            show_legend=(col == 0))

    # Row 1: SVF context (CC-Switch / CC-No-Consensus / CC-Cluster)
    svf_cat_labels = [l for l, _, _, _ in CONTEXT_CATS]
    row1_axes = []
    for col, (roi_key, roi_title) in enumerate(ROI_SPEC):
        ax = axes[1, col]
        row1_axes.append(ax)
        svf_time_vec_full = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR
        for cat_label, _, color, ls in CONTEXT_CATS:
            data  = svf_stacks[cat_label][roi_key]
            gmean = np.nanmean(data, axis=0)
            gsem  = np.nanstd(data, axis=0) / np.sqrt(svf_n)
            ax.plot(svf_time_vec_full, gmean, color=color, ls=ls, lw=2, label=cat_label)
            ax.fill_between(svf_time_vec_full, gmean - gsem, gmean + gsem, color=color, alpha=0.18)

        ax.axvline(0, color='#555555', ls='-', lw=1.2)
        ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
        ax.set_xlim(-30, 60)
        ax.set_xticks([-30, -15, 0, 15, 30, 45, 60])
        ax.tick_params(axis='x', labelsize=LABEL_FS - 2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Time from preceding word offset (s)', fontsize=LABEL_FS - 2)
        if col == 0:
            ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FS)
            ax.legend(fontsize=LABEL_FS - 4, frameon=False, loc='upper right')

    # Add row label
    axes[1, 0].annotate(
        f'SVF context: after ≥2 clusters (N={svf_n})',
        xy=(0, 0.5), xycoords='axes fraction',
        xytext=(-0.35, 0.5), textcoords='axes fraction',
        fontsize=LABEL_FS - 4, rotation=90, va='center', ha='center',
    )

    # Shared y-axis across BOTH rows
    all_row_axes = list(axes[0, :]) + row1_axes
    ymin = min(ax.get_ylim()[0] for ax in all_row_axes)
    ymax = max(ax.get_ylim()[1] for ax in all_row_axes)
    for ax in all_row_axes:
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['-1', '0', '1'], fontsize=LABEL_FS - 2)

    fig.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nCollecting SVF context data...")
    svf_stacks, svf_n = collect_svf_context()

    for retained_only in (False, True):
        tag = 'retained131' if retained_only else 'all314'
        print(f"\n{'=' * 50}")
        print(f"Collecting filmfest data — {tag}")
        print('=' * 50)

        subject_data, between_times, within_times = collect_subject_data(retained_only)
        fig = make_figure(subject_data, between_times, within_times, retained_only,
                          svf_stacks, svf_n)

        out = OUTPUT_DIR / f'filmfest_boundary_response_{tag}.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == '__main__':
    main()
