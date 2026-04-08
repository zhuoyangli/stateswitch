#!/usr/bin/env python3
"""
Macro vs Micro Boundary Response Figure

3 rows × 8 ROI columns showing boundary-locked BOLD responses.

Two alignment modes (--align):
  onset  (default) : t=0 = current word / explanation / trial onset
                     dashed markers show preceding offset for each category
  offset           : t=0 = preceding word / explanation offset  (SVF/AHC macro → trial offset)
                     dashed markers show current word/explanation onset for each category
  FilmFest is unchanged in both modes (movie offset = next movie onset).

ROI order: Hippocampus, PMC, AG, mPFC, dACC, dlPFC, EAC, EVC
Time window: -30 s to +60 s (TRS_BEFORE=20, TRS_AFTER=40)

Usage:
    uv run python srcs/fmrianalysis/macro_micro_boundary_figure.py [--align {onset,offset}]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.config import FIGS_DIR, TR, FILMFEST_SUBJECTS
from configs.schaefer_rois import (
    POSTERIOR_MEDIAL, ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL,
    DLPFC, DACC, MPFC,
)
from fmrianalysis.utils import get_parcel_data, get_movie_boundary_offsets
from fmrianalysis.filmfest_boundary_response import get_within_movie_boundaries
from fmrianalysis.svf_switch_boundary import (
    find_rated_sessions, load_multirater_events, _add_context_filter, CONTEXT_CATS,
    SCANNER_START_OFFSET as SVF_SCANNER_OFFSET,
)
from fmrianalysis.ahc_category_boundary import (
    find_ahc_sessions,
    ANNOTATIONS_DIR as AHC_ANNOT_DIR,
    SCANNER_START_OFFSET as AHC_SCANNER_OFFSET,
)
from fmrianalysis.multitask_boundary_parcel import (
    discover_sessions, find_psychopy_csv, parse_trial_onsets, parse_trial_offsets,
)

OUTPUT_DIR = FIGS_DIR / 'macro_micro_boundary'
SUBJECT_IDS = ['sub-001', 'sub-003', 'sub-004', 'sub-006', 'sub-007', 'sub-008', 'sub-009']

# === WINDOW ===
TRS_BEFORE = 20   # 30 s
TRS_AFTER  = 40   # 60 s
TIME_VEC   = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR   # -30 to +60 s

# === ROI SPEC (ordered as requested) ===
ROI_SPEC = [
    ('hipp',  'Hippocampus'),
    ('pmc',   'PMC'),
    ('ag',    'AG'),
    ('mpfc',  'mPFC'),
    ('dacc',  'dACC'),
    ('dlpfc', 'dlPFC'),
    ('eac',   'EAC'),
    ('evc',   'EVC'),
]
ROI_KEYS = [k for k, _ in ROI_SPEC]

PMC_LABELS   = POSTERIOR_MEDIAL.get('left_labels', []) + POSTERIOR_MEDIAL.get('right_labels', [])
AG_LABELS    = ANGULAR_GYRUS.get('left_labels', [])    + ANGULAR_GYRUS.get('right_labels', [])
EAC_LABELS   = EARLY_AUDITORY.get('left_labels', [])   + EARLY_AUDITORY.get('right_labels', [])
EVC_LABELS   = EARLY_VISUAL.get('left_labels', [])     + EARLY_VISUAL.get('right_labels', [])
DLPFC_LABELS = DLPFC.get('left_labels', []) + DLPFC.get('right_labels', [])
DACC_LABELS  = DACC.get('left_labels', [])  + DACC.get('right_labels', [])
MPFC_LABELS  = MPFC.get('left_labels', [])  + MPFC.get('right_labels', [])
HIPP_KEYWORDS = ['hippocampus']

# === STYLE ===
LABEL_FS = 14

# ── Systematic color scheme ──────────────────────────────────────────────────
# FilmFest → blue family   (darker = more macro)
# SVF      → red family    (darker = more macro)
# AHC/EG   → orange family (darker = more macro)

# FilmFest (blue)
FF_CONDITIONS = ['between', 'strong', 'moderate', 'weak']
FF_COLORS = {
    'between':  '#084594',   # dark navy   — macro
    'strong':   '#2171b5',   # medium-dark blue
    'moderate': '#6baed6',   # medium blue
    'weak':     '#bdd7e7',   # light blue
}
FF_LABELS = {
    'between':  'Movie title onset',
    'strong':   'Within-movie — strong',
    'moderate': 'Within-movie — moderate',
    'weak':     'Within-movie — weak',
}

# SVF (red)
SVF_MACRO_COLOR = '#67000d'   # very dark maroon — macro
SVF_MACRO_LABEL = 'Trial onset'
# micro CC colors override CONTEXT_CATS colors at plot time
SVF_MICRO_COLORS = {
    'CC-Switch':        '#cb181d',   # dark red
    'CC-No-Consensus':  '#fb6a4a',   # medium red
    'CC-Cluster':       '#fcbba1',   # light pink-red
}
SVF_MICRO_LS = {
    'CC-Switch':        '-',
    'CC-No-Consensus':  '-',
    'CC-Cluster':       '-',
}
SVF_MICRO_LABELS = {
    'CC-Switch':        'Switching word\n(prev. word offset)',
    'CC-No-Consensus':  'Ambiguous word\n(prev. word offset)',
    'CC-Cluster':       'Clustering word\n(prev. word offset)',
}

# AHC/EG (orange)
AHC_MACRO_COLOR = '#7f2704'   # dark burnt orange — macro
AHC_MACRO_LABEL = 'Trial onset'
AHC_MICRO_COLOR = '#fd8d3c'   # bright orange
AHC_MICRO_LABEL = 'Expl. transition\n(prev. expl. offset)'


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def _avg_labels(parcel_dict, label_list):
    ts = [parcel_dict[l] for l in label_list if l in parcel_dict]
    if not ts:
        return None
    return np.column_stack(ts).mean(axis=1)


def _avg_keywords(parcel_dict, keywords):
    ts = [v for l, v in parcel_dict.items()
          if l != 'Background' and any(kw in l.lower() for kw in keywords)]
    if not ts:
        return None
    return np.column_stack(ts).mean(axis=1)


def extract_roi_timeseries(subject, session, task):
    """Load parcel data and return dict of 8 ROI time series. Returns None on missing data."""
    try:
        schaefer = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
        ho_sub   = get_parcel_data(subject, session, task, atlas='HarvardOxford_sub')
    except FileNotFoundError:
        return None
    result = {
        'pmc':   _avg_labels(schaefer, PMC_LABELS),
        'ag':    _avg_labels(schaefer, AG_LABELS),
        'eac':   _avg_labels(schaefer, EAC_LABELS),
        'evc':   _avg_labels(schaefer, EVC_LABELS),
        'dlpfc': _avg_labels(schaefer, DLPFC_LABELS),
        'dacc':  _avg_labels(schaefer, DACC_LABELS),
        'mpfc':  _avg_labels(schaefer, MPFC_LABELS),
        'hipp':  _avg_keywords(ho_sub, HIPP_KEYWORDS),
    }
    # Drop any ROIs that failed to load
    return {k: v for k, v in result.items() if v is not None}


def extract_event_locked(signal, event_times_sec):
    """Extract event-locked epochs. Returns (n_events, n_tp) array or None."""
    n = len(signal)
    if len(event_times_sec) == 0:
        return None
    centers = np.round(np.array(event_times_sec) / TR).astype(int)
    offsets = np.arange(-TRS_BEFORE, TRS_AFTER + 1)
    idx  = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n), axis=1)
    if not valid.any():
        return None
    epochs = signal[idx[valid]]
    return epochs.reshape(-1, offsets.size) if epochs.ndim == 1 else epochs


# ============================================================================
# FILMFEST DATA
# ============================================================================

def collect_filmfest_data():
    """Return subject_data[cond][roi_key] = (n_subj, n_tp) array.

    Conditions: 'between', 'strong', 'moderate', 'weak'
    """
    between_times = {t: get_movie_boundary_offsets(t) for t in ('filmfest1', 'filmfest2')}
    within_times  = {t: get_within_movie_boundaries(t, retained_only=True)
                     for t in ('filmfest1', 'filmfest2')}

    subject_data = {cond: {k: [] for k in ROI_KEYS} for cond in FF_CONDITIONS}

    for subject, session in FILMFEST_SUBJECTS.items():
        subj_epochs = {cond: {k: [] for k in ROI_KEYS} for cond in FF_CONDITIONS}
        for task in ('filmfest1', 'filmfest2'):
            roi_ts = extract_roi_timeseries(subject, session, task)
            if roi_ts is None:
                continue
            for k in ROI_KEYS:
                if k not in roi_ts:
                    continue
                sig = roi_ts[k]
                cond_times = (
                    [('between', between_times[task])] +
                    [(bt, within_times[task][bt]) for bt in ('strong', 'moderate', 'weak')]
                )
                for cond, times in cond_times:
                    epochs = extract_event_locked(sig, times)
                    if epochs is not None:
                        subj_epochs[cond][k].append(epochs)

        n_tp = TRS_BEFORE + TRS_AFTER + 1
        for cond in FF_CONDITIONS:
            for k in ROI_KEYS:
                if subj_epochs[cond][k]:
                    stacked = np.vstack(subj_epochs[cond][k])
                    subject_data[cond][k].append(stacked.mean(axis=0))
                else:
                    subject_data[cond][k].append(np.full(n_tp, np.nan))

    for cond in FF_CONDITIONS:
        for k in ROI_KEYS:
            subject_data[cond][k] = np.array(subject_data[cond][k])

    n_between = sum(len(v) for v in between_times.values())
    n_within  = {bt: sum(len(within_times[t][bt]) for t in ('filmfest1', 'filmfest2'))
                 for bt in ('strong', 'moderate', 'weak')}
    print(f"  FilmFest: {len(FILMFEST_SUBJECTS)} subjects | between: {n_between} | "
          f"strong: {n_within['strong']} moderate: {n_within['moderate']} weak: {n_within['weak']}")
    return subject_data


# ============================================================================
# SVF DATA
# ============================================================================

def _collect_task_macro(task, align):
    """Generic macro collector for SVF or AHC.

    align='onset' → lock to trial onset, return (stacks, n)
    align='offset' → lock to trial offset, return (stacks, n)
    """
    parse_fn = parse_trial_onsets if align == 'onset' else parse_trial_offsets
    sub_data = defaultdict(lambda: defaultdict(list))

    for subject in SUBJECT_IDS:
        for session, task_found in discover_sessions(subject):
            if task_found != task:
                continue
            csv_path = find_psychopy_csv(subject, session, task)
            if csv_path is None:
                continue
            times = parse_fn(csv_path, task)
            if not times:
                continue
            roi_ts = extract_roi_timeseries(subject, session, task)
            if roi_ts is None:
                continue
            for k in ROI_KEYS:
                if k not in roi_ts:
                    continue
                epochs = extract_event_locked(roi_ts[k], times)
                if epochs is not None:
                    sub_data[subject][k].append(epochs.mean(0))

    subjects = sorted(sub_data.keys())
    n_tp = TRS_BEFORE + TRS_AFTER + 1
    stacks = {k: [] for k in ROI_KEYS}
    for sub in subjects:
        for k in ROI_KEYS:
            tcs = sub_data[sub][k]
            stacks[k].append(np.nanmean(tcs, axis=0) if tcs else np.full(n_tp, np.nan))
    for k in ROI_KEYS:
        stacks[k] = np.array(stacks[k])
    return stacks, len(subjects)


def collect_svf_macro(align):
    stacks, n = _collect_task_macro('svf', align)
    label = 'trial onsets' if align == 'onset' else 'trial offsets'
    print(f"  SVF macro ({label}): {n} subjects")
    return stacks, n


def collect_svf_micro(align):
    """Micro: CC-Switch / CC-No-Consensus / CC-Cluster per subject.

    align='onset'  → lock to current word onset; markers = per-category -mean(word_onset_rel)
    align='offset' → lock to preceding word offset; markers = per-category +mean(word_onset_rel)

    Returns (stacks, n_subjects, markers) where
    markers[cat_label] = float position for dashed vertical line.
    """
    cat_labels = [l for l, _, _, _ in CONTEXT_CATS]
    sub_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cat_onset_rels = {l: [] for l in cat_labels}

    for subject, session in find_rated_sessions():
        try:
            events_df = load_multirater_events(subject, session)
        except Exception:
            continue
        events_df = _add_context_filter(events_df)
        roi_ts = extract_roi_timeseries(subject, session, 'svf')
        if roi_ts is None:
            continue

        for cat_label, cond_fn, _, _ in CONTEXT_CATS:
            mask = cond_fn(events_df)
            cat_onset_rels[cat_label].extend(
                events_df.loc[mask, 'word_onset_rel'].dropna().values)
            if align == 'onset':
                # t=0 = current word onset = preceding offset + word_onset_rel
                tp = (events_df.loc[mask, 'onset']
                      + events_df.loc[mask, 'word_onset_rel']).dropna().values
            else:
                # t=0 = preceding word offset (original behaviour)
                tp = events_df.loc[mask, 'onset'].values
            for k in ROI_KEYS:
                if k not in roi_ts:
                    continue
                epochs = extract_event_locked(roi_ts[k], tp)
                if epochs is not None:
                    sub_data[subject][cat_label][k].append(epochs.mean(0))

    subjects = sorted(sub_data.keys())
    n_tp = TRS_BEFORE + TRS_AFTER + 1
    stacks = {}
    for cat_label in cat_labels:
        stacks[cat_label] = {}
        for k in ROI_KEYS:
            sub_means = []
            for sub in subjects:
                tcs = sub_data[sub][cat_label][k]
                sub_means.append(np.nanmean(tcs, axis=0) if tcs else np.full(n_tp, np.nan))
            stacks[cat_label][k] = np.array(sub_means)

    # Markers: in onset-locked, preceding offset is at -mean(rel); in offset-locked, +mean(rel)
    sign = -1.0 if align == 'onset' else +1.0
    markers = {l: sign * float(np.mean(v)) if v else 0.0
               for l, v in cat_onset_rels.items()}
    for l, m in markers.items():
        print(f"  SVF micro {l}: marker at {m:+.2f} s  (n={len(cat_onset_rels[l])} events)")
    print(f"  SVF micro: {len(subjects)} subjects")
    return stacks, len(subjects), markers


# ============================================================================
# AHC DATA
# ============================================================================

def collect_ahc_macro(align):
    stacks, n = _collect_task_macro('ahc', align)
    label = 'trial onsets' if align == 'onset' else 'trial offsets'
    print(f"  AHC macro ({label}): {n} subjects")
    return stacks, n


def _parse_ahc_boundary_events(subject, session, align):
    """Return (event_times, expl_onset_rels) for AHC micro.

    event_times : array of lock-point times in scanner seconds
    expl_onset_rels : gap (new expl. start - preceding expl. end) for each event
    """
    xlsx_path = AHC_ANNOT_DIR / f"{subject}_{session}_task-ahc_desc-sentences.xlsx"
    if not xlsx_path.exists():
        return np.array([]), []

    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()
    df['Prompt Number'] = df['Prompt Number'].ffill()
    df = df.sort_values(['Prompt Number', 'Start Time']).reset_index(drop=True)
    df['Preceding_Possibility'] = df.groupby('Prompt Number')['Possibility Number'].shift(1)
    df['Preceding_End'] = df.groupby('Prompt Number')['End Time'].shift(1)
    df['is_boundary'] = ((df['Possibility Number'] != df['Preceding_Possibility'])
                         & df['Preceding_Possibility'].notna())
    bdf = df[df['is_boundary']].copy()

    event_times = []
    gaps = []
    for _, row in bdf.iterrows():
        prev_offset = row['Preceding_End'] - AHC_SCANNER_OFFSET
        curr_onset  = row['Start Time']    - AHC_SCANNER_OFFSET
        gap = row['Start Time'] - row['Preceding_End']
        if align == 'onset':
            t = curr_onset
        else:
            t = prev_offset
        if t >= TRS_BEFORE * TR:
            event_times.append(t)
            gaps.append(gap)
    return np.array(sorted(event_times)), gaps


def collect_ahc_micro(align):
    """Micro: across-explanation transitions.

    align='onset'  → lock to new explanation onset; marker at -mean(gap)
    align='offset' → lock to preceding explanation offset; marker at +mean(gap)

    Returns (stacks, n_subjects, marker) where marker is a float.
    """
    sub_data = defaultdict(lambda: defaultdict(list))
    all_gaps = []

    for subject, session in find_ahc_sessions():
        try:
            event_times, gaps = _parse_ahc_boundary_events(subject, session, align)
        except Exception:
            continue
        if len(event_times) < 2:
            continue
        all_gaps.extend(gaps)

        roi_ts = extract_roi_timeseries(subject, session, 'ahc')
        if roi_ts is None:
            continue
        for k in ROI_KEYS:
            if k not in roi_ts:
                continue
            epochs = extract_event_locked(roi_ts[k], event_times)
            if epochs is not None:
                sub_data[subject][k].append(epochs.mean(0))

    subjects = sorted(sub_data.keys())
    n_tp = TRS_BEFORE + TRS_AFTER + 1
    stacks = {k: [] for k in ROI_KEYS}
    for sub in subjects:
        for k in ROI_KEYS:
            tcs = sub_data[sub][k]
            stacks[k].append(np.nanmean(tcs, axis=0) if tcs else np.full(n_tp, np.nan))
    for k in ROI_KEYS:
        stacks[k] = np.array(stacks[k])

    sign = -1.0 if align == 'onset' else +1.0
    marker = sign * float(np.mean(all_gaps)) if all_gaps else 0.0
    print(f"  AHC micro (expl. transitions): {len(subjects)} subjects | "
          f"marker at {marker:+.2f} s")
    return stacks, len(subjects), marker


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_line(ax, data, color, label, ls='-'):
    """Plot group mean ± SEM. data = (n_subjects, n_tp)."""
    gmean = np.nanmean(data, axis=0)
    n_valid = np.sum(~np.isnan(data[:, 0]))
    gsem  = np.nanstd(data, axis=0) / np.sqrt(max(n_valid, 1))
    ax.plot(TIME_VEC, gmean, color=color, lw=2, label=label, ls=ls)
    ax.fill_between(TIME_VEC, gmean - gsem, gmean + gsem, color=color, alpha=0.18)


def _style_ax(ax, show_ylabel=False, show_xlabel=False, xlabel='Time from boundary (s)',
              add_gray_window=False):
    _ymin, _ycap = -1.0, 1.5
    cap_frac = (_ycap - _ymin) / (1.8 - _ymin)

    if add_gray_window:
        ax.axvspan(0, 15, ymax=cap_frac, color='lightgrey', alpha=0.5, zorder=0)
    ax.axvline(0, ymax=cap_frac if add_gray_window else 1,
               color='#555555', ls='-', lw=1.2)
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)

    ax.set_xlim(-30, 60)
    ax.set_ylim(_ymin, 1.8)
    ax.set_xticks([-30, -15, 0, 15, 30, 45, 60])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['-1', '0', '1'], fontsize=LABEL_FS - 2)
    ax.tick_params(axis='x', labelsize=LABEL_FS - 2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_bounds(_ymin, _ycap)

    if show_ylabel:
        ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FS)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS - 2)


def make_figure(align, ff_data,
                svf_macro, svf_macro_n, svf_micro, svf_micro_n, svf_markers,
                ahc_macro, ahc_macro_n, ahc_micro, ahc_micro_n, ahc_marker,
                micro_only=False):
    n_rois = len(ROI_SPEC)
    fig, axes = plt.subplots(3, n_rois, figsize=(3.8 * n_rois, 10.5),
                             facecolor='white')

    # x-axis labels depend on alignment and micro_only
    ff_xlabel  = 'Time from boundary (s)'
    if align == 'onset':
        svf_xlabel = 'Time from word onset (s)' if micro_only else 'Time from trial / word onset (s)'
        ahc_xlabel = 'Time from expl. onset (s)' if micro_only else 'Time from trial / expl. onset (s)'
    else:
        svf_xlabel = 'Time from word offset (s)' if micro_only else 'Time from trial / word offset (s)'
        ahc_xlabel = 'Time from expl. offset (s)' if micro_only else 'Time from trial / expl. offset (s)'

    # ── Row 0: FilmFest ───────────────────────────────────────────────────────
    ff_conds_to_plot = [c for c in FF_CONDITIONS if not (micro_only and c == 'between')]
    for col, (roi_key, roi_title) in enumerate(ROI_SPEC):
        ax = axes[0, col]
        for cond in ff_conds_to_plot:
            _plot_line(ax, ff_data[cond][roi_key], FF_COLORS[cond], FF_LABELS[cond])
        _style_ax(ax, show_ylabel=(col == 0), show_xlabel=True, xlabel=ff_xlabel)
        ax.set_title(roi_title, fontsize=LABEL_FS, fontweight='bold', pad=6)
        if col == 0:
            ax.legend(fontsize=LABEL_FS - 5, frameon=False, loc='upper right')

    # ── Row 1: SVF ───────────────────────────────────────────────────────────
    for col, (roi_key, _) in enumerate(ROI_SPEC):
        ax = axes[1, col]
        if not micro_only:
            svf_macro_label = 'Trial onset' if align == 'onset' else 'Trial offset'
            _plot_line(ax, svf_macro[roi_key], SVF_MACRO_COLOR, svf_macro_label)
        for cat_label, _, _orig_color, _orig_ls in CONTEXT_CATS:
            color  = SVF_MICRO_COLORS[cat_label]
            label  = SVF_MICRO_LABELS[cat_label]
            ls     = SVF_MICRO_LS[cat_label]
            marker = svf_markers[cat_label]
            _plot_line(ax, svf_micro[cat_label][roi_key], color, label, ls=ls)
            ax.axvline(marker, color=color, ls=':', lw=1.2, alpha=0.8)
        _style_ax(ax, show_ylabel=(col == 0), show_xlabel=True, xlabel=svf_xlabel)
        if col == 0:
            ax.legend(fontsize=LABEL_FS - 5, frameon=False, loc='upper right')

    # ── Row 2: AHC/EG ────────────────────────────────────────────────────────
    for col, (roi_key, _) in enumerate(ROI_SPEC):
        ax = axes[2, col]
        if not micro_only:
            ahc_macro_label = 'Trial onset' if align == 'onset' else 'Trial offset'
            _plot_line(ax, ahc_macro[roi_key], AHC_MACRO_COLOR, ahc_macro_label)
        _plot_line(ax, ahc_micro[roi_key], AHC_MICRO_COLOR, AHC_MICRO_LABEL)
        ax.axvline(ahc_marker, color=AHC_MICRO_COLOR, ls=':', lw=1.2, alpha=0.8)
        _style_ax(ax, show_ylabel=(col == 0), show_xlabel=True, xlabel=ahc_xlabel)
        if col == 0:
            ax.legend(fontsize=LABEL_FS - 5, frameon=False, loc='upper right')

    # ── Shared y-axis across all panels ───────────────────────────────────────
    if micro_only:
        ymin, ymax = -0.5, 0.5
        yticks, yticklabels = [-0.5, 0, 0.5], ['-0.5', '0', '0.5']
    else:
        ymin = min(ax.get_ylim()[0] for ax in axes.flat)
        ymax = max(ax.get_ylim()[1] for ax in axes.flat)
        yticks, yticklabels = [-1, 0, 1], ['-1', '0', '1']
    for ax in axes.flat:
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=LABEL_FS - 2)
        ax.spines['left'].set_bounds(ymin, ymax)

    # ── Row labels ────────────────────────────────────────────────────────────
    row_labels = [
        'Movie watching',
        'Semantic fluency',
        'Explanation generation',
    ]
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label,
            xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-0.32 if micro_only else -0.22, 0.5), textcoords='axes fraction',
            fontsize=20, rotation=90, va='center', ha='center',
        )

    fig.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--align', choices=['onset', 'offset'], default='onset',
                        help='onset: lock to word/expl/trial onset (default); '
                             'offset: lock to preceding word/expl offset, trial offset')
    parser.add_argument('--micro-only', action='store_true',
                        help='plot only micro boundaries; use zoomed y-axis (-0.5 to 0.5)')
    args = parser.parse_args()
    align = args.align
    micro_only = args.micro_only

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 60)
    print(f'MACRO vs MICRO BOUNDARY RESPONSE — 3 tasks × 8 ROIs  [{align}-locked]')
    print('=' * 60)

    print('\n[FilmFest]')
    ff_data = collect_filmfest_data()

    print('\n[SVF — macro]')
    svf_macro, svf_macro_n = collect_svf_macro(align)

    print('\n[SVF — micro]')
    svf_micro, svf_micro_n, svf_markers = collect_svf_micro(align)

    print('\n[AHC — macro]')
    ahc_macro, ahc_macro_n = collect_ahc_macro(align)

    print('\n[AHC — micro]')
    ahc_micro, ahc_micro_n, ahc_marker = collect_ahc_micro(align)

    print('\nBuilding figure...')
    fig = make_figure(
        align, ff_data,
        svf_macro, svf_macro_n, svf_micro, svf_micro_n, svf_markers,
        ahc_macro, ahc_macro_n, ahc_micro, ahc_micro_n, ahc_marker,
        micro_only=micro_only,
    )

    suffix = f'{align}_micro_only' if micro_only else align
    out = OUTPUT_DIR / f'macro_micro_boundary_8roi_{suffix}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved → {out}')
    print('=' * 60)


if __name__ == '__main__':
    main()
