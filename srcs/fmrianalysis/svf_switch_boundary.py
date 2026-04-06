#!/usr/bin/env python3
"""
SVF Switch/Cluster Boundary Neural Analysis

ROI event-locked time courses (PMC, Hippocampus, Angular Gyrus, Auditory Cortex)
at switch vs cluster events during semantic verbal fluency, using cached parcel
time series.

Events locked to preceding word offset (Su et al. 2025 approach).

Usage:
    python svf_boundary.py
"""
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# === CONFIG ===
from configs.config import DATA_DIR, DERIVATIVES_DIR, FIGS_DIR, TR
from configs.schaefer_rois import ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL, POSTERIOR_MEDIAL, DLPFC, DACC
from fmrianalysis.utils import get_parcel_data

ANNOTATIONS_DIR = DATA_DIR / 'rec/svf_annotated'
RATINGS_DIR = DATA_DIR / 'rec/svf_ratings'
OUTPUT_DIR = FIGS_DIR / 'svf_boundary'

RATERS = ['AS', 'GL', 'JC', 'KG']

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HRF_DELAY = 4.5
WINDOW_DURATION = 6.0
TRS_BEFORE = 2
TRS_AFTER = 15

# === STYLE ===
COLORS = {'switch': '#e74c3c', 'cluster': 'gray'}
LABEL_FS = 12
TITLE_FS = 14

ROI_SPEC = [
    ('pmc',   'Posterior Medial Cortex'),
    ('hipp',  'Hippocampus'),
    ('ag',    'Angular Gyrus'),
    ('dlpfc', 'dlPFC'),
    ('dacc',  'dACC'),
    ('eac',   'Auditory Cortex'),
    ('evc',   'Early Visual Cortex'),
]

# === ROI label matching ===
PMC_LABELS   = POSTERIOR_MEDIAL.get('left_labels', []) + POSTERIOR_MEDIAL.get('right_labels', [])
AG_LABELS    = ANGULAR_GYRUS.get('left_labels', []) + ANGULAR_GYRUS.get('right_labels', [])
EAC_LABELS   = EARLY_AUDITORY.get('left_labels', []) + EARLY_AUDITORY.get('right_labels', [])
EVC_LABELS   = EARLY_VISUAL.get('left_labels', []) + EARLY_VISUAL.get('right_labels', [])
DLPFC_LABELS = DLPFC.get('left_labels', []) + DLPFC.get('right_labels', [])
DACC_LABELS  = DACC.get('left_labels', []) + DACC.get('right_labels', [])
HIPP_KEYWORDS = ['hippocampus']


# ============================================================================
# SESSION DISCOVERY
# ============================================================================

def find_svf_sessions():
    """Auto-detect subject-session pairs with both SVF annotations and BOLD data."""
    sessions = []
    for csv_path in sorted(ANNOTATIONS_DIR.glob(
            'sub-*_ses-*_task-svf_desc-wordtimestampswithswitch.csv')):
        # Skip gio_rated subdirectory files
        if 'gio_rated' in str(csv_path):
            continue
        parts = csv_path.stem.split('_')
        subject, session = parts[0], parts[1]
        bold_path = (DERIVATIVES_DIR / subject / session / "func" /
                     f"{subject}_{session}_task-svf_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
        if bold_path.exists():
            sessions.append((subject, session))
    return sessions


# ============================================================================
# EVENT EXTRACTION
# ============================================================================

def get_events(subject, session):
    """Load SVF events with preceding word offset timing.

    Filters:
      - Remove "next" words
      - Remove depletion switches (switch after switch or after "next")

    Returns DataFrame with columns: trial_type ('switch'/'cluster'), onset
    """
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV for {subject} {session}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)

    # Preceding info BEFORE filtering
    df["preceding_end"] = df["end"].shift(1)
    df["preceding_switch_flag"] = df["switch_flag"].shift(1)
    df["preceding_word"] = df["transcription"].shift(1).astype(str).str.lower()

    # Remove "next" words
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()

    # Remove depletion switches
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["preceding_switch_flag"] == 1
    prev_was_next = df["preceding_word"] == "next"
    df = df[~(is_switch & (prev_was_switch | prev_was_next))].copy()

    # Lock to preceding word offset (scanner time)
    df["onset"] = df["preceding_end"] - SCANNER_START_OFFSET
    df["trial_type"] = df["switch_flag"].map({1: "switch", 0: "cluster"})

    df = df.dropna(subset=["onset"])
    df = df[df["onset"] >= TRS_BEFORE * TR]
    return df


def get_switch_cluster_timepoints(subject, session):
    """Return (switch_tp, cluster_tp) arrays in scanner time."""
    df = get_events(subject, session)
    switch_tp = df[df["trial_type"] == "switch"]["onset"].values
    cluster_tp = df[df["trial_type"] == "cluster"]["onset"].values
    return switch_tp, cluster_tp


# ============================================================================
# ROI EXTRACTION FROM CACHED PARCEL DATA
# ============================================================================

def _avg_labels(parcel_dict, label_list):
    """Average time series across parcels matching explicit label list."""
    ts = [parcel_dict[l] for l in label_list if l in parcel_dict]
    return np.column_stack(ts).mean(axis=1)


def _avg_keywords(parcel_dict, keywords):
    """Average time series across parcels whose label contains any keyword."""
    ts = [v for l, v in parcel_dict.items()
          if l != 'Background' and any(kw in l.lower() for kw in keywords)]
    return np.column_stack(ts).mean(axis=1)


def extract_roi_timeseries(subject, session, task='svf'):
    """Load cached parcel data and return dict of ROI time series."""
    schaefer = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
    ho_sub = get_parcel_data(subject, session, task, atlas='HarvardOxford_sub')
    return {
        'pmc':   _avg_labels(schaefer, PMC_LABELS),
        'ag':    _avg_labels(schaefer, AG_LABELS),
        'dlpfc': _avg_labels(schaefer, DLPFC_LABELS),
        'dacc':  _avg_labels(schaefer, DACC_LABELS),
        'eac':   _avg_labels(schaefer, EAC_LABELS),
        'evc':   _avg_labels(schaefer, EVC_LABELS),
        'hipp':  _avg_keywords(ho_sub, HIPP_KEYWORDS),
    }


# ============================================================================
# EPOCH EXTRACTION
# ============================================================================

def extract_event_locked_epochs(signal, event_times_sec):
    """Extract event-locked epochs from signal.

    Returns array (n_valid_events, n_timepoints) or None.
    """
    n = len(signal)
    if len(event_times_sec) == 0:
        return None
    centers = np.round(np.array(event_times_sec) / TR).astype(int)
    offsets = np.arange(-TRS_BEFORE, TRS_AFTER + 1)
    idx = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n), axis=1)
    if not valid.any():
        return None
    epochs = signal[idx[valid]]
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)
    return epochs


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_by_subject(results):
    """Group session results by subject, average time courses."""
    by_sub = defaultdict(list)
    for r in results:
        by_sub[r['subject']].append(r)

    agg = {}
    for sub, sessions in sorted(by_sub.items()):
        n = len(sessions)
        d = {'time_vec': sessions[0]['time_vec'], 'n_sessions': n}
        for roi, _ in ROI_SPEC:
            for cond in ('switch', 'cluster'):
                k = f'{roi}_{cond}_tc'
                stack = np.array([s[k] for s in sessions])
                d[k] = stack.mean(0)
                d[f'{roi}_{cond}_sem'] = (
                    stack.std(0) / np.sqrt(n) if n > 1 else np.zeros_like(stack[0]))
        agg[sub] = d
    return agg


# ============================================================================
# PLOTTING -- Subject level
# ============================================================================

def plot_subject_timecourse(subject, data):
    """Per-subject: 5x1 ROI grid, switch vs cluster with SEM across sessions."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 18))
    tv = data['time_vec']
    n_ses = data['n_sessions']

    fig.suptitle(f"SVF Switch vs Cluster: {subject} (N={n_ses} sessions)\n"
                 f"Locked to preceding word offset",
                 fontsize=TITLE_FS, fontweight='bold')

    for ax, (roi, title) in zip(axes, ROI_SPEC):
        for cond, label in [('switch', 'Switch'), ('cluster', 'Cluster')]:
            c = COLORS[cond]
            ax.plot(tv, data[f'{roi}_{cond}_tc'], color=c, lw=3, label=label)
            ax.fill_between(tv,
                            data[f'{roi}_{cond}_tc'] - data[f'{roi}_{cond}_sem'],
                            data[f'{roi}_{cond}_tc'] + data[f'{roi}_{cond}_sem'],
                            color=c, alpha=0.3)
        ax.axvline(0, color='grey', ls='--', alpha=0.5)
        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.15, color='yellow')
        ax.set(xlabel='Time from preceding word offset (s)', ylabel='BOLD (z-scored)',
               title=title, xlim=(tv[0], tv[-1]))
        ax.legend(loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)

    # Unify y-axis
    all_ylims = [ax.get_ylim() for ax in axes]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    out = OUTPUT_DIR / f"{subject}_svf_switch_cluster.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# PLOTTING -- Group level
# ============================================================================

def plot_group_timecourse(subject_data, n_subjects):
    """Group-level: 5x1 ROI grid, mean across subjects with SEM."""
    tv = next(iter(subject_data.values()))['time_vec']

    stacks = {}
    for cond in ('switch', 'cluster'):
        for roi, _ in ROI_SPEC:
            stacks[f'{roi}_{cond}'] = np.array(
                [d[f'{roi}_{cond}_tc'] for d in subject_data.values()])

    fig, axes = plt.subplots(5, 1, figsize=(12, 18))
    fig.suptitle(f"SVF Switch vs Cluster: Group (N={n_subjects} subjects)\n"
                 f"Locked to preceding word offset | * p < 0.05 uncorrected",
                 fontsize=TITLE_FS, fontweight='bold')

    for ax, (roi, title) in zip(axes, ROI_SPEC):
        sw = stacks[f'{roi}_switch']
        cl = stacks[f'{roi}_cluster']
        for arr, label, c in [(sw, 'Switch', COLORS['switch']),
                               (cl, 'Cluster', COLORS['cluster'])]:
            m = arr.mean(0)
            se = arr.std(0) / np.sqrt(n_subjects)
            ax.plot(tv, m, color=c, lw=3, label=label, marker='o', ms=4)
            ax.fill_between(tv, m - se, m + se, color=c, alpha=0.3)

        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.15, color='yellow')
        ax.set(xlabel='Time from preceding word offset (s)', ylabel='BOLD (z-scored)',
               title=title, xlim=(tv[0] - 0.5, tv[-1] + 0.5))
        ax.legend(loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)

    # Unify y-axis, then add significance markers
    all_ylims = [ax.get_ylim() for ax in axes]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax, (roi, _) in zip(axes, ROI_SPEC):
        ax.set_ylim(ymin, ymax)
        sw = stacks[f'{roi}_switch']
        cl = stacks[f'{roi}_cluster']
        pvals = np.array([stats.ttest_rel(sw[:, t], cl[:, t])[1]
                          for t in range(len(tv))])
        sig = np.where(pvals < 0.05)[0]
        if len(sig):
            yp = ymin + 0.05 * (ymax - ymin)
            for i in sig:
                ax.text(tv[i], yp, '*', fontsize=14, ha='center', fontweight='bold')

    plt.tight_layout()
    out = OUTPUT_DIR / "GROUP_svf_switch_cluster.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# MULTI-RATER AGREEMENT ANALYSIS
# ============================================================================

# Rating scale in xlsx files: 0=unsure, 1=clustering, 2=switching

def find_rated_sessions():
    """Find sessions that have ratings from all 4 raters AND BOLD data."""
    sessions = []
    # Use first rater's files as reference
    for xlsx_path in sorted((RATINGS_DIR / RATERS[0]).glob(
            'sub-*_ses-*_task-svf_desc-wordtimestamps_rated.xlsx')):
        parts = xlsx_path.stem.split('_')
        subject, session = parts[0], parts[1]
        # Check all raters have this session
        all_exist = all(
            (RATINGS_DIR / r / xlsx_path.name).exists() for r in RATERS)
        # Check BOLD exists
        bold_path = (DERIVATIVES_DIR / subject / session / "func" /
                     f"{subject}_{session}_task-svf_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
        if all_exist and bold_path.exists():
            sessions.append((subject, session))
    return sessions


def load_multirater_events(subject, session):
    """Load SVF events with multi-rater agreement.

    For each word, counts how many of the 4 raters rated it as switching (2),
    clustering (1), or unsure (0).

    Filters:
      - Remove "next" words
      - Remove depletion switches (majority-switch after majority-switch or "next")

    Returns DataFrame with columns including: n_switch, n_cluster, onset
    """
    dfs = {}
    for rater in RATERS:
        path = (RATINGS_DIR / rater /
                f"{subject}_{session}_task-svf_desc-wordtimestamps_rated.xlsx")
        dfs[rater] = pd.read_excel(path)

    # Use first rater as base for word/timing info
    base = dfs[RATERS[0]][['word', 'start', 'end']].copy()

    # Collect each rater's switch_flag
    for rater in RATERS:
        base[f'r_{rater}'] = pd.to_numeric(
            dfs[rater]['switch_flag'], errors='coerce')

    rating_cols = [f'r_{r}' for r in RATERS]

    # Count agreement (rating scale: 0=unsure, 1=cluster, 2=switch)
    base['n_switch'] = (base[rating_cols] == 2).sum(axis=1)
    base['n_cluster'] = (base[rating_cols] == 1).sum(axis=1)
    base['n_unsure'] = (base[rating_cols] == 0).sum(axis=1)

    # Track preceding info BEFORE filtering
    base['preceding_end'] = base['end'].shift(1)
    base['preceding_word'] = base['word'].shift(1).astype(str).str.lower()
    base['preceding_n_switch'] = base['n_switch'].shift(1)

    # Remove "next" words
    base = base[base['word'].astype(str).str.lower() != 'next'].copy()

    # Remove depletion switches: majority-switch (>=3) after majority-switch or "next"
    is_majority_switch = base['n_switch'] >= 3
    prev_majority_switch = base['preceding_n_switch'] >= 3
    prev_was_next = base['preceding_word'] == 'next'
    base = base[~(is_majority_switch & (prev_majority_switch | prev_was_next))].copy()

    # Lock to preceding word offset
    base['onset'] = base['preceding_end'] - SCANNER_START_OFFSET
    base = base.dropna(subset=['onset'])
    base = base[base['onset'] >= TRS_BEFORE * TR]

    # Relative timing of current word w.r.t. lock point (preceding word offset)
    base['word_onset_rel'] = base['start'] - base['preceding_end']
    base['word_offset_rel'] = base['end'] - base['preceding_end']

    return base


# Agreement category definitions (3-way: consensus switch, ambiguous, consensus cluster):
#   (label, condition_func, color, linestyle)
AGREEMENT_CATS = [
    ('Consensus Switch', lambda df: df['n_switch'] >= 3, '#c0392b', '-'),
    ('No-Consensus',        lambda df: (df['n_switch'] < 3) & (df['n_cluster'] < 3), '#7f8c8d', '--'),
    ('Consensus Cluster', lambda df: df['n_cluster'] >= 3, '#2c3e50', '-'),
]


def _add_context_filter(events_df):
    """Add context columns: whether the 2 preceding words are consensus cluster.

    After all filtering, checks whether the two immediately preceding words
    (in the filtered sequence) both have n_cluster >= 3.
    """
    df = events_df.copy()
    df['is_cons_cluster'] = df['n_cluster'] >= 3
    df['prev1_cons_cluster'] = df['is_cons_cluster'].shift(1, fill_value=False)
    df['prev2_cons_cluster'] = df['is_cons_cluster'].shift(2, fill_value=False)
    df['context_cc'] = df['prev1_cons_cluster'] & df['prev2_cons_cluster']
    return df


# Context-filtered categories: only words preceded by >=2 consensus cluster words
CONTEXT_CATS = [
    ('CC-Switch',  lambda df: (df['n_switch'] >= 3) & df['context_cc'],  '#c0392b', '-'),
    ('CC-No-Consensus', lambda df: (df['n_switch'] < 3) & (df['n_cluster'] < 3) & df['context_cc'], '#7f8c8d', '--'),
    ('CC-Cluster', lambda df: (df['n_cluster'] >= 3) & df['context_cc'], '#2c3e50', '-'),
]


def _collect_agreement_data(sessions, time_vec, cat_defs, apply_context=False):
    """Collect per-subject epoch data for a set of agreement categories.

    Returns (subject_stacks, subjects, word_timing, subject_event_counts)
    where subject_stacks[cat_label][roi_key] = (n_subjects, n_tp) array
    and subject_event_counts[cat_label] = list of per-subject event counts.
    """
    roi_keys = [k for k, _ in ROI_SPEC]

    # subject -> cat_label -> roi_key -> list of session-mean TCs
    sub_cat_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # subject -> cat_label -> list of event counts per session
    sub_event_counts = defaultdict(lambda: defaultdict(list))
    # subject -> list of total words per session
    sub_total_counts = defaultdict(list)
    subject_set = set()

    all_word_onset_rel = []
    all_word_offset_rel = []

    for subject, session in sessions:
        try:
            events_df = load_multirater_events(subject, session)
            roi_ts = extract_roi_timeseries(subject, session)
        except Exception:
            continue

        if apply_context:
            events_df = _add_context_filter(events_df)

        all_word_onset_rel.extend(events_df['word_onset_rel'].dropna().values)
        all_word_offset_rel.extend(events_df['word_offset_rel'].dropna().values)

        n_total = len(events_df)
        sub_total_counts[subject].append(n_total)

        for cat_label, cond_fn, _, _ in cat_defs:
            mask = cond_fn(events_df)
            tp = events_df.loc[mask, 'onset'].values
            sub_event_counts[subject][cat_label].append(len(tp))

            for roi_key in roi_keys:
                epochs = extract_event_locked_epochs(roi_ts[roi_key], tp)
                if epochs is not None:
                    sub_cat_data[subject][cat_label][roi_key].append(
                        epochs.mean(axis=0))
                else:
                    sub_cat_data[subject][cat_label][roi_key].append(
                        np.full(len(time_vec), np.nan))

        subject_set.add(subject)

    subjects = sorted(subject_set)
    n_sub = len(subjects)

    # Build stacks
    cat_stacks = {}
    for cat_label, _, _, _ in cat_defs:
        cat_stacks[cat_label] = {}
        for roi_key in roi_keys:
            sub_means = []
            for sub in subjects:
                tcs = sub_cat_data[sub][cat_label][roi_key]
                if tcs:
                    sub_means.append(np.nanmean(tcs, axis=0))
                else:
                    sub_means.append(np.full(len(time_vec), np.nan))
            cat_stacks[cat_label][roi_key] = np.array(sub_means)

    # Per-subject event stats: sum across sessions
    event_stats = {}
    for cat_label, _, _, _ in cat_defs:
        per_sub_counts = []
        per_sub_props = []
        for sub in subjects:
            n_events = sum(sub_event_counts[sub][cat_label])
            n_total = sum(sub_total_counts[sub])
            per_sub_counts.append(n_events)
            per_sub_props.append(n_events / n_total * 100 if n_total > 0 else 0)
        event_stats[cat_label] = {
            'count_mean': np.mean(per_sub_counts),
            'count_std': np.std(per_sub_counts),
            'prop_mean': np.mean(per_sub_props),
            'prop_std': np.std(per_sub_props),
        }

    word_timing = {
        'onset': np.mean(all_word_onset_rel) if all_word_onset_rel else 0,
        'offset': np.mean(all_word_offset_rel) if all_word_offset_rel else 0,
    }

    return cat_stacks, subjects, word_timing, event_stats


def plot_agreement_group(time_vec):
    """Group-level plot: 4x2 layout.

    Left column:  all events by agreement category
    Right column: context-filtered (preceded by >=2 consensus cluster words)
    Vertical lines mark average current-word onset and offset.
    """
    sessions = find_rated_sessions()
    print(f"\nFound {len(sessions)} rated sessions with BOLD data")

    # Collect data for both panels
    print("\n  [All events]")
    all_stacks, subjects, word_timing, all_stats = _collect_agreement_data(
        sessions, time_vec, AGREEMENT_CATS, apply_context=False)

    print("\n  [Context-filtered: cluster-cluster-X]")
    ctx_stacks, _, ctx_timing, ctx_stats = _collect_agreement_data(
        sessions, time_vec, CONTEXT_CATS, apply_context=True)

    n_sub = len(subjects)
    print(f"\n{n_sub} subjects with rated data")
    print(f"  Avg word onset:  {word_timing['onset']:.2f}s | offset: {word_timing['offset']:.2f}s")

    # Print event stats
    for title, stats in [("All events", all_stats), ("Context-filtered", ctx_stats)]:
        print(f"\n  {title}:")
        for cat_label, s in stats.items():
            print(f"    {cat_label}: {s['count_mean']:.1f} +/- {s['count_std']:.1f} "
                  f"({s['prop_mean']:.1f}% +/- {s['prop_std']:.1f}%)")

    roi_keys = [k for k, _ in ROI_SPEC]

    # --- Plot (n_rois rows x 2 cols) ---
    n_rois = len(ROI_SPEC)
    fig, axes = plt.subplots(n_rois, 2, figsize=(14, 3.4 * n_rois))
    fig.suptitle(
        f"SVF by Inter-Rater Agreement: Group (N={n_sub} subjects)\n"
        f"Locked to preceding word offset | shading = SEM across subjects",
        fontsize=TITLE_FS + 1, fontweight='bold')

    col_configs = [
        (0, 'All events', AGREEMENT_CATS, all_stacks, all_stats, word_timing),
        (1, 'Context: cluster-cluster-X', CONTEXT_CATS, ctx_stacks, ctx_stats, ctx_timing),
    ]

    for col, col_title, cat_defs, stacks, stats, timing in col_configs:
        for row, (roi_key, roi_title) in enumerate(ROI_SPEC):
            ax = axes[row, col]

            for cat_label, _, color, ls in cat_defs:
                data = stacks[cat_label][roi_key]
                mean = np.nanmean(data, axis=0)
                sem = np.nanstd(data, axis=0) / np.sqrt(n_sub)
                s = stats[cat_label]
                display_label = (f"{cat_label}  "
                                 f"n={s['count_mean']:.0f}\u00b1{s['count_std']:.0f} "
                                 f"({s['prop_mean']:.0f}\u00b1{s['prop_std']:.0f}%)")
                ax.plot(time_vec, mean, color=color, ls=ls, lw=2.5,
                        label=display_label)
                ax.fill_between(time_vec, mean - sem, mean + sem,
                                color=color, alpha=0.15)

            # Vertical markers
            ax.axvline(0, color='grey', ls='--', lw=1)
            ax.axhline(0, color='k', ls='-', alpha=0.3)
            ax.axvline(timing['onset'], color='#2ecc71', ls='-', lw=1.5)
            ax.axvline(timing['offset'], color='#e67e22', ls='-', lw=1.5)

            ax.set(ylabel='BOLD (z-scored)',
                   xlim=(time_vec[0] - 0.5, time_vec[-1] + 0.5))
            ax.spines[['top', 'right']].set_visible(False)

            # ROI title on left column, column title on top row
            if col == 0:
                ax.set_ylabel(f'{roi_title}\nBOLD (z-scored)')
            if row == 0:
                ax.set_title(col_title, fontsize=TITLE_FS, fontweight='bold')

        # x-label only on bottom row
        axes[-1, col].set_xlabel('Time from preceding word offset (s)')

    # Unify y-axis across ALL subplots
    all_ylims = [ax.get_ylim() for ax in axes.flat]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax in axes.flat:
        ax.set_ylim(ymin, ymax)

    # Legend on top-left and top-right
    for col in range(2):
        axes[0, col].legend(loc='upper right', fontsize=7)

    # Add vertical line labels on top-right subplot only (avoid clutter)
    ax0 = axes[0, 1]
    ax0.text(word_timing['onset'], ymax * 0.95, 'word\nonset',
             color='#2ecc71', fontsize=7, ha='center', va='top')
    ax0.text(word_timing['offset'], ymax * 0.95, 'word\noffset',
             color='#e67e22', fontsize=7, ha='center', va='top')

    plt.tight_layout()
    out = OUTPUT_DIR / "GROUP_svf_agreement.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SVF SWITCH/CLUSTER BOUNDARY ANALYSIS")
    print("(Locked to preceding word offset)")
    print("=" * 60)

    print(f"\nROI parcels -- PMC: {len(PMC_LABELS)}, AG: {len(AG_LABELS)}, "
          f"EAC: {len(EAC_LABELS)}, EVC: {len(EVC_LABELS)}, Hipp: matched by keyword")

    # Discover sessions
    sessions = find_svf_sessions()
    print(f"\nFound {len(sessions)} SVF sessions:")
    for sub, ses in sessions:
        print(f"  {sub} {ses}")

    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR

    # Process all sessions
    all_results = []
    for subject, session in sessions:
        print(f"\n--- {subject} {session} ---")
        try:
            switch_tp, cluster_tp = get_switch_cluster_timepoints(subject, session)
            print(f"  Events: {len(switch_tp)} switch, {len(cluster_tp)} cluster")

            roi_ts = extract_roi_timeseries(subject, session)

            result = {
                'subject': subject, 'session': session, 'time_vec': time_vec,
                'n_switch': len(switch_tp), 'n_cluster': len(cluster_tp),
            }
            for roi_key, _ in ROI_SPEC:
                for cond, tp in [('switch', switch_tp), ('cluster', cluster_tp)]:
                    epochs = extract_event_locked_epochs(roi_ts[roi_key], tp)
                    if epochs is not None:
                        result[f'{roi_key}_{cond}_tc'] = epochs.mean(axis=0)
                    else:
                        result[f'{roi_key}_{cond}_tc'] = np.zeros(len(time_vec))

            all_results.append(result)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    print(f"\nProcessed {len(all_results)} sessions")

    # Print event summary
    total_sw = sum(r['n_switch'] for r in all_results)
    total_cl = sum(r['n_cluster'] for r in all_results)
    print(f"Total events: {total_sw} switch, {total_cl} cluster")

    # Aggregate by subject
    sub_agg = aggregate_by_subject(all_results)

    # Subject-level plots
    print("\n--- Subject-level plots ---")
    for sub, data in sub_agg.items():
        plot_subject_timecourse(sub, data)

    # Group-level plot
    n_subjects = len(sub_agg)
    if n_subjects >= 2:
        print("\n--- Group-level plot ---")
        plot_group_timecourse(sub_agg, n_subjects)

    # --- Multi-rater agreement plot ---
    print("\n--- Inter-rater agreement analysis ---")
    plot_agreement_group(time_vec)

    print("\n" + "=" * 60)
    print(f"DONE. Figures in {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
