#!/usr/bin/env python3
"""
AHC Boundary/Non-Boundary Neural Analysis

ROI event-locked time courses (PMC, Hippocampus, Angular Gyrus, Auditory Cortex)
at across-possibility boundaries vs non-boundary periods during ad-hoc category
generation, using cached parcel time series.

Events:
  - Boundary: onset of across-possibility transitions
  - Non-boundary: midpoint of long possibilities (>= 10s)

Usage:
    python ahc_boundary.py
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

ANNOTATIONS_DIR = DATA_DIR / 'rec/ahc_sentences'
OUTPUT_DIR = FIGS_DIR / 'ahc_boundary'

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HRF_DELAY = 4.5
WINDOW_DURATION = 6.0
MIN_POSSIBILITY_DURATION = 10.0  # seconds, for non-boundary midpoint events
TRS_BEFORE = 2
TRS_AFTER = 15

# === STYLE ===
COLORS = {'boundary': '#e74c3c', 'nonboundary': 'gray'}
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

def find_ahc_sessions():
    """Auto-detect subject-session pairs with both AHC annotations and BOLD data."""
    sessions = []
    for xlsx_path in sorted(ANNOTATIONS_DIR.glob(
            'sub-*_ses-*_task-ahc_desc-sentences.xlsx')):
        parts = xlsx_path.stem.split('_')
        subject, session = parts[0], parts[1]
        bold_path = (DERIVATIVES_DIR / subject / session / "func" /
                     f"{subject}_{session}_task-ahc_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
        if bold_path.exists():
            sessions.append((subject, session))
    return sessions


# ============================================================================
# EVENT EXTRACTION
# ============================================================================

def get_boundary_events(subject, session):
    """Get boundary and non-boundary timepoints for AHC.

    Boundary = offset (end time) of the previous possibility, i.e. the end of
               the last sentence before an across-possibility transition.
    Non-boundary = midpoint of long possibilities >= 10s (scanner time)

    Returns (boundary_tp, nonboundary_tp) arrays.
    """
    xlsx_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-ahc_desc-sentences.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"No AHC file: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()
    df['Prompt Number'] = df['Prompt Number'].ffill()
    df = df.sort_values(['Prompt Number', 'Start Time']).reset_index(drop=True)

    # --- Boundary: offset of the previous possibility ---
    # For each sentence, track the preceding sentence's end time and possibility
    df['Preceding_Possibility'] = df.groupby('Prompt Number')['Possibility Number'].shift(1)
    df['Preceding_End'] = df.groupby('Prompt Number')['End Time'].shift(1)
    df['is_boundary'] = ((df['Possibility Number'] != df['Preceding_Possibility'])
                         & df['Preceding_Possibility'].notna())

    boundary_tp = []
    for _, row in df[df['is_boundary']].iterrows():
        # Use end time of preceding sentence (= offset of previous possibility)
        prev_offset = row['Preceding_End'] - SCANNER_START_OFFSET
        if prev_offset >= TRS_BEFORE * TR:
            boundary_tp.append(prev_offset)

    # --- Non-boundary: midpoint of long possibilities ---
    df['poss_group'] = ((df['Possibility Number'] != df['Possibility Number'].shift(1)) |
                        (df['Prompt Number'] != df['Prompt Number'].shift(1))).cumsum()

    nonboundary_tp = []
    for _, group_df in df.groupby('poss_group'):
        poss_start = group_df['Start Time'].min()
        poss_end = group_df['End Time'].max()
        poss_dur = poss_end - poss_start
        if poss_dur >= MIN_POSSIBILITY_DURATION:
            middle = poss_start + poss_dur / 2 - SCANNER_START_OFFSET
            if middle >= TRS_BEFORE * TR:
                nonboundary_tp.append(middle)

    return np.array(sorted(boundary_tp)), np.array(sorted(nonboundary_tp))


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


def extract_roi_timeseries(subject, session, task='ahc'):
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
            for cond in ('boundary', 'nonboundary'):
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
    """Per-subject: 5x1 ROI grid, boundary vs non-boundary with SEM across sessions."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 18))
    tv = data['time_vec']
    n_ses = data['n_sessions']

    fig.suptitle(f"AHC Boundary vs Non-Boundary: {subject} (N={n_ses} sessions)\n"
                 f"Locked to preceding possibility offset",
                 fontsize=TITLE_FS, fontweight='bold')

    for ax, (roi, title) in zip(axes, ROI_SPEC):
        for cond, label in [('boundary', 'Boundary'), ('nonboundary', 'Non-boundary')]:
            c = COLORS[cond]
            ax.plot(tv, data[f'{roi}_{cond}_tc'], color=c, lw=3, label=label)
            ax.fill_between(tv,
                            data[f'{roi}_{cond}_tc'] - data[f'{roi}_{cond}_sem'],
                            data[f'{roi}_{cond}_tc'] + data[f'{roi}_{cond}_sem'],
                            color=c, alpha=0.3)
        ax.axvline(0, color='grey', ls='--', alpha=0.5)
        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.set(xlabel='Time from preceding possibility offset (s)', ylabel='BOLD (z-scored)',
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
    out = OUTPUT_DIR / f"{subject}_ahc_boundary.png"
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
    for cond in ('boundary', 'nonboundary'):
        for roi, _ in ROI_SPEC:
            stacks[f'{roi}_{cond}'] = np.array(
                [d[f'{roi}_{cond}_tc'] for d in subject_data.values()])

    fig, axes = plt.subplots(5, 1, figsize=(12, 18))
    fig.suptitle(f"AHC Boundary vs Non-Boundary: Group (N={n_subjects} subjects)\n"
                 f"Locked to preceding possibility offset | * p < 0.05 uncorrected",
                 fontsize=TITLE_FS, fontweight='bold')

    for ax, (roi, title) in zip(axes, ROI_SPEC):
        bnd = stacks[f'{roi}_boundary']
        nbnd = stacks[f'{roi}_nonboundary']
        for arr, label, c in [(bnd, 'Boundary', COLORS['boundary']),
                               (nbnd, 'Non-boundary', COLORS['nonboundary'])]:
            m = arr.mean(0)
            se = arr.std(0) / np.sqrt(n_subjects)
            ax.plot(tv, m, color=c, lw=3, label=label, marker='o', ms=4)
            ax.fill_between(tv, m - se, m + se, color=c, alpha=0.3)

        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.set(xlabel='Time from preceding possibility offset (s)', ylabel='BOLD (z-scored)',
               title=title, xlim=(tv[0] - 0.5, tv[-1] + 0.5))
        ax.legend(loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)

    # Unify y-axis, then add significance markers
    all_ylims = [ax.get_ylim() for ax in axes]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax, (roi, _) in zip(axes, ROI_SPEC):
        ax.set_ylim(ymin, ymax)
        bnd = stacks[f'{roi}_boundary']
        nbnd = stacks[f'{roi}_nonboundary']
        pvals = np.array([stats.ttest_rel(bnd[:, t], nbnd[:, t])[1]
                          for t in range(len(tv))])
        sig = np.where(pvals < 0.05)[0]
        if len(sig):
            yp = ymin + 0.05 * (ymax - ymin)
            for i in sig:
                ax.text(tv[i], yp, '*', fontsize=14, ha='center', fontweight='bold')

    plt.tight_layout()
    out = OUTPUT_DIR / "GROUP_ahc_boundary.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("AHC BOUNDARY/NON-BOUNDARY ANALYSIS")
    print("(Locked to preceding possibility offset)")
    print("=" * 60)

    print(f"\nROI parcels -- PMC: {len(PMC_LABELS)}, AG: {len(AG_LABELS)}, "
          f"EAC: {len(EAC_LABELS)}, EVC: {len(EVC_LABELS)}, Hipp: matched by keyword")

    # Discover sessions
    sessions = find_ahc_sessions()
    print(f"\nFound {len(sessions)} AHC sessions:")
    for sub, ses in sessions:
        print(f"  {sub} {ses}")

    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR

    # Process all sessions
    all_results = []
    for subject, session in sessions:
        print(f"\n--- {subject} {session} ---")
        try:
            boundary_tp, nonboundary_tp = get_boundary_events(subject, session)
            print(f"  Events: {len(boundary_tp)} boundary, {len(nonboundary_tp)} non-boundary")

            if len(boundary_tp) < 2:
                print(f"  SKIPPED: insufficient boundary events")
                continue

            roi_ts = extract_roi_timeseries(subject, session)

            result = {
                'subject': subject, 'session': session, 'time_vec': time_vec,
                'n_boundary': len(boundary_tp), 'n_nonboundary': len(nonboundary_tp),
            }
            for roi_key, _ in ROI_SPEC:
                for cond, tp in [('boundary', boundary_tp), ('nonboundary', nonboundary_tp)]:
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
    total_bnd = sum(r['n_boundary'] for r in all_results)
    total_nbnd = sum(r['n_nonboundary'] for r in all_results)
    print(f"Total events: {total_bnd} boundary, {total_nbnd} non-boundary")

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

    print("\n" + "=" * 60)
    print(f"DONE. Figures in {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
