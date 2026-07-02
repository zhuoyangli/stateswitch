#!/usr/bin/env python3
"""
SVF Peri-Boundary ROI Time Courses by Rater-Consensus Level

Event-locked ROI BOLD time courses around word onsets during the semantic
verbal fluency (word generation) task, split by how strongly the 7 raters
agreed that a word was a switch (vs. a within-cluster continuation).

Consensus levels come from the aggregated per-word rating file
    figs/behavior/svf_transition_irr/svf_transition_word_scores.csv
produced by srcs/behavior/svf_transition_rater_calibration.py.

Two grouping modes (--by):
  class  (default) : clustering (k<=1) / ambiguous (k=2..5) / switching (k>=6)
                     using the `consensus_class` column.
  votes            : one curve per k_switch_votes level (0..7).

Word times in the CSV are transcript-recording seconds; scan-relative time =
    t - SCANNER_START_OFFSET (12.0 s).

Alignment (--align):
  onset  (default) : lock to the word's own onset (`start`).
  offset           : lock to the offset (`end`) of the preceding word within the
                     same category (the moment just before the transition).

Usage:
    uv run python srcs/fmrianalysis/svf_boundary_consensus.py
    uv run python srcs/fmrianalysis/svf_boundary_consensus.py --by votes
    uv run python srcs/fmrianalysis/svf_boundary_consensus.py --align offset
"""
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from scipy import stats

# === CONFIG ===
from configs.config import FIGS_DIR, DERIVATIVES_DIR, TR
from configs.schaefer_rois import (
    ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL, POSTERIOR_MEDIAL,
    DLPFC, VLPFC, DACC,
)
from fmrianalysis.utils import get_parcel_data

WORD_SCORES = FIGS_DIR / 'behavior' / 'svf_transition_irr' / 'svf_transition_word_scores.csv'
OUTPUT_DIR = FIGS_DIR / 'svf_boundary_consensus'

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
TRS_BEFORE = 4          # 6 s pre
TRS_AFTER = 12          # 18 s post

# === STYLE ===
LABEL_FS = 12
TITLE_FS = 14

ROI_SPEC = [
    ('pmc',   'Posterior Medial Cortex'),
    ('hipp',  'Hippocampus'),
    ('ag',    'Angular Gyrus'),
    ('dlpfc', 'dlPFC'),
    ('vlpfc', 'vlPFC'),
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
VLPFC_LABELS = VLPFC.get('left_labels', []) + VLPFC.get('right_labels', [])
DACC_LABELS  = DACC.get('left_labels', []) + DACC.get('right_labels', [])
HIPP_KEYWORDS = ['hippocampus']

# 3-way consensus_class palette (matches svf_transition_consensus.py band colors)
CLASS_CONDITIONS = [
    ('clustering', 'Clustering (k≤1)', '#4c72b0'),
    ('ambiguous',  'Ambiguous (k=2–5)', '#bbbbbb'),
    ('switching',  'Switching (k≥6)',  '#c44e52'),
]


# ============================================================================
# CONDITIONS
# ============================================================================

def build_conditions(mode):
    """Return list of (key, label, color) for the requested grouping mode."""
    if mode == 'class':
        return list(CLASS_CONDITIONS)
    if mode == 'votes':
        cmap = cm.get_cmap('coolwarm')
        conds = []
        for k in range(8):
            color = cmap(k / 7.0)
            conds.append((f'k{k}', f'{k}/7 switch votes', color))
        return conds
    raise ValueError(f"Unknown grouping mode: {mode}")


def assign_condition(df, mode):
    """Return a Series of condition keys aligned to df rows (NaN -> dropped)."""
    if mode == 'class':
        return df['consensus_class']
    if mode == 'votes':
        return 'k' + df['k_switch_votes'].astype(int).astype(str)
    raise ValueError(f"Unknown grouping mode: {mode}")


# ============================================================================
# SESSION DISCOVERY
# ============================================================================

def find_svf_sessions(words):
    """(subject, session) pairs present in the consensus CSV that also have BOLD."""
    sessions = []
    for (subject, session), _ in words.groupby(['subject', 'session']):
        bold_path = (DERIVATIVES_DIR / subject / session / "func" /
                     f"{subject}_{session}_task-svf_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
        if bold_path.exists():
            sessions.append((subject, session))
        else:
            print(f"  (no BOLD for {subject} {session}, skipping)")
    return sessions


# ============================================================================
# EVENT EXTRACTION
# ============================================================================

def get_consensus_events(words_ses, conditions, align):
    """Scan-relative event times per condition for one (subject, session).

    Returns {cond_key: np.array of times (s)}.
    """
    df = words_ses.sort_values('start').reset_index(drop=True)

    if align == 'offset':
        # offset of the preceding word within the same category block
        df['lock'] = df.groupby('category')['end'].shift(1)
    else:  # onset
        df['lock'] = df['start']

    df['event_t'] = df['lock'] - SCANNER_START_OFFSET
    df = df[df['event_t'].notna() & (df['event_t'] >= TRS_BEFORE * TR)]

    cond_of = assign_condition(df, ARGS_MODE)
    out = {}
    for key, _, _ in conditions:
        out[key] = df.loc[cond_of == key, 'event_t'].to_numpy()
    return out


# ============================================================================
# ROI EXTRACTION FROM CACHED PARCEL DATA
# ============================================================================

def _avg_labels(parcel_dict, label_list):
    ts = [parcel_dict[l] for l in label_list if l in parcel_dict]
    return np.column_stack(ts).mean(axis=1)


def _avg_keywords(parcel_dict, keywords):
    ts = [v for l, v in parcel_dict.items()
          if l != 'Background' and any(kw in l.lower() for kw in keywords)]
    return np.column_stack(ts).mean(axis=1)


def extract_roi_timeseries(subject, session, task='svf'):
    schaefer = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
    ho_sub = get_parcel_data(subject, session, task, atlas='HarvardOxford_sub')
    return {
        'pmc':   _avg_labels(schaefer, PMC_LABELS),
        'ag':    _avg_labels(schaefer, AG_LABELS),
        'dlpfc': _avg_labels(schaefer, DLPFC_LABELS),
        'vlpfc': _avg_labels(schaefer, VLPFC_LABELS),
        'dacc':  _avg_labels(schaefer, DACC_LABELS),
        'eac':   _avg_labels(schaefer, EAC_LABELS),
        'evc':   _avg_labels(schaefer, EVC_LABELS),
        'hipp':  _avg_keywords(ho_sub, HIPP_KEYWORDS),
    }


# ============================================================================
# EPOCH EXTRACTION
# ============================================================================

def extract_event_locked_epochs(signal, event_times_sec):
    """Return (n_valid_events, n_timepoints) array, or None."""
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
# PLOTTING
# ============================================================================

def _zero_formatter():
    """Tick labels: render 0 as '0' (never '0.0'); others normally."""
    def fmt(x, _pos):
        if abs(x) < 1e-9:
            return '0'
        s = f'{x:.2f}'.rstrip('0').rstrip('.')
        return s
    return FuncFormatter(fmt)


def _finalize_axes(axes):
    """Share y-limits across all ROI panels and apply zero tick formatting."""
    ylims = [ax.get_ylim() for ax in axes]
    ymin = min(y[0] for y in ylims)
    ymax = max(y[1] for y in ylims)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_formatter(_zero_formatter())
    return ymin, ymax


def _make_axes(n_roi):
    ncol = 2
    nrow = int(np.ceil(n_roi / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.0 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_roi:]:
        ax.set_visible(False)
    return fig, axes[:n_roi]


def plot_subject_timecourse(subject, data, conditions, align, mode):
    fig, axes = _make_axes(len(ROI_SPEC))
    tv = data['time_vec']
    xlabel = ('Time from preceding-word offset (s)' if align == 'offset'
              else 'Time from word onset (s)')

    fig.suptitle(f"SVF peri-boundary ROI responses by consensus level: {subject} "
                 f"(N={data['n_sessions']} sessions)\nLocked to {xlabel.lower()}",
                 fontsize=TITLE_FS, fontweight='bold')

    for ax, (roi, title) in zip(axes, ROI_SPEC):
        for key, label, color in conditions:
            ax.plot(tv, data[f'{roi}::{key}_tc'], color=color, lw=2.2, label=label)
        ax.axvline(0, color='grey', ls='--', alpha=0.6)
        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.set(xlabel=xlabel, ylabel='BOLD (z)', title=title, xlim=(tv[0], tv[-1]))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(loc='upper right', fontsize=8, ncol=1)
    _finalize_axes(axes)

    plt.tight_layout()
    out = OUTPUT_DIR / f"{subject}_svf_boundary_consensus_by-{mode}_align-{align}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def plot_group_timecourse(subject_data, conditions, align, mode):
    tv = next(iter(subject_data.values()))['time_vec']
    n_tp = len(tv)
    subs = list(subject_data.values())
    xlabel = ('Time from preceding-word offset (s)' if align == 'offset'
              else 'Time from word onset (s)')

    fig, axes = _make_axes(len(ROI_SPEC))
    note = ('  |  * switching vs clustering, p<0.05 uncorr.' if mode == 'class' else '')
    fig.suptitle(f"SVF peri-boundary ROI responses by consensus level: "
                 f"Group (N={len(subs)} subjects)\nLocked to {xlabel.lower()}{note}",
                 fontsize=TITLE_FS, fontweight='bold')

    # per-ROI stacks: {roi: {cond: (n_sub, n_tp)}}
    for ax, (roi, title) in zip(axes, ROI_SPEC):
        cond_stacks = {}
        for key, label, color in conditions:
            arr = np.array([d[f'{roi}::{key}_tc'] for d in subs])  # (n_sub, n_tp)
            cond_stacks[key] = arr
            with np.errstate(invalid='ignore'):
                m = np.nanmean(arr, axis=0)
                nvalid = np.sum(~np.isnan(arr), axis=0)
                se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(nvalid, 1))
            ax.plot(tv, m, color=color, lw=2.5, label=label, marker='o', ms=3)
            ax.fill_between(tv, m - se, m + se, color=color, alpha=0.25)
        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.set(xlabel=xlabel, ylabel='BOLD (z)', title=title,
               xlim=(tv[0] - 0.5, tv[-1] + 0.5))
        ax.spines[['top', 'right']].set_visible(False)
        ax._cond_stacks = cond_stacks
    axes[0].legend(loc='upper right', fontsize=8, ncol=1)

    ymin, ymax = _finalize_axes(axes)

    # significance stars (class mode only): switching vs clustering, paired across subjects
    if mode == 'class':
        for ax in axes:
            sw = ax._cond_stacks['switching']
            cl = ax._cond_stacks['clustering']
            pvals = np.array([stats.ttest_rel(sw[:, t], cl[:, t], nan_policy='omit')[1]
                              for t in range(n_tp)])
            sig = np.where(pvals < 0.05)[0]
            if len(sig):
                yp = ymin + 0.05 * (ymax - ymin)
                for i in sig:
                    ax.text(tv[i], yp, '*', fontsize=13, ha='center', fontweight='bold')

    plt.tight_layout()
    out = OUTPUT_DIR / f"GROUP_svf_boundary_consensus_by-{mode}_align-{align}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# MAIN
# ============================================================================

ARGS_MODE = 'class'  # set in main(); read by assign_condition/get_consensus_events


def main():
    global ARGS_MODE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--by', choices=['class', 'votes'], default='class',
                    help="Consensus grouping: 3-way class (default) or per k_switch_votes.")
    ap.add_argument('--align', choices=['onset', 'offset'], default='onset',
                    help="Lock to word onset (default) or preceding-word offset.")
    args = ap.parse_args()
    ARGS_MODE = args.by

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args.by)

    print("\n" + "=" * 64)
    print("SVF PERI-BOUNDARY ROI TIME COURSES BY CONSENSUS LEVEL")
    print(f"  grouping = {args.by}   align = {args.align}")
    print("=" * 64)

    if not WORD_SCORES.exists():
        raise FileNotFoundError(f"Consensus word scores not found: {WORD_SCORES}")
    words = pd.read_csv(WORD_SCORES)
    print(f"Loaded {len(words)} rated words from {WORD_SCORES.name}")

    sessions = find_svf_sessions(words)
    print(f"\n{len(sessions)} SVF sessions with BOLD:")
    for sub, ses in sessions:
        print(f"  {sub} {ses}")

    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR
    n_tp = len(time_vec)

    all_results = []
    for subject, session in sessions:
        print(f"\n--- {subject} {session} ---")
        try:
            words_ses = words[(words['subject'] == subject) & (words['session'] == session)]
            events = get_consensus_events(words_ses, conditions, args.align)
            counts = {k: len(v) for k, v in events.items()}
            print("  events: " + ", ".join(f"{k}={n}" for k, n in counts.items()))

            roi_ts = extract_roi_timeseries(subject, session)

            result = {'subject': subject, 'session': session, 'time_vec': time_vec}
            for roi_key, _ in ROI_SPEC:
                for key, _, _ in conditions:
                    epochs = extract_event_locked_epochs(roi_ts[roi_key], events[key])
                    result[f'{roi_key}::{key}_tc'] = (
                        epochs.mean(axis=0) if epochs is not None else np.full(n_tp, np.nan))
            for key, _, _ in conditions:
                result[f'{key}_n'] = counts[key]
            all_results.append(result)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    print(f"\nProcessed {len(all_results)} sessions")
    for key, label, _ in conditions:
        tot = sum(r[f'{key}_n'] for r in all_results)
        print(f"  total {label}: {tot} events")

    # --- Per-subject aggregation (nanmean across sessions per ROI x condition) ---
    by_sub = defaultdict(list)
    for r in all_results:
        by_sub[r['subject']].append(r)

    sub_agg = {}
    for sub, sess in sorted(by_sub.items()):
        d = {'time_vec': time_vec, 'n_sessions': len(sess)}
        for roi_key, _ in ROI_SPEC:
            for key, _, _ in conditions:
                stack = np.array([s[f'{roi_key}::{key}_tc'] for s in sess])
                with np.errstate(invalid='ignore'):
                    d[f'{roi_key}::{key}_tc'] = np.nanmean(stack, axis=0)
        for key, _, _ in conditions:
            d[f'{key}_n'] = sum(s[f'{key}_n'] for s in sess)
        sub_agg[sub] = d

    print("\n--- Subject-level plots ---")
    for sub, data in sub_agg.items():
        plot_subject_timecourse(sub, data, conditions, args.align, args.by)

    if len(sub_agg) >= 2:
        print("\n--- Group-level plot ---")
        plot_group_timecourse(sub_agg, conditions, args.align, args.by)

    print("\n" + "=" * 64)
    print(f"DONE. Figures in {OUTPUT_DIR}")
    print("=" * 64)


if __name__ == '__main__':
    main()
