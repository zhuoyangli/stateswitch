"""
Filmfest SEG-B Boundary Neural Analysis

ROI event-locked time courses (PMC, Hippocampus, Angular Gyrus, Auditory Cortex)
at SEG-B segment boundaries during movie watching, using cached parcel time series.

The first segment of each movie (title) is excluded from analysis.

Usage:
    python filmfest_boundary.py
"""
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR, FILMFEST_SUBJECTS, MOVIE_INFO
from configs.schaefer_rois import ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL, POSTERIOR_MEDIAL
from fmrianalysis.utils import get_parcel_data, mss_to_seconds

ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
OUTPUT_DIR = FIGS_DIR / 'filmfest_boundary'
CACHE_DIR = ANALYSIS_CACHE_DIR / 'filmfest_boundary'

# === PARAMETERS ===
HRF_DELAY = 4.5
WINDOW_DURATION = 6.0
MIN_SEGMENT_DURATION = 10.0  # seconds, for non-boundary events
TRS_BEFORE = 2
TRS_AFTER = 15

# === STYLE ===
COLORS = {'boundary': '#e74c3c', 'nonboundary': 'gray'}
LABEL_FS = 12
TITLE_FS = 14
SUBJECT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ROI keys and display names
ROI_SPEC = [
    ('pmc',  'Posterior Medial Cortex'),
    ('hipp', 'Hippocampus'),
    ('ag',   'Angular Gyrus'),
    ('eac',  'Auditory Cortex'),
    ('evc',  'Early Visual Cortex'),
]

# === ROI label matching ===
PMC_LABELS = POSTERIOR_MEDIAL.get('left_labels', []) + POSTERIOR_MEDIAL.get('right_labels', [])
AG_LABELS = ANGULAR_GYRUS.get('left_labels', []) + ANGULAR_GYRUS.get('right_labels', [])
EAC_LABELS = EARLY_AUDITORY.get('left_labels', []) + EARLY_AUDITORY.get('right_labels', [])
EVC_LABELS = EARLY_VISUAL.get('left_labels', []) + EARLY_VISUAL.get('right_labels', [])
HIPP_KEYWORDS = ['hippocampus']


# ============================================================================
# HELPERS
# ============================================================================

def get_segb_events(task):
    """Get SEG-B boundary and non-boundary timepoints for a run (in seconds).

    The first segment of each movie (title) is excluded:
      - Boundaries start from the 3rd segment onset (skipping the title->seg2 transition)
      - Non-boundary midpoints exclude the title segment
    """
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    boundary_tp, nonboundary_tp = [], []

    for movie in movies:
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number']).copy()
        starts = segb['Start Time (m.ss)'].apply(mss_to_seconds).values
        ends = segb['End Time (m.ss)'].apply(mss_to_seconds).values

        # Boundaries: onset of each SEG-B from the 3rd segment onward
        # (skip index 0 = title, skip index 1 = title->seg2 boundary)
        for i in range(2, len(starts)):
            boundary_tp.append(starts[i])

        # Non-boundary: midpoint of long SEG-B segments, excluding title (index 0)
        for i in range(1, len(starts)):
            dur = ends[i] - starts[i]
            if dur >= MIN_SEGMENT_DURATION:
                nonboundary_tp.append(starts[i] + dur / 2)

    # Filter: need enough pre-event TRs
    min_t = TRS_BEFORE * TR
    boundary_tp = np.array(sorted([t for t in boundary_tp if t >= min_t]))
    nonboundary_tp = np.array(sorted([t for t in nonboundary_tp if t >= min_t]))
    return boundary_tp, nonboundary_tp


def get_movie_offsets(task):
    """Get movie offset times (end of each movie except the last) for a task."""
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    offsets = []
    for movie in movies[:-1]:  # skip last movie (no transition after it)
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        last_end = segb['End Time (m.ss)'].values[-1]
        offsets.append(mss_to_seconds(last_end))
    return offsets


def get_segb_offsets(task):
    """Get SEG-B segment offset (end) times, excluding the title (first) segment of each movie."""
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    offsets = []
    for movie in movies:
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        ends = segb['End Time (m.ss)'].apply(mss_to_seconds).values
        # Skip first segment (title), keep the rest
        for end_time in ends[1:]:
            offsets.append(end_time)
    return sorted(offsets)


def get_segb_offsets_preceding(task, min_dur=0.0, max_dur=np.inf,
                               prev_min_dur=0.0, prev_max_dur=np.inf):
    """Get offset (end) of the segment BEFORE a segment with duration in [min_dur, max_dur).

    Optionally filter the preceding segment's duration with prev_min_dur/prev_max_dur.
    i.e., the onset/transition INTO segments of a given duration range.
    Excludes title (index 0). The preceding segment must be index >= 1 (not title).

    Returns (offsets, mean_duration) where mean_duration is the mean of the
    matching (current) segments' durations.
    """
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    offsets = []
    durations = []
    for movie in movies:
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number']).copy()
        starts = segb['Start Time (m.ss)'].apply(mss_to_seconds).values
        ends = segb['End Time (m.ss)'].apply(mss_to_seconds).values
        for i in range(2, len(starts)):  # i >= 2 so previous (i-1 >= 1) is non-title
            dur = ends[i] - starts[i]
            prev_dur = ends[i - 1] - starts[i - 1]
            if (min_dur <= dur < max_dur) and (prev_min_dur <= prev_dur < prev_max_dur):
                offsets.append(ends[i - 1])
                durations.append(dur)
    mean_dur = np.mean(durations) if durations else 0.0
    return sorted(offsets), mean_dur


def get_long_segb_offsets(task, min_duration=15.0):
    """Get offset (end) of the segment before a segment >= min_duration."""
    offsets, _ = get_segb_offsets_preceding(task, min_dur=min_duration)
    return offsets


def extract_event_locked_epochs(signal, event_times_sec, trs_before, trs_after):
    """Extract event-locked epochs from signal.

    Returns array (n_valid_events, n_timepoints) or None.
    """
    n = len(signal)
    if len(event_times_sec) == 0:
        return None
    centers = np.round(np.array(event_times_sec) / TR).astype(int)
    offsets = np.arange(-trs_before, trs_after + 1)
    idx = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n), axis=1)
    if not valid.any():
        return None
    epochs = signal[idx[valid]]
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)
    return epochs


def extract_event_locked_timecourse(signal, event_tp):
    """Average event-locked time course from 1D signal."""
    n = len(signal)
    if len(event_tp) == 0:
        return np.array([]), np.array([])
    centers = np.round(np.array(event_tp) / TR).astype(int)
    offsets = np.arange(-TRS_BEFORE, TRS_AFTER + 1)
    idx = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n), axis=1)
    if not valid.any():
        return np.array([]), np.array([])
    epochs = signal[idx[valid]]
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)
    return epochs.mean(axis=0), offsets * TR


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


def extract_roi_timeseries(subject, session, task):
    """Load cached parcel data and return dict of ROI time series."""
    schaefer = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
    ho_sub = get_parcel_data(subject, session, task, atlas='HarvardOxford_sub')
    return {
        'pmc':  _avg_labels(schaefer, PMC_LABELS),
        'ag':   _avg_labels(schaefer, AG_LABELS),
        'eac':  _avg_labels(schaefer, EAC_LABELS),
        'evc':  _avg_labels(schaefer, EVC_LABELS),
        'hipp': _avg_keywords(ho_sub, HIPP_KEYWORDS),
    }


# ============================================================================
# RUN PROCESSING (with caching)
# ============================================================================

def process_run(subject, session, task):
    """Process one filmfest run. Returns result dict. Caches to disk."""
    cache_file = CACHE_DIR / f"{subject}_{session}_{task}_segb.npz"
    if cache_file.exists():
        print(f"  {subject} {session} {task}: loading cache")
        loaded = np.load(cache_file, allow_pickle=True)
        result = {k: loaded[k] for k in loaded.files}
        for k in ('subject', 'session', 'task'):
            if k in result:
                result[k] = str(result[k])
        for k in ('n_boundary', 'n_nonboundary'):
            if k in result:
                result[k] = int(result[k])
        return result

    print(f"  {subject} {session} {task}: processing ...")

    boundary_tp, nonboundary_tp = get_segb_events(task)

    # ROI time series from cached parcel data
    roi_ts = extract_roi_timeseries(subject, session, task)

    # Event-locked time courses
    result = {
        'subject': subject, 'session': session, 'task': task,
        'time_vec': None,
        'n_boundary': len(boundary_tp),
        'n_nonboundary': len(nonboundary_tp),
    }
    for roi_key, _ in ROI_SPEC:
        bnd_tc, time_vec = extract_event_locked_timecourse(roi_ts[roi_key], boundary_tp)
        nbnd_tc, _ = extract_event_locked_timecourse(roi_ts[roi_key], nonboundary_tp)
        result[f'{roi_key}_boundary_tc'] = bnd_tc
        result[f'{roi_key}_nonboundary_tc'] = nbnd_tc
    result['time_vec'] = time_vec

    np.savez_compressed(cache_file, **result)
    print(f"    Cached to {cache_file.name}")
    return result


# ============================================================================
# PLOTTING -- Subject level
# ============================================================================

def plot_subject_timecourse(subject, data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    tv = data['time_vec']
    n_runs = data['n_runs']

    fig.suptitle(f"Filmfest SEG-B Boundary: {subject} (N={n_runs} runs)",
                 fontsize=TITLE_FS, fontweight='bold')

    for ax, (roi, title) in zip(axes.flat, ROI_SPEC):
        for cond, label in [('boundary', 'Boundary'), ('nonboundary', 'Non-boundary')]:
            c = COLORS[cond]
            ax.plot(tv, data[f'{roi}_{cond}_tc'], color=c, lw=3, label=label)
            ax.fill_between(tv,
                            data[f'{roi}_{cond}_tc'] - data[f'{roi}_{cond}_sem'],
                            data[f'{roi}_{cond}_tc'] + data[f'{roi}_{cond}_sem'],
                            color=c, alpha=0.3)
        ax.axvline(0, color='grey', ls='--', alpha=0.5)
        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.15, color='yellow')
        ax.set(xlabel='Time (s)', ylabel='BOLD (z-scored)', title=title,
               xlim=(tv[0], tv[-1]))
        ax.legend(loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)

    # Unify y-axis limits
    all_ylims = [ax.get_ylim() for ax in axes.flat]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax in axes.flat:
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    out = OUTPUT_DIR / f"{subject}_filmfest_segb_timecourse.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# PLOTTING -- Group level
# ============================================================================

def plot_group_timecourse(all_results):
    n = len(all_results)
    tv = all_results[0]['time_vec']

    stacks = {}
    for cond in ('boundary', 'nonboundary'):
        for roi, _ in ROI_SPEC:
            stacks[f'{roi}_{cond}'] = np.array(
                [r[f'{roi}_{cond}_tc'] for r in all_results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Filmfest SEG-B Boundary: Group (N={n} runs)\n* p < 0.05 uncorrected",
                 fontsize=TITLE_FS, fontweight='bold')

    for ax, (roi, title) in zip(axes.flat, ROI_SPEC):
        bnd = stacks[f'{roi}_boundary']
        nbnd = stacks[f'{roi}_nonboundary']
        for arr, label, c in [(bnd, 'Boundary', COLORS['boundary']),
                                (nbnd, 'Non-boundary', COLORS['nonboundary'])]:
            m = arr.mean(0)
            se = arr.std(0) / np.sqrt(n)
            ax.plot(tv, m, color=c, lw=3, label=label, marker='o', ms=4)
            ax.fill_between(tv, m - se, m + se, color=c, alpha=0.3)

        # Significance markers
        pvals = np.array([stats.ttest_rel(bnd[:, t], nbnd[:, t])[1]
                          for t in range(len(tv))])
        sig = np.where(pvals < 0.05)[0]

        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.set(xlabel='Time (s)', ylabel='BOLD (z-scored)', title=title,
               xlim=(tv[0] - 0.5, tv[-1] + 0.5))
        ax.legend(loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)

    # Unify y-axis limits, then add significance markers
    all_ylims = [ax.get_ylim() for ax in axes.flat]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax, (roi, _) in zip(axes.flat, ROI_SPEC):
        ax.set_ylim(ymin, ymax)
        # Re-compute significance for marker placement
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
    out = OUTPUT_DIR / "GROUP_filmfest_segb_timecourse.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_by_subject(results):
    by_sub = defaultdict(list)
    for r in results:
        by_sub[r['subject']].append(r)

    agg = {}
    for sub, runs in by_sub.items():
        n = len(runs)
        d = {'time_vec': runs[0]['time_vec'], 'n_runs': n}
        for roi, _ in ROI_SPEC:
            for cond in ('boundary', 'nonboundary'):
                k = f'{roi}_{cond}_tc'
                stack = np.array([r[k] for r in runs])
                d[f'{roi}_{cond}_tc'] = stack.mean(0)
                d[f'{roi}_{cond}_sem'] = (
                    stack.std(0) / np.sqrt(n) if n > 1 else np.zeros_like(stack[0]))
        agg[sub] = d
    return agg


# ============================================================================
# PLOTTING -- Full session time series with event markers
# ============================================================================

def plot_session_timeseries(subject, session):
    """Plot full ROI time series for both tasks, marking movie and SEG-B offsets."""
    fig, axes = plt.subplots(4, 2, figsize=(24, 12), sharex='col')
    fig.suptitle(f"ROI Time Series with Event Markers: {subject} ({session})",
                 fontsize=TITLE_FS, fontweight='bold')

    for col, task in enumerate(('filmfest1', 'filmfest2')):
        roi_ts = extract_roi_timeseries(subject, session, task)
        n_trs = len(roi_ts['pmc'])
        time_sec = np.arange(n_trs) * TR

        movie_offs = get_movie_offsets(task)
        segb_offs = get_segb_offsets(task)

        for row, (roi_key, roi_title) in enumerate(ROI_SPEC):
            ax = axes[row, col]
            ax.plot(time_sec, roi_ts[roi_key], color='k', lw=0.6)

            # SEG-B offsets (thin gray)
            for t in segb_offs:
                ax.axvline(t, color='#7f8c8d', ls=':', lw=0.5, alpha=0.5)

            # Movie offsets (red, more prominent)
            for t in movie_offs:
                ax.axvline(t, color='#e74c3c', ls='--', lw=1.5, alpha=0.8)

            if col == 0:
                ax.set_ylabel(f'{roi_title}\nBOLD (z)', fontsize=10)
            if row == 0:
                ax.set_title(task, fontsize=TITLE_FS)
            if row == len(ROI_SPEC) - 1:
                ax.set_xlabel('Time (s)', fontsize=LABEL_FS)
            ax.spines[['top', 'right']].set_visible(False)

    # Unify y-axis limits across all subplots
    all_ylims = [ax.get_ylim() for ax in axes.flat]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax in axes.flat:
        ax.set_ylim(ymin, ymax)

    # Legend in first subplot
    legend_elements = [
        Line2D([0], [0], color='#e74c3c', ls='--', lw=1.5, label='Movie offset'),
        Line2D([0], [0], color='#7f8c8d', ls=':', lw=0.8, label='SEG-B offset'),
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / f"{subject}_session_timeseries.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# PLOTTING -- Offset-locked time courses for long segments
# ============================================================================

OFFSET_TRS_BEFORE = 10   # 15 s
OFFSET_TRS_AFTER = 20    # 30 s

def plot_offset_locked(min_duration=15.0):
    """Plot offset-locked time courses for long SEG-B segments.

    One figure per subject: 4 ROI rows × 2 task columns.
    Mean across events with SEM across events as shading.
    """
    time_vec = np.arange(-OFFSET_TRS_BEFORE, OFFSET_TRS_AFTER + 1) * TR
    roi_keys = [k for k, _ in ROI_SPEC]
    tasks = ('filmfest1', 'filmfest2')

    # Pre-compute event times per task
    task_events = {}
    for task in tasks:
        task_events[task] = get_long_segb_offsets(task, min_duration=min_duration)
        print(f"  {task}: {len(task_events[task])} segment offsets (dur >= {min_duration}s)")

    # Collect all y-values across subjects for unified limits
    global_ylims = []

    for subject, session in FILMFEST_SUBJECTS.items():
        fig, axes = plt.subplots(4, 2, figsize=(16, 14))
        fig.suptitle(
            f"SEG-B Offset-Locked Response: {subject} ({session})\n"
            f"Segments >= {min_duration:.0f}s, shading = SEM across events",
            fontsize=TITLE_FS, fontweight='bold')

        for col, task in enumerate(tasks):
            event_times = task_events[task]
            roi_ts = extract_roi_timeseries(subject, session, task)

            for row, (roi_key, roi_title) in enumerate(ROI_SPEC):
                ax = axes[row, col]
                epochs = extract_event_locked_epochs(
                    roi_ts[roi_key], event_times,
                    OFFSET_TRS_BEFORE, OFFSET_TRS_AFTER)

                if epochs is not None:
                    mean = epochs.mean(axis=0)
                    sem = epochs.std(axis=0) / np.sqrt(epochs.shape[0])
                    ax.plot(time_vec, mean, color='k', lw=2)
                    ax.fill_between(time_vec, mean - sem, mean + sem,
                                    color='k', alpha=0.2)

                ax.axvline(0, color='grey', ls='--', lw=1)
                ax.axhline(0, color='k', ls='-', alpha=0.3)
                if row == 0:
                    ax.set_title(f"{task}  (N={len(event_times)} events)",
                                 fontsize=TITLE_FS)
                if col == 0:
                    ax.set_ylabel(f'{roi_title}\nBOLD (z)', fontsize=10)
                if row == len(ROI_SPEC) - 1:
                    ax.set_xlabel('Time from segment onset (s)', fontsize=LABEL_FS)
                ax.spines[['top', 'right']].set_visible(False)

        # Unify y-axis across all subplots within this subject
        all_ylims = [ax.get_ylim() for ax in axes.flat]
        ymin = min(y[0] for y in all_ylims)
        ymax = max(y[1] for y in all_ylims)
        for ax in axes.flat:
            ax.set_ylim(ymin, ymax)
        global_ylims.append((ymin, ymax))

        plt.tight_layout()
        out = OUTPUT_DIR / f"{subject}_offset_locked_long_segments.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out}")


def plot_offset_locked_group(dur_threshold=15.0):
    """Group offset-locked plot: filmfest1 & filmfest2 side by side, LL vs SS transitions.

    Layout: 4 ROI rows × 2 task columns.
    LL = long segment preceded by long segment.
    SS = short segment preceded by short segment.
    """
    time_vec = np.arange(-OFFSET_TRS_BEFORE, OFFSET_TRS_AFTER + 1) * TR
    roi_keys = [k for k, _ in ROI_SPEC]
    tasks = ('filmfest1', 'filmfest2')
    thr = dur_threshold
    # Colors grouped by current segment type
    COLOR_LONG = '#2c3e50'   # LL, SL (current = long)
    COLOR_SHORT = '#e67e22'  # LS, SS (current = short)
    # (label, curr_min, curr_max, prev_min, prev_max, color, linestyle)
    conditions = [
        ('LL', thr, np.inf, thr, np.inf, COLOR_LONG, '-'),
        ('SL', thr, np.inf, 0.0, thr, COLOR_LONG, '--'),
        ('LS', 0.0, thr, thr, np.inf, COLOR_SHORT, '--'),
        ('SS', 0.0, thr, 0.0, thr, COLOR_SHORT, '-'),
    ]

    # Collect per-subject means: task -> cond -> roi -> list of (n_tp,)
    task_cond_data = {}
    task_cond_n = {}
    task_cond_mean_dur = {}
    for task in tasks:
        task_cond_data[task] = {}
        task_cond_n[task] = {}
        task_cond_mean_dur[task] = {}
        for cond_name, min_d, max_d, prev_min, prev_max, _, _ in conditions:
            events, mean_dur = get_segb_offsets_preceding(
                task, min_dur=min_d, max_dur=max_d,
                prev_min_dur=prev_min, prev_max_dur=prev_max)
            task_cond_n[task][cond_name] = len(events)
            task_cond_mean_dur[task][cond_name] = mean_dur
            print(f"  {task} {cond_name}: {len(events)} events, mean dur={mean_dur:.1f}s")

            roi_means = {k: [] for k in roi_keys}
            for subject, session in FILMFEST_SUBJECTS.items():
                roi_ts = extract_roi_timeseries(subject, session, task)
                for k in roi_keys:
                    epochs = extract_event_locked_epochs(
                        roi_ts[k], events, OFFSET_TRS_BEFORE, OFFSET_TRS_AFTER)
                    if epochs is not None:
                        roi_means[k].append(epochs.mean(axis=0))
                    else:
                        roi_means[k].append(np.full(len(time_vec), np.nan))
            for k in roi_keys:
                roi_means[k] = np.array(roi_means[k])
            task_cond_data[task][cond_name] = roi_means

    n_sub = len(FILMFEST_SUBJECTS)

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(
        f"SEG-B Onset-Locked Group Response by Transition Type\n"
        f"L >= {dur_threshold:.0f}s, S < {dur_threshold:.0f}s  "
        f"(N={n_sub} subjects, shading = SEM across subjects)",
        fontsize=TITLE_FS, fontweight='bold')

    for col, task in enumerate(tasks):
        for row, (roi_key, roi_title) in enumerate(ROI_SPEC):
            ax = axes[row, col]

            for cond_name, _, _, _, _, color, ls in conditions:
                data = task_cond_data[task][cond_name][roi_key]
                n_ev = task_cond_n[task][cond_name]
                mean = np.nanmean(data, axis=0)
                sem = np.nanstd(data, axis=0) / np.sqrt(n_sub)
                label = f"{cond_name} (n={n_ev})"
                ax.plot(time_vec, mean, color=color, ls=ls, lw=2.5, label=label)
                ax.fill_between(time_vec, mean - sem, mean + sem,
                                color=color, alpha=0.15)

            # Vertical lines: segment onset (t=0) and mean end for long / short
            ax.axvline(0, color='grey', ls='--', lw=1)
            long_durs = [task_cond_mean_dur[task][c] for c in ('LL', 'SL')
                         if task_cond_n[task].get(c, 0) > 0]
            short_durs = [task_cond_mean_dur[task][c] for c in ('LS', 'SS')
                          if task_cond_n[task].get(c, 0) > 0]
            if long_durs:
                ax.axvline(np.mean(long_durs), color=COLOR_LONG, ls='--', lw=1.5, alpha=0.5)
            if short_durs:
                ax.axvline(np.mean(short_durs), color=COLOR_SHORT, ls='--', lw=1.5, alpha=0.5)

            ax.axhline(0, color='k', ls='-', alpha=0.3)
            if row == 0:
                ax.set_title(task, fontsize=TITLE_FS)
            if col == 0:
                ax.set_ylabel(f'{roi_title}\nBOLD (z)', fontsize=10)
            if row == len(ROI_SPEC) - 1:
                ax.set_xlabel('Time from segment onset (s)', fontsize=LABEL_FS)
            ax.spines[['top', 'right']].set_visible(False)

    # Unify y-axis
    all_ylims = [ax.get_ylim() for ax in axes.flat]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax in axes.flat:
        ax.set_ylim(ymin, ymax)

    # Add mean duration annotations at top of first ROI row
    for col, task in enumerate(tasks):
        ax = axes[0, col]
        long_durs = [task_cond_mean_dur[task][c] for c in ('LL', 'SL')
                     if task_cond_n[task].get(c, 0) > 0]
        short_durs = [task_cond_mean_dur[task][c] for c in ('LS', 'SS')
                      if task_cond_n[task].get(c, 0) > 0]
        if long_durs:
            md = np.mean(long_durs)
            ax.text(md, ymax * 0.9, f'~{md:.0f}s', color=COLOR_LONG,
                    fontsize=9, fontweight='bold', ha='center')
        if short_durs:
            md = np.mean(short_durs)
            ax.text(md, ymax * 0.9, f'~{md:.0f}s', color=COLOR_SHORT,
                    fontsize=9, fontweight='bold', ha='center')

    axes[0, 0].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / "GROUP_offset_locked_long_vs_short.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("FILMFEST SEG-B BOUNDARY ANALYSIS")
    print("(first segment / title excluded)")
    print("=" * 60)

    print(f"\nROI parcels -- PMC: {len(PMC_LABELS)}, AG: {len(AG_LABELS)}, "
          f"EAC: {len(EAC_LABELS)}, Hipp: matched by keyword")

    # Print event summary
    for task in ('filmfest1', 'filmfest2'):
        bnd, nbnd = get_segb_events(task)
        print(f"  {task}: {len(bnd)} boundary, {len(nbnd)} non-boundary events")

    # Process all runs
    all_results = []
    for subject, session in FILMFEST_SUBJECTS.items():
        for task in ('filmfest1', 'filmfest2'):
            result = process_run(subject, session, task)
            if result is not None:
                all_results.append(result)

    print(f"\nProcessed {len(all_results)} runs")

    # --- Offset-locked time courses (per subject) ---
    print("\n--- Offset-locked time courses per subject (segments >= 15s) ---")
    plot_offset_locked(min_duration=15.0)

    # --- Group offset-locked: long vs short ---
    print("\n--- Group offset-locked: long vs short segments ---")
    plot_offset_locked_group(dur_threshold=15.0)

    # --- Full session time series ---
    print("\n--- Session time series with event markers ---")
    for subject, session in FILMFEST_SUBJECTS.items():
        plot_session_timeseries(subject, session)

    # --- Subject-level ---
    print("\n--- Subject-level plots ---")
    sub_agg = aggregate_by_subject(all_results)
    for sub, data in sorted(sub_agg.items()):
        plot_subject_timecourse(sub, data)

    # --- Group-level ---
    if len(all_results) >= 2:
        print("\n--- Group-level plots ---")
        plot_group_timecourse(all_results)

    print("\n" + "=" * 60)
    print(f"DONE. Figures in {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
