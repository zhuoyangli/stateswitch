#!/usr/bin/env python3
"""
Combined boundary figure: 4 columns × 5 ROI rows.

Columns:
  1. Filmfest movie boundaries (boundary vs non-boundary)
  2. AHC explanation boundaries (boundary vs non-boundary)
  3. SVF switch/cluster (all events, 3-way agreement)
  4. SVF switch/cluster (context: cluster-cluster-X)

Rows: PMC, Hippocampus, Angular Gyrus, Auditory Cortex, Early Visual Cortex

Usage:
    python combined_boundary_figure.py
"""
import subprocess
import tempfile
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
from nilearn.plotting import plot_surf

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR
from fmrianalysis.utils import get_parcel_data
from fmrianalysis.multitask_boundary_parcel import (
    _build_surface_parcellation, _parcel_to_surface,
)

# Import from each analysis module
from fmrianalysis.filmfest_segment_boundary import (
    FILMFEST_SUBJECTS, ROI_SPEC, TRS_BEFORE, TRS_AFTER,
    MOVIE_INFO, ANNOTATIONS_DIR as FF_ANNOTATIONS_DIR,
    extract_roi_timeseries as ff_extract_roi,
    extract_event_locked_epochs as ff_extract_epochs,
    get_movie_offsets, mss_to_seconds, get_segb_events,
)
from fmrianalysis.ahc_category_boundary import (
    find_ahc_sessions,
    get_boundary_events,
    extract_roi_timeseries as ahc_extract_roi,
    extract_event_locked_epochs as ahc_extract_epochs,
    TRS_BEFORE as AHC_TRS_BEFORE, TRS_AFTER as AHC_TRS_AFTER,
)
from fmrianalysis.svf_switch_boundary import (
    find_rated_sessions,
    load_multirater_events,
    extract_roi_timeseries as svf_extract_roi,
    extract_event_locked_epochs as svf_extract_epochs,
    TRS_BEFORE as SVF_TRS_BEFORE, TRS_AFTER as SVF_TRS_AFTER,
    AGREEMENT_CATS, CONTEXT_CATS,
    _add_context_filter,
)

OUTPUT_DIR = FIGS_DIR / 'combined'
TITLE_FS = 13
ROI_KEYS = [k for k, _ in ROI_SPEC]

# === AUDIO CONSTANTS ===
_FILMFEST_MP4 = {
    'filmfest1': FF_ANNOTATIONS_DIR / 'FilmFest_part1.mp4',
    'filmfest2': FF_ANNOTATIONS_DIR / 'FilmFest_part2.mp4',
}
_AUDIO_BIDS_DIR = Path('/home/datasets/stateswitch/rec/bids')
_AUDIO_SCAN_OFFSET = 12   # WAV recording starts 12 s before fMRI scan
_AUDIO_SR = 1             # Hz – downsample envelope to 1 Hz
_AUDIO_PRE  = int(TRS_BEFORE * TR)   # 3 s
_AUDIO_POST = int(TRS_AFTER  * TR)   # 22 s
_AUDIO_N_TP = _AUDIO_PRE + _AUDIO_POST + 1   # 26 samples
_AUDIO_TV   = np.linspace(-_AUDIO_PRE, _AUDIO_POST, _AUDIO_N_TP)

# === SURFACE T-MAP CONSTANTS ===
MICRO_EPOCH_DEFS = [
    ('-3 to 0 s',      0,  2),
    ('0 to 7.5 s',     2,  7),
    ('7.5 to 15 s',    7, 12),
    ('15 to 22.5 s',  12, 18),
]
LABEL_FS = 12
TMAP_TITLE_FS = 14


# ============================================================================
# AUDIO HELPERS
# ============================================================================

def _load_audio_envelope(filepath):
    """Load WAV or MP4 and return a z-scored amplitude envelope at _AUDIO_SR Hz."""
    filepath = Path(filepath)
    if filepath.suffix.lower() == '.mp4':
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', str(filepath), '-ac', '1', '-ar', '16000', '-vn', tmp_path],
                capture_output=True, check=True)
            sr, data = wavfile.read(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        sr, data = wavfile.read(str(filepath))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    envelope = np.abs(data)
    sos = butter(4, min(10.0, sr / 2 * 0.9) / (sr / 2), btype='low', output='sos')
    envelope = sosfilt(sos, envelope)
    n_out = int(len(envelope) * _AUDIO_SR / sr)
    envelope = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(envelope)), envelope)
    sd = np.std(envelope)
    if sd > 0:
        envelope = (envelope - np.mean(envelope)) / sd
    return envelope


def _extract_audio_epochs(envelope, event_times_sec):
    """Extract (_AUDIO_PRE, _AUDIO_POST) windows. Returns (epochs, _AUDIO_TV) or (None, _AUDIO_TV)."""
    n_pre, n_post = int(_AUDIO_PRE * _AUDIO_SR), int(_AUDIO_POST * _AUDIO_SR)
    n = len(envelope)
    epochs = []
    for t in event_times_sec:
        c = int(round(t * _AUDIO_SR))
        s, e = c - n_pre, c + n_post + 1
        if s >= 0 and e <= n:
            epochs.append(envelope[s:e])
    return (np.array(epochs), _AUDIO_TV) if epochs else (None, _AUDIO_TV)


def _find_audio_wav(subject, session, task):
    p = _AUDIO_BIDS_DIR / subject / session / f"{subject}_{session}_task-{task}_desc-audio.wav"
    return p if p.exists() else None


def _collect_ff_audio(get_events_fn):
    """Load filmfest MP4 audio and extract event-locked epochs.

    get_events_fn(task) -> (boundary_tp, nonboundary_tp) arrays in seconds.
    Returns stacks = {'boundary': (n_epochs, n_tp), 'nonboundary': (n_epochs, n_tp)}.
    """
    all_epochs = {'boundary': [], 'nonboundary': []}
    for task in ('filmfest1', 'filmfest2'):
        mp4 = _FILMFEST_MP4.get(task)
        if mp4 is None or not mp4.exists():
            continue
        try:
            env = _load_audio_envelope(mp4)
            bnd, nbnd = get_events_fn(task)
            for cond, tp in [('boundary', np.asarray(bnd)), ('nonboundary', np.asarray(nbnd))]:
                valid = tp[tp >= _AUDIO_PRE]
                epochs, _ = _extract_audio_epochs(env, valid)
                if epochs is not None:
                    all_epochs[cond].extend(list(epochs))
        except Exception as exc:
            print(f"  filmfest {task} audio: {exc}")
    return {c: np.array(v) if v else np.zeros((0, _AUDIO_N_TP))
            for c, v in all_epochs.items()}


# ============================================================================
# AUDIO COLLECTION
# ============================================================================

def collect_ff_between_audio():
    """Audio epochs locked to filmfest between-movie boundaries / movie midpoints."""
    def _ev(task):
        bnd  = np.array([t for t in get_movie_offsets(task)  if t >= _AUDIO_PRE])
        nbnd = np.array([t for t in get_movie_midpoints(task) if t >= _AUDIO_PRE])
        return bnd, nbnd
    return _collect_ff_audio(_ev)


def collect_ff_segb_audio():
    """Audio epochs locked to filmfest within-movie SEG-B boundaries / segment midpoints."""
    def _ev(task):
        bnd, nbnd = get_segb_events(task)
        return bnd[bnd >= _AUDIO_PRE], nbnd[nbnd >= _AUDIO_PRE]
    return _collect_ff_audio(_ev)


def collect_ahc_audio():
    """Audio epochs locked to AHC explanation boundaries per subject."""
    sub_data = defaultdict(lambda: defaultdict(list))
    for subject, session in find_ahc_sessions():
        try:
            bnd_tp, nbnd_tp = get_boundary_events(subject, session)
        except Exception:
            continue
        wav = _find_audio_wav(subject, session, 'ahc')
        if wav is None:
            continue
        try:
            env = _load_audio_envelope(wav)
            for cond, tp in [('boundary', bnd_tp), ('nonboundary', nbnd_tp)]:
                epochs, _ = _extract_audio_epochs(env, tp + _AUDIO_SCAN_OFFSET)
                if epochs is not None:
                    sub_data[subject][cond].append(epochs.mean(0))
        except Exception as exc:
            print(f"  AHC audio {subject} {session}: {exc}")
    subjects = sorted(sub_data.keys())
    stacks = {}
    for cond in ('boundary', 'nonboundary'):
        means = [np.nanmean(sub_data[s][cond], axis=0) for s in subjects if sub_data[s][cond]]
        stacks[cond] = np.array(means) if means else np.zeros((0, _AUDIO_N_TP))
    return stacks


def collect_svf_audio(cat_defs, apply_context=False):
    """Audio epochs locked to SVF word events per category per subject."""
    sub_data = defaultdict(lambda: defaultdict(list))
    for subject, session in find_rated_sessions():
        try:
            events_df = load_multirater_events(subject, session)
        except Exception:
            continue
        if apply_context:
            events_df = _add_context_filter(events_df)
        wav = _find_audio_wav(subject, session, 'svf')
        if wav is None:
            continue
        try:
            env = _load_audio_envelope(wav)
            for cat_label, cond_fn, _, _ in cat_defs:
                tp = events_df.loc[cond_fn(events_df), 'onset'].values + _AUDIO_SCAN_OFFSET
                epochs, _ = _extract_audio_epochs(env, tp)
                if epochs is not None:
                    sub_data[subject][cat_label].append(epochs.mean(0))
        except Exception as exc:
            print(f"  SVF audio {subject} {session}: {exc}")
    subjects = sorted(sub_data.keys())
    stacks = {}
    for cat_label, _, _, _ in cat_defs:
        means = [np.nanmean(sub_data[s][cat_label], axis=0)
                 for s in subjects if sub_data[s].get(cat_label)]
        stacks[cat_label] = np.array(means) if means else np.zeros((0, _AUDIO_N_TP))
    return stacks


# ============================================================================
# DATA COLLECTION
# ============================================================================

def get_movie_midpoints(task):
    """Get the temporal midpoint of each movie in a task (in seconds)."""
    import pandas as pd
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    midpoints = []
    for movie in movies:
        df = pd.read_excel(FF_ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        starts = segb['Start Time (m.ss)'].apply(mss_to_seconds).values
        ends = segb['End Time (m.ss)'].apply(mss_to_seconds).values
        movie_start = starts[0]
        movie_end = ends[-1]
        midpoints.append((movie_start + movie_end) / 2)
    return midpoints


def collect_filmfest():
    """Collect filmfest between-movie boundary vs movie midpoint per subject.

    Between-movie boundary = offset (end) of each movie (transition to next movie).
    Non-boundary           = temporal midpoint of each movie.
    """
    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR
    min_t = TRS_BEFORE * TR
    # subject -> cond -> roi -> list of run-level mean TCs
    sub_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for subject, session in FILMFEST_SUBJECTS.items():
        for task in ('filmfest1', 'filmfest2'):
            # Between-movie boundaries: movie offset times
            movie_offsets = get_movie_offsets(task)
            movie_boundary_tp = np.array([t for t in movie_offsets if t >= min_t])

            # Non-boundary: midpoint of each movie
            midpoints = get_movie_midpoints(task)
            movie_midpoint_tp = np.array([t for t in midpoints if t >= min_t])

            roi_ts = ff_extract_roi(subject, session, task)
            for roi_key in ROI_KEYS:
                for cond, tp in [('boundary', movie_boundary_tp),
                                 ('nonboundary', movie_midpoint_tp)]:
                    epochs = ff_extract_epochs(roi_ts[roi_key], tp, TRS_BEFORE, TRS_AFTER)
                    if epochs is not None:
                        sub_data[subject][cond][roi_key].append(epochs.mean(0))

    # Average runs within subject
    subjects = sorted(sub_data.keys())
    stacks = {}
    for cond in ('boundary', 'nonboundary'):
        stacks[cond] = {}
        for roi_key in ROI_KEYS:
            sub_means = []
            for sub in subjects:
                tcs = sub_data[sub][cond][roi_key]
                if tcs:
                    sub_means.append(np.nanmean(tcs, axis=0))
            stacks[cond][roi_key] = np.array(sub_means)

    return time_vec, stacks, len(subjects)


def collect_filmfest_segb():
    """Collect filmfest within-movie SEG-B boundary vs non-boundary per subject."""
    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR
    sub_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for subject, session in FILMFEST_SUBJECTS.items():
        for task in ('filmfest1', 'filmfest2'):
            boundary_tp, nonboundary_tp = get_segb_events(task)
            roi_ts = ff_extract_roi(subject, session, task)
            for roi_key in ROI_KEYS:
                for cond, tp in [('boundary', boundary_tp),
                                 ('nonboundary', nonboundary_tp)]:
                    epochs = ff_extract_epochs(roi_ts[roi_key], tp, TRS_BEFORE, TRS_AFTER)
                    if epochs is not None:
                        sub_data[subject][cond][roi_key].append(epochs.mean(0))

    subjects = sorted(sub_data.keys())
    stacks = {}
    for cond in ('boundary', 'nonboundary'):
        stacks[cond] = {}
        for roi_key in ROI_KEYS:
            sub_means = []
            for sub in subjects:
                tcs = sub_data[sub][cond][roi_key]
                if tcs:
                    sub_means.append(np.nanmean(tcs, axis=0))
            stacks[cond][roi_key] = np.array(sub_means)

    return time_vec, stacks, len(subjects)


def collect_ahc():
    """Collect AHC boundary vs non-boundary per subject."""
    time_vec = np.arange(-AHC_TRS_BEFORE, AHC_TRS_AFTER + 1) * TR
    sub_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    sessions = find_ahc_sessions()
    for subject, session in sessions:
        try:
            boundary_tp, nonboundary_tp = get_boundary_events(subject, session)
            roi_ts = ahc_extract_roi(subject, session)
        except Exception:
            continue
        for roi_key in ROI_KEYS:
            for cond, tp in [('boundary', boundary_tp), ('nonboundary', nonboundary_tp)]:
                epochs = ahc_extract_epochs(roi_ts[roi_key], tp)
                if epochs is not None:
                    sub_data[subject][cond][roi_key].append(epochs.mean(0))

    subjects = sorted(sub_data.keys())
    stacks = {}
    for cond in ('boundary', 'nonboundary'):
        stacks[cond] = {}
        for roi_key in ROI_KEYS:
            sub_means = []
            for sub in subjects:
                tcs = sub_data[sub][cond][roi_key]
                if tcs:
                    sub_means.append(np.nanmean(tcs, axis=0))
            stacks[cond][roi_key] = np.array(sub_means)

    return time_vec, stacks, len(subjects)


def collect_svf(cat_defs, apply_context=False):
    """Collect SVF agreement data per subject."""
    time_vec = np.arange(-SVF_TRS_BEFORE, SVF_TRS_AFTER + 1) * TR
    sub_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    sessions = find_rated_sessions()
    for subject, session in sessions:
        try:
            events_df = load_multirater_events(subject, session)
            roi_ts = svf_extract_roi(subject, session)
        except Exception:
            continue

        if apply_context:
            events_df = _add_context_filter(events_df)

        for cat_label, cond_fn, _, _ in cat_defs:
            mask = cond_fn(events_df)
            tp = events_df.loc[mask, 'onset'].values
            for roi_key in ROI_KEYS:
                epochs = svf_extract_epochs(roi_ts[roi_key], tp)
                if epochs is not None:
                    sub_data[subject][cat_label][roi_key].append(epochs.mean(0))
                else:
                    sub_data[subject][cat_label][roi_key].append(
                        np.full(len(time_vec), np.nan))

    subjects = sorted(sub_data.keys())
    stacks = {}
    for cat_label, _, _, _ in cat_defs:
        stacks[cat_label] = {}
        for roi_key in ROI_KEYS:
            sub_means = []
            for sub in subjects:
                tcs = sub_data[sub][cat_label][roi_key]
                if tcs:
                    sub_means.append(np.nanmean(tcs, axis=0))
                else:
                    sub_means.append(np.full(len(time_vec), np.nan))
            stacks[cat_label][roi_key] = np.array(sub_means)

    return time_vec, stacks, len(subjects)


# ============================================================================
# PARCEL-LEVEL COLLECTION (for surface t-maps)
# ============================================================================

def _get_schaefer_labels():
    """Return ordered Schaefer 400 parcel labels (no Background)."""
    from nilearn import datasets as ds
    atlas = ds.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    return [l for l in all_labels if l != 'Background']


def _extract_event_locked_2d(data_2d, event_times_sec):
    """Extract event-locked epochs from 2D data (n_timepoints, n_parcels).

    Uses TRS_BEFORE / TRS_AFTER from the micro-level analysis (2 / 15 TRs).
    Returns (n_events, 18, n_parcels) or None.
    """
    n_time = data_2d.shape[0]
    centers = np.round(np.array(event_times_sec) / TR).astype(int)
    offsets = np.arange(-TRS_BEFORE, TRS_AFTER + 1)   # length 18
    idx = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n_time), axis=1)
    if not valid.any():
        return None
    return data_2d[idx[valid]]


def _load_schaefer_2d(subject, session, task, schaefer_labels):
    """Load Schaefer 400 parcel data as (n_timepoints, 400) array."""
    schaefer = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
    return np.column_stack([schaefer[l] for l in schaefer_labels])


def collect_filmfest_parcels(schaefer_labels):
    """Collect filmfest between-movie boundary vs midpoint, full 400-parcel epochs.

    Returns {condition: (n_subj, 18, 400)}.
    """
    min_t = TRS_BEFORE * TR
    sub_data = defaultdict(lambda: defaultdict(list))

    for subject, session in FILMFEST_SUBJECTS.items():
        for task in ('filmfest1', 'filmfest2'):
            movie_boundary_tp = np.array([t for t in get_movie_offsets(task) if t >= min_t])
            movie_midpoint_tp = np.array([t for t in get_movie_midpoints(task) if t >= min_t])
            try:
                parcel_ts = _load_schaefer_2d(subject, session, task, schaefer_labels)
            except Exception:
                continue
            for cond, tp in [('boundary', movie_boundary_tp),
                             ('nonboundary', movie_midpoint_tp)]:
                if len(tp) == 0:
                    continue
                epochs = _extract_event_locked_2d(parcel_ts, tp)
                if epochs is not None:
                    sub_data[subject][cond].append(epochs.mean(axis=0))  # (18, 400)

    subjects = sorted(sub_data.keys())
    stacks = {}
    for cond in ('boundary', 'nonboundary'):
        means = [np.nanmean(sub_data[s][cond], axis=0) for s in subjects if sub_data[s][cond]]
        stacks[cond] = np.array(means) if means else np.zeros((0, TRS_BEFORE + TRS_AFTER + 1, 400))
    return stacks


def collect_filmfest_segb_parcels(schaefer_labels):
    """Collect filmfest within-movie SEG-B boundary vs midpoint, full 400-parcel epochs."""
    sub_data = defaultdict(lambda: defaultdict(list))

    for subject, session in FILMFEST_SUBJECTS.items():
        for task in ('filmfest1', 'filmfest2'):
            boundary_tp, nonboundary_tp = get_segb_events(task)
            try:
                parcel_ts = _load_schaefer_2d(subject, session, task, schaefer_labels)
            except Exception:
                continue
            for cond, tp in [('boundary', boundary_tp),
                             ('nonboundary', nonboundary_tp)]:
                if len(tp) == 0:
                    continue
                epochs = _extract_event_locked_2d(parcel_ts, tp)
                if epochs is not None:
                    sub_data[subject][cond].append(epochs.mean(axis=0))

    subjects = sorted(sub_data.keys())
    stacks = {}
    for cond in ('boundary', 'nonboundary'):
        means = [np.nanmean(sub_data[s][cond], axis=0) for s in subjects if sub_data[s][cond]]
        stacks[cond] = np.array(means) if means else np.zeros((0, TRS_BEFORE + TRS_AFTER + 1, 400))
    return stacks


def collect_ahc_parcels(schaefer_labels):
    """Collect AHC boundary vs non-boundary, full 400-parcel epochs."""
    sub_data = defaultdict(lambda: defaultdict(list))

    for subject, session in find_ahc_sessions():
        try:
            boundary_tp, nonboundary_tp = get_boundary_events(subject, session)
        except Exception:
            continue
        try:
            parcel_ts = _load_schaefer_2d(subject, session, 'ahc', schaefer_labels)
        except Exception:
            continue
        for cond, tp in [('boundary', boundary_tp),
                         ('nonboundary', nonboundary_tp)]:
            if len(tp) == 0:
                continue
            epochs = _extract_event_locked_2d(parcel_ts, tp)
            if epochs is not None:
                sub_data[subject][cond].append(epochs.mean(axis=0))

    subjects = sorted(sub_data.keys())
    stacks = {}
    for cond in ('boundary', 'nonboundary'):
        means = [np.nanmean(sub_data[s][cond], axis=0) for s in subjects if sub_data[s][cond]]
        stacks[cond] = np.array(means) if means else np.zeros((0, TRS_BEFORE + TRS_AFTER + 1, 400))
    return stacks


def collect_svf_parcels(cat_defs, apply_context, schaefer_labels):
    """Collect SVF per-category, full 400-parcel epochs."""
    sub_data = defaultdict(lambda: defaultdict(list))

    for subject, session in find_rated_sessions():
        try:
            events_df = load_multirater_events(subject, session)
        except Exception:
            continue
        if apply_context:
            events_df = _add_context_filter(events_df)
        try:
            parcel_ts = _load_schaefer_2d(subject, session, 'svf', schaefer_labels)
        except Exception:
            continue
        for cat_label, cond_fn, _, _ in cat_defs:
            tp = events_df.loc[cond_fn(events_df), 'onset'].values
            if len(tp) == 0:
                continue
            epochs = _extract_event_locked_2d(parcel_ts, tp)
            if epochs is not None:
                sub_data[subject][cat_label].append(epochs.mean(axis=0))

    subjects = sorted(sub_data.keys())
    stacks = {}
    for cat_label, _, _, _ in cat_defs:
        means = [np.nanmean(sub_data[s][cat_label], axis=0)
                 for s in subjects if sub_data[s].get(cat_label)]
        stacks[cat_label] = np.array(means) if means else np.zeros((0, TRS_BEFORE + TRS_AFTER + 1, 400))
    return stacks


# ============================================================================
# SURFACE T-MAP FIGURES
# ============================================================================

def _render_brain_cell(fig, subplot_spec, fsavg, surf_data_l, surf_data_r, vmax):
    """Render a 2×2 brain view (lateral/medial × L/R) inside a GridSpec cell."""
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=subplot_spec, wspace=0.01, hspace=0.01)
    views = [
        (0, 0, fsavg['infl_left'],  surf_data_l, 'left',  'lateral', fsavg['sulc_left']),
        (0, 1, fsavg['infl_right'], surf_data_r, 'right', 'lateral', fsavg['sulc_right']),
        (1, 0, fsavg['infl_left'],  surf_data_l, 'left',  'medial',  fsavg['sulc_left']),
        (1, 1, fsavg['infl_right'], surf_data_r, 'right', 'medial',  fsavg['sulc_right']),
    ]
    for row, col, mesh, sdata, hemi, view, bg in views:
        ax = fig.add_subplot(inner[row, col], projection='3d')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', (DeprecationWarning, RuntimeWarning, FutureWarning))
            plot_surf(
                mesh, surf_map=sdata, hemi=hemi, view=view,
                bg_map=bg, axes=ax, colorbar=False,
                cmap='RdBu_r', bg_on_data=True, darkness=0.5,
                vmin=-vmax, vmax=vmax)


def _build_surface_contrast_figure(
        all_parcel_stacks, task_defs, fsavg, surf_l, surf_r, n_subj,
        epoch_defs=None):
    """Build contrast surface t-map figure: rows=epochs, cols=tasks.

    all_parcel_stacks: list of dicts {condition: (n_subj, 18, 400)}
    task_defs: list of (title, cond_a, cond_b) tuples where contrast = a - b
    epoch_defs: list of (label, idx_start, idx_end) tuples; defaults to MICRO_EPOCH_DEFS
    """
    if epoch_defs is None:
        epoch_defs = MICRO_EPOCH_DEFS
    n_tasks = len(task_defs)
    n_epochs = len(epoch_defs)

    # Compute t-maps: keyed (task_idx, epoch_idx)
    t_maps = {}
    for ti, (title, cond_a, cond_b) in enumerate(task_defs):
        stacks = all_parcel_stacks[ti]
        if stacks[cond_a].shape[0] < 2 or stacks[cond_b].shape[0] < 2:
            continue
        for ei, (_, idx_s, idx_e) in enumerate(epoch_defs):
            mean_a = stacks[cond_a][:, idx_s:idx_e, :].mean(axis=1)
            mean_b = stacks[cond_b][:, idx_s:idx_e, :].mean(axis=1)
            diff = mean_a - mean_b
            t_vals, _ = stats.ttest_1samp(diff, 0, axis=0)
            t_maps[(ti, ei)] = t_vals

    if not t_maps:
        return None

    all_t = np.concatenate(list(t_maps.values()))
    vmax = min(np.nanmax(np.abs(all_t)), 8.0)

    # Layout: rows = epoch windows, cols = tasks
    fig_w = max(6 * n_tasks + 2, 16)
    fig_h = max(6 * n_epochs + 2, 10)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_epochs + 1, n_tasks + 1,
        height_ratios=[0.06] + [1] * n_epochs,
        width_ratios=[0.10] + [1] * n_tasks,
        wspace=0.05, hspace=0.10, top=0.93, bottom=0.04,
        left=0.02, right=0.92)
    fig.suptitle(
        f"Parcel-wise Surface t-maps: Contrast (N={n_subj})",
        fontsize=TMAP_TITLE_FS, fontweight='bold', y=0.96)

    # Column headers (task titles) on top
    for ti, (title, _, _) in enumerate(task_defs):
        ax_hdr = fig.add_subplot(gs[0, ti + 1])
        ax_hdr.set_axis_off()
        ax_hdr.text(0.5, 0.5, title, ha='center', va='center',
                    fontsize=LABEL_FS, fontweight='bold', transform=ax_hdr.transAxes)

    # Row labels (epoch windows) on left
    for ei, (elabel, _, _) in enumerate(epoch_defs):
        ax_label = fig.add_subplot(gs[ei + 1, 0])
        ax_label.set_axis_off()
        ax_label.text(0.5, 0.5, elabel, ha='center', va='center',
                      fontsize=LABEL_FS, fontweight='bold', rotation=90,
                      transform=ax_label.transAxes)

    for ti in range(n_tasks):
        for ei in range(n_epochs):
            if (ti, ei) not in t_maps:
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 2, subplot_spec=gs[ei + 1, ti + 1],
                    wspace=0.01, hspace=0.01)
                for r2 in range(2):
                    for c2 in range(2):
                        fig.add_subplot(inner[r2, c2]).set_axis_off()
                continue
            t_vals = t_maps[(ti, ei)]
            sd_l, sd_r = _parcel_to_surface(t_vals, surf_l, surf_r)
            _render_brain_cell(fig, gs[ei + 1, ti + 1], fsavg, sd_l, sd_r, vmax)

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('t-statistic', fontsize=LABEL_FS)

    return fig


def _build_surface_percond_figure(
        all_parcel_stacks, percond_defs, fsavg, surf_l, surf_r, n_subj,
        epoch_defs=None):
    """Build per-condition surface t-map figure: rows=epochs, cols=conditions.

    percond_defs: list of (task_title, condition_label, stack_index) tuples,
                  one per column.
    epoch_defs: list of (label, idx_start, idx_end) tuples; defaults to MICRO_EPOCH_DEFS
    """
    if epoch_defs is None:
        epoch_defs = MICRO_EPOCH_DEFS
    n_conds = len(percond_defs)
    n_epochs = len(epoch_defs)

    # Compute t-maps: keyed (cond_idx, epoch_idx)
    t_maps = {}
    for ci, (_, cond_label, stack_idx) in enumerate(percond_defs):
        stacks = all_parcel_stacks[stack_idx]
        data = stacks.get(cond_label)
        if data is None or data.shape[0] < 2:
            continue
        for ei, (_, idx_s, idx_e) in enumerate(epoch_defs):
            epoch_means = data[:, idx_s:idx_e, :].mean(axis=1)
            t_vals, _ = stats.ttest_1samp(epoch_means, 0, axis=0)
            t_maps[(ci, ei)] = t_vals

    if not t_maps:
        return None

    all_t = np.concatenate(list(t_maps.values()))
    vmax = min(np.nanmax(np.abs(all_t)), 8.0)

    fig_w = max(5 * n_conds + 2, 16)
    fig_h = max(6 * n_epochs + 2, 10)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_epochs + 1, n_conds + 1,
        height_ratios=[0.06] + [1] * n_epochs,
        width_ratios=[0.10] + [1] * n_conds,
        wspace=0.05, hspace=0.08, top=0.94, bottom=0.03,
        left=0.02, right=0.92)
    fig.suptitle(
        f"Parcel-wise Surface t-maps: Per-condition vs 0 (N={n_subj})",
        fontsize=TMAP_TITLE_FS, fontweight='bold', y=0.97)

    # Column headers (condition titles) on top
    for ci, (task_title, cond_label, _) in enumerate(percond_defs):
        ax_hdr = fig.add_subplot(gs[0, ci + 1])
        ax_hdr.set_axis_off()
        ax_hdr.text(0.5, 0.5, f"{task_title}\n{cond_label}",
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', transform=ax_hdr.transAxes)

    # Row labels (epoch windows) on left
    for ei, (elabel, _, _) in enumerate(epoch_defs):
        ax_label = fig.add_subplot(gs[ei + 1, 0])
        ax_label.set_axis_off()
        ax_label.text(0.5, 0.5, elabel, ha='center', va='center',
                      fontsize=LABEL_FS, fontweight='bold', rotation=90,
                      transform=ax_label.transAxes)

    for ci in range(n_conds):
        for ei in range(n_epochs):
            if (ci, ei) not in t_maps:
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 2, subplot_spec=gs[ei + 1, ci + 1],
                    wspace=0.01, hspace=0.01)
                for r2 in range(2):
                    for c2 in range(2):
                        fig.add_subplot(inner[r2, c2]).set_axis_off()
                continue
            t_vals = t_maps[(ci, ei)]
            sd_l, sd_r = _parcel_to_surface(t_vals, surf_l, surf_r)
            _render_brain_cell(fig, gs[ei + 1, ci + 1], fsavg, sd_l, sd_r, vmax)

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.10, 0.015, 0.75])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('t-statistic', fontsize=LABEL_FS)

    return fig


# ============================================================================
# PLOTTING
# ============================================================================

def _to_tvals(data):
    """Convert (n_subjects, n_tp) array to one-sample t-values against zero."""
    n = data.shape[0]
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1)
    std[std == 0] = np.nan
    return mean / (std / np.sqrt(n))


def _plot_two_cond(ax, time_vec, stacks, n_sub, cond_styles, use_tvals=False):
    """Plot 2 conditions (boundary/nonboundary style) on one axis."""
    for cond, label, color, ls in cond_styles:
        data = stacks[cond]
        if use_tvals:
            y = _to_tvals(data)
            ax.plot(time_vec, y, color=color, ls=ls, lw=2, label=label)
        else:
            mean = np.nanmean(data, axis=0)
            sem = np.nanstd(data, axis=0) / np.sqrt(n_sub)
            ax.plot(time_vec, mean, color=color, ls=ls, lw=2, label=label)
            ax.fill_between(time_vec, mean - sem, mean + sem, color=color, alpha=0.15)


def _plot_agreement(ax, time_vec, stacks, n_sub, cat_defs, use_tvals=False):
    """Plot 3 agreement categories on one axis."""
    for cat_label, _, color, ls in cat_defs:
        data = stacks[cat_label]
        if use_tvals:
            y = _to_tvals(data)
            ax.plot(time_vec, y, color=color, ls=ls, lw=2, label=cat_label)
        else:
            mean = np.nanmean(data, axis=0)
            sem = np.nanstd(data, axis=0) / np.sqrt(n_sub)
            ax.plot(time_vec, mean, color=color, ls=ls, lw=2, label=cat_label)
            ax.fill_between(time_vec, mean - sem, mean + sem, color=color, alpha=0.15)


def _build_figure(ff_tv, ff_stacks, ff_n, segb_tv, segb_stacks, segb_n,
                   ahc_tv, ahc_stacks, ahc_n,
                   svf_tv, svf_all_stacks, svf_ctx_stacks, svf_n,
                   ff_audio_stacks, segb_audio_stacks,
                   ahc_audio_stacks, svf_all_audio_stacks, svf_ctx_audio_stacks,
                   use_tvals=False):
    """Build the 6x5 combined figure (5 ROI rows + audio envelope row after EAC)."""
    # Audio row inserted after EAC (ROI index 3) → grid row 4; EVC shifts to row 5
    # Audio row is omitted in the t-value figure.
    show_audio = not use_tvals
    AUDIO_GRID_ROW = 4
    N_ROWS = len(ROI_SPEC) + (1 if show_audio else 0)   # 6 or 5

    fig, axes = plt.subplots(N_ROWS, 5, figsize=(25, 18 if show_audio else 15))

    col_titles = [
        f'Filmfest: between-movie\nboundaries (N={ff_n})',
        f'Filmfest: within-movie\nSEG-B boundaries (N={segb_n})',
        f'Explanation generation:\nbetween-explanation boundaries (N={ahc_n})',
        f'Semantic fluency: switching\nsubcategories, all (N={svf_n})',
        f'Semantic fluency: switching subcategories,\nafter \u22652 word clusters (N={svf_n})',
    ]

    bnd_styles_ff = [
        ('boundary', 'Between-movie boundary', '#e74c3c', '-'),
        ('nonboundary', 'Movie midpoint', 'gray', '-'),
    ]
    bnd_styles_segb = [
        ('boundary', 'SEG-B boundary', '#e74c3c', '-'),
        ('nonboundary', 'Segment midpoint', 'gray', '-'),
    ]
    bnd_styles_ahc = [
        ('boundary', 'Boundary', '#e74c3c', '-'),
        ('nonboundary', 'Non-boundary', 'gray', '-'),
    ]

    # --- ROI rows ---
    for roi_idx, (roi_key, roi_title) in enumerate(ROI_SPEC):
        grid_row = roi_idx if (not show_audio or roi_idx < AUDIO_GRID_ROW) else roi_idx + 1

        ax = axes[grid_row, 0]
        roi_stacks = {c: ff_stacks[c][roi_key] for c in ('boundary', 'nonboundary')}
        _plot_two_cond(ax, ff_tv, roi_stacks, ff_n, bnd_styles_ff, use_tvals=use_tvals)
        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)

        ax = axes[grid_row, 1]
        roi_stacks = {c: segb_stacks[c][roi_key] for c in ('boundary', 'nonboundary')}
        _plot_two_cond(ax, segb_tv, roi_stacks, segb_n, bnd_styles_segb, use_tvals=use_tvals)
        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)

        ax = axes[grid_row, 2]
        roi_stacks = {c: ahc_stacks[c][roi_key] for c in ('boundary', 'nonboundary')}
        _plot_two_cond(ax, ahc_tv, roi_stacks, ahc_n, bnd_styles_ahc, use_tvals=use_tvals)
        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)

        ax = axes[grid_row, 3]
        roi_stacks = {c: svf_all_stacks[c][roi_key] for c in [l for l, _, _, _ in AGREEMENT_CATS]}
        _plot_agreement(ax, svf_tv, roi_stacks, svf_n, AGREEMENT_CATS, use_tvals=use_tvals)
        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)

        ax = axes[grid_row, 4]
        roi_stacks = {c: svf_ctx_stacks[c][roi_key] for c in [l for l, _, _, _ in CONTEXT_CATS]}
        _plot_agreement(ax, svf_tv, roi_stacks, svf_n, CONTEXT_CATS, use_tvals=use_tvals)
        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)

    # --- Audio envelope row (mean ± SEM, omitted in t-value figure) ---
    if show_audio:
        audio_col_data = [
            (ff_audio_stacks,       'two_cond',  bnd_styles_ff),
            (segb_audio_stacks,     'two_cond',  bnd_styles_segb),
            (ahc_audio_stacks,      'two_cond',  bnd_styles_ahc),
            (svf_all_audio_stacks,  'agreement', AGREEMENT_CATS),
            (svf_ctx_audio_stacks,  'agreement', CONTEXT_CATS),
        ]
        for col, (stacks, plot_type, styles) in enumerate(audio_col_data):
            ax = axes[AUDIO_GRID_ROW, col]
            if plot_type == 'two_cond':
                n = max(stacks['boundary'].shape[0], stacks['nonboundary'].shape[0], 1)
                _plot_two_cond(ax, _AUDIO_TV, stacks, n, styles, use_tvals=False)
            else:
                labels = [l for l, _, _, _ in styles]
                n = max((stacks[l].shape[0] for l in labels if stacks[l].size > 0), default=1)
                _plot_agreement(ax, _AUDIO_TV, stacks, n, styles, use_tvals=False)
            ax.axvline(0, color='grey', ls='--', lw=1)
            ax.axhline(0, color='k', ls='-', alpha=0.3)

    # --- Labels and formatting ---
    ylabel = 't-value (vs. 0)' if use_tvals else 'BOLD (z-scored)'
    for roi_idx, (_, roi_title) in enumerate(ROI_SPEC):
        grid_row = roi_idx if (not show_audio or roi_idx < AUDIO_GRID_ROW) else roi_idx + 1
        axes[grid_row, 0].set_ylabel(f'{roi_title}\n{ylabel}', fontsize=10)
    if show_audio:
        axes[AUDIO_GRID_ROW, 0].set_ylabel('Audio Envelope\n(z-scored)', fontsize=10)

    col_xlabels = [
        'Time from movie offset (s)',
        'Time from segment onset (s)',
        'Time from preceding explanation offset (s)',
        'Time from preceding word offset (s)',
        'Time from preceding word offset (s)',
    ]
    for col in range(5):
        axes[0, col].set_title(col_titles[col], fontsize=TITLE_FS, fontweight='bold')
        axes[-1, col].set_xlabel(col_xlabels[col], fontsize=9)

    xlim = (ff_tv[0] - 0.5, ff_tv[-1] + 0.5)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(*xlim)

    # Unify y-axis for ROI rows
    roi_row_indices = [r for r in range(N_ROWS) if not show_audio or r != AUDIO_GRID_ROW]
    roi_axes = [axes[r, c] for r in roi_row_indices for c in range(5)]
    all_ylims = [ax.get_ylim() for ax in roi_axes]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax in roi_axes:
        ax.set_ylim(ymin, ymax)

    # Unify y-axis for audio row
    if show_audio:
        audio_axes = [axes[AUDIO_GRID_ROW, c] for c in range(5)]
        audio_ylims = [ax.get_ylim() for ax in audio_axes]
        aymin = min(y[0] for y in audio_ylims)
        aymax = max(y[1] for y in audio_ylims)
        for ax in audio_axes:
            ax.set_ylim(aymin, aymax)

    # Legends on top row (and audio row when shown)
    for col in range(5):
        axes[0, col].legend(loc='upper right', fontsize=7)
    if show_audio:
        for col in range(5):
            axes[AUDIO_GRID_ROW, col].legend(loc='upper right', fontsize=7)

    subtitle = 't-values (one-sample vs. 0)' if use_tvals else 'Shading = SEM across subjects'
    fig.suptitle(
        f'Event-Locked ROI Time Courses Across Tasks\n{subtitle}',
        fontsize=15, fontweight='bold', y=1.01)

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting filmfest data...")
    ff_tv, ff_stacks, ff_n = collect_filmfest()
    print(f"  {ff_n} subjects")

    print("Collecting filmfest SEG-B (within-movie) data...")
    segb_tv, segb_stacks, segb_n = collect_filmfest_segb()
    print(f"  {segb_n} subjects")

    print("Collecting AHC data...")
    ahc_tv, ahc_stacks, ahc_n = collect_ahc()
    print(f"  {ahc_n} subjects")

    print("Collecting SVF (all events)...")
    svf_tv, svf_all_stacks, svf_n = collect_svf(AGREEMENT_CATS, apply_context=False)
    print(f"  {svf_n} subjects")

    print("Collecting SVF (context-filtered)...")
    _, svf_ctx_stacks, _ = collect_svf(CONTEXT_CATS, apply_context=True)

    print("Collecting audio envelopes...")
    ff_audio_stacks   = collect_ff_between_audio()
    segb_audio_stacks = collect_ff_segb_audio()
    ahc_audio_stacks  = collect_ahc_audio()
    svf_all_audio_stacks = collect_svf_audio(AGREEMENT_CATS, apply_context=False)
    svf_ctx_audio_stacks = collect_svf_audio(CONTEXT_CATS,   apply_context=True)

    data_args = (ff_tv, ff_stacks, ff_n, segb_tv, segb_stacks, segb_n,
                 ahc_tv, ahc_stacks, ahc_n, svf_tv, svf_all_stacks, svf_ctx_stacks, svf_n,
                 ff_audio_stacks, segb_audio_stacks,
                 ahc_audio_stacks, svf_all_audio_stacks, svf_ctx_audio_stacks)

    # Mean ± SEM figure
    fig = _build_figure(*data_args, use_tvals=False)
    fig.tight_layout()
    out = OUTPUT_DIR / "combined_boundary_figure.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out}")

    # t-value figure
    fig = _build_figure(*data_args, use_tvals=True)
    fig.tight_layout()
    out = OUTPUT_DIR / "combined_boundary_tvals.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ==================================================================
    # SURFACE T-MAP FIGURES
    # ==================================================================
    print("\nPreparing surface parcellation...")
    fsavg, surf_l, surf_r, schaefer_labels = _build_surface_parcellation()

    parcel_cache = OUTPUT_DIR / "parcel_stacks_cache.npz"
    if parcel_cache.exists():
        print(f"Loading cached parcel stacks from {parcel_cache.name}...")
        loaded = np.load(parcel_cache, allow_pickle=True)
        all_parcel_stacks = list(loaded['all_parcel_stacks'])
        parcel_n = int(loaded['parcel_n'])
    else:
        print("Collecting parcel-level data (400 parcels)...")
        print("  Filmfest between-movie parcels...")
        ff_parcel = collect_filmfest_parcels(schaefer_labels)
        print("  Filmfest SEG-B parcels...")
        segb_parcel = collect_filmfest_segb_parcels(schaefer_labels)
        print("  AHC parcels...")
        ahc_parcel = collect_ahc_parcels(schaefer_labels)
        print("  SVF all parcels...")
        svf_all_parcel = collect_svf_parcels(AGREEMENT_CATS, False, schaefer_labels)
        print("  SVF context parcels...")
        svf_ctx_parcel = collect_svf_parcels(CONTEXT_CATS, True, schaefer_labels)

        all_parcel_stacks = [ff_parcel, segb_parcel, ahc_parcel,
                             svf_all_parcel, svf_ctx_parcel]

        parcel_n = max(
            ff_parcel['boundary'].shape[0],
            segb_parcel['boundary'].shape[0],
            ahc_parcel['boundary'].shape[0],
            max((v.shape[0] for v in svf_all_parcel.values()), default=0),
            max((v.shape[0] for v in svf_ctx_parcel.values()), default=0),
        )

        np.savez_compressed(parcel_cache,
                            all_parcel_stacks=np.array(all_parcel_stacks, dtype=object),
                            parcel_n=parcel_n)
        print(f"Cached parcel stacks to {parcel_cache.name}")

    # --- Contrast figure ---
    # For SVF: contrast = first condition (consensus switch / CC-switch) minus
    # last condition (consensus cluster / CC-cluster)
    agr_labels = [l for l, _, _, _ in AGREEMENT_CATS]
    ctx_labels = [l for l, _, _, _ in CONTEXT_CATS]
    col_defs_contrast = [
        ('Filmfest:\nbetween-movie', 'boundary', 'nonboundary'),
        ('Filmfest:\nSEG-B', 'boundary', 'nonboundary'),
        ('Explanation\ngeneration', 'boundary', 'nonboundary'),
        ('SVF: all\n(switch - cluster)', agr_labels[0], agr_labels[-1]),
        ('SVF: context\n(switch - cluster)', ctx_labels[0], ctx_labels[-1]),
    ]

    print("\nBuilding contrast surface t-map figure...")
    fig_contrast = _build_surface_contrast_figure(
        all_parcel_stacks, col_defs_contrast, fsavg, surf_l, surf_r, parcel_n)
    if fig_contrast is not None:
        out = OUTPUT_DIR / "combined_boundary_surface_tmaps_contrast.png"
        fig_contrast.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig_contrast)
        print(f"Saved {out}")

    # --- Per-condition figure ---
    # Each row = one condition from one task column
    percond_defs = []
    # Col 0: filmfest between-movie (2 conditions)
    for cond in ('boundary', 'nonboundary'):
        percond_defs.append(('FF between', cond, 0))
    # Col 1: filmfest SEG-B (2 conditions)
    for cond in ('boundary', 'nonboundary'):
        percond_defs.append(('FF SEG-B', cond, 1))
    # Col 2: AHC (2 conditions)
    for cond in ('boundary', 'nonboundary'):
        percond_defs.append(('AHC', cond, 2))
    # Col 3: SVF all (3 conditions)
    for cat_label, _, _, _ in AGREEMENT_CATS:
        percond_defs.append(('SVF all', cat_label, 3))
    # Col 4: SVF context (3 conditions)
    for cat_label, _, _, _ in CONTEXT_CATS:
        percond_defs.append(('SVF ctx', cat_label, 4))

    print("Building per-condition surface t-map figure...")
    fig_percond = _build_surface_percond_figure(
        all_parcel_stacks, percond_defs, fsavg, surf_l, surf_r, parcel_n)
    if fig_percond is not None:
        out = OUTPUT_DIR / "combined_boundary_surface_tmaps_percond.png"
        fig_percond.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig_percond)
        print(f"Saved {out}")

    # --- Single-window (4.5–10.5 s) contrast figure ---
    # 4.5s = index 5, 10.5s = index 9 (inclusive) → slice [5:10]
    hrf_epoch = [('4.5 to 10.5 s', 5, 10)]

    print("\nBuilding single-window (4.5–10.5 s) contrast surface t-map...")
    fig_hrf = _build_surface_contrast_figure(
        all_parcel_stacks, col_defs_contrast, fsavg, surf_l, surf_r, parcel_n,
        epoch_defs=hrf_epoch)
    if fig_hrf is not None:
        out = OUTPUT_DIR / "combined_boundary_surface_tmaps_contrast_hrf.png"
        fig_hrf.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig_hrf)
        print(f"Saved {out}")


if __name__ == '__main__':
    main()
