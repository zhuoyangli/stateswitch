"""
Task Boundary Time Course Analysis

Per-subject figures showing ROI time courses (PMC, Hippocampus, Angular Gyrus,
Auditory Cortex, Early Visual Cortex, dlPFC, dACC) locked to the offset of each
SVF and AHC trial.

Layout: 7 rows (ROIs) × 2 columns (SVF, AHC) per subject.
Time window: -30 to +60 s from trial offset.
Vertical marker at t=15 s (next trial onset).

Usage:
    python task_boundary.py
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy import stats
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
import nibabel.freesurfer as nib_fs
from nilearn import datasets, surface
from nilearn.plotting import plot_surf, plot_surf_contours
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, CACHE_DIR, FIGS_DIR, TR, FILMFEST_SUBJECTS, MOVIE_INFO
from configs.schaefer_rois import POSTERIOR_MEDIAL, ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL, DLPFC, DACC, MPFC
from fmrianalysis.utils import get_parcel_data

PSYCHOPY_DIR = Path('/home/datasets/stateswitch/psychopy')
ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
AUDIO_BIDS_DIR = Path('/home/datasets/stateswitch/rec/bids')
FILMFEST_MP4 = {
    'filmfest1': ANNOTATIONS_DIR / 'FilmFest_part1.mp4',
    'filmfest2': ANNOTATIONS_DIR / 'FilmFest_part2.mp4',
}
OUTPUT_DIR = FIGS_DIR / 'task_boundary'

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT  = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT  = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'

SUBJECT_IDS = ['sub-001', 'sub-003', 'sub-004', 'sub-006', 'sub-007', 'sub-008', 'sub-009']

# === PARAMETERS ===
TRS_BEFORE = 20   # 30s / 1.5s
TRS_AFTER = 40    # 60s / 1.5s
NEXT_TRIAL_ONSET = 15      # seconds after offset (SVF/AHC inter-trial interval)
TITLE_SCENE_OFFSET = 6.0   # seconds to shift filmfest boundaries for onset alignment
AUDIO_TARGET_SR = 1  # downsample envelope to 1 Hz
AUDIO_PRE = 30   # seconds before event
AUDIO_POST = 60  # seconds after event
AUDIO_SCAN_OFFSET = 12  # WAV recording starts 12s before fMRI scan

TASK_KEYS = ('svf', 'ahc', 'movie')
TASK_DISPLAY = {
    'svf': 'Semantic fluency\n(category offset; participant says "next")',
    'ahc': 'Explanation generation\n(prompt offset; participant says "next")',
    'movie': 'FilmFest\n(movie offset)',
}

# Epoch windows for surface t-maps (indices into 61-timepoint epoch array)
# Each epoch spans 10 TRs = 15 s; slice indices (start inclusive, end exclusive)
EPOCH_DEFS = [
    ('-30 to -15 s',  0,  10),
    ('-15 to 0 s',   10,  20),
    ('0 to 15 s',    20,  30),
    ('15 to 30 s',   30,  40),
    ('30 to 45 s',   40,  50),
]

# === STYLE ===
SUBJECT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
TITLE_FS = 14
LABEL_FS = 12

# === ROI definitions ===
ROI_SPEC = [
    ('pmc',   'Posterior Medial Cortex'),
    ('hipp',  'Hippocampus'),
    ('ag',    'Angular Gyrus'),
    ('mpfc',  'mPFC'),
    ('dlpfc', 'dlPFC'),
    ('dacc',  'dACC'),
    ('evc',   'Early Visual Cortex'),
    ('eac',   'Auditory Cortex'),
]
PMC_LABELS   = POSTERIOR_MEDIAL.get('left_labels', []) + POSTERIOR_MEDIAL.get('right_labels', [])
AG_LABELS    = ANGULAR_GYRUS.get('left_labels', []) + ANGULAR_GYRUS.get('right_labels', [])
EAC_LABELS   = EARLY_AUDITORY.get('left_labels', []) + EARLY_AUDITORY.get('right_labels', [])
EVC_LABELS   = EARLY_VISUAL.get('left_labels', []) + EARLY_VISUAL.get('right_labels', [])
DLPFC_LABELS = DLPFC.get('left_labels', []) + DLPFC.get('right_labels', [])
DACC_LABELS  = DACC.get('left_labels', []) + DACC.get('right_labels', [])
MPFC_LABELS  = MPFC.get('left_labels', []) + MPFC.get('right_labels', [])
HIPP_KEYWORDS = ['hippocampus']

# === ROI surface visualization ===
ROI_SURFACE = {
    'pmc': {
        'left': POSTERIOR_MEDIAL['left'], 'right': POSTERIOR_MEDIAL['right'],
        'view': 'medial',
    },
    'ag': {
        'left': ANGULAR_GYRUS['left'], 'right': ANGULAR_GYRUS['right'],
        'view': 'lateral',
    },
    'eac': {
        'left': EARLY_AUDITORY['left'], 'right': EARLY_AUDITORY['right'],
        'view': 'lateral',
    },
    'evc': {
        'left': EARLY_VISUAL['left'], 'right': EARLY_VISUAL['right'],
        'view': 'medial',
    },
    'dlpfc': {
        'left': DLPFC['left'], 'right': DLPFC['right'],
        'view': 'lateral',
    },
    'dacc': {
        'left': DACC['left'], 'right': DACC['right'],
        'view': 'medial',
    },
    'mpfc': {
        'left': MPFC['left'], 'right': MPFC['right'],
        'view': 'medial',
    },
}
ROI_COLOR = '#e74c3c'
ROI_CMAP = LinearSegmentedColormap.from_list('roi', ['#888888', ROI_COLOR], N=256)


# ============================================================================
# AUDIO ENVELOPE HELPERS
# ============================================================================

def load_audio_envelope(filepath, target_sr=AUDIO_TARGET_SR):
    """Load audio from WAV or MP4 and compute amplitude envelope.

    Returns (envelope, sample_rate) where envelope is normalized to [0, 1].
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() == '.mp4':
        # Extract audio from MP4 via ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(filepath), '-ac', '1', '-ar', '16000',
             '-vn', tmp_path],
            capture_output=True, check=True)
        sr, data = wavfile.read(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
    else:
        sr, data = wavfile.read(str(filepath))

    # Convert to float mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)

    # Amplitude envelope: abs -> block-average downsample to target_sr
    envelope = np.abs(data)

    # Downsample by block-averaging (each output sample is the mean of one
    # target_sr-sized block). This is proper anti-aliasing and avoids
    # instantaneous transients (breaths, clicks) dominating individual samples.
    block = int(round(sr / target_sr))
    n_blocks = len(envelope) // block
    envelope = envelope[:n_blocks * block].reshape(n_blocks, block).mean(axis=1)

    # Z-score across time (like BOLD)
    mu = np.mean(envelope)
    sd = np.std(envelope)
    if sd > 0:
        envelope = (envelope - mu) / sd

    return envelope, target_sr


def extract_audio_event_locked(envelope, sr, event_times_sec,
                               pre=AUDIO_PRE, post=AUDIO_POST):
    """Extract event-locked audio epochs at native audio resolution.

    Returns (epochs, time_vec) where epochs is (n_events, n_timepoints).
    """
    n_pre = int(pre * sr)
    n_post = int(post * sr)
    n_total = n_pre + n_post + 1
    time_vec = np.linspace(-pre, post, n_total)

    n = len(envelope)
    epochs = []
    for t in event_times_sec:
        center = int(round(t * sr))
        start = center - n_pre
        end = center + n_post + 1
        if start >= 0 and end <= n:
            epochs.append(envelope[start:end])
    if not epochs:
        return None, time_vec
    return np.array(epochs), time_vec


def find_audio_wav(subject, session, task):
    """Find audio WAV file for an SVF/AHC run."""
    wav_path = (AUDIO_BIDS_DIR / subject / session /
                f"{subject}_{session}_task-{task}_desc-audio.wav")
    if wav_path.exists():
        return wav_path
    return None


# ============================================================================
# SURFACE HELPERS
# ============================================================================

def _build_surface_parcellation():
    """Project Schaefer atlas to fsaverage surface (cached across calls)."""
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=17, resolution_mm=2)
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l)
                  for l in atlas['labels']]
    schaefer_labels = [l for l in all_labels if l != 'Background']
    fsavg = datasets.fetch_surf_fsaverage()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        surf_l = surface.vol_to_surf(
            atlas['maps'], fsavg['pial_left'],
            interpolation='nearest_most_frequent').astype(int)
        surf_r = surface.vol_to_surf(
            atlas['maps'], fsavg['pial_right'],
            interpolation='nearest_most_frequent').astype(int)
    return fsavg, surf_l, surf_r, schaefer_labels


def plot_roi_brain(ax, roi_key, fsavg, surf_l, surf_r):
    """Render an ROI on inflated brain surface (gray background, red ROI)."""
    if roi_key == 'hipp':
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'Hippocampus', ha='center', va='center',
                fontsize=LABEL_FS, transform=ax.transAxes)
        return

    spec = ROI_SURFACE[roi_key]
    view = spec['view']

    mask_l = np.where(np.isin(surf_l, spec['left']), 1.0, np.nan)
    mask_r = np.where(np.isin(surf_r, spec['right']), 1.0, np.nan)

    if view == 'medial':
        mesh, bg, mask, hemi = (fsavg['infl_right'], fsavg['sulc_right'],
                                mask_r, 'right')
    else:
        mesh, bg, mask, hemi = (fsavg['infl_left'], fsavg['sulc_left'],
                                mask_l, 'left')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', (DeprecationWarning, RuntimeWarning))
        plot_surf(
            mesh, surf_map=mask, hemi=hemi, view=view,
            bg_map=bg, axes=ax, colorbar=False,
            cmap=ROI_CMAP, bg_on_data=True, darkness=0.5,
            vmin=0, vmax=1,
        )


# ============================================================================
# SESSION DISCOVERY
# ============================================================================

def discover_sessions(subject):
    """Discover sessions with SVF or AHC fMRI data from derivatives.

    Returns list of (session, task) tuples, e.g. [('ses-06', 'svf'), ('ses-06', 'ahc'), ...].
    """
    sub_dir = DERIVATIVES_DIR / subject
    if not sub_dir.exists():
        return []
    hits = []
    for ses_dir in sorted(sub_dir.glob('ses-*')):
        session = ses_dir.name
        func_dir = ses_dir / 'func'
        if not func_dir.exists():
            continue
        for task in ('svf', 'ahc'):
            bold = list(func_dir.glob(f'*_task-{task}_*_bold.nii.gz'))
            if bold:
                hits.append((session, task))
    return hits


def find_psychopy_csv(subject, session, task):
    """Find the psychopy CSV containing data for a given task.

    If multiple CSVs exist (e.g. program crashed and resumed), return the one
    that has the relevant task columns.
    """
    ses_dir = PSYCHOPY_DIR / subject / session
    if not ses_dir.exists():
        return None
    stopped_col = f'{task}_trial.stopped'
    for csv_path in sorted(ses_dir.glob('*.csv')):
        df = pd.read_csv(csv_path, nrows=0)
        if stopped_col in df.columns:
            return csv_path
    return None


def parse_trial_offsets(csv_path, task):
    """Parse psychopy CSV and return trial offset times in scan-relative seconds.

    Returns list of offset times (seconds from the start of that task's scan),
    excluding the last trial (no boundary after it).

    The scanner trigger coincides with the onset of the first trial.
    """
    df = pd.read_csv(csv_path)
    started_col = f'{task}_trial.started'
    stopped_col = f'{task}_trial.stopped'
    if stopped_col not in df.columns:
        return []
    mask = df[stopped_col].notna()
    if not mask.any():
        return []
    rows = df[mask]
    start = rows[started_col].iloc[0]
    offsets = (rows[stopped_col].values - start).tolist()
    # Skip the last trial (no boundary after it)
    return offsets[:-1]


def parse_trial_onsets(csv_path, task):
    """Parse psychopy CSV and return trial onset times in scan-relative seconds.

    Returns list of onset times (seconds from the start of that task's scan),
    excluding the first trial (onset is always 0 by definition).
    """
    df = pd.read_csv(csv_path)
    started_col = f'{task}_trial.started'
    stopped_col = f'{task}_trial.stopped'
    if stopped_col not in df.columns:
        return []
    mask = df[stopped_col].notna()
    if not mask.any():
        return []
    rows = df[mask]
    start = rows[started_col].iloc[0]
    onsets = (rows[started_col].values - start).tolist()
    # Skip the first trial (onset is 0, no silence can precede it)
    return onsets[1:]


def parse_all_trial_boundaries(csv_path, task):
    """Return all trial onset and offset times in scan-relative seconds.

    Unlike parse_trial_offsets/parse_trial_onsets, this includes every onset
    and offset without skipping the first or last, for use as exclusion zones.
    """
    df = pd.read_csv(csv_path)
    started_col = f'{task}_trial.started'
    stopped_col = f'{task}_trial.stopped'
    if stopped_col not in df.columns:
        return []
    mask = df[stopped_col].notna()
    if not mask.any():
        return []
    rows = df[mask]
    start = rows[started_col].iloc[0]
    onsets  = (rows[started_col].values - start).tolist()
    offsets = (rows[stopped_col].values  - start).tolist()
    return onsets + offsets


# ============================================================================
# MOVIE BOUNDARY HELPERS
# ============================================================================

def _mss_to_seconds(mss):
    minutes = int(mss)
    seconds = round((mss - minutes) * 100)
    return minutes * 60 + seconds


def get_movie_boundary_offsets(task):
    """Get movie boundary times (offset of previous movie) for a given filmfest task."""
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    boundaries = []
    for movie in movies[:-1]:
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        last_end = segb['End Time (m.ss)'].values[-1]
        boundaries.append(_mss_to_seconds(last_end))
    return boundaries


def collect_movie_epochs(subject, roi_keys):
    """Collect movie boundary epochs for a filmfest subject.

    Returns dict of roi_key -> list of epoch arrays, or empty dict if not a filmfest subject.
    """
    if subject not in FILMFEST_SUBJECTS:
        return {k: [] for k in roi_keys}

    session = FILMFEST_SUBJECTS[subject]
    boundaries = {}
    for task in ('filmfest1', 'filmfest2'):
        boundaries[task] = get_movie_boundary_offsets(task)

    movie_epochs = {k: [] for k in roi_keys}
    for task in ('filmfest1', 'filmfest2'):
        roi_ts = extract_roi_timeseries(subject, session, task)
        if roi_ts is None:
            continue
        for k in roi_keys:
            epochs = extract_event_locked(roi_ts[k], boundaries[task])
            if epochs is not None:
                movie_epochs[k].append(epochs)
    return movie_epochs


# ============================================================================
# ROI HELPERS
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
    """Load parcel data and return dict of ROI time series.

    Extracts from BOLD if not cached; returns None only if BOLD file is missing.
    """
    try:
        schaefer = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
        ho_sub = get_parcel_data(subject, session, task, atlas='HarvardOxford_sub')
    except FileNotFoundError:
        return None
    return {
        'pmc':   _avg_labels(schaefer, PMC_LABELS),
        'ag':    _avg_labels(schaefer, AG_LABELS),
        'eac':   _avg_labels(schaefer, EAC_LABELS),
        'evc':   _avg_labels(schaefer, EVC_LABELS),
        'hipp':  _avg_keywords(ho_sub, HIPP_KEYWORDS),
        'dlpfc': _avg_labels(schaefer, DLPFC_LABELS),
        'dacc':  _avg_labels(schaefer, DACC_LABELS),
        'mpfc':  _avg_labels(schaefer, MPFC_LABELS),
    }


def extract_event_locked(signal, event_times_sec):
    """Extract event-locked epochs from signal.

    Returns array of shape (n_events, n_timepoints) or None if no valid epochs.
    """
    n = len(signal)
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


def extract_event_locked_2d(data_2d, event_times_sec):
    """Extract event-locked epochs from 2D data (n_timepoints, n_parcels).

    Returns array of shape (n_events, n_epoch_timepoints, n_parcels) or None.
    """
    n_time = data_2d.shape[0]
    centers = np.round(np.array(event_times_sec) / TR).astype(int)
    offsets = np.arange(-TRS_BEFORE, TRS_AFTER + 1)
    idx = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n_time), axis=1)
    if not valid.any():
        return None
    return data_2d[idx[valid]]


def _roi_boundary_edges(faces, coords, roi_mask):
    """Return (N, 2, 3) edge coordinate pairs on the ROI boundary (vectorized).

    A boundary edge has exactly one endpoint inside the ROI.
    """
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    m0, m1, m2 = roi_mask[v0], roi_mask[v1], roi_mask[v2]
    segs = []
    for b, ia, ib in [(m0 != m1, v0, v1), (m1 != m2, v1, v2), (m0 != m2, v0, v2)]:
        if b.any():
            segs.append(np.stack([coords[ia[b]], coords[ib[b]]], axis=1))
    return np.concatenate(segs, axis=0) if segs else np.zeros((0, 2, 3))


def _build_roi_outline_edges(surf_l, surf_r, fsavg):
    """Pre-compute PMC/dlPFC/dACC/AG boundary edges on the inflated fsaverage surface."""
    from nilearn import surface as surf_mod
    coords_l, faces_l = surf_mod.load_surf_mesh(fsavg['infl_left'])
    coords_r, faces_r = surf_mod.load_surf_mesh(fsavg['infl_right'])
    coords_l = np.asarray(coords_l)
    coords_r = np.asarray(coords_r)
    faces_l = np.asarray(faces_l)
    faces_r = np.asarray(faces_r)
    roi_outline_keys = ('pmc', 'dlpfc', 'dacc', 'ag', 'mpfc')
    edges_l_all, edges_r_all = [], []
    for roi_key in roi_outline_keys:
        spec = ROI_SURFACE[roi_key]
        mask_l = np.isin(surf_l, spec['left'])
        mask_r = np.isin(surf_r, spec['right'])
        e_l = _roi_boundary_edges(faces_l, coords_l, mask_l)
        e_r = _roi_boundary_edges(faces_r, coords_r, mask_r)
        if len(e_l):
            edges_l_all.append(e_l)
        if len(e_r):
            edges_r_all.append(e_r)
    outline_l = np.concatenate(edges_l_all) if edges_l_all else np.zeros((0, 2, 3))
    outline_r = np.concatenate(edges_r_all) if edges_r_all else np.zeros((0, 2, 3))
    return outline_l, outline_r


def _parcel_to_surface(t_values, surf_l, surf_r):
    """Map 400-parcel values to fsaverage surface vertices.

    Parameters
    ----------
    t_values : (400,) array, ordered by Schaefer parcel ID 1-400.
    surf_l, surf_r : vertex-level parcel ID arrays from _build_surface_parcellation.

    Returns (surf_data_l, surf_data_r) with NaN for non-parcel vertices.
    """
    t_map = np.full(401, np.nan)
    t_map[1:] = t_values
    surf_data_l = np.where(surf_l > 0, t_map[surf_l], np.nan)
    surf_data_r = np.where(surf_r > 0, t_map[surf_r], np.nan)
    return surf_data_l, surf_data_r


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--align', choices=['offset', 'onset'], default='offset',
                        help='Align epochs to trial offset (default) or onset')
    parser.add_argument('--conjunction-only', action='store_true',
                        help='Skip all figures except the conjunction z-maps')
    parser.add_argument('--vertical-only', action='store_true',
                        help='Only plot the vertical ROI layout figure')
    args = parser.parse_args()
    align = args.align
    conjunction_only = args.conjunction_only
    vertical_only = args.vertical_only

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing surface parcellation for ROI insets...")
    fsavg, surf_l, surf_r, schaefer_labels = _build_surface_parcellation()
    print("Computing ROI outline edges (PMC, dlPFC, dACC, AG)...")
    roi_outline_l, roi_outline_r = _build_roi_outline_edges(surf_l, surf_r, fsavg)

    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR
    roi_keys = [k for k, _ in ROI_SPEC]

    # Movie boundary times (same for all subjects)
    # For onset alignment, shift forward by TITLE_SCENE_OFFSET to skip the title card
    movie_boundaries = {}
    for task in ('filmfest1', 'filmfest2'):
        bnd = get_movie_boundary_offsets(task)
        if align == 'onset':
            bnd = [t + TITLE_SCENE_OFFSET for t in bnd]
        movie_boundaries[task] = bnd
        print(f"  {task} movie boundaries (s): {movie_boundaries[task]}")

    # Audio time vector (at native audio resolution)
    n_audio_pre = int(AUDIO_PRE * AUDIO_TARGET_SR)
    n_audio_post = int(AUDIO_POST * AUDIO_TARGET_SR)
    n_audio_total = n_audio_pre + n_audio_post + 1
    audio_time_vec = np.linspace(-AUDIO_PRE, AUDIO_POST, n_audio_total)

    # Pre-compute movie audio epochs (same stimulus for all subjects)
    movie_audio_epochs = []  # list of per-boundary epoch arrays
    for ff_task in ('filmfest1', 'filmfest2'):
        mp4_path = FILMFEST_MP4.get(ff_task)
        if mp4_path is not None and mp4_path.exists():
            try:
                env, env_sr = load_audio_envelope(mp4_path)
                audio_ep, _ = extract_audio_event_locked(
                    env, env_sr, movie_boundaries[ff_task])
                if audio_ep is not None:
                    for row in audio_ep:
                        movie_audio_epochs.append(row)
            except Exception as e:
                print(f"  {ff_task} audio: {e}")
    if movie_audio_epochs:
        movie_audio_epochs = np.array(movie_audio_epochs)
        print(f"  Movie audio: {movie_audio_epochs.shape[0]} boundary epochs")

    # Group-level accumulators
    group_data = {t: {k: [] for k in roi_keys} for t in TASK_KEYS}
    group_audio = {t: [] for t in TASK_KEYS}
    group_parcel_data = {t: [] for t in TASK_KEYS}
    group_labels = []

    for subject in SUBJECT_IDS:
        print(f"\n{'='*60}")
        print(f"  {subject}")
        print(f"{'='*60}")

        # Discover which (session, task) pairs have fMRI data
        fmri_hits = discover_sessions(subject)
        if not fmri_hits:
            print(f"  No SVF/AHC fMRI data found, skipping")
            continue

        # Group by session
        sessions_with_tasks = {}
        for session, task in fmri_hits:
            sessions_with_tasks.setdefault(session, []).append(task)
        print(f"  fMRI sessions: {dict(sessions_with_tasks)}")

        # Collect epochs across sessions for SVF, AHC, and movie
        task_epochs = {t: {k: [] for k in roi_keys} for t in TASK_KEYS}
        task_audio_epochs = {t: [] for t in TASK_KEYS}
        task_parcel_epochs = {t: [] for t in TASK_KEYS}

        for session, tasks in sorted(sessions_with_tasks.items()):
            for task_key in tasks:
                csv_path = find_psychopy_csv(subject, session, task_key)
                if csv_path is None:
                    print(f"  {session} {task_key}: no psychopy CSV found, skipping")
                    continue

                if align == 'onset':
                    trial_events = parse_trial_onsets(csv_path, task_key)
                    ev_label = 'onsets'
                else:
                    trial_events = parse_trial_offsets(csv_path, task_key)
                    ev_label = 'offsets'
                if not trial_events:
                    print(f"  {session} {task_key}: no trial {ev_label}, skipping")
                    continue

                roi_ts = extract_roi_timeseries(subject, session, task_key)
                if roi_ts is None:
                    print(f"  {session} {task_key}: no cached parcel data, skipping")
                    continue

                print(f"  {session} {task_key}: {len(roi_ts['pmc'])} TRs, "
                      f"{len(trial_events)} boundary events")

                for k in roi_keys:
                    epochs = extract_event_locked(roi_ts[k], trial_events)
                    if epochs is not None:
                        task_epochs[task_key][k].append(epochs)

                # Audio envelope epochs
                wav_path = find_audio_wav(subject, session, task_key)
                if wav_path is not None:
                    try:
                        env, env_sr = load_audio_envelope(wav_path)
                        # Shift event times by scan offset (WAV starts 12s before scan)
                        wav_event_times = [t + AUDIO_SCAN_OFFSET for t in trial_events]
                        audio_ep, _ = extract_audio_event_locked(
                            env, env_sr, wav_event_times)
                        if audio_ep is not None:
                            task_audio_epochs[task_key].append(audio_ep)
                            print(f"  {session} {task_key}: {audio_ep.shape[0]} audio epochs")
                    except Exception as e:
                        print(f"  {session} {task_key}: audio load failed: {e}")

                # Per-parcel epochs for surface t-maps
                schaefer = get_parcel_data(subject, session, task_key, atlas='Schaefer400_17Nets')
                parcel_ts = np.column_stack(
                    [schaefer[l] for l in schaefer_labels])
                pe = extract_event_locked_2d(parcel_ts, trial_events)
                if pe is not None:
                    task_parcel_epochs[task_key].append(pe)

        # Collect movie boundary epochs
        if subject in FILMFEST_SUBJECTS:
            ff_session = FILMFEST_SUBJECTS[subject]
            for ff_task in ('filmfest1', 'filmfest2'):
                roi_ts = extract_roi_timeseries(subject, ff_session, ff_task)
                if roi_ts is None:
                    continue
                bnd_times = movie_boundaries[ff_task]
                for k in roi_keys:
                    epochs = extract_event_locked(roi_ts[k], bnd_times)
                    if epochs is not None:
                        task_epochs['movie'][k].append(epochs)
                # Per-parcel epochs for surface t-maps
                schaefer = get_parcel_data(subject, ff_session, ff_task, atlas='Schaefer400_17Nets')
                parcel_ts = np.column_stack(
                    [schaefer[l] for l in schaefer_labels])
                pe = extract_event_locked_2d(parcel_ts, bnd_times)
                if pe is not None:
                    task_parcel_epochs['movie'].append(pe)
            n_movie = sum(e.shape[0] for e in task_epochs['movie'].get('pmc', []))
            if n_movie:
                print(f"  movie: {n_movie} boundary epochs")

        # Check we have data
        has_any = any(
            all(len(task_epochs[t][k]) > 0 for k in roi_keys)
            for t in TASK_KEYS
        )
        if not has_any:
            print(f"  {subject}: no valid epochs, skipping figure")
            continue

        # Compute mean, SEM, and epoch count per ROI per task
        task_stats = {}
        task_audio_stats = {}
        for task_key in TASK_KEYS:
            task_stats[task_key] = {}
            for k in roi_keys:
                if task_epochs[task_key][k]:
                    stacked = np.vstack(task_epochs[task_key][k])
                    n = stacked.shape[0]
                    task_stats[task_key][k] = {
                        'mean': stacked.mean(axis=0),
                        'sem': stacked.std(axis=0) / np.sqrt(n),
                        'n': n,
                    }
                    print(f"  {subject} {task_key} {k}: {n} epochs")
                else:
                    task_stats[task_key][k] = None

            # Audio stats
            if task_audio_epochs[task_key]:
                stacked = np.vstack(task_audio_epochs[task_key])
                n = stacked.shape[0]
                task_audio_stats[task_key] = {
                    'mean': stacked.mean(axis=0),
                    'sem': stacked.std(axis=0) / np.sqrt(n),
                    'n': n,
                }
            else:
                task_audio_stats[task_key] = None

        # Store per-subject means for group plot
        for task_key in TASK_KEYS:
            for k in roi_keys:
                if task_stats[task_key][k] is not None:
                    group_data[task_key][k].append((subject, task_stats[task_key][k]['mean']))
            if task_audio_stats[task_key] is not None:
                group_audio[task_key].append((subject, task_audio_stats[task_key]['mean']))
        # Store per-subject parcel means for group surface t-maps
        for task_key in TASK_KEYS:
            if task_parcel_epochs[task_key]:
                stacked = np.concatenate(task_parcel_epochs[task_key], axis=0)
                group_parcel_data[task_key].append(stacked.mean(axis=0))
        group_labels.append(subject)

        # (Per-subject figures skipped for speed)

    ZMAP_VMAX = 0.6
    n_subj = len(group_labels)

    # ===================================================================
    # CONJUNCTION MAPS (fast — runs before slow figure sections)
    # ===================================================================
    # Build ROI outline masks from FreeSurfer annot files (same vertex space
    # as plot_surf on inflated fsaverage6) — avoids distortion from vol_to_surf.
    _lh_labels, _, _ = nib_fs.read_annot(str(LH_ANNOT))
    _rh_labels, _, _ = nib_fs.read_annot(str(RH_ANNOT))
    annot_surf_l = _lh_labels.astype(int)
    annot_surf_r = np.where(_rh_labels > 0, _rh_labels + 200, 0).astype(int)

    pmc_mask_l   = np.isin(annot_surf_l, POSTERIOR_MEDIAL['left']).astype(int)
    pmc_mask_r   = np.isin(annot_surf_r, POSTERIOR_MEDIAL['right']).astype(int)
    mpfc_mask_l  = np.isin(annot_surf_l, MPFC['left']).astype(int)
    mpfc_mask_r  = np.isin(annot_surf_r, MPFC['right']).astype(int)
    dlpfc_mask_l = np.isin(annot_surf_l, DLPFC['left']).astype(int)
    dlpfc_mask_r = np.isin(annot_surf_r, DLPFC['right']).astype(int)
    dacc_mask_l  = np.isin(annot_surf_l, DACC['left']).astype(int)
    dacc_mask_r  = np.isin(annot_surf_r, DACC['right']).astype(int)

    def _print_top_parcels(conj_map, labels, all_z, n=20, header="  Top parcels", label_filter=None):
        """Print top-N parcels by min-z across tasks, with per-task z-scores.

        label_filter : str, optional — if given, only print parcels whose label
                       contains this substring (case-insensitive).
        """
        valid = np.where(~np.isnan(conj_map))[0]
        if label_filter:
            valid = np.array([i for i in valid if label_filter.lower() in labels[i].lower()])
        if len(valid) == 0:
            print(f"{header}: (no parcels)")
            return
        ranked = valid[np.argsort(conj_map[valid])[::-1]][:n]
        task_names = list(TASK_KEYS)
        header_row = f"  {'Rank':>4}  {'Min-z':>6}  " + "  ".join(f"{t:>8}" for t in task_names) + "  Label"
        print(header)
        print(header_row)
        for rank, idx in enumerate(ranked, 1):
            per_task = "  ".join(f"{all_z[ti, idx]:>8.3f}" for ti in range(len(task_names)))
            print(f"  {rank:>4}  {conj_map[idx]:>6.3f}  {per_task}  {labels[idx]}")

    def _plot_conj_figure(conj_map, vmin_val, vmax_val, title, outpath, cbar_label,
                          extra_masks_l=None, extra_masks_r=None,
                          subplot_wspace=-0.15, subplot_hspace=-0.15,
                          cbar_rect=None, cbar_orientation='vertical',
                          gs_right=0.90, gs_bottom=0.05, thresh=None):
        """Plot conjunction figure; extra_masks_l/r are lists of additional ROI masks to outline."""
        # Build colormap: figure background (white) below thresh, Reds above
        if thresh is not None and thresh > vmin_val:
            thresh_frac = (thresh - vmin_val) / (vmax_val - vmin_val)
            reds = plt.cm.get_cmap('Reds')
            n = 256
            colors = []
            for i in range(n):
                t = i / (n - 1)
                if t < thresh_frac:
                    colors.append((0.88, 0.88, 0.88, 1.0))  # light grey
                else:
                    t_reds = (t - thresh_frac) / (1.0 - thresh_frac)
                    colors.append(reds(t_reds))
            cmap = LinearSegmentedColormap.from_list('reds_thresh', colors, N=n)
        else:
            cmap = plt.cm.get_cmap('Reds')

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(title, fontsize=TITLE_FS, fontweight='bold', y=1.01)
        surf_data_l, surf_data_r = _parcel_to_surface(conj_map, surf_l, surf_r)
        inner_gs = gridspec.GridSpec(
            2, 2, wspace=subplot_wspace, hspace=subplot_hspace,
            top=0.92, bottom=gs_bottom, left=0.02, right=gs_right)
        views = [
            (0, 0, fsavg['infl_left'],  surf_data_l, pmc_mask_l, mpfc_mask_l, 'left',  'lateral', fsavg['sulc_left']),
            (0, 1, fsavg['infl_right'], surf_data_r, pmc_mask_r, mpfc_mask_r, 'right', 'lateral', fsavg['sulc_right']),
            (1, 0, fsavg['infl_left'],  surf_data_l, pmc_mask_l, mpfc_mask_l, 'left',  'medial',  fsavg['sulc_left']),
            (1, 1, fsavg['infl_right'], surf_data_r, pmc_mask_r, mpfc_mask_r, 'right', 'medial',  fsavg['sulc_right']),
        ]
        for row, col, mesh, sdata, pmc_m, mpfc_m, hemi, view, bg in views:
            ax = fig.add_subplot(inner_gs[row, col], projection='3d')
            extra_l = extra_masks_l or []
            extra_r = extra_masks_r or []
            extra_m = extra_l if hemi == 'left' else extra_r
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', (DeprecationWarning, RuntimeWarning))
                plot_surf(
                    mesh, surf_map=sdata, hemi=hemi, view=view,
                    bg_map=bg, axes=ax, colorbar=False,
                    cmap='Reds', bg_on_data=True, darkness=0.5,
                    vmin=vmin_val, vmax=vmax_val)
                if pmc_m.any():
                    plot_surf_contours(mesh, roi_map=pmc_m, hemi=hemi,
                                       levels=[1], colors=['black'], axes=ax,
                                       linewidths=0.2)
                if mpfc_m.any():
                    plot_surf_contours(mesh, roi_map=mpfc_m, hemi=hemi,
                                       levels=[1], colors=['black'], axes=ax,
                                       linewidths=0.2)
                for em in extra_m:
                    if em.any():
                        plot_surf_contours(mesh, roi_map=em, hemi=hemi,
                                           levels=[1], colors=['black'], axes=ax,
                                           linewidths=0.2)
                ax.set_title('')
        if cbar_rect is None:
            cbar_rect = [0.92, 0.15, 0.015, 0.65]
        cbar_ax = fig.add_axes(cbar_rect)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_val, vmax=vmax_val))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation=cbar_orientation)
        cb.set_label(cbar_label, fontsize=16)
        cb.ax.tick_params(labelsize=16)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {outpath}")

    print(f"\n{'='*60}")
    print("  Conjunction maps (0-9 TR, all three tasks)")
    print(f"{'='*60}")

    ws0, we0 = TRS_BEFORE + 0, TRS_BEFORE + 10
    task_zmaps_0to9 = {}
    for task_key in TASK_KEYS:
        if not group_parcel_data[task_key]:
            continue
        data = np.array(group_parcel_data[task_key])
        task_zmaps_0to9[task_key] = data[:, ws0:we0, :].mean(axis=1).mean(axis=0)

    if all(k in task_zmaps_0to9 for k in TASK_KEYS):
        all_z = np.array([task_zmaps_0to9[k] for k in TASK_KEYS])
        _thresh09 = 0.2 if align == 'onset' else 0.2
        _vmax09   = 0.5 if align == 'onset' else ZMAP_VMAX
        conj_mask_z = np.all(all_z > _thresh09, axis=0)
        conj_zmap = np.where(conj_mask_z, all_z.min(axis=0), np.nan)
        n_conj_z = int(conj_mask_z.sum())
        print(f"  Z-map conjunction: {n_conj_z} parcels with z > {_thresh09} in all tasks")
        _print_top_parcels(conj_zmap, schaefer_labels, all_z, n=20, header="  Top parcels (0-9 TR)")
        _print_top_parcels(conj_zmap, schaefer_labels, all_z, n=20, header="  Top PFC parcels (0-9 TR)", label_filter='PFC')
        _plot_conj_figure(
            conj_zmap, 0, _vmax09,
            f"Conjunction z-map: 0\u20139 TR, z > {_thresh09} in all tasks (N={n_subj})",
            OUTPUT_DIR / f"task_boundary_group_conjunction_zmap_0to9TR_align-{align}.png",
            cbar_label='min z-score (BOLD) across tasks')
        if align == 'onset':
            _plot_conj_figure(
                conj_zmap, 0, 0.5,
                f"Conjunction z-map: 0\u20139 TR, z > {_thresh09} in all tasks (N={n_subj})",
                OUTPUT_DIR / f"task_boundary_group_conjunction_zmap_0to9TR_align-{align}_clim0-0.5.png",
                cbar_label='min z-score (BOLD) across tasks')
            _plot_conj_figure(
                conj_zmap, 0, 0.5,
                '',
                OUTPUT_DIR / f"task_boundary_group_conjunction_zmap_0to9TR_align-{align}_clim0-0.5_dlpfc-dacc.png",
                cbar_label='min z-score (BOLD) across tasks',
                extra_masks_l=[dlpfc_mask_l, dacc_mask_l],
                extra_masks_r=[dlpfc_mask_r, dacc_mask_r],
                subplot_wspace=-0.35, subplot_hspace=0.05,
                cbar_rect=[0.325, 0.04, 0.35, 0.025],
                cbar_orientation='horizontal',
                gs_right=0.98, gs_bottom=0.12,
                thresh=_thresh09)
            # No-threshold version: show min-z across tasks for all parcels
            nothresh_zmap = all_z.min(axis=0)
            _plot_conj_figure(
                nothresh_zmap, 0, 0.6,
                f"Min z-map: 0\u20139 TR, no threshold (N={n_subj})",
                OUTPUT_DIR / f"task_boundary_group_conjunction_zmap_0to9TR_align-{align}_nothresh.png",
                cbar_label='min z-score (BOLD) across tasks')

    ws06, we06 = TRS_BEFORE + 0, TRS_BEFORE + 7
    task_zmaps_0to6 = {}
    for task_key in TASK_KEYS:
        if not group_parcel_data[task_key]:
            continue
        data = np.array(group_parcel_data[task_key])
        task_zmaps_0to6[task_key] = data[:, ws06:we06, :].mean(axis=1).mean(axis=0)

    if all(k in task_zmaps_0to6 for k in TASK_KEYS):
        all_z06 = np.array([task_zmaps_0to6[k] for k in TASK_KEYS])
        conj_mask_z06 = np.all(all_z06 > 0.2, axis=0)
        conj_zmap06 = np.where(conj_mask_z06, all_z06.min(axis=0), np.nan)
        n_conj_z06 = int(conj_mask_z06.sum())
        print(f"  Z-map conjunction (0-6 TR): {n_conj_z06} parcels with z > 0.2 in all tasks")
        _print_top_parcels(conj_zmap06, schaefer_labels, all_z06, n=20, header="  Top parcels (0-6 TR)")
        _print_top_parcels(conj_zmap06, schaefer_labels, all_z06, n=20, header="  Top PFC parcels (0-6 TR)", label_filter='PFC')
        _plot_conj_figure(
            conj_zmap06, 0.2, ZMAP_VMAX,
            f"Conjunction z-map: 0\u20136 TR, z > 0.2 in all tasks (N={n_subj})",
            OUTPUT_DIR / f"task_boundary_group_conjunction_zmap_0to6TR_align-{align}.png",
            cbar_label='min z-score (BOLD) across tasks')

    ws3, we3 = TRS_BEFORE + 3, TRS_BEFORE + 10
    task_zmaps_3to9 = {}
    for task_key in TASK_KEYS:
        if not group_parcel_data[task_key]:
            continue
        data = np.array(group_parcel_data[task_key])
        task_zmaps_3to9[task_key] = data[:, ws3:we3, :].mean(axis=1).mean(axis=0)

    if all(k in task_zmaps_3to9 for k in TASK_KEYS):
        all_z3 = np.array([task_zmaps_3to9[k] for k in TASK_KEYS])
        conj_mask_z3 = np.all(all_z3 > 0.2, axis=0)
        conj_zmap3 = np.where(conj_mask_z3, all_z3.min(axis=0), np.nan)
        n_conj_z3 = int(conj_mask_z3.sum())
        print(f"  Z-map conjunction (3-9 TR): {n_conj_z3} parcels with z > 0.2 in all tasks")
        _print_top_parcels(conj_zmap3, schaefer_labels, all_z3, n=20, header="  Top parcels (3-9 TR)")
        _print_top_parcels(conj_zmap3, schaefer_labels, all_z3, n=20, header="  Top PFC parcels (3-9 TR)", label_filter='PFC')
        _vmin3, _vmax3 = (0.1, 0.4) if align == 'onset' else (0.2, ZMAP_VMAX)
        _plot_conj_figure(
            conj_zmap3, _vmin3, _vmax3,
            f"Conjunction z-map: 3\u20139 TR, z > 0.2 in all tasks (N={n_subj})",
            OUTPUT_DIR / f"task_boundary_group_conjunction_zmap_3to9TR_align-{align}.png",
            cbar_label='min z-score (BOLD) across tasks')

    ws4, we4 = TRS_BEFORE + 4, TRS_BEFORE + 14
    task_zmaps_4to13 = {}
    for task_key in TASK_KEYS:
        if not group_parcel_data[task_key]:
            continue
        data = np.array(group_parcel_data[task_key])
        task_zmaps_4to13[task_key] = data[:, ws4:we4, :].mean(axis=1).mean(axis=0)

    if all(k in task_zmaps_4to13 for k in TASK_KEYS):
        all_z4 = np.array([task_zmaps_4to13[k] for k in TASK_KEYS])
        conj_mask_z4 = np.all(all_z4 > 0.2, axis=0)
        conj_zmap4 = np.where(conj_mask_z4, all_z4.min(axis=0), np.nan)
        n_conj_z4 = int(conj_mask_z4.sum())
        print(f"  Z-map conjunction (4-13 TR): {n_conj_z4} parcels with z > 0.2 in all tasks")
        _print_top_parcels(conj_zmap4, schaefer_labels, all_z4, n=20, header="  Top parcels (4-13 TR)")
        _print_top_parcels(conj_zmap4, schaefer_labels, all_z4, n=20, header="  Top PFC parcels (4-13 TR)", label_filter='PFC')
        _plot_conj_figure(
            conj_zmap4, 0.2, ZMAP_VMAX,
            f"Conjunction z-map: 4\u201313 TR, z > 0.2 in all tasks (N={n_subj})",
            OUTPUT_DIR / f"task_boundary_group_conjunction_zmap_4to13TR_align-{align}.png",
            cbar_label='min z-score (BOLD) across tasks')

    if conjunction_only:
        return

    # ===================================================================
    # VERTICAL ROI LAYOUT — PMC, mPFC, dlPFC, Hippocampus, tasks overlaid
    # ===================================================================
    print(f"\n{'='*60}")
    print("  Vertical ROI layout (4 ROIs, tasks overlaid)")
    print(f"{'='*60}")

    VERT_ROI_SPEC = [
        ('pmc',   'PMC'),
        ('mpfc',  'mPFC'),
        ('dlpfc', 'dlPFC'),
        ('hipp',  'Hippocampus'),
    ]
    VERT_TASK_COLORS = {'svf': '#e41a1c', 'ahc': '#ff7f00', 'movie': '#377eb8'}
    VERT_TASK_LABELS = {'svf': 'SF', 'ahc': 'EG', 'movie': 'MW'}

    n_vert_rois = len(VERT_ROI_SPEC)
    fig_v, axes_v = plt.subplots(
        n_vert_rois, 1,
        figsize=(5, 3.0 * n_vert_rois),
        gridspec_kw={'hspace': 0.45})
    tc_axes_v = []
    for ri, (roi_key, roi_title) in enumerate(VERT_ROI_SPEC):
        ax = axes_v[ri]
        tc_axes_v.append(ax)

        for task_key in TASK_KEYS:
            subj_means = group_data[task_key][roi_key]
            if not subj_means:
                continue
            data = np.array([m for _, m in subj_means])
            gmean = data.mean(axis=0)
            gsem = data.std(axis=0) / np.sqrt(len(data))
            ax.plot(time_vec, gmean,
                    color=VERT_TASK_COLORS[task_key], lw=2,
                    label=VERT_TASK_LABELS[task_key])
            ax.fill_between(time_vec, gmean - gsem, gmean + gsem,
                            color=VERT_TASK_COLORS[task_key], alpha=0.18)

        # Grey shaded region: 0 to 15 s post-boundary
        ax.axvspan(0, 15, color='lightgrey', alpha=0.5, zorder=0)
        # Dashed vertical lines for pre-boundary events
        ax.axvline(-6, color='#377eb8', ls='--', lw=1.2)
        ax.axvline(-15, color='#FF5200', ls='--', lw=1.2)
        ax.axvline(0, color='#555555', ls='-', lw=1.2)
        ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
        ax.set_xlim(-30, 60)
        ax.set_xlabel('Time from boundary (s)', fontsize=LABEL_FS - 1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FS - 1)
        ax.set_title(roi_title, fontsize=LABEL_FS, fontweight='bold')
        if ri == 0:
            ax.legend(fontsize=LABEL_FS - 3, frameon=False, loc='upper right')

    for ax in tc_axes_v:
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['-1', '0', '1'])

    # Annotate vertical lines using fixed y coordinates on top two panels
    tc_axes_v[1].text(-6, 1.55, 'offset of previous\nmovie (MW)',
                      color='#377eb8', fontsize=8, ha='center', va='bottom')
    tc_axes_v[0].text(-15, 1.55, 'offset of previous\nsubtask (SF, EG)',
                      color='#FF5200', fontsize=8, ha='center', va='bottom')

    out_v = OUTPUT_DIR / f"task_boundary_group_vertical_{align}.png"
    plt.savefig(out_v, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_v}")

    if vertical_only:
        return

    # ===================================================================
    # GROUP-LEVEL PLOT
    # ===================================================================
    print(f"\n{'='*60}")
    print("  Group-level plot")
    print(f"{'='*60}")

    n_rois = len(ROI_SPEC)
    n_rows = n_rois + 1  # +1 for audio envelope row
    fig = plt.figure(figsize=(26, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 4, width_ratios=[1, 3, 3, 3], wspace=0.15, hspace=0.35,
                           top=0.93)
    n_subj = len(group_labels)
    fig.suptitle(f"Boundary Response ({align}-locked), N={n_subj}",
                 fontsize=TITLE_FS, fontweight='bold', y=0.96)

    tc_axes = []

    # Insert audio envelope row after EVC (last row)
    AUDIO_ROW = n_rois
    ROW_ITEMS = []  # (grid_row, type, roi_index_or_None)
    roi_idx = 0
    for grid_row in range(n_rows):
        if grid_row == AUDIO_ROW:
            ROW_ITEMS.append((grid_row, 'audio', None))
        else:
            ROW_ITEMS.append((grid_row, 'roi', roi_idx))
            roi_idx += 1

    for grid_row, row_type, ri in ROW_ITEMS:
        if row_type == 'audio':
            # Audio envelope row
            ax_label = fig.add_subplot(gs[grid_row, 0])
            ax_label.set_axis_off()
            ax_label.text(0.5, 0.5, 'Audio\nEnvelope', ha='center', va='center',
                          fontsize=LABEL_FS, fontweight='bold')

            for col_idx, task_key in enumerate(TASK_KEYS, start=1):
                ax = fig.add_subplot(gs[grid_row, col_idx])
                tc_axes.append(ax)

                if task_key == 'movie':
                    if len(movie_audio_epochs) > 0:
                        gmean = movie_audio_epochs.mean(axis=0)
                        gsem = movie_audio_epochs.std(axis=0) / np.sqrt(len(movie_audio_epochs))
                        ax.plot(audio_time_vec, gmean, color='k', lw=3)
                        ax.fill_between(audio_time_vec, gmean - gsem, gmean + gsem,
                                        color='k', alpha=0.2)
                else:
                    subj_audio = group_audio[task_key]
                    if subj_audio:
                        audio_labels = [s for s, _ in subj_audio]
                        data = np.array([m for _, m in subj_audio])

                        for i, (tc, label) in enumerate(zip(data, audio_labels)):
                            ci = group_labels.index(label) if label in group_labels else i
                            ax.plot(audio_time_vec, tc,
                                    color=SUBJECT_COLORS[ci % len(SUBJECT_COLORS)],
                                    lw=1, alpha=0.5, label=label)

                        gmean = data.mean(axis=0)
                        gsem = data.std(axis=0) / np.sqrt(len(data))
                        ax.plot(audio_time_vec, gmean, color='k', lw=3)
                        ax.fill_between(audio_time_vec, gmean - gsem, gmean + gsem,
                                        color='k', alpha=0.2)

                ax.axvline(0, color='grey', ls='--', lw=1)
                if task_key != 'movie' and align == 'offset':
                    ax.axvline(NEXT_TRIAL_ONSET, color='blue', ls=':', lw=1,
                               label='next trial')
                ax.axhline(0, color='k', ls='-', alpha=0.3)
                ax.set(xlabel='Time from boundary (s)', ylabel='Amplitude (z-scored)',
                       xlim=(-30, 60))
                ax.spines[['top', 'right']].set_visible(False)
                if grid_row == 0:
                    ax.set_title(TASK_DISPLAY[task_key],
                                 fontsize=TITLE_FS, fontweight='bold')
        else:
            # ROI row
            roi_key, roi_title = ROI_SPEC[ri]
            if roi_key == 'hipp':
                ax_brain = fig.add_subplot(gs[grid_row, 0])
            else:
                ax_brain = fig.add_subplot(gs[grid_row, 0], projection='3d')
            plot_roi_brain(ax_brain, roi_key, fsavg, surf_l, surf_r)
            if roi_key != 'hipp':
                ax_brain.set_title(roi_title, fontsize=LABEL_FS, pad=-10, y=0)

            for col_idx, task_key in enumerate(TASK_KEYS, start=1):
                ax = fig.add_subplot(gs[grid_row, col_idx])
                tc_axes.append(ax)

                subj_means = group_data[task_key][roi_key]
                if subj_means:
                    roi_labels = [s for s, _ in subj_means]
                    data = np.array([m for _, m in subj_means])

                    for i, (tc, label) in enumerate(zip(data, roi_labels)):
                        ci = group_labels.index(label) if label in group_labels else i
                        ax.plot(time_vec, tc, color=SUBJECT_COLORS[ci % len(SUBJECT_COLORS)],
                                lw=1, alpha=0.5, label=label)

                    gmean = data.mean(axis=0)
                    gsem = data.std(axis=0) / np.sqrt(len(data))
                    ax.plot(time_vec, gmean, color='k', lw=3)
                    ax.fill_between(time_vec, gmean - gsem, gmean + gsem,
                                    color='k', alpha=0.2)

                ax.axvline(0, color='grey', ls='--', lw=1)
                if task_key != 'movie' and align == 'offset':
                    ax.axvline(NEXT_TRIAL_ONSET, color='blue', ls=':', lw=1,
                               label='next trial')
                ax.axhline(0, color='k', ls='-', alpha=0.3)
                ax.set(xlabel='Time from boundary (s)', ylabel='BOLD (z-scored)',
                       xlim=(-30, 60))
                ax.spines[['top', 'right']].set_visible(False)
                if grid_row == 0:
                    ax.set_title(TASK_DISPLAY[task_key],
                                 fontsize=TITLE_FS, fontweight='bold')
                    if ax.get_legend_handles_labels()[1]:
                        ax.legend(loc='lower left', fontsize=7, ncol=4, framealpha=0.8)

    # Match y-axis limits for ROI axes
    yticks = np.arange(-1.5, 2.0, 0.5)
    for ax in tc_axes:
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks(yticks)


    out = OUTPUT_DIR / f"task_boundary_group_{align}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")

    # ===================================================================
    # HORIZONTAL ROI LAYOUT — all tasks overlaid per ROI
    # ===================================================================
    print(f"\n{'='*60}")
    print("  Horizontal ROI layout (tasks overlaid)")
    print(f"{'='*60}")

    TASK_COLORS = {'svf': '#1f77b4', 'ahc': '#d62728', 'movie': '#2ca02c'}
    TASK_LABELS = {'svf': 'SVF', 'ahc': 'AHC', 'movie': 'FilmFest'}

    n_rois = len(ROI_SPEC)
    n_cols = (n_rois + 1) // 2   # ceil(n_rois / 2)
    # 4 grid rows: brain_row0, tc_row0, brain_row1, tc_row1
    fig_h = plt.figure(figsize=(5.5 * n_cols, 11))
    fig_h.suptitle(f"Boundary Response by ROI ({align}-locked), N={n_subj}",
                   fontsize=TITLE_FS, fontweight='bold', y=1.01)
    gs_h = gridspec.GridSpec(
        4, n_cols,
        height_ratios=[0.45, 1, 0.45, 1],
        hspace=0.12, wspace=0.35,
        top=0.95, bottom=0.07, left=0.07, right=0.98)

    tc_axes_h = []
    for ri, (roi_key, roi_title) in enumerate(ROI_SPEC):
        grid_row_brain = (ri // n_cols) * 2       # 0 or 2
        grid_row_tc    = grid_row_brain + 1        # 1 or 3
        col            = ri % n_cols

        # Brain inset
        if roi_key == 'hipp':
            ax_brain = fig_h.add_subplot(gs_h[grid_row_brain, col])
        else:
            ax_brain = fig_h.add_subplot(gs_h[grid_row_brain, col], projection='3d')
        plot_roi_brain(ax_brain, roi_key, fsavg, surf_l, surf_r)
        if roi_key != 'hipp':
            ax_brain.set_title(roi_title, fontsize=LABEL_FS, fontweight='bold',
                               pad=-10, y=0)

        # Time course panel
        ax = fig_h.add_subplot(gs_h[grid_row_tc, col])
        tc_axes_h.append(ax)

        for task_key in TASK_KEYS:
            subj_means = group_data[task_key][roi_key]
            if not subj_means:
                continue
            data = np.array([m for _, m in subj_means])
            gmean = data.mean(axis=0)
            gsem = data.std(axis=0) / np.sqrt(len(data))
            ax.plot(time_vec, gmean,
                    color=TASK_COLORS[task_key], lw=2,
                    label=TASK_LABELS[task_key])
            ax.fill_between(time_vec, gmean - gsem, gmean + gsem,
                            color=TASK_COLORS[task_key], alpha=0.18)

        ax.axvline(0, color='grey', ls='--', lw=1)
        if align == 'offset':
            ax.axvline(NEXT_TRIAL_ONSET, color='grey', ls=':', lw=1)
        ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
        ax.set_xlim(-30, 60)
        ax.set_xlabel('Time from boundary (s)', fontsize=LABEL_FS - 1)
        ax.spines[['top', 'right']].set_visible(False)
        # y-label and legend on leftmost panel of each row
        if col == 0:
            ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FS - 1)
        if ri == 0:
            ax.legend(fontsize=LABEL_FS - 2, framealpha=0.8, loc='upper left')

    # Shared y-axis limits; hide y-ticks on non-leftmost columns
    for ax in tc_axes_h:
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks(np.arange(-1.5, 2.0, 0.5))
    for ax in tc_axes_h:
        col_pos = tc_axes_h.index(ax) % n_cols
        if col_pos != 0:
            ax.set_yticklabels([])

    out_h = OUTPUT_DIR / f"task_boundary_group_horizontal_{align}.png"
    plt.savefig(out_h, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_h}")

    # ===================================================================
    # PARCEL-WISE SURFACE T-MAP FIGURE
    # ===================================================================
    print(f"\n{'='*60}")
    print("  Parcel-wise surface t-maps")
    print(f"{'='*60}")

    # Compute t-maps for each epoch x task
    t_maps = {}
    for ti, (epoch_label, idx_start, idx_end) in enumerate(EPOCH_DEFS):
        for tj, task_key in enumerate(TASK_KEYS):
            if len(group_parcel_data[task_key]) < 2:
                continue
            data = np.array(group_parcel_data[task_key])  # (n_subj, 61, 400)
            epoch_means = data[:, idx_start:idx_end, :].mean(axis=1)  # (n_subj, 400)
            t_vals, _ = stats.ttest_1samp(epoch_means, 0, axis=0)
            t_maps[(ti, tj)] = t_vals
            print(f"  {TASK_DISPLAY[task_key]} | {epoch_label}: "
                  f"max |t| = {np.nanmax(np.abs(t_vals)):.2f}")

    if t_maps:
        all_t = np.concatenate(list(t_maps.values()))
        vmax = min(np.nanmax(np.abs(all_t)), 8.0)

        n_epochs = len(EPOCH_DEFS)
        n_tasks = len(TASK_KEYS)

        fig_surf = plt.figure(figsize=(32, 20))
        # Rows = tasks, Columns = time epochs (+ row label column)
        outer_gs = gridspec.GridSpec(
            n_tasks + 1, n_epochs + 1,
            height_ratios=[1] * n_tasks + [0.06],
            width_ratios=[0.08] + [1] * n_epochs,
            wspace=0.05, hspace=0.12, top=0.92, bottom=0.05,
            left=0.02, right=0.92)
        fig_surf.suptitle(
            f"Parcel-wise Boundary Response t-maps (N={n_subj})",
            fontsize=TITLE_FS, fontweight='bold', y=0.96)

        # Column headers (time epoch labels) on bottom
        for ti, (epoch_label, _, _) in enumerate(EPOCH_DEFS):
            ax_hdr = fig_surf.add_subplot(outer_gs[n_tasks, ti + 1])
            ax_hdr.set_axis_off()
            ax_hdr.text(0.5, 0.5, epoch_label, ha='center', va='center',
                        fontsize=LABEL_FS, fontweight='bold',
                        transform=ax_hdr.transAxes)

        for tj, task_key in enumerate(TASK_KEYS):
            # Row label (task name)
            ax_label = fig_surf.add_subplot(outer_gs[tj, 0])
            ax_label.set_axis_off()
            ax_label.text(0.5, 0.5, TASK_DISPLAY[task_key],
                          ha='center', va='center',
                          fontsize=LABEL_FS, fontweight='bold', rotation=90,
                          transform=ax_label.transAxes)

            for ti, (epoch_label, _, _) in enumerate(EPOCH_DEFS):
                if (ti, tj) not in t_maps:
                    inner = gridspec.GridSpecFromSubplotSpec(
                        2, 2, subplot_spec=outer_gs[tj, ti + 1],
                        wspace=0.01, hspace=0.01)
                    for r in range(2):
                        for c in range(2):
                            ax = fig_surf.add_subplot(inner[r, c])
                            ax.set_axis_off()
                    continue

                t_vals = t_maps[(ti, tj)]
                surf_data_l, surf_data_r = _parcel_to_surface(
                    t_vals, surf_l, surf_r)

                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 2, subplot_spec=outer_gs[tj, ti + 1],
                    wspace=0.01, hspace=0.01)

                # 2x2 layout: top row = lateral views, bottom row = medial views
                views = [
                    (0, 0, fsavg['infl_left'],  surf_data_l, 'left',  'lateral',
                     fsavg['sulc_left']),
                    (0, 1, fsavg['infl_right'], surf_data_r, 'right', 'lateral',
                     fsavg['sulc_right']),
                    (1, 0, fsavg['infl_left'],  surf_data_l, 'left',  'medial',
                     fsavg['sulc_left']),
                    (1, 1, fsavg['infl_right'], surf_data_r, 'right', 'medial',
                     fsavg['sulc_right']),
                ]

                for row, col, mesh, sdata, hemi, view, bg in views:
                    ax = fig_surf.add_subplot(
                        inner[row, col], projection='3d')
                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            'ignore', (DeprecationWarning, RuntimeWarning))
                        plot_surf(
                            mesh, surf_map=sdata, hemi=hemi, view=view,
                            bg_map=bg, axes=ax, colorbar=False,
                            cmap='RdBu_r', bg_on_data=True, darkness=0.5,
                            vmin=-vmax, vmax=vmax)

        # Colorbar
        cbar_ax = fig_surf.add_axes([0.93, 0.15, 0.015, 0.65])
        sm = plt.cm.ScalarMappable(
            cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
        sm.set_array([])
        cb = fig_surf.colorbar(sm, cax=cbar_ax)
        cb.set_label('t-statistic', fontsize=LABEL_FS)

        out_surf = OUTPUT_DIR / f"task_boundary_group_surface_tmaps_{align}.png"
        plt.savefig(out_surf, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_surf}")

        # ===============================================================
        # ANNOTATED T-MAP FIGURE (surface + top 5 parcels text)
        # ===============================================================
        print(f"\n{'='*60}")
        print("  Annotated t-map figure (top 5 parcels)")
        print(f"{'='*60}")

        def _short_label(label):
            """Shorten Schaefer label: strip '17Networks_' prefix."""
            return label.replace('17Networks_', '')

        fig_ann = plt.figure(figsize=(48, 22))
        # Each cell: surface (2x2) on left, text panel on right
        # width_ratios: row_label, then for each epoch: [surface, text]
        col_ratios = [0.04]
        for _ in range(n_epochs):
            col_ratios.extend([1, 0.45])
        ann_gs = gridspec.GridSpec(
            n_tasks + 1, 1 + 2 * n_epochs,
            height_ratios=[1] * n_tasks + [0.06],
            width_ratios=col_ratios,
            wspace=0.03, hspace=0.12, top=0.92, bottom=0.05,
            left=0.02, right=0.98)
        fig_ann.suptitle(
            f"Parcel-wise Boundary Response t-maps with Top 5 Parcels (N={n_subj})",
            fontsize=TITLE_FS, fontweight='bold', y=0.96)

        # Column headers (time epoch labels) on bottom — span surface+text
        for ti, (epoch_label, _, _) in enumerate(EPOCH_DEFS):
            ax_hdr = fig_ann.add_subplot(ann_gs[n_tasks, 1 + 2 * ti])
            ax_hdr.set_axis_off()
            ax_hdr.text(0.5, 0.5, epoch_label, ha='center', va='center',
                        fontsize=LABEL_FS, fontweight='bold',
                        transform=ax_hdr.transAxes)

        for tj, task_key in enumerate(TASK_KEYS):
            # Row label
            ax_label = fig_ann.add_subplot(ann_gs[tj, 0])
            ax_label.set_axis_off()
            ax_label.text(0.5, 0.5, TASK_DISPLAY[task_key],
                          ha='center', va='center',
                          fontsize=LABEL_FS, fontweight='bold', rotation=90,
                          transform=ax_label.transAxes)

            for ti, (epoch_label, _, _) in enumerate(EPOCH_DEFS):
                surf_col = 1 + 2 * ti
                text_col = 2 + 2 * ti

                if (ti, tj) not in t_maps:
                    # Empty surface
                    inner = gridspec.GridSpecFromSubplotSpec(
                        2, 2, subplot_spec=ann_gs[tj, surf_col],
                        wspace=0.01, hspace=0.01)
                    for r in range(2):
                        for c in range(2):
                            ax = fig_ann.add_subplot(inner[r, c])
                            ax.set_axis_off()
                    # Empty text
                    ax_txt = fig_ann.add_subplot(ann_gs[tj, text_col])
                    ax_txt.set_axis_off()
                    continue

                t_vals = t_maps[(ti, tj)]
                surf_data_l, surf_data_r = _parcel_to_surface(
                    t_vals, surf_l, surf_r)

                # Surface 2x2
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 2, subplot_spec=ann_gs[tj, surf_col],
                    wspace=0.01, hspace=0.01)
                views = [
                    (0, 0, fsavg['infl_left'],  surf_data_l, 'left',  'lateral',
                     fsavg['sulc_left']),
                    (0, 1, fsavg['infl_right'], surf_data_r, 'right', 'lateral',
                     fsavg['sulc_right']),
                    (1, 0, fsavg['infl_left'],  surf_data_l, 'left',  'medial',
                     fsavg['sulc_left']),
                    (1, 1, fsavg['infl_right'], surf_data_r, 'right', 'medial',
                     fsavg['sulc_right']),
                ]
                for row, col, mesh, sdata, hemi, view, bg in views:
                    ax = fig_ann.add_subplot(
                        inner[row, col], projection='3d')
                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            'ignore', (DeprecationWarning, RuntimeWarning))
                        plot_surf(
                            mesh, surf_map=sdata, hemi=hemi, view=view,
                            bg_map=bg, axes=ax, colorbar=False,
                            cmap='RdBu_r', bg_on_data=True, darkness=0.5,
                            vmin=-vmax, vmax=vmax)

                # Text panel: top 5 positive t-value parcels
                top5_idx = np.argsort(t_vals)[::-1][:5]
                lines = []
                for rank, idx in enumerate(top5_idx, 1):
                    label = _short_label(schaefer_labels[idx])
                    lines.append(f"{rank}. {label}\n   t = {t_vals[idx]:.2f}")
                txt = '\n'.join(lines)

                ax_txt = fig_ann.add_subplot(ann_gs[tj, text_col])
                ax_txt.set_axis_off()
                ax_txt.text(0.05, 0.95, txt, ha='left', va='top',
                            fontsize=7, family='monospace',
                            transform=ax_txt.transAxes)

        # Colorbar
        cbar_ax = fig_ann.add_axes([0.99, 0.15, 0.008, 0.65])
        sm = plt.cm.ScalarMappable(
            cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
        sm.set_array([])
        cb = fig_ann.colorbar(sm, cax=cbar_ax)
        cb.set_label('t-statistic', fontsize=LABEL_FS)

        out_ann = OUTPUT_DIR / f"task_boundary_group_surface_tmaps_annotated_{align}.png"
        plt.savefig(out_ann, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_ann}")

        # ---------------------------------------------------------------
        # Shared helper for per-task window t-map figures (with ROI outlines)
        # ---------------------------------------------------------------
        n_task_cols = len(TASK_KEYS)
        THRESH = 2.0

        def _plot_window_figure(task_tmaps, vmax_val, title, outpath, thresh=None,
                                cbar_label='t-statistic'):
            fig = plt.figure(figsize=(6 * n_task_cols, 8))
            fig.suptitle(title, fontsize=TITLE_FS, fontweight='bold', y=1.01)
            # Two rows: thin label row on top, brain views below
            gs = gridspec.GridSpec(
                2, n_task_cols,
                height_ratios=[0.07, 1],
                wspace=0.05, hspace=0.0,
                top=0.92, bottom=0.05, left=0.02, right=0.90)
            for tj, task_key in enumerate(TASK_KEYS):
                # Centered task label
                ax_lbl = fig.add_subplot(gs[0, tj])
                ax_lbl.set_axis_off()
                ax_lbl.text(0.5, 0.5, TASK_DISPLAY[task_key],
                            ha='center', va='center',
                            fontsize=LABEL_FS, fontweight='bold',
                            transform=ax_lbl.transAxes)

                cell_gs = gridspec.GridSpecFromSubplotSpec(
                    2, 2, subplot_spec=gs[1, tj], wspace=0.01, hspace=0.01)
                if task_key not in task_tmaps:
                    for r in range(2):
                        for c in range(2):
                            fig.add_subplot(cell_gs[r, c]).set_axis_off()
                    continue
                t_vals = task_tmaps[task_key].copy()
                if thresh is not None:
                    t_vals[np.abs(t_vals) < thresh] = np.nan
                surf_data_l, surf_data_r = _parcel_to_surface(t_vals, surf_l, surf_r)
                views = [
                    (0, 0, fsavg['infl_left'],  surf_data_l, 'left',  'lateral',
                     fsavg['sulc_left']),
                    (0, 1, fsavg['infl_right'], surf_data_r, 'right', 'lateral',
                     fsavg['sulc_right']),
                    (1, 0, fsavg['infl_left'],  surf_data_l, 'left',  'medial',
                     fsavg['sulc_left']),
                    (1, 1, fsavg['infl_right'], surf_data_r, 'right', 'medial',
                     fsavg['sulc_right']),
                ]
                for row, col, mesh, sdata, hemi, view, bg in views:
                    ax = fig.add_subplot(cell_gs[row, col], projection='3d')
                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            'ignore', (DeprecationWarning, RuntimeWarning))
                        plot_surf(
                            mesh, surf_map=sdata, hemi=hemi, view=view,
                            bg_map=bg, axes=ax, colorbar=False,
                            cmap='RdBu_r', bg_on_data=True, darkness=0.5,
                            vmin=-vmax_val, vmax=vmax_val)
                    # ROI outlines (PMC, dlPFC, dACC, AG)
                    edges = roi_outline_l if hemi == 'left' else roi_outline_r
                    if len(edges):
                        lc = Line3DCollection(
                            edges, colors='black', linewidths=0.3, alpha=0.8)
                        ax.add_collection3d(lc)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
            sm = plt.cm.ScalarMappable(
                cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax_val, vmax=vmax_val))
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cbar_ax)
            cb.set_label(cbar_label, fontsize=LABEL_FS)
            plt.savefig(outpath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved {outpath}")

        # ===============================================================
        # 4-13 TR WINDOW T-MAP FIGURE
        # ===============================================================
        print(f"\n{'='*60}")
        print("  4-13 TR window surface t-maps (per task)")
        print(f"{'='*60}")

        WIN_START = TRS_BEFORE + 4   # 24
        WIN_END   = TRS_BEFORE + 14  # 34 (exclusive, covers TRs 4-13 inclusive)

        task_tmaps_win = {}
        for task_key in TASK_KEYS:
            if len(group_parcel_data[task_key]) < 2:
                continue
            data = np.array(group_parcel_data[task_key])
            epoch_means = data[:, WIN_START:WIN_END, :].mean(axis=1)
            t_vals, _ = stats.ttest_1samp(epoch_means, 0, axis=0)
            task_tmaps_win[task_key] = t_vals
            print(f"  {task_key} 4-13 TR: max |t| = {np.nanmax(np.abs(t_vals)):.2f}")

        if task_tmaps_win:
            vmax_win = min(np.nanmax(np.abs(np.concatenate(
                list(task_tmaps_win.values())))), 8.0)
            _plot_window_figure(
                task_tmaps_win, vmax_win,
                f"Boundary Response t-map: 4\u201313 TR post-offset (N={n_subj})",
                OUTPUT_DIR / "task_boundary_group_surface_tmaps_4to13TR_roi.png")
            _plot_window_figure(
                task_tmaps_win, vmax_win,
                f"Boundary Response t-map: 4\u201313 TR post-offset, |t|\u22652 (N={n_subj})",
                OUTPUT_DIR / "task_boundary_group_surface_tmaps_4to13TR_thresh2_roi.png",
                thresh=THRESH)

        # ===============================================================
        # 3-12 TR WINDOW T-MAP FIGURE
        # ===============================================================
        print(f"\n{'='*60}")
        print("  3-12 TR window surface t-maps (per task)")
        print(f"{'='*60}")

        WIN2_START = TRS_BEFORE + 3   # 23
        WIN2_END   = TRS_BEFORE + 13  # 33 (exclusive, covers TRs 3-12 inclusive)

        task_tmaps_win2 = {}
        for task_key in TASK_KEYS:
            if len(group_parcel_data[task_key]) < 2:
                continue
            data = np.array(group_parcel_data[task_key])
            epoch_means = data[:, WIN2_START:WIN2_END, :].mean(axis=1)
            t_vals, _ = stats.ttest_1samp(epoch_means, 0, axis=0)
            task_tmaps_win2[task_key] = t_vals
            print(f"  {task_key} 3-12 TR: max |t| = {np.nanmax(np.abs(t_vals)):.2f}")

        if task_tmaps_win2:
            vmax_win2 = min(np.nanmax(np.abs(np.concatenate(
                list(task_tmaps_win2.values())))), 8.0)
            _plot_window_figure(
                task_tmaps_win2, vmax_win2,
                f"Boundary Response t-map: 3\u201312 TR post-offset (N={n_subj})",
                OUTPUT_DIR / "task_boundary_group_surface_tmaps_3to12TR_roi.png")
            _plot_window_figure(
                task_tmaps_win2, vmax_win2,
                f"Boundary Response t-map: 3\u201312 TR post-offset, |t|\u22652 (N={n_subj})",
                OUTPUT_DIR / "task_boundary_group_surface_tmaps_3to12TR_thresh2_roi.png",
                thresh=THRESH)

        # ===============================================================
        # 0-9 TR WINDOW T-MAP FIGURE
        # ===============================================================
        print(f"\n{'='*60}")
        print("  0-9 TR window surface t-maps (per task)")
        print(f"{'='*60}")

        WIN3_START = TRS_BEFORE + 0   # 20
        WIN3_END   = TRS_BEFORE + 10  # 30 (exclusive, covers TRs 0-9 inclusive)

        task_tmaps_win3 = {}
        for task_key in TASK_KEYS:
            if len(group_parcel_data[task_key]) < 2:
                continue
            data = np.array(group_parcel_data[task_key])
            epoch_means = data[:, WIN3_START:WIN3_END, :].mean(axis=1)
            t_vals, _ = stats.ttest_1samp(epoch_means, 0, axis=0)
            task_tmaps_win3[task_key] = t_vals
            print(f"  {task_key} 0-9 TR: max |t| = {np.nanmax(np.abs(t_vals)):.2f}")

        if task_tmaps_win3:
            vmax_win3 = min(np.nanmax(np.abs(np.concatenate(
                list(task_tmaps_win3.values())))), 8.0)
            _plot_window_figure(
                task_tmaps_win3, vmax_win3,
                f"Boundary Response t-map: 0\u20139 TR post-offset (N={n_subj})",
                OUTPUT_DIR / "task_boundary_group_surface_tmaps_0to9TR_roi.png")
            _plot_window_figure(
                task_tmaps_win3, vmax_win3,
                f"Boundary Response t-map: 0\u20139 TR post-offset, |t|\u22652 (N={n_subj})",
                OUTPUT_DIR / "task_boundary_group_surface_tmaps_0to9TR_thresh2_roi.png",
                thresh=THRESH)

        # ===============================================================
        # Z-SCORE SURFACE MAPS (group-mean BOLD z, ±0.6 limit, ±0.2 threshold)
        # ===============================================================
        print(f"\n{'='*60}")
        print("  Z-score surface maps (per window, per task)")
        print(f"{'='*60}")

        ZMAP_VMAX  = 0.6
        ZMAP_THRESH = 0.3

        window_defs = [
            ('0to9TR',   TRS_BEFORE + 0,  TRS_BEFORE + 10),
            ('3to12TR',  TRS_BEFORE + 3,  TRS_BEFORE + 13),
            ('4to13TR',  TRS_BEFORE + 4,  TRS_BEFORE + 14),
        ]

        for win_name, ws, we in window_defs:
            print(f"  {win_name}...")
            task_zmaps = {}
            for task_key in TASK_KEYS:
                if not group_parcel_data[task_key]:
                    continue
                data = np.array(group_parcel_data[task_key])  # (n_subj, 61, 400)
                # Mean over time window then over subjects
                group_mean = data[:, ws:we, :].mean(axis=1).mean(axis=0)  # (400,)
                task_zmaps[task_key] = group_mean
                print(f"    {task_key}: max |z| = {np.nanmax(np.abs(group_mean)):.3f}")

            if not task_zmaps:
                continue

            _plot_window_figure(
                task_zmaps, ZMAP_VMAX,
                f"Boundary Response z-score: {win_name.replace('TR', ' TR')} post-offset (N={n_subj})",
                OUTPUT_DIR / f"task_boundary_group_surface_zmaps_{win_name}_roi.png",
                cbar_label='BOLD (z-score)')
            _plot_window_figure(
                task_zmaps, ZMAP_VMAX,
                f"Boundary Response z-score: {win_name.replace('TR', ' TR')} post-offset, |z|\u2265{ZMAP_THRESH} (N={n_subj})",
                OUTPUT_DIR / f"task_boundary_group_surface_zmaps_{win_name}_thresh_roi.png",
                thresh=ZMAP_THRESH,
                cbar_label='BOLD (z-score)')

        # ===============================================================
        # T-MAP CONJUNCTION — 0-9 TR window (requires task_tmaps_win3)
        # ===============================================================
        print(f"\n{'='*60}")
        print("  T-map conjunction (0-9 TR, all three tasks)")
        print(f"{'='*60}")

        def _plot_conj_figure(conj_map, vmin_val, vmax_val, title, outpath, cbar_label):
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(title, fontsize=TITLE_FS, fontweight='bold', y=1.01)
            surf_data_l, surf_data_r = _parcel_to_surface(conj_map, surf_l, surf_r)
            inner_gs = gridspec.GridSpec(
                2, 2, wspace=0.01, hspace=0.01,
                top=0.92, bottom=0.05, left=0.02, right=0.90)
            views = [
                (0, 0, fsavg['infl_left'],  surf_data_l, 'left',  'lateral',
                 fsavg['sulc_left']),
                (0, 1, fsavg['infl_right'], surf_data_r, 'right', 'lateral',
                 fsavg['sulc_right']),
                (1, 0, fsavg['infl_left'],  surf_data_l, 'left',  'medial',
                 fsavg['sulc_left']),
                (1, 1, fsavg['infl_right'], surf_data_r, 'right', 'medial',
                 fsavg['sulc_right']),
            ]
            for row, col, mesh, sdata, hemi, view, bg in views:
                ax = fig.add_subplot(inner_gs[row, col], projection='3d')
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        'ignore', (DeprecationWarning, RuntimeWarning))
                    plot_surf(
                        mesh, surf_map=sdata, hemi=hemi, view=view,
                        bg_map=bg, axes=ax, colorbar=False,
                        cmap='Reds', bg_on_data=True, darkness=0.5,
                        vmin=vmin_val, vmax=vmax_val)
                edges = roi_outline_l if hemi == 'left' else roi_outline_r
                if len(edges):
                    lc = Line3DCollection(
                        edges, colors='black', linewidths=0.3, alpha=0.8)
                    ax.add_collection3d(lc)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
            sm = plt.cm.ScalarMappable(
                cmap='Reds', norm=plt.Normalize(vmin=vmin_val, vmax=vmax_val))
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cbar_ax)
            cb.set_label(cbar_label, fontsize=LABEL_FS)
            plt.savefig(outpath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved {outpath}")

        # T-map conjunction: t > 2 in all three tasks → show minimum t across tasks
        if (task_tmaps_win3 and
                all(k in task_tmaps_win3 for k in TASK_KEYS)):
            all_t = np.array([task_tmaps_win3[k] for k in TASK_KEYS])
            conj_mask_t = np.all(all_t > 2.0, axis=0)
            conj_tmap = np.where(conj_mask_t, all_t.min(axis=0), np.nan)
            n_conj_t = int(conj_mask_t.sum())
            print(f"  T-map conjunction: {n_conj_t} parcels with t > 2 in all tasks")
            vmax_conj_t = min(float(np.nanmax(conj_tmap)) if n_conj_t else 5.0, 8.0)
            _plot_conj_figure(
                conj_tmap, 2.0, vmax_conj_t,
                f"Conjunction t-map: 0\u20139 TR, t > 2 in all tasks (N={n_subj})",
                OUTPUT_DIR / f"task_boundary_group_conjunction_tmap_0to9TR_align-{align}.png",
                cbar_label='min t-statistic across tasks')



if __name__ == '__main__':
    main()
