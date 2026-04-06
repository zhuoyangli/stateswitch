"""
Utility functions for fMRI analysis
"""
import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd
from nilearn import datasets
from nilearn.surface import load_surf_data
from nilearn.maskers import NiftiLabelsMasker

# === CONFIG SETUP ===
try:
    from configs.config import DATA_DIR, CACHE_DIR, DERIVATIVES_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR, MOVIE_INFO
except ImportError:
    print("Error: Could not import 'configs.config'. Ensure your directory structure is correct.")
    sys.exit(1)

HIGH_PASS_HZ = 0.01

from scipy.signal import butter, filtfilt

PSYCHOPY_DIR = DATA_DIR / 'psychopy'
ANNOTATIONS_DIR = DATA_DIR / 'filmfest_annotations'

_atlas_resampled_cache = {}  # bold_path_str -> resampled atlas array

## Localizers functions
def generate_events_dataframe(task, run_num):
    """Generate events for specified task and run"""
    if task == 'langloc':
        contrast = 'intact - degraded'
        conditions_run1 = [[1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 1, 0]]
        
        conditions_run2 = [[0, 1, 1, 0],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 0, 1]]
        
        conditions = conditions_run1 if run_num == 1 else conditions_run2
        
        events = []
        current_time = 15.0  # Skip initial fixation
        
        for loop_conditions in conditions:
            for is_intact in loop_conditions:
                condition = 'intact' if is_intact else 'degraded'
                events.append({
                    'onset': current_time,
                    'duration': 18.0,
                    'trial_type': condition
                })
                current_time += 18.0
            current_time += 15.0  # Inter-loop fixation

        return pd.DataFrame(events), contrast

    elif task == 'mdloc':
        contrast = 'hard - easy'
        conditions_run1 = [[1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 1, 0]]
        
        conditions_run2 = [[0, 1, 1, 0],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 0, 1]]
        
        conditions = conditions_run1 if run_num == 1 else conditions_run2
        
        events = []
        current_time = 15.0 # Skip initial fixation
        
        for iloop, loop_conditions in enumerate(conditions):
            for itrial, is_hard in enumerate(loop_conditions):
                condition = 'hard' if is_hard else 'easy'
                
                duration = 9.0
                events.append({
                    'onset': current_time,
                    'duration': duration,
                    'trial_type': condition
                })
                current_time += duration
            current_time += 15.0 # Inter-loop fixation
        
        return pd.DataFrame(events), contrast
    
    elif task == 'tomloc':
        contrast = 'belief - photo'
        conditions_run1 = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
        conditions_run2 = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]

        conditions = conditions_run1 if run_num == 1 else conditions_run2

        events = []
        current_time = 12.0 # Skip initial fixation

        for is_belief in conditions:
            condition = 'belief' if is_belief else 'photo'
            events.append({
                'onset': current_time,
                'duration': 16.5,
                'trial_type': condition
            })
            current_time += 16.5
            current_time += 12 # Response + Inter-trial fixation

        return pd.DataFrame(events), contrast

def load_surface_data(subject, session, task, hemi, data_dir, fsaverage='fsaverage6'):
    """
    Load fMRIPrep surface data for a subject
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-001')
    session : str
        Session ID (e.g., 'ses-01')
    task : str
        Task name (e.g., 'rest')
    hemi : str
        Hemisphere ('L' or 'R')
    data_dir : str or Path
        Path to fMRIPrep derivatives
        
    Returns
    -------
    data : numpy array
        Time series data (n_timepoints, n_vertices)
    """
    filename = f"{subject}_{session}_task-{task}_hemi-{hemi}_space-{fsaverage}_bold.func.gii"
    filepath = Path(data_dir) / subject / session / "func" / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = load_surf_data(filepath)
    
    return data

def load_surface_image(subject, session, task, data_dir, fsaverage='fsaverage6'):
    """
    Load fMRIPrep surface data and create a SurfaceImage object
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-001')
    session : str
        Session ID (e.g., 'ses-01')
    task : str
        Task name (e.g., 'rest')
    data_dir : str or Path
        Path to fMRIPrep derivatives
    fsaverage : str
        Name of the fsaverage template (default: 'fsaverage6')
        
    Returns
    -------
    surface_image : SurfaceImage
        Nilearn SurfaceImage object containing left and right hemisphere data
    """
    from nilearn import datasets
    from nilearn.surface import SurfaceImage
    
    # Load left hemisphere data
    left_data = load_surface_data(subject, session, task, 'L', data_dir, fsaverage=fsaverage)
    
    # Load right hemisphere data
    right_data = load_surface_data(subject, session, task, 'R', data_dir, fsaverage=fsaverage)

    # Fetch fsaverage surfaces
    fsavg = datasets.fetch_surf_fsaverage(fsaverage)
    
    # Create SurfaceImage
    surface_image = SurfaceImage(
        mesh={'left': fsavg.infl_left, 'right': fsavg.infl_right},
        data={'left': left_data, 'right': right_data}
    )
    
    return surface_image

def get_parcel_data(
    subject: str,
    session: str,
    task: str,
    atlas: str = 'Schaefer400_17Nets',
    highpass: float = 0.01,
    zscore: bool = True,
    cache_dir: Path = None,
) -> dict:
    """
    Load and cache atlas parcel data.

    Parameters
    ----------
    subject, session, task : str
    atlas : str
        Atlas label: 'Schaefer400_17Nets', 'HarvardOxford_sub', 'HarvardOxford_cort'
    highpass : float or None
        High-pass cutoff in Hz (None = no filtering)
    zscore : bool
        Whether to z-score each parcel's time series
    cache_dir : Path or None
        Override cache directory; defaults to CACHE_DIR / 'parcels'
    """
    parcels_dir = (cache_dir if cache_dir is not None else CACHE_DIR / 'parcels')
    parcels_dir.mkdir(parents=True, exist_ok=True)

    nilearn_dir = CACHE_DIR / 'nilearn'
    nilearn_dir.mkdir(parents=True, exist_ok=True)

    hp_str = f"{highpass}" if highpass is not None else "None"
    z_str = str(zscore)
    cache_file = parcels_dir / f"{subject}_{session}_task-{task}_atlas-{atlas}_hp-{hp_str}_zscore-{z_str}.npz"

    if cache_file.exists():
        print(f"Loading cached parcel data from: {cache_file.name}...")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            return loaded['parcel_data'].item()
        except Exception as e:
            print(f"Cache load failed ({e}), re-computing...")

    if atlas == 'Schaefer400_17Nets':
        atlas_obj = datasets.fetch_atlas_schaefer_2018(
            n_rois=400, yeo_networks=17, resolution_mm=2
        )
    elif atlas == 'HarvardOxford_sub':
        atlas_obj = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    elif atlas == 'HarvardOxford_cort':
        atlas_obj = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    else:
        raise ValueError(f"Unknown atlas: {atlas}")

    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas_obj['labels']]
    labels = [l for l in all_labels if l != 'Background']

    standardize = 'zscore_sample' if zscore else False

    masker = NiftiLabelsMasker(
        labels_img=atlas_obj['maps'],
        labels=all_labels,
        standardize=standardize,
        high_pass=highpass,
        t_r=TR,
        verbose=0,
        memory=str(nilearn_dir),
    )

    bold_path = (DERIVATIVES_DIR / subject / session / "func" /
                 f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")

    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD file not found: {bold_path}")

    print(f"Extracting signal for {subject} {session} using {atlas}...")

    data = masker.fit_transform(bold_path)

    parcel_data = {label: d for label, d in zip(labels, data.T)}

    np.savez_compressed(cache_file, parcel_data=parcel_data)

    return parcel_data


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def highpass_filter(data, cutoff=0.01, tr=TR, order=5):
    """Zero-phase Butterworth high-pass filter along axis 0 (time).

    Parameters
    ----------
    data   : np.ndarray, shape (T, ...)
    cutoff : float — cutoff frequency in Hz (default 0.01 Hz)
    tr     : float — repetition time in seconds
    order  : int   — filter order (default 5)
    """
    nyq = 0.5 / tr
    b, a = butter(order, cutoff / nyq, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)


# ============================================================================
# BOLD PATH HELPERS
# ============================================================================

def get_bold_path(subject, session, task):
    """Return path to preprocessed BOLD NIfTI file."""
    return (DERIVATIVES_DIR / subject / session / 'func' /
            f'{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')


def get_atlas_data(bold_path, atlas_maps):
    """Return atlas resampled to BOLD voxel space (cached by bold directory).

    Parameters
    ----------
    bold_path  : Path — path to BOLD file (used for resampling reference)
    atlas_maps : NIfTI image — atlas parcellation maps

    Returns
    -------
    atlas_data : np.ndarray, shape (X, Y, Z), dtype int
    """
    from nilearn import image
    key = str(bold_path)
    if key not in _atlas_resampled_cache:
        bold_ref = image.index_img(str(bold_path), 0)
        atlas_resampled = image.resample_to_img(
            atlas_maps, bold_ref, interpolation='nearest'
        )
        _atlas_resampled_cache[key] = np.round(atlas_resampled.get_fdata()).astype(int)
        print(f"  Atlas resampled to BOLD space: {_atlas_resampled_cache[key].shape}")
    return _atlas_resampled_cache[key]


# ============================================================================
# TIME CONVERSION
# ============================================================================

def mss_to_seconds(mss):
    """Convert m.ss timestamp to seconds.

    Format: integer part = minutes, decimal part * 100 = seconds.
    E.g. 6.30 -> 6 min 30 sec -> 390 sec.
    """
    minutes = int(mss)
    seconds = round((mss - minutes) * 100)
    return minutes * 60 + seconds


# ============================================================================
# PSYCHOPY / TRIAL TIMING
# ============================================================================

def find_psychopy_csv(subject, session, task):
    """Find the psychopy CSV containing data for a given task.

    If multiple CSVs exist (e.g. program crashed and resumed), return the one
    that has the relevant task columns.

    Returns path or None.
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


def get_trial_times(subject, session, task):
    """Return scan-relative (onsets, offsets) in seconds for all trials.

    Returns
    -------
    onsets, offsets : np.ndarray — empty arrays if unavailable
    """
    csv_path = find_psychopy_csv(subject, session, task)
    if csv_path is None:
        return np.array([]), np.array([])
    df = pd.read_csv(csv_path)
    started_col = f'{task}_trial.started'
    stopped_col = f'{task}_trial.stopped'
    if stopped_col not in df.columns:
        return np.array([]), np.array([])
    mask = df[stopped_col].notna()
    if not mask.any():
        return np.array([]), np.array([])
    rows = df[mask]
    first_start = rows[started_col].iloc[0]
    onsets = rows[started_col].values - first_start
    offsets = rows[stopped_col].values - first_start
    return onsets, offsets


def discover_svf_ahc_sessions(subject):
    """Return (session, task) pairs with both BOLD and psychopy data."""
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
            if bold and find_psychopy_csv(subject, session, task) is not None:
                hits.append((session, task))
    return hits


# ============================================================================
# FILMFEST ANNOTATIONS
# ============================================================================

def get_movie_boundary_offsets(task):
    """Return movie-end times (seconds) between consecutive movies for filmfest task.

    These are the between-movie transitions (offset of movie n = onset of movie n+1).
    The last movie has no subsequent boundary and is excluded.
    """
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    boundaries = []
    for movie in movies[:-1]:
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        last_end = segb['End Time (m.ss)'].values[-1]
        boundaries.append(mss_to_seconds(last_end))
    return boundaries


# ============================================================================
# CORRELATION UTILITIES
# ============================================================================

def cross_corrcoef(A, B):
    """Pearson r between every row of A and every row of B.

    Parameters
    ----------
    A, B : np.ndarray, shape (T, n_verts)

    Returns
    -------
    corr : np.ndarray, shape (T, T)
        corr[i, j] = Pearson r between A[i] and B[j]
    """
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    A = np.where(A_norm == 0, 0.0, A / np.where(A_norm == 0, 1.0, A_norm))
    B = np.where(B_norm == 0, 0.0, B / np.where(B_norm == 0, 1.0, B_norm))
    return A @ B.T


def group_average(maps):
    """Fisher z-transform before averaging, back-transform after.

    Parameters
    ----------
    maps : list of np.ndarray, each shape (T_i, T_i)

    Returns
    -------
    avg : np.ndarray, shape (T_min, T_min)
    """
    T = min(m.shape[0] for m in maps)
    z = np.arctanh(
        np.clip(np.stack([m[:T, :T] for m in maps], axis=0), -0.999, 0.999)
    )
    return np.tanh(z.mean(axis=0))


# ============================================================================
# EVENT-LOCKED EXTRACTION
# ============================================================================

def extract_event_locked(signal, event_tp, trs_before, trs_after,
                         return_epochs=False, return_sem=False):
    """Extract event-locked epochs from a 1D signal.

    Parameters
    ----------
    signal       : np.ndarray, shape (T,)
    event_tp     : array-like — event times in seconds
    trs_before   : int — TRs before event to include
    trs_after    : int — TRs after event to include
    return_epochs : bool — if True, return raw epochs array instead of mean
    return_sem    : bool — if True, also return SEM (only when return_epochs=False)

    Returns
    -------
    If return_epochs=True  : np.ndarray, shape (n_valid_events, n_timepoints) or None
    If return_sem=True     : (mean_tc, sem_tc) tuple or (None, None)
    Otherwise              : mean_tc np.ndarray or None
    """
    n = len(signal)
    event_tp = np.asarray(event_tp)
    if len(event_tp) == 0:
        if return_epochs:
            return None
        return (None, None) if return_sem else None

    centers = np.round(event_tp / TR).astype(int)
    offsets = np.arange(-trs_before, trs_after + 1)
    idx = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n), axis=1)

    if not valid.any():
        if return_epochs:
            return None
        return (None, None) if return_sem else None

    epochs = signal[idx[valid]]
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)

    if return_epochs:
        return epochs

    mean_tc = epochs.mean(axis=0)
    if return_sem:
        sem_tc = epochs.std(axis=0) / np.sqrt(len(epochs))
        return mean_tc, sem_tc
    return mean_tc