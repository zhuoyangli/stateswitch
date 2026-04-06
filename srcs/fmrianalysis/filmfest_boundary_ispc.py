"""
filmfest_boundary_ispc.py — Filmfest Inter-Subject Pattern Correlation

For each of 8 movie-onset boundaries (movies 2–5 and 7–10; first movie of each
run excluded), extracts 5 time-window activation patterns per ROI. Computes
pairwise LOO inter-subject pattern correlation across all 40 conditions
(8 movies × 5 windows), producing a 40×40 ISPC matrix per ROI.

Condition ordering (movies nested in windows):
  [M2_pre, M3_pre, ..., M5_pre, M7_pre, ..., M10_pre,
   M2_onset, ..., M10_onset, ..., M2_late40, ..., M10_late40]

Preprocessing pipeline (per voxel time series):
  1. Spatial smoothing (FWHM=6mm) — applied to full BOLD image, results cached
  2. Linear detrend
  3. High-pass filter at 0.01 Hz (off by default, --hp flag)
  4. Z-score per voxel

Level-1 cache (reusable across analyses):
  ANALYSIS_CACHE_DIR/roi_voxels/{sub}_{ses}_task-{task}_roi-{roi}_sm6.npz

Usage:
    uv run python srcs/fmrianalysis/filmfest_ispc.py
    uv run python srcs/fmrianalysis/filmfest_ispc.py --hp
    uv run python srcs/fmrianalysis/filmfest_ispc.py --no-cache
    uv run python srcs/fmrianalysis/filmfest_ispc.py --n_jobs 4
    uv run python srcs/fmrianalysis/filmfest_ispc.py --roi pmc ag eac
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal as sp_signal
from scipy.stats import zscore as sp_zscore
from nilearn import datasets, image
from nilearn.maskers import NiftiMasker
from joblib import Parallel, delayed

from configs.config import (
    DATA_DIR, DERIVATIVES_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR,
    FILMFEST_SUBJECTS, MOVIE_INFO,
)
from configs.schaefer_rois import (
    EARLY_AUDITORY, EARLY_VISUAL, POSTERIOR_MEDIAL, ANGULAR_GYRUS,
    DLPFC, DACC, get_bilateral_ids,
)
from fmrianalysis.utils import highpass_filter, get_bold_path, cross_corrcoef

# ============================================================================
# CONSTANTS
# ============================================================================

ANNOTATIONS_DIR = DATA_DIR / 'filmfest_annotations'
VOXEL_CACHE_DIR = ANALYSIS_CACHE_DIR / 'roi_voxels'
OUTPUT_DIR      = FIGS_DIR / 'filmfest_ispc' / 'default'  # reassigned in main
RESULT_CACHE_DIR = ANALYSIS_CACHE_DIR / 'filmfest_ispc'

SMOOTH_FWHM = 6.0
HP_CUTOFF   = 0.01

# Movies 2–5 (run 1) and 7–10 (run 2); first movie of each run excluded
N_MOVIES = 8

# Window presets (inclusive TR ranges relative to movie onset)
WINDOW_PRESETS = {
    'default': {
        'windows': [
            ('pre',   -5, -1),
            ('onset',  0,  4),
            ('post',   5,  9),
            ('late',  10, 14),
            ('late2', 15, 19),
        ],
        'labels': [
            'pre-boundary\n-5 to -1 TRs',
            'post-boundary\n0 to 4 TRs',
            'post-boundary\n5 to 9 TRs',
            'post-boundary\n10 to 14 TRs',
            'post-boundary\n15 to 19 TRs',
        ],
        'tag': '',
    },
    'hrf': {
        'windows': [
            ('pre',    -6,  3),
            ('onset',   4, 13),
            ('post',   14, 23),
            ('late',   24, 33),
        ],
        'labels': [
            'pre-boundary\n-6 to 3 TRs',
            'post-boundary\n4 to 13 TRs',
            'post-boundary\n14 to 23 TRs',
            'post-boundary\n24 to 33 TRs',
        ],
        'tag': '_hrf',
    },
    'post4': {
        'windows': [
            ('onset',   4, 13),
            ('post',   14, 23),
            ('late',   24, 33),
            ('vlate',  34, 43),
        ],
        'labels': [
            'post-boundary\n4 to 13 TRs',
            'post-boundary\n14 to 23 TRs',
            'post-boundary\n24 to 33 TRs',
            'post-boundary\n34 to 43 TRs',
        ],
        'tag': '_post4',
    },
}

# Active window set — reassigned by apply_window_preset() before any processing
WINDOWS      = WINDOW_PRESETS['default']['windows']
N_WINDOWS    = len(WINDOWS)
N_CONDITIONS = N_MOVIES * N_WINDOWS
WIN_TAG      = ''


def apply_window_preset(name):
    """Update module-level window globals to the named preset ('default' or 'hrf')."""
    global WINDOWS, N_WINDOWS, N_CONDITIONS, _WINDOW_LABELS, WIN_TAG
    preset       = WINDOW_PRESETS[name]
    WINDOWS      = preset['windows']
    N_WINDOWS    = len(WINDOWS)
    N_CONDITIONS = N_MOVIES * N_WINDOWS
    _WINDOW_LABELS = preset['labels']
    WIN_TAG      = preset['tag']

ROI_SPEC = [
    ('eac',   'EAC',         get_bilateral_ids(EARLY_AUDITORY)),
    ('evc',   'EVC',         get_bilateral_ids(EARLY_VISUAL)),
    ('pmc',   'PMC',         get_bilateral_ids(POSTERIOR_MEDIAL)),
    ('ag',    'AG',          get_bilateral_ids(ANGULAR_GYRUS)),
    ('hipp',  'Hippocampus', None),  # None = use H-O mask
    ('dlpfc', 'dlPFC',       get_bilateral_ids(DLPFC)),
    ('dacc',  'dACC',        get_bilateral_ids(DACC)),
]

# ============================================================================
# ATLAS SETUP (module level, loaded once)
# ============================================================================

print("Loading atlases...")
SCHAEFER = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)

_ho         = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
_ho_img     = nib.load(_ho['maps']) if isinstance(_ho['maps'], (str, Path)) else _ho['maps']
_ho_labels  = _ho['labels']
_hipp_ids   = [i for i, l in enumerate(_ho_labels) if 'hippocampus' in l.lower()]
_hipp_data  = np.isin(_ho_img.get_fdata().astype(int), _hipp_ids).astype(np.int8)
HIPP_MASK_IMG = nib.Nifti1Image(_hipp_data, _ho_img.affine, _ho_img.header)
print(f"  Hippocampus mask: {_hipp_data.sum()} voxels.")

_schaefer_resampled = {}  # bold_dir -> (X,Y,Z) int array


def _get_schaefer_atlas(bold_path):
    """Return Schaefer atlas resampled to BOLD voxel space (cached by directory)."""
    key = str(bold_path.parent)
    if key not in _schaefer_resampled:
        bold_ref = image.index_img(str(bold_path), 0)
        atlas_img = image.resample_to_img(SCHAEFER['maps'], bold_ref, interpolation='nearest')
        _schaefer_resampled[key] = np.round(atlas_img.get_fdata()).astype(int)
    return _schaefer_resampled[key]


# ============================================================================
# STEP 1: MOVIE ONSETS
# ============================================================================

def get_all_movie_onsets(task):
    """Return start times (seconds) for all 5 movies in one filmfest run.

    Uses the start time of the first SEG-B segment as the movie onset.
    """
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    onsets = []
    for movie in movies:
        df   = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        first_start_raw = segb['Start Time (m.ss)'].values[0]
        onsets.append(_mss_to_seconds(first_start_raw))
    return onsets  # list of 5 floats


def _mss_to_seconds(mss):
    """Convert m.ss timestamp to total seconds."""
    minutes = int(mss)
    seconds = round((float(mss) - minutes) * 100)
    return minutes * 60 + seconds


# ============================================================================
# STEP 2: VOXEL EXTRACTION & CACHING
# ============================================================================

def load_roi_voxels(subject, session, task, roi_key, parcel_ids, fwhm=SMOOTH_FWHM,
                    force=False):
    """Load smoothed ROI voxel time series, with Level-1 caching.

    Returns np.ndarray shape (T, V), float32 (raw smoothed, no temporal preprocessing).

    Cache: ANALYSIS_CACHE_DIR/roi_voxels/{sub}_{ses}_task-{task}_roi-{roi}_sm{fwhm}.npz
    """
    VOXEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sm_tag    = f'sm{int(fwhm)}'
    cache_path = VOXEL_CACHE_DIR / f'{subject}_{session}_task-{task}_roi-{roi_key}_{sm_tag}.npz'

    if not force and cache_path.exists():
        return np.load(cache_path)['voxels']

    bold_path = get_bold_path(subject, session, task)
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD not found: {bold_path}")

    print(f"  Smoothing {bold_path.name} (fwhm={fwhm}mm) ...")
    bold_img    = nib.load(str(bold_path))
    bold_smooth = image.smooth_img(bold_img, fwhm=fwhm)

    if roi_key == 'hipp':
        masker   = NiftiMasker(mask_img=HIPP_MASK_IMG, standardize=False, verbose=0)
        voxels   = masker.fit_transform(bold_smooth).astype(np.float32)  # (T, V)
    else:
        bold_data  = bold_smooth.get_fdata(dtype=np.float32)  # (X, Y, Z, T)
        atlas_data = _get_schaefer_atlas(bold_path)            # (X, Y, Z)
        parts = []
        for pid in parcel_ids:
            mask = atlas_data == pid
            if mask.sum() < 2:
                continue
            parts.append(bold_data[mask].T)  # (T, n_vox)
        del bold_data
        if not parts:
            raise ValueError(f"No usable parcels: {subject} {task} roi={roi_key}")
        voxels = np.concatenate(parts, axis=1).astype(np.float32)

    del bold_smooth
    np.savez_compressed(cache_path, voxels=voxels)
    print(f"  Cached → {cache_path.name}  shape={voxels.shape}")
    return voxels


def preprocess_voxels(voxels, do_hp=False):
    """Apply temporal preprocessing to (T, V) float array.

    Steps: linear detrend → optional HP filter → z-score per voxel.
    Returns float64 array.
    """
    data = sp_signal.detrend(voxels.astype(np.float64), axis=0)
    if do_hp:
        data = highpass_filter(data, cutoff=HP_CUTOFF, tr=TR)
    data = sp_zscore(data, axis=0, nan_policy='omit')
    return np.nan_to_num(data)


# ============================================================================
# STEP 3: BOUNDARY-LOCKED PATTERNS
# ============================================================================

def extract_boundary_patterns(voxel_data, boundary_trs):
    """Extract 4 time-window patterns around each boundary.

    Parameters
    ----------
    voxel_data   : (T, V) preprocessed voxel array
    boundary_trs : list of int — boundary TR indices (0-based)

    Returns
    -------
    patterns : (N_valid_boundaries, N_WINDOWS, V) array
    valid_mask : bool array length len(boundary_trs) — True if boundary was valid
    """
    T = voxel_data.shape[0]
    valid_patterns = []
    valid_mask = []

    for btr in boundary_trs:
        win_patterns = []
        ok = True
        for _, tr_start_rel, tr_end_rel in WINDOWS:
            start = btr + tr_start_rel
            end   = btr + tr_end_rel
            if start < 0 or end >= T:
                ok = False
                break
            win_patterns.append(voxel_data[start:end + 1].mean(axis=0))  # (V,)
        valid_mask.append(ok)
        if ok:
            valid_patterns.append(np.stack(win_patterns, axis=0))  # (4, V)

    if not valid_patterns:
        return np.empty((0, N_WINDOWS, voxel_data.shape[1])), np.array(valid_mask)

    return np.stack(valid_patterns, axis=0), np.array(valid_mask)


def get_subject_patterns(subject, roi_key, parcel_ids, do_hp=False, force_voxels=False):
    """Load and process patterns for one subject, both filmfest runs.

    Returns
    -------
    patterns : (10, 4, V) or None if data unavailable
    valid_mask : (10,) bool array — which of the 10 boundaries were valid
    """
    session = FILMFEST_SUBJECTS[subject]
    all_patterns = []
    all_valid    = []

    # 4 boundaries per run (movies 2–5 and 7–10; first excluded)
    N_PER_RUN = 4
    for task in ('filmfest1', 'filmfest2'):
        try:
            raw = load_roi_voxels(subject, session, task, roi_key,
                                  parcel_ids, force=force_voxels)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {task} roi={roi_key}: {e}")
            all_patterns.append(None)
            all_valid.extend([False] * N_PER_RUN)
            continue

        voxel_data    = preprocess_voxels(raw, do_hp=do_hp)
        onset_secs    = get_all_movie_onsets(task)[1:]  # skip first movie of each run
        boundary_trs  = [int(round(s / TR)) for s in onset_secs]
        pats, vmask   = extract_boundary_patterns(voxel_data, boundary_trs)
        all_patterns.append(pats)
        all_valid.extend(vmask.tolist())

    # Concatenate across runs (4 + 4 = 8 boundaries)
    valid_parts = [p for p in all_patterns if p is not None and p.shape[0] > 0]
    if not valid_parts:
        return None, np.array(all_valid)

    V = valid_parts[0].shape[2]

    # Build (8, N_WINDOWS, V) array, filling invalid slots with NaN
    full_patterns = np.full((N_MOVIES, N_WINDOWS, V), np.nan, dtype=np.float64)
    run_valid_idx = 0
    for task_idx, (task, pats) in enumerate(zip(('filmfest1', 'filmfest2'), all_patterns)):
        if pats is None or pats.shape[0] == 0:
            run_valid_idx += N_PER_RUN
            continue
        valid_flags = all_valid[run_valid_idx:run_valid_idx + N_PER_RUN]
        valid_count = 0
        for bi, is_valid in enumerate(valid_flags):
            movie_idx = task_idx * N_PER_RUN + bi
            if is_valid and valid_count < pats.shape[0]:
                full_patterns[movie_idx] = pats[valid_count]
                valid_count += 1
        run_valid_idx += N_PER_RUN

    return full_patterns, np.array(all_valid)


# ============================================================================
# STEP 4: LOO INTER-SUBJECT PATTERN CORRELATION
# ============================================================================

def flatten_patterns(patterns):
    """Reshape (8, 5, V) → (40, V) in movies-nested-in-windows order.

    Output order: [M2_W0..M10_W0, M2_W1..M10_W1, ..., M2_W4..M10_W4]
    i.e., all 8 movies for each window block in sequence.
    """
    # patterns: (N_movies, N_windows, V)
    return patterns.transpose(1, 0, 2).reshape(N_CONDITIONS, -1)


def compute_group_ispc(all_patterns):
    """Compute LOO ISPC matrix averaged across subjects.

    Parameters
    ----------
    all_patterns : (N_subj, 40, V) array — NaN indicates missing condition

    Returns
    -------
    group_ispc    : (40, 40) Fisher-z averaged correlation matrix
    per_subj_ispc : (N_subj, 40, 40) per-subject correlation matrices
    """
    N = all_patterns.shape[0]
    group_sum = all_patterns.sum(axis=0)  # (40, V)

    per_subj_ispc = np.full((N, N_CONDITIONS, N_CONDITIONS), np.nan)

    for i in range(N):
        loo_mean = (group_sum - all_patterns[i]) / (N - 1)  # (40, V)

        # Compute correlation row by row to handle NaN conditions
        corr_mat = np.full((N_CONDITIONS, N_CONDITIONS), np.nan)
        for row in range(N_CONDITIONS):
            if np.all(np.isnan(all_patterns[i, row])):
                continue
            for col in range(N_CONDITIONS):
                if np.all(np.isnan(loo_mean[col])):
                    continue
                a = all_patterns[i, row]
                b = loo_mean[col]
                valid = ~(np.isnan(a) | np.isnan(b))
                if valid.sum() < 2:
                    continue
                # Pearson r manually (avoids overhead)
                a_, b_ = a[valid], b[valid]
                a_ -= a_.mean(); b_ -= b_.mean()
                denom = np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum())
                if denom == 0:
                    continue
                corr_mat[row, col] = np.dot(a_, b_) / denom

        per_subj_ispc[i] = corr_mat

    # Fisher-z average across subjects
    z_stack = np.arctanh(np.clip(per_subj_ispc, -0.999, 0.999))
    group_z = np.nanmean(z_stack, axis=0)
    group_ispc = np.tanh(group_z)

    return group_ispc, per_subj_ispc


# ============================================================================
# STEP 5: FIGURE
# ============================================================================

# _WINDOW_LABELS is set by apply_window_preset() at startup
_WINDOW_LABELS = WINDOW_PRESETS['default']['labels']
# Movies 2–5 (run 1) and 7–10 (run 2)
_MOVIE_LABELS = ['M2', 'M3', 'M4', 'M5', 'M7', 'M8', 'M9', 'M10']


def make_ispc_figure(ispc_matrix, roi_name, vmax=0.3):
    """Plot 40×40 ISPC heatmap with block structure.

    Layout: movies nested in windows — thick grid lines between window blocks
    (every 8 conditions), thin grid lines between movies within a block.
    Window block labels on top (x) and to the left of the y movie labels.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.subplots_adjust(left=0.28, right=0.88, top=0.88, bottom=0.12)

    im = ax.imshow(ispc_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal', origin='upper', interpolation='none')

    # ---- Tick labels: movie numbers, repeated for each window block ----
    tick_pos    = np.arange(N_CONDITIONS)
    tick_labels = _MOVIE_LABELS * N_WINDOWS
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=90)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    # ---- Thin grid lines between all cells ----
    for pos in np.arange(0.5, N_CONDITIONS - 1, 1):
        ax.axhline(pos, color='white', linewidth=0.3, alpha=0.5)
        ax.axvline(pos, color='white', linewidth=0.3, alpha=0.5)

    # ---- Thick grid lines between window blocks ----
    for block_edge in range(N_MOVIES, N_CONDITIONS, N_MOVIES):
        ax.axhline(block_edge - 0.5, color='black', linewidth=1.8)
        ax.axvline(block_edge - 0.5, color='black', linewidth=1.8)

    # ---- Window block labels on top (secondary x-axis) ----
    block_centers = np.arange(N_MOVIES / 2 - 0.5, N_CONDITIONS, N_MOVIES)
    ax2_top = ax.secondary_xaxis('top')
    ax2_top.set_xticks(block_centers)
    ax2_top.set_xticklabels(_WINDOW_LABELS, fontsize=7.5)
    ax2_top.tick_params(length=0)
    for spine in ax2_top.spines.values():
        spine.set_visible(False)

    # ---- Window block labels on the y-axis: rotated 90°, close to matrix ----
    # Use blended transform: x in axes coords, y in data coords so labels
    # align exactly with each block regardless of aspect-ratio padding.
    from matplotlib.transforms import blended_transform_factory
    blend = blended_transform_factory(ax.transAxes, ax.transData)
    for data_y, label in zip(block_centers, _WINDOW_LABELS):
        ax.text(-0.035, data_y, label, ha='center', va='center',
                fontsize=7.5, rotation=90, rotation_mode='anchor',
                transform=blend, clip_on=False)

    ax.set_xlabel('Condition (movie × window)', fontsize=10)
    ax.set_ylabel('Condition (movie × window)', fontsize=10)
    ax.set_title(roi_name, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.12, shrink=0.4)
    cbar.set_label('inter-subject pattern correlation (r)', fontsize=9)

    return fig


def make_combined_figure(ispc_by_roi, vmax=0.3, horizontal=False):
    """Plot all ROIs in a grid (2 columns) or single row (horizontal=True)."""
    n_rois = len(ispc_by_roi)

    if horizontal:
        ncols, nrows = n_rois, 1
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.2 * ncols + 1.5, 4.5))
        fig.subplots_adjust(left=0.10, right=0.91, top=0.90,
                            bottom=0.04, wspace=0.12)
        axes_flat = np.array(axes).flatten()
        last_im   = None

        block_centers = np.arange(N_MOVIES / 2 - 0.5, N_CONDITIONS, N_MOVIES)
        # TR-range-only column labels, e.g. "-6 - 3 TRs"
        tr_range_labels = [f"{w[1]} - {w[2]} TRs" for w in WINDOWS]
        tick_pos    = np.arange(N_CONDITIONS)
        tick_labels = _MOVIE_LABELS * N_WINDOWS

        from matplotlib.transforms import blended_transform_factory

        for idx, (roi_key, roi_name, _) in enumerate(ROI_SPEC):
            ax  = axes_flat[idx]
            mat = ispc_by_roi.get(roi_key)
            if mat is None:
                ax.set_visible(False)
                continue

            im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           aspect='equal', origin='upper', interpolation='none')
            last_im = im

            # Thick block dividers
            for block_edge in range(N_MOVIES, N_CONDITIONS, N_MOVIES):
                ax.axhline(block_edge - 0.5, color='black', linewidth=1.2)
                ax.axvline(block_edge - 0.5, color='black', linewidth=1.2)

            # Movie tick labels on bottom x-axis for every panel
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=4, rotation=90)
            ax.xaxis.set_tick_params(length=1)

            # No y tick labels; first panel gets rotated window block labels
            ax.set_yticks([])
            if idx == 0:
                blend = blended_transform_factory(ax.transAxes, ax.transData)
                for data_y, label in zip(block_centers, _WINDOW_LABELS):
                    ax.text(-0.08, data_y, label, ha='center', va='center',
                            fontsize=5.5, rotation=90, rotation_mode='anchor',
                            transform=blend, clip_on=False)

            # TR-range labels on top for every panel
            ax2 = ax.secondary_xaxis('top')
            ax2.set_xticks(block_centers)
            ax2.set_xticklabels(tr_range_labels, fontsize=5, rotation=0, ha='center')
            ax2.tick_params(length=0)
            for spine in ax2.spines.values():
                spine.set_visible(False)

            ax.set_title(roi_name, fontsize=8.5, fontweight='bold', pad=0)

        # Single shared colorbar
        if last_im is not None:
            cbar = fig.colorbar(last_im,
                                ax=axes_flat[axes_flat != None].tolist(),
                                fraction=0.012, pad=0.01, shrink=0.6)
            cbar.set_label('inter-subject pattern correlation (r)', fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        fig.suptitle(
            'ISPC between mean activation patterns of peri-boundary windows in each movie boundary',
            fontsize=12, fontweight='bold', y=0.98)

    else:
        ncols  = 2
        nrows  = (n_rois + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(9 * ncols, 9 * nrows))
        axes_flat = np.array(axes).flatten()

        block_centers = np.arange(N_MOVIES / 2 - 0.5, N_CONDITIONS, N_MOVIES)

        for idx, (roi_key, roi_name, _) in enumerate(ROI_SPEC):
            ax  = axes_flat[idx]
            mat = ispc_by_roi.get(roi_key)
            if mat is None:
                ax.set_visible(False)
                continue

            im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           aspect='equal', origin='upper', interpolation='none')

            ax.set_xticks([])
            ax.set_yticks([])
            for block_edge in range(N_MOVIES, N_CONDITIONS, N_MOVIES):
                ax.axhline(block_edge - 0.5, color='black', linewidth=1.5)
                ax.axvline(block_edge - 0.5, color='black', linewidth=1.5)

            for center, label in zip(block_centers, _WINDOW_LABELS):
                ax.text(center, -1.5, label.split('\n')[0],
                        ha='center', va='bottom', fontsize=7,
                        transform=ax.get_xaxis_transform())

            ax.set_title(roi_name, fontsize=10, fontweight='bold')
            cb = fig.colorbar(im, ax=ax, fraction=0.023, pad=0.03, shrink=0.4)
            cb.set_label('inter-subject pattern correlation (r)', fontsize=7)

        for idx in range(n_rois, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle('Filmfest ISPC — All ROIs', fontsize=13, fontweight='bold')
        plt.tight_layout()

    return fig


# ============================================================================
# PER-ROI PIPELINE
# ============================================================================

def run_roi(roi_key, roi_name, parcel_ids, subjects, do_hp=False,
            force_voxels=False, force_ispc=False, vmax=0.3):
    """Full pipeline for one ROI: extract → compute ISPC → save figure.

    Returns group ISPC (40, 40) array.
    """
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hp_tag   = '_hp' if do_hp else ''
    ispc_cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6{hp_tag}{WIN_TAG}_ispc.npz'

    if not force_ispc and ispc_cache.exists():
        print(f"  [{roi_key}] Loading ISPC from cache ...")
        data = np.load(ispc_cache, allow_pickle=True)
        group_ispc = data['group_ispc']
    else:
        print(f"\n{'='*60}")
        print(f"ROI: {roi_name} ({roi_key})")

        # Collect patterns from all subjects
        subject_patterns = []
        for subj in subjects:
            print(f"  Subject: {subj}")
            pats, vmask = get_subject_patterns(
                subj, roi_key, parcel_ids, do_hp=do_hp, force_voxels=force_voxels)
            if pats is None:
                print(f"  SKIP {subj} roi={roi_key}: no data")
                continue
            flat = flatten_patterns(pats)  # (40, V)
            subject_patterns.append(flat)
            n_valid = (~np.isnan(flat).all(axis=1)).sum()
            print(f"    {n_valid}/40 conditions valid")

        if len(subject_patterns) < 2:
            print(f"  [{roi_key}] Not enough subjects, skipping.")
            return None

        all_patterns = np.stack(subject_patterns, axis=0)  # (N_subj, 40, V)
        print(f"  [{roi_key}] Computing group ISPC ({all_patterns.shape[0]} subjects)...")
        group_ispc, per_subj = compute_group_ispc(all_patterns)

        np.savez_compressed(
            ispc_cache,
            group_ispc=group_ispc,
            per_subject=per_subj,
            subjects=np.array(subjects),
            do_hp=do_hp,
        )
        print(f"  [{roi_key}] Cached → {ispc_cache.name}")

    # Save figure (subfolder already encodes hp and window preset)
    fig_path = OUTPUT_DIR / f'roi-{roi_key}_ispc.png'
    fig = make_ispc_figure(group_ispc, roi_name, vmax=vmax)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{roi_key}] Figure saved → {fig_path.name}")

    return group_ispc


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Filmfest inter-subject pattern correlation (40×40 ISPC).'
    )
    parser.add_argument('--roi', nargs='+',
                        default=[k for k, _, _ in ROI_SPEC],
                        help='ROI keys to process (default: all)')
    parser.add_argument('--hrf', action='store_true',
                        help='Use HRF-shifted windows (-6:3, 4:13, 14:23, 24:33 TRs)')
    parser.add_argument('--post4', action='store_true',
                        help='Use post4 windows (4:13, 14:23, 24:33, 34:43 TRs); no pre-boundary')
    parser.add_argument('--hp', action='store_true',
                        help='Enable high-pass filtering at 0.01 Hz (default: off)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force recompute voxel extraction and ISPC')
    parser.add_argument('--no-cache-ispc', action='store_true',
                        help='Force recompute ISPC (reuse voxel cache)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Parallel jobs for voxel extraction (default: 1)')
    parser.add_argument('--vmax', type=float, default=0.5,
                        help='Colorbar limit for heatmap (default: 0.3)')
    args = parser.parse_args()

    global OUTPUT_DIR
    preset_name  = 'post4' if args.post4 else ('hrf' if args.hrf else 'default')
    hp_suffix    = '_hp' if args.hp else ''
    apply_window_preset(preset_name)
    OUTPUT_DIR = FIGS_DIR / 'filmfest_ispc' / f'{preset_name}{hp_suffix}'

    subjects = sorted(FILMFEST_SUBJECTS.keys())
    print(f"Subjects: {subjects}")
    print(f"ROIs: {args.roi}")
    print(f"Window preset: {'hrf' if args.hrf else 'default'} ({N_CONDITIONS} conditions)")
    print(f"HP filter: {args.hp}")

    # Build ROI spec for requested ROIs
    roi_spec_filtered = [(k, n, p) for k, n, p in ROI_SPEC if k in args.roi]
    if not roi_spec_filtered:
        print("No valid ROIs specified.")
        return

    VOXEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-extract voxels in parallel if requested
    if args.n_jobs > 1:
        print(f"\nPre-extracting voxels with n_jobs={args.n_jobs} ...")

        def _extract_job(subj, roi_key, parcel_ids):
            try:
                session = FILMFEST_SUBJECTS[subj]
                for task in ('filmfest1', 'filmfest2'):
                    load_roi_voxels(subj, session, task, roi_key, parcel_ids,
                                    force=args.no_cache)
                return subj, roi_key, True
            except Exception as e:
                print(f"  ERROR {subj} roi={roi_key}: {e}")
                return subj, roi_key, False

        Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_extract_job)(subj, roi_key, parcel_ids)
            for roi_key, _, parcel_ids in roi_spec_filtered
            for subj in subjects
        )

    # Run ISPC pipeline per ROI
    ispc_by_roi = {}
    for roi_key, roi_name, parcel_ids in roi_spec_filtered:
        mat = run_roi(
            roi_key, roi_name, parcel_ids, subjects,
            do_hp=args.hp,
            force_voxels=args.no_cache,
            force_ispc=(args.no_cache or args.no_cache_ispc),
            vmax=args.vmax,
        )
        ispc_by_roi[roi_key] = mat

    # Combined figure if more than one ROI
    if sum(v is not None for v in ispc_by_roi.values()) > 1:
        hp_tag   = '_hp' if args.hp else ''
        combined_path = OUTPUT_DIR / 'all_rois_ispc.png'
        fig = make_combined_figure(ispc_by_roi, vmax=args.vmax, horizontal=(args.hrf or args.post4))
        fig.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\nCombined figure saved → {combined_path.name}")

    print(f"\nDone.\n  Figures: {OUTPUT_DIR}\n  Cache: {RESULT_CACHE_DIR}")


if __name__ == '__main__':
    main()
