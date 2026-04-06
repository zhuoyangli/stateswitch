"""
Group-Average Time-Time Correlation Maps (filmfest only)

Unlike time_time_correlation.py, which computes per-subject T×T corrmaps and
then Fisher-z averages them, this script averages spatial patterns across
subjects first (at each timepoint), then computes the T×T Pearson correlation
matrix from the group-mean pattern. This gives a single corrmap reflecting the
shared response geometry rather than averaging individual corrmaps.

Output:
  - GROUP_avg_time_time_corr_filmfest.png  (4 ROIs × filmfest1 | filmfest2)

Usage:
    python srcs/fmrianalysis/time_time_correlation_group.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import datasets, image
from scipy.signal import butter, filtfilt
from scipy.stats import zscore as sp_zscore

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR
from configs.schaefer_rois import POSTERIOR_MEDIAL, ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL

ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')

CACHE_SUFFIX  = '_vol_roicat_z_groupavg_hp'
OUTPUT_DIR    = FIGS_DIR / 'time_time_corr'
GROUP_AVG_DIR = OUTPUT_DIR / 'group_average'
CACHE_DIR     = OUTPUT_DIR / 'cache'

FILMFEST_SUBJECTS = {
    'sub-003': 'ses-10',
    'sub-004': 'ses-10',
    'sub-006': 'ses-08',
    'sub-007': 'ses-08',
    'sub-008': 'ses-08',
    'sub-009': 'ses-07',
}

MOVIE_INFO = [
    {'id': 1,  'file': 'FilmFest_01_CMIYC_Segments.xlsx',         'task': 'filmfest1'},
    {'id': 2,  'file': 'FilmFest_02_The_Record_Segments.xlsx',     'task': 'filmfest1'},
    {'id': 3,  'file': 'FilmFest_03_The_Boyfriend_Segments.xlsx',  'task': 'filmfest1'},
    {'id': 4,  'file': 'FilmFest_04_The_Shoe_Segments.xlsx',       'task': 'filmfest1'},
    {'id': 5,  'file': 'FilmFest_05_Keith_Reynolds_Segments.xlsx', 'task': 'filmfest1'},
    {'id': 6,  'file': 'FilmFest_06_The_Rock_Segments.xlsx',       'task': 'filmfest2'},
    {'id': 7,  'file': 'FilmFest_07_The_Prisoner_Segments.xlsx',   'task': 'filmfest2'},
    {'id': 8,  'file': 'FilmFest_08_The_Black_Hole_Segments.xlsx', 'task': 'filmfest2'},
    {'id': 9,  'file': 'FilmFest_09_Post-it_Love_Segments.xlsx',   'task': 'filmfest2'},
    {'id': 10, 'file': 'FilmFest_10_Bus_Stop_Segments.xlsx',       'task': 'filmfest2'},
]

ROI_SPEC = [
    ('pmc', 'Posterior Medial Cortex',
     POSTERIOR_MEDIAL['left'] + POSTERIOR_MEDIAL['right']),
    ('ag',  'Angular Gyrus',
     ANGULAR_GYRUS['left'] + ANGULAR_GYRUS['right']),
    ('eac', 'Auditory Cortex',
     EARLY_AUDITORY['left'] + EARLY_AUDITORY['right']),
    ('evc', 'Early Visual Cortex',
     EARLY_VISUAL['left'] + EARLY_VISUAL['right']),
]

# === LOAD ATLAS (module level) ===
print("Loading Schaefer atlas...", flush=True)
SCHAEFER = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
_atlas_resampled = None
print("  Done.", flush=True)


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def _highpass_filter(data, cutoff=0.01, order=2):
    """Zero-phase Butterworth high-pass filter along axis 0 (time).

    Parameters
    ----------
    data   : np.ndarray, shape (T, n_voxels)
    cutoff : float — cutoff frequency in Hz (default 0.01 Hz)
    order  : int   — filter order (default 2)
    """
    nyq = 0.5 / TR
    b, a = butter(order, cutoff / nyq, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)


# ============================================================================
# CORE HELPERS
# ============================================================================

def _get_bold_path(subject, session, task):
    return (DERIVATIVES_DIR / subject / session / 'func' /
            f'{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')


def _get_atlas_data(bold_path):
    """Return Schaefer atlas resampled to BOLD voxel space (cached on first call)."""
    global _atlas_resampled
    if _atlas_resampled is None:
        bold_ref = image.index_img(str(bold_path), 0)
        atlas_resampled = image.resample_to_img(
            SCHAEFER['maps'], bold_ref, interpolation='nearest'
        )
        _atlas_resampled = np.round(atlas_resampled.get_fdata()).astype(int)
        print(f"  Atlas resampled to BOLD space: {_atlas_resampled.shape}")
    return _atlas_resampled


def _mss_to_seconds(mss):
    minutes = int(mss)
    seconds = round((mss - minutes) * 100)
    return minutes * 60 + seconds


def get_movie_boundary_offsets(task):
    """Return between-movie transition times (seconds) for a filmfest task."""
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    boundaries = []
    for movie in movies[:-1]:
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        last_end = segb['End Time (m.ss)'].values[-1]
        boundaries.append(_mss_to_seconds(last_end))
    return boundaries


def _add_boundary_lines(ax, T, movie_boundaries=None):
    """Draw movie boundary lines on a time-time axes."""
    if movie_boundaries is not None:
        for t_s in movie_boundaries:
            idx = int(round(t_s / TR))
            if 0 <= idx < T:
                ax.axvline(idx, color='k', lw=0.6, alpha=0.8)
                ax.axhline(idx, color='k', lw=0.6, alpha=0.8)


def _vminmax(panel_maps):
    """±99th percentile of off-diagonal values across a dict of {key: array|None}."""
    all_vals = []
    for m in panel_maps.values():
        if m is not None:
            T = m.shape[0]
            all_vals.append(m[~np.eye(T, dtype=bool)])
    if not all_vals:
        return -0.5, 0.5
    vmax = float(np.percentile(np.abs(np.concatenate(all_vals)), 99))
    return -vmax, vmax


# ============================================================================
# COMPUTATION
# ============================================================================

def load_subject_roi_data(task):
    """Load BOLD once per subject and extract all ROIs in one pass.

    Returns
    -------
    roi_data : dict  roi_key -> list of (T_subj, n_voxels) per subject
    T_list   : list of int — number of timepoints per subject
    """
    first_subj, first_ses = next(iter(FILMFEST_SUBJECTS.items()))
    atlas_data = _get_atlas_data(_get_bold_path(first_subj, first_ses, task))

    all_roi_ids = {roi_key: roi_ids for roi_key, _, roi_ids in ROI_SPEC}
    roi_data = {roi_key: [] for roi_key in all_roi_ids}
    T_list = []
    n_subjects = len(FILMFEST_SUBJECTS)

    for i, (subj, ses) in enumerate(FILMFEST_SUBJECTS.items(), 1):
        bold_path = _get_bold_path(subj, ses, task)
        print(f"  [{i}/{n_subjects}] Loading {subj} {ses}...", flush=True)
        bold_data = nib.load(str(bold_path)).get_fdata()  # (X, Y, Z, T)
        T_list.append(bold_data.shape[3])
        print(f"    BOLD shape: {bold_data.shape}", flush=True)
        for roi_key, roi_ids in all_roi_ids.items():
            parcel_patterns = []
            for parcel_id in roi_ids:
                mask = atlas_data == parcel_id
                if mask.sum() < 2:
                    continue
                parcel_patterns.append(bold_data[mask].T)
            if parcel_patterns:
                roi_data[roi_key].append(np.concatenate(parcel_patterns, axis=1))
        del bold_data
        print(f"    All ROIs extracted.", flush=True)

    return roi_data, T_list


def compute_group_avg_corrmap_from_data(roi_key, subject_patterns, T):
    """Compute T×T corrmap from pre-loaded per-subject ROI patterns.

    Parameters
    ----------
    roi_key          : str
    subject_patterns : list of (T_subj, n_voxels) arrays
    T                : int — common timepoints to trim to

    Returns
    -------
    corrmap : np.ndarray, shape (T, T)
    """
    processed = []
    for sp in subject_patterns:
        sp = sp[:T]
        sp = _highpass_filter(sp)
        sp = sp_zscore(sp, axis=0, nan_policy='omit')
        sp = np.nan_to_num(sp)
        processed.append(sp)
    group_mean = np.mean(np.stack(processed, axis=0), axis=0)  # (T, total_vox)
    print(f"    {roi_key}: group mean shape {group_mean.shape}", flush=True)
    return np.corrcoef(group_mean)  # (T, T)


def compute_and_cache_task_corrmaps(task):
    """Load BOLD once per subject, compute + cache all ROI corrmaps for task.

    Skips ROIs whose cache already exists.
    """
    missing = []
    cached = {}
    for roi_key, _, _ in ROI_SPEC:
        cache_file = CACHE_DIR / f"GROUP_task-{task}_roi-{roi_key}{CACHE_SUFFIX}_corrmap.npz"
        if cache_file.exists():
            print(f"  {task} roi={roi_key}: loading cache", flush=True)
            cached[roi_key] = np.load(cache_file)['corrmap']
        else:
            missing.append(roi_key)

    if not missing:
        return cached

    print(f"\n  Loading BOLD for {task} ({len(missing)} ROIs to compute: {missing})...", flush=True)
    roi_data, T_list = load_subject_roi_data(task)
    T = min(T_list)
    print(f"\n  Common T={T}. Computing corrmaps...", flush=True)

    for roi_key in missing:
        corrmap = compute_group_avg_corrmap_from_data(roi_key, roi_data[roi_key], T)
        cache_file = CACHE_DIR / f"GROUP_task-{task}_roi-{roi_key}{CACHE_SUFFIX}_corrmap.npz"
        np.savez_compressed(cache_file, corrmap=corrmap)
        print(f"    Cached → {cache_file.name}", flush=True)
        cached[roi_key] = corrmap

    return cached


# ============================================================================
# FIGURE
# ============================================================================

def make_group_avg_figure(vmin=None, vmax=None):
    """4 rows × 2 cols: group-average-pattern T×T corrmap for filmfest1 and filmfest2."""
    print(f"\n{'=' * 60}")
    print("BUILDING FIGURE: group-average-pattern TTC (filmfest1 | filmfest2)")
    print(f"{'=' * 60}")

    col_tasks = ('filmfest1', 'filmfest2')
    col_titles = ['Filmfest 1', 'Filmfest 2']

    panel_maps = {}
    for task in col_tasks:
        print(f"\n--- {task} ---", flush=True)
        roi_corrmaps = compute_and_cache_task_corrmaps(task)
        panel_maps[task] = roi_corrmaps
        for roi_key, corrmap in roi_corrmaps.items():
            print(f"  roi={roi_key}: ({corrmap.shape[0]}, {corrmap.shape[0]})", flush=True)

    if vmin is None or vmax is None:
        all_vals = {f'{t}_{k}': panel_maps[t].get(k)
                    for t in col_tasks for k, _, _ in ROI_SPEC}
        vmin, vmax = _vminmax(all_vals)
    print(f"\nColor scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

    movie_bounds = {
        'filmfest1': get_movie_boundary_offsets('filmfest1'),
        'filmfest2': get_movie_boundary_offsets('filmfest2'),
    }
    for task, bounds in movie_bounds.items():
        print(f"  {task} boundaries: {len(bounds)} at {[round(t) for t in bounds]} s")

    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    fig.suptitle('Group-Average-Pattern Time-Time Correlation Maps',
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.94)

    for row, (roi_key, roi_name, _) in enumerate(ROI_SPEC):
        for col, task in enumerate(col_tasks):
            ax = axes[row, col]
            m = panel_maps[task].get(roi_key)
            if m is None:
                ax.set_visible(False)
                continue

            T = m.shape[0]
            ax.imshow(m, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                      aspect='equal', origin='upper')
            _add_boundary_lines(ax, T, movie_boundaries=movie_bounds[task])

            n_ticks = 6
            tick_idxs = np.linspace(0, T - 1, n_ticks, dtype=int)
            tick_labels = [str(int(i * TR)) for i in tick_idxs]
            ax.set_xticks(tick_idxs)
            ax.set_xticklabels(tick_labels, fontsize=7)
            ax.set_yticks(tick_idxs)
            ax.set_yticklabels(tick_labels, fontsize=7)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(roi_name, fontsize=10)
            if row == len(ROI_SPEC) - 1:
                ax.set_xlabel('Time (s)', fontsize=9)

    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.02, shrink=0.6)
    cbar.set_label('Pearson r', fontsize=10)

    out = GROUP_AVG_DIR / 'GROUP_avg_time_time_corr_filmfest.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    for d in (OUTPUT_DIR, GROUP_AVG_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GROUP-AVERAGE-PATTERN TIME-TIME CORRELATION MAPS")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    make_group_avg_figure(vmin=-0.5, vmax=0.5)

    print("\n" + "=" * 60)
    print(f"DONE. Figure saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
