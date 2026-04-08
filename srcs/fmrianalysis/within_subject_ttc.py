"""
Time-Time Correlation Maps

For SVF, AHC, and Filmfest tasks, compute T×T pairwise Pearson correlation
matrices of multi-vertex spatial patterns within 4 ROIs (PMC, AG, EAC, EVC).
Instead of averaging vertices into a 1D time course, the full spatial pattern
at each timepoint is used — capturing representational similarity over time.

Output:
  - GROUP_time_time_corr_filmfest1.png  (4 ROIs × [SVF, AHC, Filmfest1])
  - GROUP_time_time_corr_filmfest2.png  (4 ROIs × [SVF, AHC, Filmfest2])

Usage:
    python srcs/fmrianalysis/time_time_correlation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import smooth_img
from scipy.stats import zscore as sp_zscore
from joblib import Parallel, delayed

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, SUBJECT_IDS, ANALYSIS_CACHE_DIR, FILMFEST_SUBJECTS, MOVIE_INFO
from configs.schaefer_rois import POSTERIOR_MEDIAL, ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL
from fmrianalysis.utils import (
    highpass_filter, get_bold_path, get_atlas_data, cross_corrcoef, group_average,
    find_psychopy_csv, get_trial_times, get_movie_boundary_offsets, discover_svf_ahc_sessions,
)

CACHE_SUFFIX     = '_vol_roicat_z_hp'   # temporal z-score + high-pass per voxel
ISC_CACHE_SUFFIX = '_vol_roicat_z_hp'   # same
OUTPUT_DIR       = FIGS_DIR / 'time_time_corr'
WITHIN_SUBJ_DIR  = OUTPUT_DIR / 'within_subject'
ISC_DIR          = OUTPUT_DIR / 'isc'
GROUP_AVG_DIR    = OUTPUT_DIR / 'group_average'
CACHE_DIR        = ANALYSIS_CACHE_DIR / 'time_time_corr'

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

# === LOAD ATLASES (module level) ===
print("Loading atlases...")
SCHAEFER = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
_atlas_resampled = None  # cached atlas array resampled to BOLD voxel space
print("  Schaefer atlas loaded.")


# ============================================================================
# CORE COMPUTATION
# ============================================================================


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


def compute_roi_corrmap(subject, session, task, roi_parcel_ids):
    """Return T×T correlation map for an ROI.

    Voxels from all parcels in the ROI are concatenated into a single (T, V)
    matrix. Each voxel's time course is temporally z-scored (zero mean, unit
    variance across time), then np.corrcoef gives the T×T Pearson correlation
    of spatial patterns across timepoints.

    Parameters
    ----------
    subject, session, task : str
    roi_parcel_ids : list of int
        1-based Schaefer parcel IDs (combined left + right).

    Returns
    -------
    corrmap : np.ndarray, shape (T, T)
    """
    bold_path = get_bold_path(subject, session, task)
    bold_data = nib.load(str(bold_path)).get_fdata()  # (X, Y, Z, T)
    atlas_data = _get_atlas_data(bold_path)           # (X, Y, Z)

    parcel_patterns = []
    for parcel_id in roi_parcel_ids:
        mask = atlas_data == parcel_id  # (X, Y, Z)
        if mask.sum() < 2:
            continue
        pdata = bold_data[mask].T  # (T, n_voxels)
        pdata = highpass_filter(pdata, order=2)
        # Round 1: temporal z-score per voxel (axis=0 = across time)
        pdata = sp_zscore(pdata, axis=0, nan_policy='omit')
        pdata = np.nan_to_num(pdata)
        parcel_patterns.append(pdata)

    if not parcel_patterns:
        raise ValueError(f"No usable parcels found for parcel IDs {roi_parcel_ids}")
    print(f"    {len(parcel_patterns)}/{len(roi_parcel_ids)} parcels used")

    roi_pattern = np.concatenate(parcel_patterns, axis=1)  # (T, total_voxels)
    return np.corrcoef(roi_pattern)  # (T, T)


def load_or_compute_corrmap(subject, session, task, roi_key, roi_parcel_ids):
    """Load T×T corrmap from cache; compute and cache if missing."""
    cache_file = (CACHE_DIR /
                  f"{subject}_{session}_task-{task}_roi-{roi_key}{CACHE_SUFFIX}_corrmap.npz")
    if cache_file.exists():
        print(f"  {subject} {session} {task} roi={roi_key}: loading cache")
        return np.load(cache_file)['corrmap']
    print(f"  {subject} {session} {task} roi={roi_key}: computing...")
    corrmap = compute_roi_corrmap(subject, session, task, roi_parcel_ids)
    np.savez_compressed(cache_file, corrmap=corrmap)
    print(f"    Cached → {cache_file.name}")
    return corrmap


def load_subject_roi_pattern(subject, session, task, roi_parcel_ids, fwhm=4):
    """Load smoothed, hp-filtered, temporally z-scored voxel pattern for one subject/run.

    Returns (T, n_voxels) float64.
    """
    bold_path  = get_bold_path(subject, session, task)
    bold_data  = smooth_img(nib.load(str(bold_path)), fwhm=fwhm).get_fdata()
    atlas_data = _get_atlas_data(bold_path)

    parcel_parts = []
    for parcel_id in roi_parcel_ids:
        mask = atlas_data == parcel_id
        if mask.sum() < 2:
            continue
        pdata = bold_data[mask].T          # (T, n_voxels)
        pdata = highpass_filter(pdata, order=2)
        pdata = sp_zscore(pdata, axis=0, nan_policy='omit')
        pdata = np.nan_to_num(pdata)
        parcel_parts.append(pdata)

    del bold_data

    if not parcel_parts:
        raise ValueError(f"No usable parcels for {subject} {session} {task}")
    return np.concatenate(parcel_parts, axis=1)   # (T, total_voxels)


def compute_group_pattern_corrmap(task, roi_key, roi_parcel_ids, fwhm=4):
    """Average spatial patterns across filmfest subjects, then compute TTC.

    Pipeline:
      1. Load each subject's smoothed, hp-filtered, z-scored voxel pattern (T, V)
      2. Trim to common T and average across subjects → group pattern (T, V)
      3. Return corrcoef(group_pattern) → (T, T)

    Cached to CACHE_DIR/GROUP_task-{task}_roi-{roi_key}_group_pattern_sm{fwhm}_corrmap.npz
    """
    cache_file = (CACHE_DIR /
                  f"GROUP_task-{task}_roi-{roi_key}_group_pattern_sm{fwhm}_corrmap.npz")
    if cache_file.exists():
        print(f"  {task} roi={roi_key}: loading group-pattern cache")
        return np.load(cache_file)['corrmap']

    print(f"  {task} roi={roi_key}: loading subject patterns (sm{fwhm}mm)...")
    patterns = []
    for subject, session in FILMFEST_SUBJECTS.items():
        try:
            p = load_subject_roi_pattern(subject, session, task, roi_parcel_ids, fwhm=fwhm)
            patterns.append(p)
            print(f"    {subject}: shape {p.shape}")
        except Exception as e:
            print(f"    SKIP {subject} {session} {task}: {e}")

    if not patterns:
        raise ValueError(f"No patterns loaded for {task} roi={roi_key}")

    T = min(p.shape[0] for p in patterns)
    group_pattern = np.mean(np.stack([p[:T] for p in patterns], axis=0), axis=0)  # (T, V)
    print(f"    Group pattern shape: {group_pattern.shape}")

    corrmap = np.corrcoef(group_pattern)   # (T, T)
    np.savez_compressed(cache_file, corrmap=corrmap)
    print(f"    Cached → {cache_file.name}")
    return corrmap


def run_group_pattern_ttc_filmfest(fwhm=4, vmin=-0.3, vmax=0.3):
    """Compute TTC on group-averaged spatial patterns for filmfest1 and filmfest2.

    Output (does not overwrite existing figures):
        GROUP_group_pattern_ttc_filmfest_sm{fwhm}.png
    """
    for d in (OUTPUT_DIR, GROUP_AVG_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    col_tasks  = ('filmfest1', 'filmfest2')
    col_titles = ['Filmfest 1', 'Filmfest 2']

    panel_maps = {}
    for task in col_tasks:
        print(f"\n--- Group-pattern TTC: {task} ---")
        panel_maps[task] = {}
        for roi_key, _, roi_ids in ROI_SPEC:
            try:
                panel_maps[task][roi_key] = compute_group_pattern_corrmap(
                    task, roi_key, roi_ids, fwhm=fwhm)
                T = panel_maps[task][roi_key].shape[0]
                print(f"  {task} roi={roi_key}: ({T}, {T})")
            except Exception as e:
                print(f"  SKIP {task} roi={roi_key}: {e}")
                panel_maps[task][roi_key] = None

    movie_bounds = {
        'filmfest1': get_movie_boundary_offsets('filmfest1'),
        'filmfest2': get_movie_boundary_offsets('filmfest2'),
    }
    boundary_info = {
        'filmfest1': dict(movie_boundaries=movie_bounds['filmfest1']),
        'filmfest2': dict(movie_boundaries=movie_bounds['filmfest2']),
    }

    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    fig.suptitle(f'Group-Pattern TTC (sm{fwhm}mm) — Filmfest',
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.94)
    _render_panels(fig, axes, col_tasks, col_titles, panel_maps, boundary_info, vmin, vmax)

    out = GROUP_AVG_DIR / f'GROUP_group_pattern_ttc_filmfest_sm{fwhm}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {out}")


def compute_all_isc_corrmaps(task, roi_parcel_ids, fwhm=None, do_hp=True):
    """Leave-one-out ISC T×T maps for all filmfest subjects.

    For each subject, entry (t1, t2) of the map is the Pearson correlation
    between that subject's spatial pattern at t1 and the LOO group mean's
    spatial pattern at t2.

    Parameters
    ----------
    task : str
    roi_parcel_ids : list of int
    fwhm : float or None
        Spatial smoothing FWHM in mm. None = no smoothing.
    do_hp : bool
        If True (default), apply 0.01 Hz Butterworth high-pass before z-scoring.

    Returns
    -------
    maps : dict subject -> np.ndarray, shape (T, T)
    """
    subjects = list(FILMFEST_SUBJECTS.keys())
    n_subjects = len(subjects)

    # Get atlas in BOLD voxel space
    first_subj, first_ses = next(iter(FILMFEST_SUBJECTS.items()))
    atlas_data = _get_atlas_data(get_bold_path(first_subj, first_ses, task))

    # Load each subject's BOLD and extract parcel voxels
    all_roi_data = {subj: [] for subj in subjects}  # subj -> list of (T, n_voxels) per parcel
    T_list = []
    for subj, ses in FILMFEST_SUBJECTS.items():
        bold_img = nib.load(str(get_bold_path(subj, ses, task)))
        if fwhm is not None:
            bold_img = smooth_img(bold_img, fwhm=fwhm)
        bold_data = bold_img.get_fdata()  # (X, Y, Z, T)
        T_list.append(bold_data.shape[3])
        for parcel_id in roi_parcel_ids:
            mask = atlas_data == parcel_id
            if mask.sum() < 2:
                continue
            all_roi_data[subj].append(bold_data[mask].T)  # (T_subj, n_voxels)
        del bold_data, bold_img

    if not all_roi_data[subjects[0]]:
        raise ValueError(f"No usable parcels found for parcel IDs {roi_parcel_ids}")

    # Trim to common T and concatenate parcels into one ROI pattern per subject
    T = min(T_list)
    n_used = sum(1 for parcel_id in roi_parcel_ids
                 if (atlas_data == parcel_id).sum() >= 2)
    print(f"    {n_used}/{len(roi_parcel_ids)} parcels used")

    # Preprocess: optional high-pass then temporal z-score per voxel within subject
    roi_data = {}
    for subj in subjects:
        pdata = np.concatenate([p[:T] for p in all_roi_data[subj]], axis=1)
        if do_hp:
            pdata = highpass_filter(pdata, order=2)
        pdata = sp_zscore(pdata, axis=0, nan_policy='omit')
        pdata = np.nan_to_num(pdata)
        roi_data[subj] = pdata

    # LOO group mean from temporally z-scored data (no spatial z-score).
    # Spatial z-scoring would subtract the spatial mean at each timepoint, removing
    # the shared "all voxels up/down together" component that drives ISC in
    # stimulus-driven ROIs. Pearson r in cross_corrcoef handles centering/scaling
    # internally, so no additional normalization is needed.
    all_pdata = np.stack([roi_data[s] for s in subjects], axis=0)  # (n_subj, T, total_voxels)
    group_mean = all_pdata.mean(axis=0)  # (T, total_voxels)

    result = {}
    for subj in subjects:
        pdata = roi_data[subj]
        loo_mean = (group_mean * n_subjects - pdata) / (n_subjects - 1)
        result[subj] = cross_corrcoef(pdata, loo_mean)
    return result


def load_or_compute_isc_corrmaps(task, roi_key, roi_parcel_ids, fwhm=None, do_hp=True):
    """Load per-subject ISC corrmaps from cache; compute and cache if missing."""
    subjects = list(FILMFEST_SUBJECTS.keys())
    sm_suffix = f'_sm{int(fwhm)}' if fwhm is not None else ''
    hp_suffix = '_hp' if do_hp else '_nohp'
    cache_suffix = f'_vol_roicat_z{sm_suffix}{hp_suffix}'
    cache_files = {
        subj: CACHE_DIR / f"{subj}_{FILMFEST_SUBJECTS[subj]}_task-{task}_roi-{roi_key}{cache_suffix}_isc_corrmap.npz"
        for subj in subjects
    }
    if all(f.exists() for f in cache_files.values()):
        print(f"  {task} roi={roi_key}: loading ISC cache")
        return {subj: np.load(cache_files[subj])['corrmap'] for subj in subjects}
    print(f"  {task} roi={roi_key}: computing ISC (sm={fwhm}, hp={do_hp})...")
    maps = compute_all_isc_corrmaps(task, roi_parcel_ids, fwhm=fwhm, do_hp=do_hp)
    for subj, corrmap in maps.items():
        np.savez_compressed(cache_files[subj], corrmap=corrmap)
        print(f"    Cached → {cache_files[subj].name}")
    return maps


# ============================================================================
# PIPELINE
# ============================================================================

def collect_task_corrmaps(task):
    """Collect per-session corrmaps for a task.

    For SVF/AHC: iterates over all subjects and their sessions.
    For filmfest: iterates over FILMFEST_SUBJECTS.

    Returns
    -------
    all_maps : dict roi_key -> list of (T, T) arrays
    """
    all_maps = {roi_key: [] for roi_key, _, _ in ROI_SPEC}

    if task in ('svf', 'ahc'):
        for subject in SUBJECT_IDS:
            for session, t in discover_svf_ahc_sessions(subject):
                if t != task:
                    continue
                for roi_key, _, roi_ids in ROI_SPEC:
                    try:
                        cm = load_or_compute_corrmap(
                            subject, session, task, roi_key, roi_ids)
                        all_maps[roi_key].append(cm)
                    except Exception as e:
                        print(f"  SKIP {subject} {session} {task} {roi_key}: {e}")
    else:
        # filmfest1 or filmfest2
        for subject, session in FILMFEST_SUBJECTS.items():
            for roi_key, _, roi_ids in ROI_SPEC:
                try:
                    cm = load_or_compute_corrmap(
                        subject, session, task, roi_key, roi_ids)
                    all_maps[roi_key].append(cm)
                except Exception as e:
                    print(f"  SKIP {subject} {session} {task} {roi_key}: {e}")

    return all_maps


def get_representative_trial_times(task):
    """Get trial onsets/offsets from the first subject with valid data for overlay."""
    if task not in ('svf', 'ahc'):
        return np.array([]), np.array([])
    for subject in SUBJECT_IDS:
        for session, t in discover_svf_ahc_sessions(subject):
            if t != task:
                continue
            onsets, offsets = get_trial_times(subject, session, task)
            if len(onsets) > 0:
                print(f"  {task} trial overlay: using {subject} {session} "
                      f"({len(onsets)} trials)")
                return onsets, offsets
    return np.array([]), np.array([])


def _add_boundary_lines(ax, T, onsets=None, offsets=None, movie_boundaries=None):
    """Draw trial/movie boundary lines on a time-time axes."""
    line_kw = dict(color='k', lw=0.6, alpha=0.8)
    for times in (onsets, offsets, movie_boundaries):
        if times is not None:
            for t_s in times:
                idx = int(round(t_s / TR))
                if 0 <= idx < T:
                    ax.axvline(idx, **line_kw)
                    ax.axhline(idx, **line_kw)


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


def _render_panels(fig, axes, col_tasks, col_titles, panel_maps, boundary_info,
                   vmin, vmax):
    """Render imshow panels onto a pre-created axes grid (rows=ROIs, cols=tasks)."""
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

            _add_boundary_lines(ax, T, **boundary_info[task])

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

    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.02, shrink=0.6)
    cbar.set_label('Pearson r', fontsize=10)


def _render_panels_horizontal(fig, axes, row_tasks, row_titles, panel_maps,
                               boundary_info, vmin, vmax):
    """Render imshow panels onto a pre-created axes grid (rows=tasks, cols=ROIs)."""
    for row, (task, row_title) in enumerate(zip(row_tasks, row_titles)):
        for col, (roi_key, roi_name, _) in enumerate(ROI_SPEC):
            ax = axes[row, col]
            m = panel_maps[task].get(roi_key)
            if m is None:
                ax.set_visible(False)
                continue

            T = m.shape[0]
            ax.imshow(m, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                      aspect='equal', origin='upper')

            _add_boundary_lines(ax, T, **boundary_info[task])

            n_ticks = 6
            tick_idxs = np.linspace(0, T - 1, n_ticks, dtype=int)
            tick_labels = [str(int(i * TR)) for i in tick_idxs]
            ax.set_xticks(tick_idxs)
            ax.set_xticklabels(tick_labels, fontsize=7)
            ax.set_yticks(tick_idxs)
            ax.set_yticklabels(tick_labels, fontsize=7)

            if row == 0:
                ax.set_title(roi_name, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{row_title}\nTime (s)', fontsize=10)
            if row == len(row_tasks) - 1:
                ax.set_xlabel('Time (s)', fontsize=9)

    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.02, shrink=0.6)
    cbar.set_label('Pearson r', fontsize=10)


# ============================================================================
# FIGURE 1: group-average filmfest (filmfest1 | filmfest2)
# ============================================================================

def make_group_filmfest_figure(vmin=None, vmax=None):
    """4 rows × 2 cols: group-average filmfest1 and filmfest2."""
    print(f"\n{'=' * 60}")
    print("BUILDING FIGURE: group filmfest")
    print(f"{'=' * 60}")

    col_tasks = ('filmfest1', 'filmfest2')
    col_titles = ['Filmfest 1', 'Filmfest 2']

    panel_maps = {}
    for task in col_tasks:
        print(f"\n--- Collecting corrmaps: {task} ---")
        all_maps = collect_task_corrmaps(task)
        panel_maps[task] = {}
        for roi_key, _, _ in ROI_SPEC:
            maps = all_maps[roi_key]
            if not maps:
                print(f"  WARNING: no maps for {task} {roi_key}")
                panel_maps[task][roi_key] = None
            else:
                panel_maps[task][roi_key] = group_average(maps)
                T = panel_maps[task][roi_key].shape[0]
                print(f"  {task} {roi_key}: N={len(maps)} → ({T}, {T})")

    # Shared color scale: use provided global limits, fall back to local
    if vmin is None or vmax is None:
        all_panel_maps = {f'{t}_{k}': panel_maps[t].get(k)
                          for t in col_tasks for k, _, _ in ROI_SPEC}
        vmin, vmax = _vminmax(all_panel_maps)
    print(f"\nColor scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

    movie_bounds = {
        'filmfest1': get_movie_boundary_offsets('filmfest1'),
        'filmfest2': get_movie_boundary_offsets('filmfest2'),
    }
    for task, bounds in movie_bounds.items():
        print(f"  {task} boundaries: {len(bounds)} at {[round(t) for t in bounds]} s")

    boundary_info = {
        'filmfest1': dict(movie_boundaries=movie_bounds['filmfest1']),
        'filmfest2': dict(movie_boundaries=movie_bounds['filmfest2']),
    }

    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    fig.suptitle('Group-Average Within-Subject Time-Time Correlation Maps',
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.94)
    _render_panels(fig, axes, col_tasks, col_titles, panel_maps, boundary_info,
                   vmin, vmax)

    out = WITHIN_SUBJ_DIR / 'GROUP_within_subj_ttc_filmfest.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {out}")
    return panel_maps   # return so per-subject figures can reuse it


# ============================================================================
# FIGURE: ISC time-time (group + per-subject)
# ============================================================================

def make_filmfest_isc_figures(vmin=None, vmax=None, fwhm=None, do_hp=True):
    """Group-average and per-subject ISC T×T figures for filmfest1 and filmfest2."""
    sm_label = f' sm{int(fwhm)}mm' if fwhm is not None else ''
    hp_label = '' if do_hp else ' no-HP'
    print(f"\n{'=' * 60}")
    print(f"BUILDING ISC FIGURES: filmfest{sm_label}{hp_label}")
    print(f"{'=' * 60}")

    col_tasks = ('filmfest1', 'filmfest2')
    col_titles = ['Filmfest 1', 'Filmfest 2']

    # Collect per-subject ISC maps: task -> roi_key -> subject -> (T, T)
    subject_task_maps = {}
    for task in col_tasks:
        print(f"\n--- Collecting ISC corrmaps: {task} ---")
        subject_task_maps[task] = {}
        for roi_key, _, roi_ids in ROI_SPEC:
            try:
                subject_task_maps[task][roi_key] = load_or_compute_isc_corrmaps(
                    task, roi_key, roi_ids, fwhm=fwhm, do_hp=do_hp)
            except Exception as e:
                print(f"  WARNING {task} {roi_key}: {e}")
                subject_task_maps[task][roi_key] = {}

    # Group-average ISC maps
    group_panel_maps = {}
    for task in col_tasks:
        group_panel_maps[task] = {}
        for roi_key, _, _ in ROI_SPEC:
            maps = list(subject_task_maps[task][roi_key].values())
            if maps:
                group_panel_maps[task][roi_key] = group_average(maps)
                T = group_panel_maps[task][roi_key].shape[0]
                print(f"  {task} {roi_key}: N={len(maps)} → ({T}, {T})")
            else:
                group_panel_maps[task][roi_key] = None

    if vmin is None or vmax is None:
        all_panel_maps = {f'{t}_{k}': group_panel_maps[t].get(k)
                          for t in col_tasks for k, _, _ in ROI_SPEC}
        vmin, vmax = _vminmax(all_panel_maps)
    print(f"\nISC color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

    movie_bounds = {
        'filmfest1': get_movie_boundary_offsets('filmfest1'),
        'filmfest2': get_movie_boundary_offsets('filmfest2'),
    }
    boundary_info = {
        'filmfest1': dict(movie_boundaries=movie_bounds['filmfest1']),
        'filmfest2': dict(movie_boundaries=movie_bounds['filmfest2']),
    }

    # Group ISC figure — original layout (4 rows=ROIs × 2 cols=runs)
    sm_file = f'_sm{int(fwhm)}' if fwhm is not None else ''
    hp_file = '_hp' if do_hp else '_nohp'
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    fig.suptitle(f'Group-Average ISC TTC (LOO){sm_label}{hp_label}',
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.94)
    _render_panels(fig, axes, col_tasks, col_titles, group_panel_maps, boundary_info,
                   vmin, vmax)
    out = ISC_DIR / f'GROUP_isc_ttc_filmfest{sm_file}{hp_file}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {out}")

    # Group ISC figure — horizontal layout (2 rows=runs × 4 cols=ROIs)
    n_rois = len(ROI_SPEC)
    fig_h, axes_h = plt.subplots(2, n_rois, figsize=(5 * n_rois, 10))
    fig_h.suptitle(f'Group-Average ISC TTC (LOO){sm_label}{hp_label}',
                   fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.94)
    _render_panels_horizontal(fig_h, axes_h, col_tasks, col_titles,
                               group_panel_maps, boundary_info, vmin, vmax)
    out_h = ISC_DIR / f'GROUP_isc_ttc_filmfest{sm_file}{hp_file}_horizontal.png'
    fig_h.savefig(out_h, dpi=300, bbox_inches='tight')
    plt.close(fig_h)
    print(f"Saved → {out_h}")

    # Per-subject ISC figures
    for subject, session in FILMFEST_SUBJECTS.items():
        panel_maps = {
            task: {roi_key: subject_task_maps[task][roi_key].get(subject)
                   for roi_key, _, _ in ROI_SPEC}
            for task in col_tasks
        }
        fig, axes = plt.subplots(4, 2, figsize=(10, 16))
        fig.suptitle(f'ISC TTC (LOO){sm_label}{hp_label} — {subject} {session}',
                     fontsize=14, fontweight='bold')
        plt.subplots_adjust(top=0.94)
        _render_panels(fig, axes, col_tasks, col_titles, panel_maps, boundary_info,
                       vmin, vmax)
        out = ISC_DIR / f'{subject}_{session}_isc_ttc_filmfest{sm_file}{hp_file}.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved → {out.name}")


# ============================================================================
# FIGURE 2: per-subject per-session (SVF | AHC | filmfest1 | filmfest2)
# ============================================================================

def make_subject_session_figure(subject, session, filmfest_maps, tasks,
                                vmin=None, vmax=None):
    """4 rows × N cols for one subject/session.

    Columns: [SVF], [AHC], [filmfest1, filmfest2 if subject has filmfest data].
    filmfest_maps : dict task -> {roi_key: array} — pre-computed, reused across sessions.
    tasks : sequence of expanded task names (e.g. ['svf', 'ahc', 'filmfest1', 'filmfest2'])
    """
    _title_map = {
        'svf': 'Semantic Fluency',
        'ahc': 'Explanation Generation',
        'filmfest1': 'Filmfest 1',
        'filmfest2': 'Filmfest 2',
    }
    col_tasks = [t for t in ('svf', 'ahc') if t in tasks]
    col_titles = [_title_map[t] for t in col_tasks]

    if subject in FILMFEST_SUBJECTS and 'filmfest1' in tasks:
        col_tasks += ['filmfest1', 'filmfest2']
        col_titles += ['Filmfest 1', 'Filmfest 2']

    if not col_tasks:
        return

    panel_maps = {}

    # SVF and AHC: compute for this specific session
    for task in ('svf', 'ahc'):
        panel_maps[task] = {}
        for roi_key, _, roi_ids in ROI_SPEC:
            try:
                panel_maps[task][roi_key] = load_or_compute_corrmap(
                    subject, session, task, roi_key, roi_ids)
            except Exception as e:
                print(f"  SKIP {subject} {session} {task} {roi_key}: {e}")
                panel_maps[task][roi_key] = None

    # Filmfest: reuse pre-computed subject-level maps
    if subject in FILMFEST_SUBJECTS:
        for task in ('filmfest1', 'filmfest2'):
            panel_maps[task] = filmfest_maps[task]

    # Shared color scale: use provided global limits, fall back to local
    if vmin is None or vmax is None:
        all_panel_maps = {f'{t}_{k}': panel_maps[t].get(k)
                          for t in col_tasks for k, _, _ in ROI_SPEC}
        vmin, vmax = _vminmax(all_panel_maps)

    # Boundary overlays
    svf_onsets, svf_offsets = get_trial_times(subject, session, 'svf')
    ahc_onsets, ahc_offsets = get_trial_times(subject, session, 'ahc')
    boundary_info = {
        'svf': dict(onsets=svf_onsets, offsets=svf_offsets),
        'ahc': dict(onsets=ahc_onsets, offsets=ahc_offsets),
    }
    if subject in FILMFEST_SUBJECTS:
        boundary_info['filmfest1'] = dict(
            movie_boundaries=get_movie_boundary_offsets('filmfest1'))
        boundary_info['filmfest2'] = dict(
            movie_boundaries=get_movie_boundary_offsets('filmfest2'))

    n_cols = len(col_tasks)
    fig, axes = plt.subplots(4, n_cols, figsize=(5 * n_cols, 16))
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(f'Within-Subject Time-Time Correlation Maps — {subject} {session}',
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.94)
    _render_panels(fig, axes, col_tasks, col_titles, panel_maps, boundary_info,
                   vmin, vmax)

    out = WITHIN_SUBJ_DIR / f'{subject}_{session}_within_subj_ttc_filmfest.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out.name}")


# ============================================================================
# PARALLEL CACHE WARMING
# ============================================================================

def _cache_job(subject, session, task, roi_key, roi_ids):
    """Worker: compute and cache one corrmap. Safe to call in a subprocess."""
    try:
        load_or_compute_corrmap(subject, session, task, roi_key, roi_ids)
    except Exception as e:
        print(f"  SKIP {subject} {session} {task} {roi_key}: {e}")


def warm_cache_parallel(n_jobs=-1, tasks=('svf', 'ahc', 'filmfest1', 'filmfest2')):
    """Build all (subject, session, task, roi_key) jobs and run in parallel."""
    jobs = []

    # SVF + AHC
    for subject in SUBJECT_IDS:
        for session, task in discover_svf_ahc_sessions(subject):
            if task not in tasks:
                continue
            for roi_key, _, roi_ids in ROI_SPEC:
                jobs.append((subject, session, task, roi_key, roi_ids))

    # Filmfest
    for subject, session in FILMFEST_SUBJECTS.items():
        for task in ('filmfest1', 'filmfest2'):
            if task not in tasks:
                continue
            for roi_key, _, roi_ids in ROI_SPEC:
                jobs.append((subject, session, task, roi_key, roi_ids))

    # Cap parallelism to avoid loading too many large BOLD volumes simultaneously
    safe_n_jobs = 4 if (n_jobs == -1 or n_jobs > 4) else n_jobs
    print(f"\nParallel cache warming: {len(jobs)} jobs, n_jobs={safe_n_jobs}")
    Parallel(n_jobs=safe_n_jobs, verbose=10)(
        delayed(_cache_job)(subj, ses, task, roi_key, roi_ids)
        for subj, ses, task, roi_key, roi_ids in jobs
    )
    print("Cache warming complete.")


# ============================================================================
# MAIN
# ============================================================================

def main(n_jobs=-1, tasks=('svf', 'ahc', 'filmfest'), isc=False, fwhm=None, do_hp=True):
    # Expand 'filmfest' shorthand into the two task names used internally
    expanded = []
    for t in tasks:
        if t == 'filmfest':
            expanded.extend(['filmfest1', 'filmfest2'])
        else:
            expanded.append(t)
    tasks = expanded

    for d in (OUTPUT_DIR, WITHIN_SUBJ_DIR, ISC_DIR, GROUP_AVG_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if isc and 'filmfest1' not in tasks:
        print("WARNING: --isc is only supported with --tasks filmfest. Ignoring --isc.")
        isc = False

    print("=" * 60)
    print("TIME-TIME CORRELATION MAPS")
    print(f"Tasks: {tasks}")
    print(f"ISC: {isc}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # --- Parallel cache warming (skips already-cached jobs) ---
    warm_cache_parallel(n_jobs=n_jobs, tasks=tasks)

    # --- Fixed global color scale ---
    vmin_global, vmax_global = -0.3, 0.3
    print(f"\nGlobal color scale: [{vmin_global:.3f}, {vmax_global:.3f}]")

    # --- Group filmfest figure + pre-load per-subject filmfest maps ---
    subject_filmfest_maps = {}
    if 'filmfest1' in tasks:
        make_group_filmfest_figure(vmin=vmin_global, vmax=vmax_global)

        print(f"\n{'=' * 60}")
        print("PRE-LOADING PER-SUBJECT FILMFEST MAPS")
        print(f"{'=' * 60}")
        for subject, session in FILMFEST_SUBJECTS.items():
            subject_filmfest_maps[subject] = {}
            for task in ('filmfest1', 'filmfest2'):
                subject_filmfest_maps[subject][task] = {}
                for roi_key, _, roi_ids in ROI_SPEC:
                    try:
                        subject_filmfest_maps[subject][task][roi_key] = \
                            load_or_compute_corrmap(subject, session, task, roi_key, roi_ids)
                    except Exception as e:
                        print(f"  SKIP {subject} {session} {task} {roi_key}: {e}")
                        subject_filmfest_maps[subject][task][roi_key] = None

        print(f"\n{'=' * 60}")
        print("BUILDING PER-SUBJECT FILMFEST FIGURES")
        print(f"{'=' * 60}")
        for subject, session in FILMFEST_SUBJECTS.items():
            print(f"\n  {subject} {session}")
            make_subject_session_figure(subject, session, subject_filmfest_maps.get(subject),
                                        tasks, vmin=vmin_global, vmax=vmax_global)

        if isc:
            make_filmfest_isc_figures(vmin=vmin_global, vmax=vmax_global,
                                     fwhm=fwhm, do_hp=do_hp)

    # --- Per-subject SVF/AHC figures ---
    if any(t in tasks for t in ('svf', 'ahc')):
        print(f"\n{'=' * 60}")
        print("BUILDING PER-SUBJECT SVF/AHC FIGURES")
        print(f"{'=' * 60}")
        for subject in SUBJECT_IDS:
            sessions_tasks = discover_svf_ahc_sessions(subject)
            sessions = sorted(set(s for s, _ in sessions_tasks))
            filmfest_maps = subject_filmfest_maps.get(subject)
            for session in sessions:
                print(f"\n  {subject} {session}")
                make_subject_session_figure(subject, session, filmfest_maps, tasks,
                                            vmin=vmin_global, vmax=vmax_global)

    print("\n" + "=" * 60)
    print(f"DONE. Figures saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all cores)')
    parser.add_argument('--tasks', nargs='+', default=['svf', 'ahc', 'filmfest'],
                        choices=['svf', 'ahc', 'filmfest'],
                        help='Tasks to run (default: all)')
    parser.add_argument('--isc', action='store_true',
                        help='Compute inter-subject T×T correlation maps (filmfest only)')
    parser.add_argument('--fwhm', type=float, default=None,
                        help='Spatial smoothing FWHM in mm for ISC mode (e.g. --fwhm 6)')
    parser.add_argument('--no-hp', dest='no_hp', action='store_true',
                        help='Skip 0.01 Hz high-pass filtering in ISC mode')
    parser.add_argument('--smoothed_only', action='store_true',
                        help='Only produce group-average smoothed (4mm) TTC figures')
    args = parser.parse_args()
    if args.smoothed_only:
        run_group_pattern_ttc_filmfest()
    else:
        main(n_jobs=args.n_jobs, tasks=args.tasks, isc=args.isc,
             fwhm=args.fwhm, do_hp=not args.no_hp)
