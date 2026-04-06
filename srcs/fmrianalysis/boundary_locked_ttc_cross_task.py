"""
Cross-Task Boundary-Locked TTC Analysis

Extends boundary_ttc.py to compare neural patterns across three tasks:
filmfest (movie boundaries), SVF (trial boundaries), and AHC (trial boundaries).

Figure layout: 4 ROIs × 6 combinations
  Col 0: filmfest within-task
  Col 1: SVF within-task
  Col 2: AHC within-task
  Col 3: filmfest × SVF
  Col 4: filmfest × AHC
  Col 5: SVF × AHC

All matrices are 50×50 TRs (TRIAL_WIN_SIZE, -20 to +29 TRs around boundary).
Filmfest voxel data is cached at 81-TR size but windows are trimmed to 50 TRs.
Spatial smoothing: 4mm FWHM Gaussian applied before parcel extraction.

Usage:
    uv run python srcs/fmrianalysis/boundary_ttc_cross_task.py
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.stats import zscore as sp_zscore
import matplotlib.pyplot as plt
from nilearn.image import smooth_img
from joblib import Parallel, delayed

from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, SUBJECT_IDS, ANALYSIS_CACHE_DIR, FILMFEST_SUBJECTS
from fmrianalysis.utils import (
    highpass_filter, get_bold_path, cross_corrcoef, group_average,
    get_movie_boundary_offsets, get_trial_times, discover_svf_ahc_sessions,
)
from fmrianalysis.boundary_ttc import (
    extract_windows,
    compute_cross_ttc,
    ROI_SPEC,
    SCHAEFER,
    _get_atlas_data,
    TRIAL_WIN_SIZE,
    TRIAL_TRS_BEFORE,
    TRIAL_NEXT_ONSET_WITHIN_WIN,
    VMIN_CROSS,
    VMAX_CROSS,
    _add_boundary_markers,
    _setup_time_axis_ticks,
)

# ============================================================================
# CONSTANTS
# ============================================================================

OUTPUT_DIR = FIGS_DIR / 'boundary_ttc_cross_task'
CACHE_DIR  = ANALYSIS_CACHE_DIR / 'boundary_ttc_cross_task'

# All 6 task-combination columns in display order
COMBO_KEYS = [
    'filmfest_within',
    'svf_within',
    'ahc_within',
    'filmfest_svf',
    'filmfest_ahc',
    'svf_ahc',
]

COMBO_TITLES = [
    'Filmfest',
    'SVF',
    'AHC',
    'Filmfest × SVF',
    'Filmfest × AHC',
    'SVF × AHC',
]

# For cross-task combos: (row_task_label, col_task_label)
CROSS_TASK_AXIS_LABELS = {
    'filmfest_svf': ('Filmfest', 'SVF'),
    'filmfest_ahc': ('Filmfest', 'AHC'),
    'svf_ahc':      ('SVF',      'AHC'),
}

# Combos that use filmfest boundaries — suppress next-onset marker
FILMFEST_COMBOS_NO_ONSET = {'filmfest_within', 'filmfest_svf', 'filmfest_ahc'}

# Subject sets per combo: filmfest combos use FILMFEST_SUBJECTS, SVF/AHC use all subjects
FILMFEST_COMBOS = {'filmfest_within', 'filmfest_svf', 'filmfest_ahc'}


# ============================================================================
# SMOOTHED VOXEL DATA LOADING
# ============================================================================

def load_roi_voxel_data_smoothed(subject, session, task, roi_parcel_ids, roi_key):
    """Load multi-voxel ROI pattern for one run with 4mm spatial smoothing.

    Applies smooth_img(bold, fwhm=4) before parcel extraction, then high-pass
    filters and temporally z-scores per voxel.

    Cache key: {subject}_{session}_task-{task}_roi-{roi_key}_voxels_hp_sm4.npz
    Stored in: boundary_ttc_cross_task/cache/

    Returns
    -------
    voxel_data : np.ndarray, shape (T, n_voxels)
    """
    cache_file = (CACHE_DIR /
                  f"{subject}_{session}_task-{task}_roi-{roi_key}_voxels_hp_sm4.npz")
    if cache_file.exists():
        return np.load(cache_file)['voxel_data']

    bold_path = get_bold_path(subject, session, task)
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD not found: {bold_path}")

    print(f"    Smoothing {bold_path.name} (4mm FWHM)...")
    bold_img    = nib.load(str(bold_path))
    bold_smooth = smooth_img(bold_img, fwhm=4)
    bold_data   = bold_smooth.get_fdata()      # (X, Y, Z, T)
    atlas_data  = _get_atlas_data(bold_path)   # (X, Y, Z)

    parcel_parts = []
    for parcel_id in roi_parcel_ids:
        mask = atlas_data == parcel_id
        if mask.sum() < 2:
            continue
        pdata = bold_data[mask].T   # (T, n_voxels)
        pdata = highpass_filter(pdata, order=2)
        pdata = sp_zscore(pdata, axis=0, nan_policy='omit')
        pdata = np.nan_to_num(pdata)
        parcel_parts.append(pdata)

    del bold_data, bold_smooth

    if not parcel_parts:
        raise ValueError(
            f"No usable parcels for {subject} {session} {task} roi={roi_key}")

    voxel_data = np.concatenate(parcel_parts, axis=1).astype(np.float32)
    np.savez_compressed(cache_file, voxel_data=voxel_data)
    print(f"    Cached voxels → {cache_file.name}")
    return voxel_data


# ============================================================================
# WINDOW COLLECTION (smoothed)
# ============================================================================

def get_filmfest_windows_trial_size(subject, roi_key, roi_parcel_ids):
    """Filmfest boundary windows trimmed to TRIAL_WIN_SIZE (50 TRs).

    Uses smoothed voxel data. Anchored at movie-end boundaries but windows
    are extracted with TRIAL_WIN_SIZE / TRIAL_TRS_BEFORE so all tasks share
    the same 50×50 matrix size.
    """
    session = FILMFEST_SUBJECTS[subject]
    all_windows = []
    for task in ('filmfest1', 'filmfest2'):
        try:
            voxel_data = load_roi_voxel_data_smoothed(
                subject, session, task, roi_parcel_ids, roi_key)
        except (FileNotFoundError, ValueError) as e:
            print(f"    SKIP filmfest {subject} {task}: {e}")
            continue
        boundary_secs = get_movie_boundary_offsets(task)
        boundary_trs  = [int(round(s / TR)) for s in boundary_secs]
        wins = extract_windows(voxel_data, boundary_trs,
                               TRIAL_WIN_SIZE, TRIAL_TRS_BEFORE)
        all_windows.extend(wins)
        print(f"    {subject} {task}: {len(wins)}/{len(boundary_trs)} windows")
    return all_windows


def get_trial_windows_smoothed(subject, task, roi_key, roi_parcel_ids):
    """Trial-boundary windows for SVF or AHC using smoothed voxel data."""
    all_windows = []
    sessions_tasks = discover_svf_ahc_sessions(subject)
    for session, t in sessions_tasks:
        if t != task:
            continue
        try:
            voxel_data = load_roi_voxel_data_smoothed(
                subject, session, task, roi_parcel_ids, roi_key)
        except (FileNotFoundError, ValueError) as e:
            print(f"    SKIP {subject} {session} {task}: {e}")
            continue
        onsets, offsets = get_trial_times(subject, session, task)
        if len(offsets) < 2:
            continue
        boundary_trs = [int(round(off / TR)) for off in offsets[:-1]]
        wins = extract_windows(voxel_data, boundary_trs,
                               TRIAL_WIN_SIZE, TRIAL_TRS_BEFORE)
        all_windows.extend(wins)
        print(f"    {subject} {session} {task}: {len(wins)}/{len(boundary_trs)} windows")
    return all_windows


# ============================================================================
# CROSS-TASK TTC
# ============================================================================

def compute_crosstask_ttc(windows_A, windows_B):
    """Fisher-z average of cross_corrcoef(w_A_i, w_B_j) for all pairs (i, j).

    Row axis = task A time, col axis = task B time.
    No symmetrization (unlike within-task cross-boundary TTC).

    Returns
    -------
    (TRIAL_WIN_SIZE, TRIAL_WIN_SIZE) array or None if either list is empty.
    """
    if not windows_A or not windows_B:
        return None
    z_maps = []
    for w_a in windows_A:
        for w_b in windows_B:
            cij = cross_corrcoef(w_a, w_b)   # (W, W)
            z_maps.append(np.arctanh(np.clip(cij, -0.999, 0.999)))
    return np.tanh(np.stack(z_maps, axis=0).mean(axis=0))


# ============================================================================
# PER-SUBJECT RESULT COMPUTATION (with caching)
# ============================================================================

def compute_subject_cross_task(subject, roi_key, roi_parcel_ids):
    """Compute all 6 combo TTC matrices for one subject/ROI.

    Returns dict combo_key -> (50, 50) array or None.
    Only computes combos where the subject has data.
    """
    # Check which combos this subject contributes to
    has_filmfest = subject in FILMFEST_SUBJECTS
    has_svf_ahc  = subject in SUBJECT_IDS

    # Load windows (only tasks this subject has)
    windows = {}

    if has_filmfest:
        print(f"  {subject} roi={roi_key}: loading filmfest windows...")
        windows['filmfest'] = get_filmfest_windows_trial_size(
            subject, roi_key, roi_parcel_ids)
        print(f"    filmfest total windows: {len(windows['filmfest'])}")

    if has_svf_ahc:
        print(f"  {subject} roi={roi_key}: loading SVF windows...")
        windows['svf'] = get_trial_windows_smoothed(
            subject, 'svf', roi_key, roi_parcel_ids)
        print(f"    svf total windows: {len(windows['svf'])}")

        print(f"  {subject} roi={roi_key}: loading AHC windows...")
        windows['ahc'] = get_trial_windows_smoothed(
            subject, 'ahc', roi_key, roi_parcel_ids)
        print(f"    ahc total windows: {len(windows['ahc'])}")

    results = {}

    if has_filmfest:
        results['filmfest_within'] = compute_cross_ttc(windows.get('filmfest', []))

    if has_svf_ahc:
        results['svf_within'] = compute_cross_ttc(windows.get('svf', []))
        results['ahc_within'] = compute_cross_ttc(windows.get('ahc', []))

    if has_filmfest and has_svf_ahc:
        results['filmfest_svf'] = compute_crosstask_ttc(
            windows.get('filmfest', []), windows.get('svf', []))
        results['filmfest_ahc'] = compute_crosstask_ttc(
            windows.get('filmfest', []), windows.get('ahc', []))

    if has_svf_ahc:
        results['svf_ahc'] = compute_crosstask_ttc(
            windows.get('svf', []), windows.get('ahc', []))

    return results


def load_or_compute_subject(subject, roi_key, roi_parcel_ids):
    """Load cached combo TTC results for subject/ROI, or compute and cache."""
    # Check if all applicable caches exist
    has_filmfest = subject in FILMFEST_SUBJECTS
    has_svf_ahc  = subject in SUBJECT_IDS

    applicable_combos = []
    if has_filmfest:
        applicable_combos.append('filmfest_within')
    if has_svf_ahc:
        applicable_combos += ['svf_within', 'ahc_within', 'svf_ahc']
    if has_filmfest and has_svf_ahc:
        applicable_combos += ['filmfest_svf', 'filmfest_ahc']

    cache_files = {
        combo: CACHE_DIR / f"{subject}_roi-{roi_key}_{combo}_ttc.npz"
        for combo in applicable_combos
    }

    if all(f.exists() for f in cache_files.values()):
        print(f"  {subject} roi={roi_key}: loading TTC cache")
        results = {}
        for combo, f in cache_files.items():
            arr = np.load(f)['ttc']
            results[combo] = arr if arr.ndim == 2 else None
        return results

    print(f"  {subject} roi={roi_key}: computing cross-task TTC...")
    results = compute_subject_cross_task(subject, roi_key, roi_parcel_ids)

    # Cache all results
    for combo in applicable_combos:
        m = results.get(combo)
        f = cache_files[combo]
        if m is not None:
            np.savez_compressed(f, ttc=m)
        else:
            np.savez_compressed(f, ttc=np.array([]))
        print(f"    Cached → {f.name}")

    return results


# ============================================================================
# FIGURE
# ============================================================================

def _make_figure(subject_label, roi_results):
    """Create 4-row × 6-col cross-task TTC figure.

    Parameters
    ----------
    subject_label : str — e.g. 'sub-003' or 'GROUP'
    roi_results   : dict roi_key -> dict combo_key -> (50,50) array or None
    """
    n_rows = len(ROI_SPEC)
    n_cols = len(COMBO_KEYS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Cross-Task Boundary-Locked TTC (Within-Subject) — {subject_label}',
        fontsize=13, fontweight='bold'
    )
    plt.subplots_adjust(top=0.94, hspace=0.4, wspace=0.4)

    vlo, vhi = VMIN_CROSS, VMAX_CROSS

    for row, (roi_key, roi_name, _) in enumerate(ROI_SPEC):
        for col, (combo_key, combo_title) in enumerate(zip(COMBO_KEYS, COMBO_TITLES)):
            ax = axes[row, col]
            m  = roi_results.get(roi_key, {}).get(combo_key)

            if m is None or np.ndim(m) != 2:
                ax.set_visible(False)
                continue

            ax.imshow(m, cmap='RdBu_r', vmin=vlo, vmax=vhi,
                      aspect='equal', origin='upper')

            # t=0 lines always; suppress next-onset for filmfest boundaries
            btype = 'filmfest_movie' if combo_key in FILMFEST_COMBOS_NO_ONSET else 'svf_trial'
            _add_boundary_markers(ax, TRIAL_WIN_SIZE, TRIAL_TRS_BEFORE, btype)
            # For cross-task cols where col=SVF/AHC, add next-onset on x-axis only
            if combo_key in ('filmfest_svf', 'filmfest_ahc'):
                ax.axvline(TRIAL_NEXT_ONSET_WITHIN_WIN,
                           color='k', lw=1.0, alpha=0.9, linestyle='--')
            _setup_time_axis_ticks(ax, TRIAL_WIN_SIZE, TRIAL_TRS_BEFORE)

            if row == 0:
                ax.set_title(combo_title, fontsize=9, fontweight='bold')

            cross_labels = CROSS_TASK_AXIS_LABELS.get(combo_key)
            if cross_labels:
                row_task, col_task = cross_labels
                ax.set_ylabel(f'{row_task} time (s)', fontsize=8)
                ax.set_xlabel(f'{col_task} time (s)', fontsize=8)
            else:
                # Within-task: ROI label on col 0, time label on bottom row
                if col == 0:
                    ax.set_ylabel(f'{roi_name}\nTime rel. boundary (s)', fontsize=8)
                if row == n_rows - 1:
                    ax.set_xlabel('Time rel. boundary (s)', fontsize=8)

    # Single shared colorbar for all subplots
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                               norm=plt.Normalize(vmin=vlo, vmax=vhi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.015, pad=0.03, shrink=0.6)
    cbar.set_label('Pearson r', fontsize=9)

    return fig


# ============================================================================
# PARALLEL WORKER
# ============================================================================

def _compute_job(subject, roi_key, roi_ids):
    """Worker: compute and cache all combo TTC matrices for one subject/ROI."""
    try:
        results = load_or_compute_subject(subject, roi_key, roi_ids)
        return subject, roi_key, results
    except Exception as e:
        print(f"  ERROR {subject} roi={roi_key}: {e}")
        return subject, roi_key, {}


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(n_jobs=4):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CROSS-TASK BOUNDARY-LOCKED TTC ANALYSIS")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"n_jobs: {n_jobs}")
    print("=" * 60)

    # All subjects that appear in either filmfest or SVF/AHC
    all_subjects = sorted(set(list(FILMFEST_SUBJECTS.keys()) + list(SUBJECT_IDS)))

    # Build list of (subject, roi_key, roi_ids) jobs
    jobs = [
        (subject, roi_key, roi_ids)
        for subject in all_subjects
        for roi_key, _, roi_ids in ROI_SPEC
    ]
    print(f"Total jobs: {len(jobs)}  (subjects × ROIs = {len(all_subjects)} × {len(ROI_SPEC)})")

    # Run in parallel; each worker returns (subject, roi_key, results_dict)
    job_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_compute_job)(subj, roi_key, roi_ids)
        for subj, roi_key, roi_ids in jobs
    )

    # Reassemble into per_subject_results[subject][roi_key]
    per_subject_results = {subj: {} for subj in all_subjects}
    for subject, roi_key, results in job_results:
        per_subject_results[subject][roi_key] = results
        for combo, m in results.items():
            shape_str = str(m.shape) if m is not None else 'None'
            print(f"  {subject} roi={roi_key} {combo}: {shape_str}")

    # ---- Per-subject figures ----
    print("\n" + "=" * 60)
    print("SAVING PER-SUBJECT FIGURES")
    print("=" * 60)

    for subject in all_subjects:
        # Build roi_results[roi_key][combo_key] for this subject
        roi_results = {}
        for roi_key, _, _ in ROI_SPEC:
            roi_results[roi_key] = per_subject_results[subject].get(roi_key, {})

        fig = _make_figure(subject, roi_results)
        out = OUTPUT_DIR / f'{subject}_cross_task_boundary_ttc.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out.name}")

    # ---- Group-average figure ----
    print("\n" + "=" * 60)
    print("BUILDING GROUP FIGURE")
    print("=" * 60)

    # For each combo, average across subjects that have data
    # Filmfest combos: only FILMFEST_SUBJECTS; SVF/AHC combos: all SUBJECT_IDS
    combo_subject_sets = {
        'filmfest_within': list(FILMFEST_SUBJECTS.keys()),
        'svf_within':      list(SUBJECT_IDS),
        'ahc_within':      list(SUBJECT_IDS),
        'filmfest_svf':    [s for s in FILMFEST_SUBJECTS if s in SUBJECT_IDS],
        'filmfest_ahc':    [s for s in FILMFEST_SUBJECTS if s in SUBJECT_IDS],
        'svf_ahc':         list(SUBJECT_IDS),
    }

    group_roi_results = {}
    for roi_key, _, _ in ROI_SPEC:
        group_roi_results[roi_key] = {}
        for combo_key in COMBO_KEYS:
            subj_set = combo_subject_sets[combo_key]
            maps = [
                per_subject_results[subj][roi_key].get(combo_key)
                for subj in subj_set
                if roi_key in per_subject_results.get(subj, {})
                   and per_subject_results[subj][roi_key].get(combo_key) is not None
            ]
            if maps:
                group_roi_results[roi_key][combo_key] = group_average(maps)
                print(f"  Group roi={roi_key} {combo_key}: N={len(maps)}")
            else:
                group_roi_results[roi_key][combo_key] = None
                print(f"  Group roi={roi_key} {combo_key}: no data")

    fig = _make_figure('GROUP', group_roi_results)
    out = OUTPUT_DIR / 'GROUP_cross_task_boundary_ttc.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out.name}")

    print("\n" + "=" * 60)
    print(f"DONE. Figures saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cross-task boundary-locked TTC analysis.'
    )
    parser.add_argument(
        '--n_jobs', type=int, default=4,
        help='Parallel workers for subject×ROI jobs (default: 4)',
    )
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
