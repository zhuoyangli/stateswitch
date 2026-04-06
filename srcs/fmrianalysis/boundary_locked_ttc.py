"""
Boundary-Locked Time-Time Correlation Analysis

Computes within-boundary and cross-boundary pattern similarity matrices
locked to three types of cognitive/narrative boundaries:
  1. filmfest_movie  — 8 between-movie transitions per subject
  2. svf_trial       — SVF trial-offset boundaries (trial i end → trial i+1 start)
  3. ahc_trial       — AHC trial-offset boundaries (same structure)

Within-boundary TTC: average corrcoef(W×W) across all boundary instances.
Cross-boundary TTC:  average cross-corrcoef(W×W) across all between-boundary pairs.

Both use Fisher-z averaging. The cross-boundary matrix is symmetrized per pair
before averaging.

Usage:
    uv run python srcs/fmrianalysis/boundary_ttc.py
    uv run python srcs/fmrianalysis/boundary_ttc.py --boundary_types filmfest_movie svf_trial
"""

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import nibabel as nib
from scipy.stats import zscore as sp_zscore
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, SUBJECT_IDS, ANALYSIS_CACHE_DIR, FILMFEST_SUBJECTS
from configs.schaefer_rois import POSTERIOR_MEDIAL, ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL
from fmrianalysis.utils import (
    highpass_filter, get_bold_path, cross_corrcoef, group_average,
    get_movie_boundary_offsets, get_trial_times, discover_svf_ahc_sessions,
)

# ROI_SPEC: (key, display_name, parcel_id_list)
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

# Load Schaefer atlas for voxel-level analysis
print("Loading Schaefer atlas (boundary_ttc)...")
from nilearn import datasets as _datasets, image as _image
SCHAEFER = _datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
_atlas_resampled = None  # cached atlas array resampled to BOLD voxel space

def _get_atlas_data(bold_path):
    """Return Schaefer atlas resampled to BOLD voxel space (module-level cache)."""
    global _atlas_resampled
    if _atlas_resampled is None:
        import numpy as np
        bold_ref = _image.index_img(str(bold_path), 0)
        atlas_resampled = _image.resample_to_img(
            SCHAEFER['maps'], bold_ref, interpolation='nearest'
        )
        _atlas_resampled = np.round(atlas_resampled.get_fdata()).astype(int)
        print(f"  Atlas resampled to BOLD space: {_atlas_resampled.shape}")
    return _atlas_resampled

# ============================================================================
# CONSTANTS
# ============================================================================

OUTPUT_DIR = FIGS_DIR / 'boundary_ttc'
CACHE_DIR  = ANALYSIS_CACHE_DIR / 'boundary_ttc'

# Filmfest window: ±40 TRs centered on movie-end boundary (81 TRs total = ±60s)
FILMFEST_TRS_BEFORE = 40
FILMFEST_WIN_SIZE   = 81    # data[btr-40 : btr+41]

# Trial window: -20 to +29 TRs relative to trial offset (50 TRs total)
# t=0  = trial offset (current trial end)
# t=+10 TRs = +15 s = next trial onset (after ~15 s ITI)
TRIAL_TRS_BEFORE = 20
TRIAL_WIN_SIZE   = 50    # data[btr-20 : btr+30]
TRIAL_NEXT_ONSET_WITHIN_WIN = TRIAL_TRS_BEFORE + 10   # = 30

BOUNDARY_TYPES = ('filmfest_movie', 'svf_trial', 'ahc_trial')

# Fixed color limits applied uniformly across all subjects and the group figure
VMIN_WITHIN, VMAX_WITHIN = -1.0,  1.0
VMIN_CROSS,  VMAX_CROSS  = -0.1,  0.1


# ============================================================================
# VOXEL DATA LOADING (temporally z-scored, cached)
# ============================================================================

def load_roi_voxel_data(subject, session, task, roi_parcel_ids, roi_key):
    """Load multi-voxel ROI pattern for one run, temporally z-scored (Round 1).

    Returns
    -------
    voxel_data : np.ndarray, shape (T, n_voxels)
    """
    cache_file = (CACHE_DIR /
                  f"{subject}_{session}_task-{task}_roi-{roi_key}_voxels_hp.npz")
    if cache_file.exists():
        return np.load(cache_file)['voxel_data']

    bold_path = get_bold_path(subject, session, task)
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD not found: {bold_path}")

    bold_data  = nib.load(str(bold_path)).get_fdata()   # (X, Y, Z, T)
    atlas_data = _get_atlas_data(bold_path)              # (X, Y, Z)

    parcel_parts = []
    for parcel_id in roi_parcel_ids:
        mask = atlas_data == parcel_id
        if mask.sum() < 2:
            continue
        pdata = bold_data[mask].T   # (T, n_voxels)
        # High-pass filter at 0.01 Hz before z-scoring
        pdata = highpass_filter(pdata, order=2)
        # Round 1: temporal z-score per voxel
        pdata = sp_zscore(pdata, axis=0, nan_policy='omit')
        pdata = np.nan_to_num(pdata)
        parcel_parts.append(pdata)

    del bold_data   # free memory early

    if not parcel_parts:
        raise ValueError(f"No usable parcels for {subject} {session} {task} roi={roi_key}")

    voxel_data = np.concatenate(parcel_parts, axis=1).astype(np.float32)  # (T, total_voxels)
    np.savez_compressed(cache_file, voxel_data=voxel_data)
    print(f"    Cached voxels → {cache_file.name}")
    return voxel_data


# ============================================================================
# WINDOW EXTRACTION (spatially z-scored, Round 2)
# ============================================================================

def extract_windows(voxel_data, boundary_trs, win_size, trs_before):
    """Extract ±window TRs around each boundary; apply spatial z-score per TR.

    Parameters
    ----------
    voxel_data   : (T, n_voxels) float array (already temporally z-scored)
    boundary_trs : list of int — boundary anchor TR indices (0-based)
    win_size     : int — total window length in TRs
    trs_before   : int — TRs before boundary anchor included in window

    Returns
    -------
    windows : list of (win_size, n_voxels) arrays, spatially z-scored
    """
    T = voxel_data.shape[0]
    trs_after = win_size - trs_before   # exclusive: data[start : start+win_size]
    windows = []
    for btr in boundary_trs:
        start = btr - trs_before
        end   = start + win_size   # exclusive
        if start < 0 or end > T:
            print(f"    Skipping boundary TR={btr}: window [{start}, {end}) out of range [0, {T})")
            continue
        w = voxel_data[start:end, :].copy().astype(np.float64)
        # Round 2: spatial z-score per timepoint across voxels
        w = sp_zscore(w, axis=1, nan_policy='omit')
        w = np.nan_to_num(w)
        windows.append(w)
    return windows


# ============================================================================
# WITHIN- AND CROSS-BOUNDARY TTC
# ============================================================================

def compute_within_ttc(windows):
    """Fisher-z average of per-window corrcoef matrices.

    Returns (W, W) array or None if < 1 window.
    """
    if not windows:
        return None
    z_maps = []
    for w in windows:
        c = np.corrcoef(w)   # (W, W)
        z_maps.append(np.arctanh(np.clip(c, -0.999, 0.999)))
    return np.tanh(np.stack(z_maps, axis=0).mean(axis=0))


def compute_cross_ttc(windows):
    """Fisher-z average of symmetrized cross-boundary TTCs for all pairs (i,j), i<j.

    Entry [t1,t2] = mean Pearson r between pattern at t1 in one window
    and pattern at t2 in a different window.

    Returns (W, W) array or None if < 2 windows.
    """
    if len(windows) < 2:
        return None
    z_maps = []
    for i, j in combinations(range(len(windows)), 2):
        cij = cross_corrcoef(windows[i], windows[j])   # (W, W)
        # Symmetrize: avg(cij, cij.T) so entry [t1,t2] = avg corr(win_i[t1], win_j[t2])
        #             and corr(win_i[t2], win_j[t1])
        sym = (cij + cij.T) / 2.0
        z_maps.append(np.arctanh(np.clip(sym, -0.999, 0.999)))
    return np.tanh(np.stack(z_maps, axis=0).mean(axis=0))


# ============================================================================
# BOUNDARY-SPECIFIC WINDOW COLLECTION
# ============================================================================

def get_filmfest_windows(subject, roi_key, roi_parcel_ids):
    """Collect all 8 filmfest boundary windows for one subject/ROI."""
    session = FILMFEST_SUBJECTS[subject]
    all_windows = []
    for task in ('filmfest1', 'filmfest2'):
        try:
            voxel_data = load_roi_voxel_data(subject, session, task,
                                             roi_parcel_ids, roi_key)
        except (FileNotFoundError, ValueError) as e:
            print(f"    SKIP filmfest {subject} {task}: {e}")
            continue
        boundary_secs = get_movie_boundary_offsets(task)
        boundary_trs  = [int(round(s / TR)) for s in boundary_secs]
        wins = extract_windows(voxel_data, boundary_trs,
                               FILMFEST_WIN_SIZE, FILMFEST_TRS_BEFORE)
        all_windows.extend(wins)
        print(f"    {subject} {task}: {len(wins)}/{len(boundary_trs)} windows extracted")
    return all_windows


def get_trial_windows(subject, task, roi_key, roi_parcel_ids):
    """Collect all trial-boundary windows across sessions for one subject/ROI."""
    all_windows = []
    sessions_tasks = discover_svf_ahc_sessions(subject)
    for session, t in sessions_tasks:
        if t != task:
            continue
        try:
            voxel_data = load_roi_voxel_data(subject, session, task,
                                             roi_parcel_ids, roi_key)
        except (FileNotFoundError, ValueError) as e:
            print(f"    SKIP {subject} {session} {task}: {e}")
            continue
        onsets, offsets = get_trial_times(subject, session, task)
        if len(offsets) < 2:
            continue
        # Anchor at offset of trial i (t=0), skip last trial (no next trial follows)
        boundary_trs = [int(round(off / TR)) for off in offsets[:-1]]
        wins = extract_windows(voxel_data, boundary_trs,
                               TRIAL_WIN_SIZE, TRIAL_TRS_BEFORE)
        all_windows.extend(wins)
        print(f"    {subject} {session} {task}: {len(wins)}/{len(boundary_trs)} windows extracted")
    return all_windows


# ============================================================================
# PER-SUBJECT TTC COMPUTATION (with caching)
# ============================================================================

def compute_subject_ttc(subject, boundary_type, roi_key, roi_parcel_ids):
    """Compute within- and cross-boundary TTC for one subject/ROI.

    Returns dict {'within': (W,W)|None, 'cross': (W,W)|None}.
    """
    within_cache = (CACHE_DIR /
                    f"{subject}_roi-{roi_key}_{boundary_type}_within_ttc_hp.npz")
    cross_cache  = (CACHE_DIR /
                    f"{subject}_roi-{roi_key}_{boundary_type}_cross_ttc_hp.npz")

    if within_cache.exists() and cross_cache.exists():
        print(f"  {subject} roi={roi_key} {boundary_type}: loading TTC cache")
        def _load(f):
            arr = np.load(f)['ttc']
            return arr if arr.ndim == 2 else None
        return {'within': _load(within_cache), 'cross': _load(cross_cache)}

    print(f"  {subject} roi={roi_key} {boundary_type}: collecting windows...")

    if boundary_type == 'filmfest_movie':
        windows = get_filmfest_windows(subject, roi_key, roi_parcel_ids)
    elif boundary_type == 'svf_trial':
        windows = get_trial_windows(subject, 'svf', roi_key, roi_parcel_ids)
    elif boundary_type == 'ahc_trial':
        windows = get_trial_windows(subject, 'ahc', roi_key, roi_parcel_ids)
    else:
        raise ValueError(f"Unknown boundary_type: {boundary_type}")

    print(f"    Total windows: {len(windows)}")

    within = compute_within_ttc(windows)
    cross  = compute_cross_ttc(windows)

    # Cache results (save sentinel with shape if None)
    if within is not None:
        np.savez_compressed(within_cache, ttc=within)
    else:
        np.savez_compressed(within_cache, ttc=np.array([]))   # empty sentinel
    if cross is not None:
        np.savez_compressed(cross_cache, ttc=cross)
    else:
        np.savez_compressed(cross_cache, ttc=np.array([]))

    return {'within': within, 'cross': cross}


def load_subject_ttc(subject, boundary_type, roi_key, roi_parcel_ids):
    """Load cached TTC or compute if missing."""
    within_cache = (CACHE_DIR /
                    f"{subject}_roi-{roi_key}_{boundary_type}_within_ttc_hp.npz")
    cross_cache  = (CACHE_DIR /
                    f"{subject}_roi-{roi_key}_{boundary_type}_cross_ttc_hp.npz")

    if within_cache.exists() and cross_cache.exists():
        within_arr = np.load(within_cache)['ttc']
        cross_arr  = np.load(cross_cache)['ttc']
        within = within_arr if within_arr.ndim == 2 else None
        cross  = cross_arr  if cross_arr.ndim  == 2 else None
        if within is not None or cross is not None:
            print(f"  {subject} roi={roi_key} {boundary_type}: loading TTC cache")
            return {'within': within, 'cross': cross}

    return compute_subject_ttc(subject, boundary_type, roi_key, roi_parcel_ids)


# ============================================================================
# FIGURE HELPERS
# ============================================================================

def _time_axis(win_size, trs_before):
    """Return time axis in seconds (t=0 at trs_before)."""
    return (np.arange(win_size) - trs_before) * TR


def _add_boundary_markers(ax, win_size, trs_before, boundary_type):
    """Draw t=0 and (for trial types) next-trial-onset lines."""
    kw = dict(color='k', lw=1.0, alpha=0.9, linestyle='--')
    ax.axvline(trs_before, **kw)
    ax.axhline(trs_before, **kw)
    if boundary_type in ('svf_trial', 'ahc_trial'):
        onset_tr = TRIAL_NEXT_ONSET_WITHIN_WIN
        ax.axvline(onset_tr, color='k', lw=1.0, alpha=0.9, linestyle='--')
        ax.axhline(onset_tr, color='k', lw=1.0, alpha=0.9, linestyle='--')


def _setup_time_axis_ticks(ax, win_size, trs_before, tick_interval_s=15):
    """Set x/y tick labels every tick_interval_s seconds relative to boundary."""
    tick_interval_tr = int(round(tick_interval_s / TR))   # 10 TRs at 1.5s
    t_start_tr = -trs_before
    t_end_tr   = win_size - 1 - trs_before
    k_start = int(np.ceil(t_start_tr  / tick_interval_tr))
    k_end   = int(np.floor(t_end_tr   / tick_interval_tr))
    tick_idxs   = [k * tick_interval_tr + trs_before for k in range(k_start, k_end + 1)]
    tick_labels = [f'{(idx - trs_before) * TR:.0f}' for idx in tick_idxs]
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_yticks(tick_idxs)
    ax.set_yticklabels(tick_labels, fontsize=7)


def _col_vlims(roi_results, col_key):
    """±99th percentile of off-diagonal values for one column (within or cross)."""
    vals = []
    for rd in roi_results.values():
        m = rd.get(col_key)
        if m is not None and m.ndim == 2:
            W = m.shape[0]
            vals.append(m[~np.eye(W, dtype=bool)])
    if not vals:
        return -0.3, 0.3
    vmax = float(np.percentile(np.abs(np.concatenate(vals)), 99))
    return -vmax, vmax


def _make_figure(subject_label, boundary_type, roi_results, win_size, trs_before,
                 vmin_within=None, vmax_within=None, vmin_cross=None, vmax_cross=None):
    """Create a 4×2 figure: rows=ROIs, cols=[within, cross].

    Within and cross columns use independent color scales so the very different
    correlation ranges (within: all-positive ~0–0.6; cross: near-zero ±0.15)
    are each displayed with full colormap resolution.

    Parameters
    ----------
    subject_label              : str — used in title (e.g. 'sub-003' or 'GROUP')
    roi_results                : dict roi_key -> {'within': array|None, 'cross': array|None}
    vmin_within/vmax_within    : color limits for within column (auto if None)
    vmin_cross/vmax_cross      : color limits for cross column (auto if None)
    """
    _boundary_title = {
        'filmfest_movie': 'Filmfest Movie Boundary',
        'svf_trial':      'SVF Trial Boundary',
        'ahc_trial':      'AHC Trial Boundary',
    }

    # Per-column color limits (computed from roi_results if not supplied)
    if vmin_within is None or vmax_within is None:
        vmin_within, vmax_within = _col_vlims(roi_results, 'within')
    if vmin_cross is None or vmax_cross is None:
        vmin_cross, vmax_cross = _col_vlims(roi_results, 'cross')

    col_vlims  = {'within': (vmin_within, vmax_within),
                  'cross':  (vmin_cross,  vmax_cross)}

    n_rows = len(ROI_SPEC)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Boundary-Locked TTC — {_boundary_title[boundary_type]} — {subject_label}',
        fontsize=13, fontweight='bold'
    )
    plt.subplots_adjust(top=0.94, hspace=0.35, wspace=0.35)

    col_titles = ['Within-Boundary TTC', 'Cross-Boundary TTC']
    col_keys   = ['within', 'cross']

    for row, (roi_key, roi_name, _) in enumerate(ROI_SPEC):
        for col, (col_key, col_title) in enumerate(zip(col_keys, col_titles)):
            ax  = axes[row, col]
            m   = roi_results.get(roi_key, {}).get(col_key)
            vlo, vhi = col_vlims[col_key]

            if m is None or np.ndim(m) != 2:
                ax.set_visible(False)
                continue

            ax.imshow(m, cmap='RdBu_r', vmin=vlo, vmax=vhi,
                      aspect='equal', origin='upper')

            _add_boundary_markers(ax, win_size, trs_before, boundary_type)
            _setup_time_axis_ticks(ax, win_size, trs_before)

            if row == 0:
                ax.set_title(col_title, fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{roi_name}\nTime rel. boundary (s)', fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel('Time rel. boundary (s)', fontsize=9)

    # One colorbar per column (independent scales)
    for col, col_key in enumerate(col_keys):
        vlo, vhi = col_vlims[col_key]
        sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                   norm=plt.Normalize(vmin=vlo, vmax=vhi))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:, col], orientation='vertical',
                            fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label('Pearson r', fontsize=9)

    return fig


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_boundary_type(boundary_type):
    """Run full analysis for one boundary type."""
    print(f"\n{'=' * 60}")
    print(f"BOUNDARY TYPE: {boundary_type}")
    print(f"{'=' * 60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if boundary_type == 'filmfest_movie':
        subjects = list(FILMFEST_SUBJECTS.keys())
        win_size   = FILMFEST_WIN_SIZE
        trs_before = FILMFEST_TRS_BEFORE
    else:
        subjects   = list(SUBJECT_IDS)
        win_size   = TRIAL_WIN_SIZE
        trs_before = TRIAL_TRS_BEFORE

    # Collect per-subject TTC results: roi_key -> subject -> {'within':…, 'cross':…}
    # Structure: results[roi_key][subject] = {'within': array|None, 'cross': array|None}
    results = {roi_key: {} for roi_key, _, _ in ROI_SPEC}

    for subject in subjects:
        print(f"\n--- {subject} ---")
        for roi_key, roi_name, roi_ids in ROI_SPEC:
            res = load_subject_ttc(subject, boundary_type, roi_key, roi_ids)
            results[roi_key][subject] = res
            for key, m in res.items():
                shape_str = str(m.shape) if m is not None else 'None'
                print(f"    roi={roi_key} {key}: {shape_str}")

    # Build per-subject roi_results dicts for figure-making
    # roi_results_by_subj[subject][roi_key][within|cross] = array|None
    roi_results_by_subj = {subj: {roi_key: results[roi_key][subj]
                                  for roi_key, _, _ in ROI_SPEC}
                           for subj in subjects}

    vmin_within, vmax_within = VMIN_WITHIN, VMAX_WITHIN
    vmin_cross,  vmax_cross  = VMIN_CROSS,  VMAX_CROSS
    print(f"\nColor scale — within: [{vmin_within:.3f}, {vmax_within:.3f}]  "
          f"cross: [{vmin_cross:.3f}, {vmax_cross:.3f}]")

    # ---- Per-subject figures ----
    for subject in subjects:
        fig = _make_figure(
            subject_label=subject,
            boundary_type=boundary_type,
            roi_results=roi_results_by_subj[subject],
            win_size=win_size, trs_before=trs_before,
            vmin_within=vmin_within, vmax_within=vmax_within,
            vmin_cross=vmin_cross,   vmax_cross=vmax_cross,
        )
        out = OUTPUT_DIR / f'{subject}_{boundary_type}_boundary_ttc.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out.name}")

    # ---- Group-average figure ----
    group_roi_results = {}
    for roi_key, _, _ in ROI_SPEC:
        group_roi_results[roi_key] = {}
        for key in ('within', 'cross'):
            maps = [results[roi_key][subj][key]
                    for subj in subjects
                    if results[roi_key][subj][key] is not None]
            if maps:
                group_roi_results[roi_key][key] = group_average(maps)
            else:
                group_roi_results[roi_key][key] = None

    fig = _make_figure(
        subject_label='GROUP',
        boundary_type=boundary_type,
        roi_results=group_roi_results,
        win_size=win_size, trs_before=trs_before,
        vmin_within=vmin_within, vmax_within=vmax_within,
        vmin_cross=vmin_cross,   vmax_cross=vmax_cross,
    )
    out = OUTPUT_DIR / f'GROUP_{boundary_type}_boundary_ttc.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out.name}")


def main(boundary_types=BOUNDARY_TYPES):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BOUNDARY-LOCKED TIME-TIME CORRELATION ANALYSIS")
    print(f"Boundary types: {boundary_types}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    for boundary_type in boundary_types:
        run_boundary_type(boundary_type)

    print("\n" + "=" * 60)
    print(f"DONE. Figures saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Boundary-locked TTC analysis for filmfest, SVF, and AHC boundaries.'
    )
    parser.add_argument(
        '--boundary_types', nargs='+',
        default=list(BOUNDARY_TYPES),
        choices=list(BOUNDARY_TYPES),
        help='Boundary types to process (default: all three)',
    )
    args = parser.parse_args()
    main(boundary_types=args.boundary_types)
