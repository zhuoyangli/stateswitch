#!/usr/bin/env python3
"""
story_ttc.py — Inter-Subject Time-Time Correlation for Story-Listening Tasks

For each story and ROI, computes the LOO inter-subject TTC:
  1. Extract voxels from all ROI parcels, concatenated into (T, V)
  2. Preprocessing: 4mm spatial smoothing → 0.01 Hz Butterworth HP → z-score
  3. For each subject i: cross_corrcoef(subject_pattern, loo_group_mean) → (T, T)
  4. Fisher-z average across subjects → group ISC TTC (T, T)

Outputs:
  CACHE_DIR/analyses/story_ttc/{story}/{roi_key}_isc_ttc.npz
      (keys: 'group_ttc' (T, T), 'per_subject' (n_subjects, T, T),
             'subjects', 'sessions')
  FIGS_DIR/story_ttc/ttc_{story}.png
      (N_rois panels, each showing the group ISC TTC heatmap)

Usage:
    uv run python srcs/fmrianalysis/story_ttc.py
    uv run python srcs/fmrianalysis/story_ttc.py --story swimming adollshouse
    uv run python srcs/fmrianalysis/story_ttc.py --roi pmc ag eac
    uv run python srcs/fmrianalysis/story_ttc.py --no-cache
    uv run python srcs/fmrianalysis/story_ttc.py --vmax 0.1
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import datasets, image
from scipy.stats import zscore as sp_zscore

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR
from configs.schaefer_rois import PRIMARY_ROIS, get_bilateral_ids
from fmrianalysis.utils import highpass_filter, get_bold_path, cross_corrcoef

# ============================================================================
# CONSTANTS
# ============================================================================

STORIES = [
    'adollshouse', 'adventuresinsayingyes', 'beneaththemushroomcloud',
    'breakingupintheageofgoogle', 'buck', 'christmas1940', 'howtodraw',
    'inamoment', 'notontheusualtour', 'odetostepfather', 'penpal',
    'shoppinginchina', 'swimming', 'theclosetthatateeverything',
    'treasureisland', 'undertheinfluence', 'vixen', 'wheretheressmoke',
]

# Default ROIs (key -> (label, parcel_ids))
DEFAULT_ROI_KEYS = ['evc', 'eac', 'pmc', 'ag']

SMOOTH_FWHM = 4.0
HP_ORDER    = 2
HP_CUTOFF   = 0.01

OUTPUT_DIR = FIGS_DIR / 'story_ttc'
CACHE_BASE = ANALYSIS_CACHE_DIR / 'story_ttc'

# Load Schaefer atlas once
print("Loading Schaefer atlas...")
SCHAEFER = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
_atlas_resampled = None
print("  Schaefer atlas loaded.")


# ============================================================================
# DISCOVERY  (reuse same logic as story_pisc.py)
# ============================================================================

def discover_story_subjects(story):
    """Return (subject, session) pairs, one per subject (earliest session)."""
    pattern = (f"sub-*/ses-*/func/sub-*_ses-*_task-{story}_space-MNI152NLin6Asym"
               f"_res-2_desc-preproc_bold.nii.gz")
    seen = {}
    for f in sorted(DERIVATIVES_DIR.glob(pattern)):
        parts = f.name.split('_')
        subj, ses = parts[0], parts[1]
        if subj not in seen:
            seen[subj] = ses
    return list(seen.items())


# ============================================================================
# ATLAS
# ============================================================================

def _get_atlas_data(bold_path):
    global _atlas_resampled
    if _atlas_resampled is None:
        bold_ref = image.index_img(str(bold_path), 0)
        atlas_img = image.resample_to_img(
            SCHAEFER['maps'], bold_ref, interpolation='nearest'
        )
        _atlas_resampled = np.round(atlas_img.get_fdata()).astype(int)
        print(f"  Atlas resampled: {_atlas_resampled.shape}")
    return _atlas_resampled


# ============================================================================
# ROI PATTERN LOADING
# ============================================================================

def load_roi_pattern(subject, session, story, parcel_ids, atlas_data,
                     do_hp=True, fwhm=SMOOTH_FWHM):
    """Load smoothed, z-scored (optionally HP-filtered) voxel pattern.

    Returns np.ndarray (T, V) float64.
    """
    bold_path = get_bold_path(subject, session, story)
    bold_img  = nib.load(str(bold_path))
    bold_sm   = image.smooth_img(bold_img, fwhm=fwhm).get_fdata(dtype=np.float32)

    parts = []
    for pid in parcel_ids:
        mask = atlas_data == pid
        if mask.sum() < 2:
            continue
        pdata = bold_sm[mask].T.astype(np.float64)  # (T, n_vox)
        if do_hp:
            pdata = highpass_filter(pdata, cutoff=HP_CUTOFF, tr=TR, order=HP_ORDER)
        pdata = sp_zscore(pdata, axis=0, nan_policy='omit')
        pdata = np.nan_to_num(pdata)
        parts.append(pdata)

    del bold_sm
    if not parts:
        raise ValueError(f"No usable parcels for {subject} {session} {story}")
    return np.concatenate(parts, axis=1)  # (T, V_roi)


# ============================================================================
# ISC TTC COMPUTATION
# ============================================================================

def compute_isc_ttc(story, subjects_sessions, parcel_ids, do_hp=True, fwhm=SMOOTH_FWHM):
    """Compute LOO ISC TTC for one story and one ROI.

    Returns
    -------
    group_ttc    : np.ndarray (T, T)
    per_subj_ttc : np.ndarray (n_subjects, T, T)
    """
    subjects   = [s for s, _ in subjects_sessions]
    n_subjects = len(subjects)

    first_path = get_bold_path(subjects_sessions[0][0], subjects_sessions[0][1], story)
    atlas_data = _get_atlas_data(first_path)

    # Load all subject patterns
    patterns = []
    for subj, ses in subjects_sessions:
        p = load_roi_pattern(subj, ses, story, parcel_ids, atlas_data, do_hp=do_hp, fwhm=fwhm)
        patterns.append(p)
        print(f"    {subj} ({ses}): pattern shape {p.shape}")

    T_common = min(p.shape[0] for p in patterns)
    patterns  = [p[:T_common] for p in patterns]
    all_pdata = np.stack(patterns, axis=0)  # (n_subjects, T, V)
    group_mean = all_pdata.mean(axis=0)     # (T, V)

    per_subj_ttc = np.empty((n_subjects, T_common, T_common))
    for i in range(n_subjects):
        loo_mean = (group_mean * n_subjects - all_pdata[i]) / (n_subjects - 1)
        per_subj_ttc[i] = cross_corrcoef(all_pdata[i], loo_mean)  # (T, T)

    # Fisher-z average across subjects
    group_ttc = np.tanh(
        np.arctanh(np.clip(per_subj_ttc, -0.999, 0.999)).mean(axis=0)
    )
    return group_ttc, per_subj_ttc


# ============================================================================
# CACHE PATHS
# ============================================================================

def _cache_file(story, roi_key, do_hp=True, fwhm=SMOOTH_FWHM):
    hp_suffix  = '' if do_hp else '_nohp'
    sm_suffix  = '' if fwhm == SMOOTH_FWHM else f'_sm{int(fwhm)}'
    return CACHE_BASE / story / f'{roi_key}_isc_ttc{sm_suffix}{hp_suffix}.npz'


# ============================================================================
# LOAD OR COMPUTE
# ============================================================================

def load_or_compute_ttc(story, subjects_sessions, roi_key, parcel_ids,
                        force=False, do_hp=True, fwhm=SMOOTH_FWHM):
    """Load ISC TTC from cache or compute and save."""
    cf = _cache_file(story, roi_key, do_hp=do_hp, fwhm=fwhm)
    if not force and cf.exists():
        print(f"  [{story}] {roi_key}: loading from cache")
        data = np.load(cf)
        return data['group_ttc'], data['per_subject']

    print(f"  [{story}] {roi_key}: computing ISC TTC (hp={do_hp}, fwhm={fwhm})...")
    group_ttc, per_subj = compute_isc_ttc(story, subjects_sessions, parcel_ids,
                                           do_hp=do_hp, fwhm=fwhm)

    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cf,
        group_ttc=group_ttc,
        per_subject=per_subj,
        subjects=np.array([s for s, _ in subjects_sessions]),
        sessions=np.array([ses for _, ses in subjects_sessions]),
        do_hp=do_hp,
    )
    print(f"    Cached → {cf.name}")
    return group_ttc, per_subj


# ============================================================================
# FIGURE
# ============================================================================

def make_ttc_figure(story, ttc_by_roi, roi_spec, vmax, out_path):
    """Plot group ISC TTC heatmaps for all ROIs.

    Layout: 1 row × N_rois cols (or 2 rows if N_rois > 4).
    Each panel: T×T heatmap, RdBu_r, vmin=-vmax, vmax=vmax.
    """
    n_rois = len(roi_spec)
    ncols  = min(n_rois, 4)
    nrows  = (n_rois + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle(f'Story ISC TTC: {story}', fontsize=13, fontweight='bold')

    for idx, (roi_key, roi_label) in enumerate(roi_spec):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ttc = ttc_by_roi[roi_key]
        T   = ttc.shape[0]

        im = ax.imshow(ttc, aspect='auto', origin='upper',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       extent=[0, T * TR, T * TR, 0])
        ax.set_title(roi_label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Time (s)', fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='ISC (r)')

    # Hide unused axes
    for idx in range(n_rois, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved → {out_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Story inter-subject time-time correlation for ROIs."
    )
    parser.add_argument('--story', nargs='+', default=None,
                        help="Stories to process (default: all 18)")
    parser.add_argument('--roi', nargs='+', default=DEFAULT_ROI_KEYS,
                        help=f"ROI keys (default: {DEFAULT_ROI_KEYS}). "
                             f"Available: {list(PRIMARY_ROIS.keys())}")
    parser.add_argument('--no-hp', action='store_true',
                        help="Skip high-pass filtering (saves to *_nohp cache/figures)")
    parser.add_argument('--no-cache', action='store_true',
                        help="Recompute even if cache exists")
    parser.add_argument('--no-fig', action='store_true',
                        help="Skip figure generation")
    parser.add_argument('--vmax', type=float, default=0.3,
                        help="Symmetric colorbar limit (default: 0.3)")
    parser.add_argument('--fwhm', type=float, default=SMOOTH_FWHM,
                        help=f"Spatial smoothing FWHM in mm (default: {SMOOTH_FWHM})")
    args = parser.parse_args()

    stories = args.story or STORIES

    # Build ROI spec: list of (key, label, parcel_ids)
    roi_spec_full = []
    for key in args.roi:
        if key not in PRIMARY_ROIS:
            print(f"WARNING: unknown ROI '{key}', skipping")
            continue
        roi = PRIMARY_ROIS[key]
        roi_spec_full.append((key, roi['name'], get_bilateral_ids(roi)))

    if not roi_spec_full:
        print("No valid ROIs specified.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_BASE.mkdir(parents=True, exist_ok=True)

    for story in stories:
        print(f"\n{'='*60}")
        print(f"Story: {story}")
        pairs = discover_story_subjects(story)
        if not pairs:
            print(f"  WARNING: no data found for '{story}', skipping")
            continue
        print(f"  {len(pairs)} subjects: " +
              ", ".join(f"{s}({ses})" for s, ses in pairs))

        do_hp = not args.no_hp
        fwhm  = args.fwhm
        ttc_by_roi = {}
        for roi_key, roi_label, parcel_ids in roi_spec_full:
            group_ttc, _ = load_or_compute_ttc(
                story, pairs, roi_key, parcel_ids,
                force=args.no_cache, do_hp=do_hp, fwhm=fwhm,
            )
            ttc_by_roi[roi_key] = group_ttc

        if not args.no_fig:
            hp_suffix = '' if do_hp else '_nohp'
            sm_suffix = '' if fwhm == SMOOTH_FWHM else f'_sm{int(fwhm)}'
            out_path = OUTPUT_DIR / f'ttc_{story}{sm_suffix}{hp_suffix}.png'
            roi_spec_labels = [(k, l) for k, l, _ in roi_spec_full]
            make_ttc_figure(story, ttc_by_roi, roi_spec_labels,
                            vmax=args.vmax, out_path=out_path)

    print("\nDone.")
    print(f"  Figures: {OUTPUT_DIR}")
    print(f"  Cache:   {CACHE_BASE}")


if __name__ == '__main__':
    main()
