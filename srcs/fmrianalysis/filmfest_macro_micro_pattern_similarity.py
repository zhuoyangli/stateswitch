#!/usr/bin/env python3
"""
filmfest_macro_micro_pattern_similarity.py

Compare neural activation pattern similarity between macro boundaries
(between-movie transitions) and micro boundaries (within-movie "strong"
boundaries from the retained 131 set), per TR within a -20 to +40 TR
window (-30 to +60 s).

Preprocessing: spatial smoothing FWHM=6mm + high-pass filter 0.01 Hz
(always applied).

Two aggregation methods (--method flag):
  mean_then_correlate  : average patterns across instances within each
                         boundary type, then correlate the two means
  correlate_then_mean  : compute pairwise correlations between every
                         macro-micro instance pair, then average r values

Output: one figure per method, one subplot per ROI (2D TR×TR heatmap).
  Rows = macro boundary TR offset (-20 to +40), cols = micro boundary TR offset.

Usage:
    uv run python srcs/fmrianalysis/filmfest_macro_micro_pattern_similarity.py
    uv run python srcs/fmrianalysis/filmfest_macro_micro_pattern_similarity.py --method correlate_then_mean
    uv run python srcs/fmrianalysis/filmfest_macro_micro_pattern_similarity.py --roi pmc hipp
    uv run python srcs/fmrianalysis/filmfest_macro_micro_pattern_similarity.py --no-cache
    uv run python srcs/fmrianalysis/filmfest_macro_micro_pattern_similarity.py --n_jobs 4
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from configs.config import FILMFEST_SUBJECTS, TR, FIGS_DIR, ANALYSIS_CACHE_DIR
from fmrianalysis.filmfest_boundary_wspc import (
    load_roi_voxels, preprocess_voxels, ROI_SPEC,
)
from fmrianalysis.filmfest_boundary_response import get_within_movie_boundaries
from fmrianalysis.utils import get_movie_boundary_offsets

# ============================================================================
# CONSTANTS
# ============================================================================

TRS_BEFORE  = 20     # 30 s before boundary
TRS_AFTER   = 40     # 60 s after boundary
N_TRS       = TRS_BEFORE + TRS_AFTER + 1   # 61

OUTPUT_DIR       = FIGS_DIR / 'filmfest_macro_micro_pattern_similarity'
RESULT_CACHE_DIR = ANALYSIS_CACHE_DIR / 'macro_micro_pattern_sim'

LABEL_FS = 14

TASKS = ('filmfest1', 'filmfest2')


# ============================================================================
# STEP 1: BOUNDARY TRs
# ============================================================================

def get_macro_boundary_trs(task):
    """Return list of int TRs for between-movie (macro) boundaries."""
    times_sec = get_movie_boundary_offsets(task)
    return [int(round(t / TR)) for t in times_sec]


def get_micro_boundary_trs(task):
    """Return list of int TRs for strong within-movie (micro) boundaries."""
    times_sec = get_within_movie_boundaries(task, retained_only=True)['strong']
    return [int(round(t / TR)) for t in times_sec]


# ============================================================================
# STEP 2: PER-TR PATTERN EXTRACTION
# ============================================================================

def extract_patterns_per_tr(voxel_data, boundary_trs):
    """Extract voxel activation pattern at each TR offset for each boundary.

    Parameters
    ----------
    voxel_data   : (T, V) preprocessed voxel array
    boundary_trs : list of int, boundary TR indices (0-based)

    Returns
    -------
    patterns : (N_valid, N_TRS, V) float64 array
               Only boundaries where the full window fits are included.
    n_total  : int, number of boundary instances supplied (for reporting)
    """
    T, V = voxel_data.shape
    offsets = np.arange(-TRS_BEFORE, TRS_AFTER + 1)   # shape (N_TRS,)
    valid = []
    for btr in boundary_trs:
        idx = btr + offsets
        if idx[0] >= 0 and idx[-1] < T:
            valid.append(voxel_data[idx])   # (N_TRS, V)
    if not valid:
        return np.empty((0, N_TRS, V), dtype=np.float64), len(boundary_trs)
    return np.stack(valid, axis=0), len(boundary_trs)   # (N_valid, N_TRS, V)


# ============================================================================
# STEP 3: SIMILARITY AT EACH TR
# ============================================================================

def _pearson(a, b):
    """Pearson r between two 1-D arrays (mean-centered)."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return np.nan
    return np.dot(a, b) / denom


def compute_similarity_matrix(macro_pats, micro_pats, method):
    """Compute 2D pattern similarity matrix: macro TR offset × micro TR offset.

    Parameters
    ----------
    macro_pats : (N_macro, N_TRS, V)
    micro_pats : (N_micro, N_TRS, V)
    method     : 'mean_then_correlate' or 'correlate_then_mean'

    Returns
    -------
    sim : (N_TRS, N_TRS) array  — rows = macro TR offset, cols = micro TR offset
    """
    N_macro = macro_pats.shape[0]
    N_micro = micro_pats.shape[0]
    sim = np.full((N_TRS, N_TRS), np.nan)

    if N_macro == 0 or N_micro == 0:
        return sim

    if method == 'mean_then_correlate':
        # Pre-compute means across instances for each TR offset
        mean_macro = macro_pats.mean(axis=0)   # (N_TRS, V)
        mean_micro = micro_pats.mean(axis=0)   # (N_TRS, V)
        for ti in range(N_TRS):
            for tj in range(N_TRS):
                sim[ti, tj] = _pearson(mean_macro[ti], mean_micro[tj])

    else:  # correlate_then_mean
        for ti in range(N_TRS):
            for tj in range(N_TRS):
                rs = []
                for i in range(N_macro):
                    for j in range(N_micro):
                        r = _pearson(macro_pats[i, ti], micro_pats[j, tj])
                        if not np.isnan(r):
                            rs.append(r)
                sim[ti, tj] = np.mean(rs) if rs else np.nan

    return sim


# ============================================================================
# STEP 4: SUBJECT-LEVEL PIPELINE
# ============================================================================

def compute_subject_similarity(subject, roi_key, parcel_ids, method,
                                comparison='macro_vs_micro', force_voxels=False):
    """Run full pipeline for one subject, one ROI.

    comparison options:
      macro_vs_micro : macro (run1+run2) vs micro (run1+run2)  [default]
      macro_vs_macro : macro run1 vs macro run2
      micro_vs_micro : micro run1 vs micro run2

    Returns (N_TRS, N_TRS) similarity matrix, or None on missing data.
    """
    session = FILMFEST_SUBJECTS[subject]

    # Collect patterns per run
    run_pats = {}   # task -> {'macro': array, 'micro': array}
    for task in TASKS:
        try:
            raw = load_roi_voxels(subject, session, task, roi_key,
                                  parcel_ids, force=force_voxels)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {task} roi={roi_key}: {e}")
            continue

        vox = preprocess_voxels(raw, do_hp=True)

        mac_pats, n_mac = extract_patterns_per_tr(vox, get_macro_boundary_trs(task))
        mic_pats, n_mic = extract_patterns_per_tr(vox, get_micro_boundary_trs(task))
        print(f"    {task}: macro {mac_pats.shape[0]}/{n_mac} valid, "
              f"micro {mic_pats.shape[0]}/{n_mic} valid")

        run_pats[task] = {'macro': mac_pats, 'micro': mic_pats}

    task1, task2 = TASKS

    if comparison == 'macro_vs_micro':
        # Pool both runs for each type
        all_macro = [run_pats[t]['macro'] for t in TASKS
                     if t in run_pats and run_pats[t]['macro'].shape[0] > 0]
        all_micro = [run_pats[t]['micro'] for t in TASKS
                     if t in run_pats and run_pats[t]['micro'].shape[0] > 0]
        if not all_macro or not all_micro:
            print(f"  SKIP {subject} roi={roi_key}: insufficient data")
            return None
        group_a = np.concatenate(all_macro, axis=0)
        group_b = np.concatenate(all_micro, axis=0)

    else:
        # Same type, split by run — run1 = group A, run2 = group B
        btype = 'macro' if comparison == 'macro_vs_macro' else 'micro'
        if task1 not in run_pats or task2 not in run_pats:
            print(f"  SKIP {subject} roi={roi_key}: need both runs for {comparison}")
            return None
        group_a = run_pats[task1][btype]
        group_b = run_pats[task2][btype]
        if group_a.shape[0] == 0 or group_b.shape[0] == 0:
            print(f"  SKIP {subject} roi={roi_key}: no valid {btype} boundaries in one run")
            return None

    print(f"  {subject} roi={roi_key}: group_a n={group_a.shape[0]}, "
          f"group_b n={group_b.shape[0]}")

    return compute_similarity_matrix(group_a, group_b, method)


# ============================================================================
# STEP 5: GROUP STATISTICS
# ============================================================================

def group_similarity(subject_sims):
    """Fisher-z average across subjects.

    Parameters
    ----------
    subject_sims : (N_subj, N_TRS, N_TRS) array

    Returns
    -------
    mean_r : (N_TRS, N_TRS) group mean correlation matrix
    """
    z_stack = np.arctanh(np.clip(subject_sims, -0.999, 0.999))
    mean_z  = np.nanmean(z_stack, axis=0)
    return np.tanh(mean_z)


# ============================================================================
# STEP 6: PLOTTING
# ============================================================================

AXIS_LABELS = {
    'macro_vs_micro': (
        'Time from between-movie boundary (s)',
        'Time from within-movie boundary (s)',
    ),
    'macro_vs_macro': (
        'Time from between-movie boundary — run 1 (s)',
        'Time from between-movie boundary — run 2 (s)',
    ),
    'micro_vs_micro': (
        'Time from within-movie boundary — run 1 (s)',
        'Time from within-movie boundary — run 2 (s)',
    ),
}


def make_figure(results_by_roi, method, n_subjects, comparison='macro_vs_micro', vmax=0.5):
    """Plot 2D TR×TR group similarity heatmap per ROI.

    Parameters
    ----------
    results_by_roi : dict {roi_key: mean_r} — each (N_TRS, N_TRS)
    method         : str
    n_subjects     : int
    vmax           : colorbar limit (symmetric, default 0.5)
    """
    roi_items = [(k, name) for k, name, _ in ROI_SPEC if k in results_by_roi]
    n_rois = len(roi_items)
    if n_rois == 0:
        return None

    # Tick positions and labels in seconds (every 10 TRs = 15 s)
    tick_step = 10
    tick_idx  = np.arange(0, N_TRS, tick_step)
    tick_sec  = (tick_idx - TRS_BEFORE) * TR
    tick_labels = [f'{int(s):+d}' for s in tick_sec]

    # Colorbar dimensions in absolute inches — constant regardless of n_rois
    CBAR_W_IN   = 0.15  # colorbar bar width
    CBAR_GAP_IN = 0.40  # gap between last panel and colorbar bar
                        # (must fit the rotated label to the left of the bar)
    fig_w = 4.0 * n_rois + 1.0
    fig_h = 4.0

    # Reserve exactly enough right margin for the colorbar + gap
    right_frac = 1.0 - (CBAR_W_IN + CBAR_GAP_IN) / fig_w

    fig, axes = plt.subplots(1, n_rois, figsize=(fig_w, fig_h),
                              facecolor='white')
    if n_rois == 1:
        axes = [axes]

    last_im = None

    for col, (roi_key, roi_name) in enumerate(roi_items):
        ax = axes[col]
        mat = results_by_roi[roi_key]   # (N_TRS, N_TRS)

        # Transpose: rows=micro (y), cols=macro (x)
        im = ax.imshow(mat.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       aspect='equal', origin='upper', interpolation='none')
        last_im = im

        ax.axvline(TRS_BEFORE - 0.5, color='k', lw=0.8, ls='-', alpha=0.6)
        ax.axhline(TRS_BEFORE - 0.5, color='k', lw=0.8, ls='-', alpha=0.6)

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, fontsize=LABEL_FS - 4)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(tick_labels if col == 0 else [], fontsize=LABEL_FS - 4)

        xlabel, ylabel = AXIS_LABELS[comparison]
        ax.set_title(roi_name, fontsize=LABEL_FS, fontweight='bold', pad=6)
        ax.set_xlabel(xlabel, fontsize=LABEL_FS - 3)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=LABEL_FS - 3)

    fig.tight_layout(rect=[0, 0, right_frac, 1])
    fig.subplots_adjust(wspace=0)

    if last_im is not None:
        # Place colorbar in absolute-inch terms, anchored to last axis height
        pos      = axes[-1].get_position()
        cbar_gap = CBAR_GAP_IN / fig_w
        cbar_w   = CBAR_W_IN  / fig_w
        cbar_h   = pos.height * 0.5
        cbar_b   = pos.y0 + pos.height * 0.25
        cbar_ax  = fig.add_axes([pos.x1 + cbar_gap, cbar_b, cbar_w, cbar_h])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label('Within-subject pattern corr ($r$)', fontsize=LABEL_FS - 5,
                       labelpad=6)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.tick_params(labelsize=LABEL_FS - 6)

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Macro vs micro boundary neural pattern similarity, per TR.'
    )
    parser.add_argument(
        '--comparison',
        choices=['macro_vs_micro', 'macro_vs_macro', 'micro_vs_micro'],
        default='macro_vs_micro',
        help='Which boundary types to compare (default: macro_vs_micro)',
    )
    parser.add_argument(
        '--method',
        choices=['mean_then_correlate', 'correlate_then_mean'],
        default='mean_then_correlate',
        help='Aggregation method (default: mean_then_correlate)',
    )
    parser.add_argument(
        '--roi', nargs='+',
        default=[k for k, _, _ in ROI_SPEC],
        help='ROI keys to process (default: all)',
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Force recompute voxel extraction',
    )
    parser.add_argument(
        '--n_jobs', type=int, default=1,
        help='Parallel jobs for voxel pre-extraction (default: 1)',
    )
    parser.add_argument(
        '--vmax', type=float, default=0.5,
        help='Colorbar limit ±vmax (default: 0.5)',
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    roi_spec_filtered = [(k, name, pids) for k, name, pids in ROI_SPEC
                         if k in args.roi]
    if not roi_spec_filtered:
        print("No valid ROIs specified.")
        return

    subjects   = sorted(FILMFEST_SUBJECTS.keys())
    method     = args.method
    comparison = args.comparison

    print(f"\n{'='*60}")
    print(f"BOUNDARY PATTERN SIMILARITY  [{comparison}  |  {method}]")
    print(f"Subjects: {subjects}")
    print(f"ROIs: {[k for k, _, _ in roi_spec_filtered]}")
    print(f"Window: -{TRS_BEFORE} to +{TRS_AFTER} TRs  ({N_TRS} time points)")
    print(f"Preprocessing: smoothing FWHM=6mm + HP filter 0.01 Hz")
    print('='*60)

    # Pre-extract voxels in parallel if requested
    if args.n_jobs > 1:
        print(f"\nPre-extracting voxels (n_jobs={args.n_jobs}) ...")

        def _extract_job(subj, roi_key, parcel_ids):
            try:
                session = FILMFEST_SUBJECTS[subj]
                for task in TASKS:
                    load_roi_voxels(subj, session, task, roi_key,
                                    parcel_ids, force=args.no_cache)
                return True
            except Exception as e:
                print(f"  ERROR {subj} roi={roi_key}: {e}")
                return False

        Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_extract_job)(subj, roi_key, parcel_ids)
            for roi_key, _, parcel_ids in roi_spec_filtered
            for subj in subjects
        )

    # Per-ROI computation
    results_by_roi = {}
    for roi_key, roi_name, parcel_ids in roi_spec_filtered:
        print(f"\n{'='*60}")
        print(f"ROI: {roi_name} ({roi_key})")

        cache_path = RESULT_CACHE_DIR / f'roi-{roi_key}_{comparison}_{method}_simmat.npz'
        if not args.no_cache and cache_path.exists():
            print(f"  Loading from cache: {cache_path.name}")
            data = np.load(cache_path)
            results_by_roi[roi_key] = data['mean_r']
            print(f"  Group: {int(data['n_subjects'])} subjects")
            continue

        subject_sims = []
        for subj in subjects:
            print(f"\n  Subject: {subj}")
            sim = compute_subject_similarity(
                subj, roi_key, parcel_ids, method,
                comparison=comparison, force_voxels=args.no_cache,
            )
            if sim is not None:
                subject_sims.append(sim)

        if not subject_sims:
            print(f"  [{roi_key}] No valid subjects, skipping.")
            continue

        subject_sims_arr = np.array(subject_sims)   # (N_subj, N_TRS, N_TRS)
        mean_r = group_similarity(subject_sims_arr)
        results_by_roi[roi_key] = mean_r

        np.savez_compressed(
            cache_path,
            mean_r=mean_r,
            subject_sims=subject_sims_arr,
            n_subjects=len(subject_sims),
            method=method,
        )
        print(f"  Cached → {cache_path.name}")

        # Save individual ROI figure immediately after computing
        fig_roi = make_figure({roi_key: mean_r}, method, len(subject_sims),
                              comparison=comparison, vmax=args.vmax)
        if fig_roi is not None:
            roi_path = OUTPUT_DIR / f'boundary_pattern_similarity_{comparison}_{method}_{roi_key}.png'
            fig_roi.savefig(roi_path, dpi=300, bbox_inches='tight')
            plt.close(fig_roi)
            print(f"  Saved → {roi_path.name}")

    if not results_by_roi:
        print("No results to plot.")
        return

    # Also save individual figures for any ROIs that were loaded from cache
    for roi_key, mean_r in results_by_roi.items():
        roi_path = OUTPUT_DIR / f'boundary_pattern_similarity_{comparison}_{method}_{roi_key}.png'
        if not roi_path.exists() or args.no_cache:
            fig_roi = make_figure({roi_key: mean_r}, method, len(subjects),
                                  comparison=comparison, vmax=args.vmax)
            if fig_roi is not None:
                fig_roi.savefig(roi_path, dpi=300, bbox_inches='tight')
                plt.close(fig_roi)
                print(f"  Saved → {roi_path.name}")

    if len(results_by_roi) > 1:
        print(f"\nBuilding all-ROIs figure...")
        fig = make_figure(results_by_roi, method, len(subjects),
                          comparison=comparison, vmax=args.vmax)
        if fig is not None:
            out_path = OUTPUT_DIR / f'boundary_pattern_similarity_{comparison}_{method}_all_rois.png'
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved → {out_path}")
    print('='*60)


if __name__ == '__main__':
    main()
