"""
Filmfest SEG-B Boundary Neural Analysis

Vertex-wise surface maps + PMC/Hippocampus event-locked time courses
at SEG-B segment boundaries during movie watching.

Supports two lock modes:
  - onset:  t=0 = start of new segment (boundary onset)
  - offset: t=0 = end of previous segment (boundary offset)

Outputs per-run (filmfest1 / filmfest2) results for cross-run consistency.

Usage:
    python filmfest_boundary_parcel.py
"""
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, filtfilt
from scipy.stats import zscore as sp_zscore
from nilearn import datasets, plotting, surface
from nilearn.maskers import NiftiMasker

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR, FILMFEST_SUBJECTS, MOVIE_INFO
from fmrianalysis.utils import load_surface_data, highpass_filter, mss_to_seconds

ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
OUTPUT_DIR = FIGS_DIR / 'filmfest_boundary_parcel'
CACHE_DIR = ANALYSIS_CACHE_DIR / 'filmfest_boundary_parcel'

RUN_LABELS = {'filmfest1': 'Run 1 (Movies 1–5)', 'filmfest2': 'Run 2 (Movies 6–10)'}

# === PARAMETERS ===
HIGH_PASS_HZ = 0.01
SURFACE_WINDOW_START = 9.0   # seconds after event for surface contrast
SURFACE_WINDOW_END = 15.0    # seconds after event for surface contrast
MIN_SEGMENT_DURATION = 10.0
TRS_BEFORE = 2
TRS_AFTER = 15

# === STYLE ===
COLORS = {'boundary': '#e74c3c', 'nonboundary': 'gray'}
LABEL_FS = 12
TITLE_FS = 14
Y_LIM_SINGLE = (-0.6, 0.6)
Y_LIM_GROUP = (-0.2, 0.25)
SURFACE_THRESHOLD = 1.5  # display threshold for surface plots
SURFACE_VMAX = 4.0
FDR_Q = 0.05

# === LOAD ATLASES & MASKS ===
print("Loading atlases and masks...")
FSAVERAGE = datasets.fetch_surf_fsaverage('fsaverage6')

# PMC mask on fsaverage6 surface
SCHAEFER = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
SCHAEFER_LABELS = [l.decode() if isinstance(l, bytes) else str(l) for l in SCHAEFER['labels']]
_pmc_ids = [i + 1 for i, l in enumerate(SCHAEFER_LABELS) if 'pCunPCC' in l]
_schaefer_L = surface.vol_to_surf(SCHAEFER['maps'], FSAVERAGE['pial_left'],
                                   interpolation='nearest')
_schaefer_R = surface.vol_to_surf(SCHAEFER['maps'], FSAVERAGE['pial_right'],
                                   interpolation='nearest')
PMC_MASK_L = np.isin(np.round(_schaefer_L).astype(int), _pmc_ids)
PMC_MASK_R = np.isin(np.round(_schaefer_R).astype(int), _pmc_ids)

# Hippocampus binary mask
_ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
_ho_img = nib.load(_ho['maps']) if isinstance(_ho['maps'], (str, Path)) else _ho['maps']
_ho_labels = _ho['labels']
_hipp_ids = [i for i, l in enumerate(_ho_labels) if 'hippocampus' in l.lower()]
_hipp_data = np.isin(_ho_img.get_fdata().astype(int), _hipp_ids).astype(np.int8)
HIPP_MASK_IMG = nib.Nifti1Image(_hipp_data, _ho_img.affine, _ho_img.header)

print(f"PMC: {PMC_MASK_L.sum()} L + {PMC_MASK_R.sum()} R vertices. "
      f"Hippocampus mask: {_hipp_data.sum()} voxels.")


# ============================================================================
# HELPERS
# ============================================================================

def get_segb_events(task, lock_mode='onset'):
    """Get SEG-B boundary and non-boundary timepoints (in seconds).

    lock_mode:
        'onset'  — t=0 at start of new segment
        'offset' — t=0 at end of previous segment
    """
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    boundary_tp, nonboundary_tp = [], []

    for movie in movies:
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number']).copy()
        starts = segb['Start Time (m.ss)'].apply(mss_to_seconds).values
        ends = segb['End Time (m.ss)'].apply(mss_to_seconds).values

        for i in range(1, len(starts)):
            if lock_mode == 'onset':
                boundary_tp.append(starts[i])
            else:
                boundary_tp.append(ends[i - 1])

        for i in range(len(starts)):
            dur = ends[i] - starts[i]
            if dur >= MIN_SEGMENT_DURATION:
                nonboundary_tp.append(starts[i] + dur / 2)

    min_t = TRS_BEFORE * TR
    boundary_tp = np.array(sorted([t for t in boundary_tp if t >= min_t]))
    nonboundary_tp = np.array(sorted([t for t in nonboundary_tp if t >= min_t]))
    return boundary_tp, nonboundary_tp


def extract_event_locked_timecourse(signal, event_tp):
    """Average event-locked time course from 1D signal.

    Returns (mean, sem, time_vec).
    """
    n = len(signal)
    if len(event_tp) == 0:
        return np.array([]), np.array([]), np.array([])
    centers = np.round(np.array(event_tp) / TR).astype(int)
    offsets = np.arange(-TRS_BEFORE, TRS_AFTER + 1)
    idx = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n), axis=1)
    if not valid.any():
        return np.array([]), np.array([]), np.array([])
    epochs = signal[idx[valid]]
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)
    n_epochs = epochs.shape[0]
    mean = epochs.mean(axis=0)
    sem = epochs.std(axis=0) / np.sqrt(n_epochs) if n_epochs > 1 else np.zeros_like(mean)
    return mean, sem, offsets * TR


def compute_vertex_contrast(surface_ts, bnd_tp, nbnd_tp):
    """Compute t-value per vertex: boundary > non-boundary.

    Uses window from SURFACE_WINDOW_START to SURFACE_WINDOW_END seconds
    after event onset.
    """
    n_scans = surface_ts.shape[0]

    def window_means(timepoints):
        means = []
        for onset in timepoints:
            s = int(np.floor((onset + SURFACE_WINDOW_START) / TR))
            e = int(np.ceil((onset + SURFACE_WINDOW_END) / TR))
            if 0 <= s and e < n_scans:
                means.append(surface_ts[s:e].mean(axis=0))
        return np.array(means)

    act_bnd = window_means(bnd_tp)
    act_nbnd = window_means(nbnd_tp)

    if len(act_bnd) < 2 or len(act_nbnd) < 2:
        return np.zeros(surface_ts.shape[1])

    t_vals, _ = stats.ttest_ind(act_bnd, act_nbnd, axis=0)
    return np.nan_to_num(t_vals)


# ============================================================================
# RUN PROCESSING (with caching)
# ============================================================================

def process_run(subject, session, task, lock_mode='onset'):
    """Process one filmfest run. Returns result dict. Caches to disk."""
    cache_file = CACHE_DIR / f"{subject}_{session}_{task}_segb_w9-15_sem_{lock_mode}.npz"
    if cache_file.exists():
        print(f"  {subject} {session} {task} [{lock_mode}]: loading cache")
        loaded = np.load(cache_file, allow_pickle=True)
        result = {k: loaded[k] for k in loaded.files}
        for k in ('subject', 'session', 'task', 'lock_mode'):
            if k in result:
                result[k] = str(result[k])
        for k in ('n_boundary', 'n_nonboundary'):
            if k in result:
                result[k] = int(result[k])
        return result

    print(f"  {subject} {session} {task} [{lock_mode}]: processing ...")

    boundary_tp, nonboundary_tp = get_segb_events(task, lock_mode)

    # --- Surface data (fsaverage6) ---
    surf_L = load_surface_data(subject, session, task, 'L', DERIVATIVES_DIR)
    surf_L = surf_L.astype(np.float64).T  # (T, V)
    surf_L = highpass_filter(surf_L)
    surf_L = sp_zscore(surf_L, axis=0, nan_policy='omit')
    surf_L = np.nan_to_num(surf_L)

    surf_R = load_surface_data(subject, session, task, 'R', DERIVATIVES_DIR)
    surf_R = surf_R.astype(np.float64).T
    surf_R = highpass_filter(surf_R)
    surf_R = sp_zscore(surf_R, axis=0, nan_policy='omit')
    surf_R = np.nan_to_num(surf_R)

    # PMC time series (bilateral average)
    pmc_L = surf_L[:, PMC_MASK_L].mean(axis=1) if PMC_MASK_L.any() else np.zeros(surf_L.shape[0])
    pmc_R = surf_R[:, PMC_MASK_R].mean(axis=1) if PMC_MASK_R.any() else np.zeros(surf_R.shape[0])
    pmc_ts = (pmc_L + pmc_R) / 2

    # Vertex-wise t-maps (using 9–15s window)
    print(f"    Computing vertex-wise contrasts (window {SURFACE_WINDOW_START}–{SURFACE_WINDOW_END}s) ...")
    t_map_L = compute_vertex_contrast(surf_L, boundary_tp, nonboundary_tp)
    t_map_R = compute_vertex_contrast(surf_R, boundary_tp, nonboundary_tp)

    del surf_L, surf_R

    # --- Hippocampus (from MNI volume) ---
    bold_path = (
        DERIVATIVES_DIR / subject / session / "func"
        / f"{subject}_{session}_task-{task}"
          f"_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    )
    if bold_path.exists():
        print(f"    Extracting hippocampus ...")
        masker = NiftiMasker(mask_img=HIPP_MASK_IMG, standardize=False, verbose=0)
        hipp_voxels = masker.fit_transform(str(bold_path))
        hipp_raw = hipp_voxels.mean(axis=1)
        hipp_ts = highpass_filter(hipp_raw.reshape(-1, 1)).flatten()
        hipp_ts = sp_zscore(hipp_ts, nan_policy='omit')
        hipp_ts = np.nan_to_num(hipp_ts)
    else:
        print(f"    WARNING: BOLD volume not found, skipping hippocampus")
        hipp_ts = np.zeros_like(pmc_ts)

    # --- Event-locked time courses ---
    pmc_bnd_tc, pmc_bnd_sem, time_vec = extract_event_locked_timecourse(pmc_ts, boundary_tp)
    pmc_nbnd_tc, pmc_nbnd_sem, _ = extract_event_locked_timecourse(pmc_ts, nonboundary_tp)
    hipp_bnd_tc, hipp_bnd_sem, _ = extract_event_locked_timecourse(hipp_ts, boundary_tp)
    hipp_nbnd_tc, hipp_nbnd_sem, _ = extract_event_locked_timecourse(hipp_ts, nonboundary_tp)

    result = {
        'subject': subject, 'session': session, 'task': task,
        'lock_mode': lock_mode,
        'pmc_boundary_tc': pmc_bnd_tc,
        'pmc_nonboundary_tc': pmc_nbnd_tc,
        'hipp_boundary_tc': hipp_bnd_tc,
        'hipp_nonboundary_tc': hipp_nbnd_tc,
        'pmc_boundary_sem': pmc_bnd_sem,
        'pmc_nonboundary_sem': pmc_nbnd_sem,
        'hipp_boundary_sem': hipp_bnd_sem,
        'hipp_nonboundary_sem': hipp_nbnd_sem,
        'time_vec': time_vec,
        't_map_left': t_map_L,
        't_map_right': t_map_R,
        'n_boundary': len(boundary_tp),
        'n_nonboundary': len(nonboundary_tp),
    }

    np.savez_compressed(cache_file, **result)
    print(f"    Cached to {cache_file.name}")
    return result


# ============================================================================
# FDR
# ============================================================================

def fdr_threshold(p_values, q=0.05):
    p = np.asarray(p_values)
    n = len(p)
    si = np.argsort(p)
    sp = p[si]
    bh = q * np.arange(1, n + 1) / n
    below = sp <= bh
    p_thr = sp[np.max(np.where(below)[0])] if below.any() else 0
    return p_thr, p <= p_thr


# ============================================================================
# PLOTTING — Group time courses (per-run comparison)
# ============================================================================

def plot_group_timecourse_by_run(run1_results, run2_results, lock_mode):
    """Plot group time courses for filmfest1 and filmfest2 side by side (2x2)."""
    mode_label = 'Onset-locked' if lock_mode == 'onset' else 'Offset-locked'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Filmfest SEG-B ({mode_label}): Per-Run Comparison\n"
                 f"* p < 0.05 paired t-test",
                 fontsize=TITLE_FS, fontweight='bold')

    for row, (results, run_key) in enumerate([
        (run1_results, 'filmfest1'),
        (run2_results, 'filmfest2'),
    ]):
        n = len(results)
        if n < 2:
            continue
        tv = results[0]['time_vec']
        run_label = RUN_LABELS[run_key]

        stacks = {}
        for cond in ('boundary', 'nonboundary'):
            for roi in ('pmc', 'hipp'):
                stacks[f'{roi}_{cond}'] = np.array(
                    [r[f'{roi}_{cond}_tc'] for r in results])

        for col, (roi, roi_title) in enumerate([('pmc', 'PMC'), ('hipp', 'Hippocampus')]):
            ax = axes[row, col]
            bnd = stacks[f'{roi}_boundary']
            nbnd = stacks[f'{roi}_nonboundary']

            for arr, label, c in [(bnd, 'Boundary', COLORS['boundary']),
                                    (nbnd, 'Non-boundary', COLORS['nonboundary'])]:
                m = arr.mean(0)
                se = arr.std(0) / np.sqrt(n)
                ax.plot(tv, m, color=c, lw=3, label=label, marker='o', ms=4)
                ax.fill_between(tv, m - se, m + se, color=c, alpha=0.3)

            pvals = np.array([stats.ttest_rel(bnd[:, t], nbnd[:, t])[1]
                              for t in range(len(tv))])
            sig = np.where(pvals < 0.05)[0]
            if len(sig):
                yp = Y_LIM_GROUP[0] + 0.05 * (Y_LIM_GROUP[1] - Y_LIM_GROUP[0])
                for i in sig:
                    ax.text(tv[i], yp, '*', fontsize=14, ha='center', fontweight='bold')

            ax.axvline(0, color='grey', ls='--', lw=1)
            ax.axhline(0, color='k', ls='-', alpha=0.3)
            ax.axvspan(SURFACE_WINDOW_START, SURFACE_WINDOW_END, alpha=0.15, color='yellow')
            ax.set(xlabel='Time (s)', ylabel='BOLD (z-scored)',
                   title=f'{roi_title} — {run_label} (N={n})',
                   xlim=(tv[0] - 0.5, tv[-1] + 0.5), ylim=Y_LIM_GROUP)
            ax.legend(loc='upper right')
            ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / f"GROUP_filmfest_segb_timecourse_byrun_{lock_mode}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# PLOTTING — Group surface (per-run comparison)
# ============================================================================

def plot_group_surface_by_run(run1_results, run2_results, lock_mode,
                              threshold=SURFACE_THRESHOLD):
    """Plot side-by-side surface contrast for filmfest1 vs filmfest2 (2x4).

    Uses a fixed display threshold (uncorrected) since N per run is small.
    """
    mode_label = 'Onset-locked' if lock_mode == 'onset' else 'Offset-locked'

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})
    fig.suptitle(f"Filmfest SEG-B ({mode_label}): Boundary > Non-Boundary\n"
                 f"Per-Run Comparison (window {SURFACE_WINDOW_START}–{SURFACE_WINDOW_END}s), "
                 f"display |t|>{threshold}",
                 fontsize=TITLE_FS, fontweight='bold', y=1.02)

    for row, (results, run_key) in enumerate([
        (run1_results, 'filmfest1'),
        (run2_results, 'filmfest2'),
    ]):
        n = len(results)
        run_label = RUN_LABELS[run_key]
        if n < 2:
            continue

        # Group t-test per vertex, per hemisphere
        for hemi, key in [('left', 't_map_left'), ('right', 't_map_right')]:
            t_matrix = np.array([r[key] for r in results])
            n_verts = t_matrix.shape[1]
            group_t = np.zeros(n_verts)
            group_p = np.ones(n_verts)
            for v in range(n_verts):
                if np.std(t_matrix[:, v]) > 0:
                    group_t[v], group_p[v] = stats.ttest_1samp(t_matrix[:, v], 0)
            if hemi == 'left':
                group_t_L = group_t
            else:
                group_t_R = group_t

        print(f"  {run_key}: displaying |t|>{threshold} (N={n})")

        views = [
            (group_t_L, 'left',  'lateral'),
            (group_t_L, 'left',  'medial'),
            (group_t_R, 'right', 'lateral'),
            (group_t_R, 'right', 'medial'),
        ]

        for col, (data, hemi, view) in enumerate(views):
            ax = axes[row, col]
            plotting.plot_surf_stat_map(
                FSAVERAGE[f'infl_{hemi}'], data,
                hemi=hemi, view=view,
                bg_map=FSAVERAGE[f'sulc_{hemi}'],
                cmap='coolwarm', threshold=threshold,
                vmax=SURFACE_VMAX, colorbar=False,
                axes=ax, figure=fig,
            )
            title = f'{hemi.capitalize()} {view}'
            if col == 0:
                title = f"{run_label}\n{title}"
            ax.set_title(title, fontsize=10)

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-SURFACE_VMAX, vmax=SURFACE_VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.05, shrink=0.5)
    cbar.set_label('t-value', fontsize=10)

    out = OUTPUT_DIR / f"GROUP_filmfest_segb_surface_byrun_{lock_mode}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# PLOTTING — Group combined
# ============================================================================

def plot_group_timecourse(all_results, lock_mode):
    mode_label = 'Onset-locked' if lock_mode == 'onset' else 'Offset-locked'
    n = len(all_results)
    tv = all_results[0]['time_vec']

    stacks = {}
    for cond in ('boundary', 'nonboundary'):
        for roi in ('pmc', 'hipp'):
            stacks[f'{roi}_{cond}'] = np.array(
                [r[f'{roi}_{cond}_tc'] for r in all_results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Filmfest SEG-B ({mode_label}): Group (N={n} runs)\n"
                 f"* p < 0.05 paired t-test + FDR",
                 fontsize=TITLE_FS, fontweight='bold')

    for ax, roi, title in [(axes[0], 'pmc', 'Posterior Medial Cortex'),
                            (axes[1], 'hipp', 'Hippocampus')]:
        bnd = stacks[f'{roi}_boundary']
        nbnd = stacks[f'{roi}_nonboundary']
        for arr, label, c in [(bnd, 'Boundary', COLORS['boundary']),
                                (nbnd, 'Non-boundary', COLORS['nonboundary'])]:
            m = arr.mean(0)
            se = arr.std(0) / np.sqrt(n)
            ax.plot(tv, m, color=c, lw=3, label=label, marker='o', ms=4)
            ax.fill_between(tv, m - se, m + se, color=c, alpha=0.3)

        pvals = np.array([stats.ttest_rel(bnd[:, t], nbnd[:, t])[1]
                          for t in range(len(tv))])
        _, fdr_mask = fdr_threshold(pvals, q=FDR_Q)
        sig = np.where(fdr_mask)[0]
        if len(sig):
            yp = Y_LIM_GROUP[0] + 0.05 * (Y_LIM_GROUP[1] - Y_LIM_GROUP[0])
            for i in sig:
                ax.text(tv[i], yp, '*', fontsize=14, ha='center', fontweight='bold')

        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.axvspan(SURFACE_WINDOW_START, SURFACE_WINDOW_END, alpha=0.15, color='yellow')
        ax.set(xlabel='Time (s)', ylabel='BOLD (z-scored)', title=title,
               xlim=(tv[0] - 0.5, tv[-1] + 0.5), ylim=Y_LIM_GROUP)
        ax.legend(loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / f"GROUP_filmfest_segb_timecourse_{lock_mode}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def plot_group_surface(all_results, lock_mode, threshold=SURFACE_THRESHOLD):
    """Plot group surface maps per run (2 rows x 4 cols), using all subjects."""
    mode_label = 'Onset-locked' if lock_mode == 'onset' else 'Offset-locked'
    run1_results = [r for r in all_results if r['task'] == 'filmfest1']
    run2_results = [r for r in all_results if r['task'] == 'filmfest2']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})
    fig.suptitle(f"Filmfest SEG-B ({mode_label}): Boundary > Non-Boundary\n"
                 f"Group (N={len(all_results)} runs), "
                 f"window {SURFACE_WINDOW_START}–{SURFACE_WINDOW_END}s, "
                 f"display |t|>{threshold}",
                 fontsize=TITLE_FS, fontweight='bold', y=1.02)

    for row, (results, run_key) in enumerate([
        (run1_results, 'filmfest1'),
        (run2_results, 'filmfest2'),
    ]):
        n = len(results)
        run_label = RUN_LABELS[run_key]
        if n < 2:
            for col in range(4):
                axes[row, col].set_visible(False)
            continue

        for hemi, key in [('left', 't_map_left'), ('right', 't_map_right')]:
            t_matrix = np.array([r[key] for r in results])
            n_verts = t_matrix.shape[1]
            group_t = np.zeros(n_verts)
            for v in range(n_verts):
                if np.std(t_matrix[:, v]) > 0:
                    group_t[v], _ = stats.ttest_1samp(t_matrix[:, v], 0)
            if hemi == 'left':
                group_t_L = group_t
            else:
                group_t_R = group_t

        print(f"  {run_key}: displaying |t|>{threshold} (N={n})")

        views = [
            (group_t_L, 'left',  'lateral'),
            (group_t_L, 'left',  'medial'),
            (group_t_R, 'right', 'lateral'),
            (group_t_R, 'right', 'medial'),
        ]
        for col, (data, hemi, view) in enumerate(views):
            ax = axes[row, col]
            plotting.plot_surf_stat_map(
                FSAVERAGE[f'infl_{hemi}'], data,
                hemi=hemi, view=view,
                bg_map=FSAVERAGE[f'sulc_{hemi}'],
                cmap='coolwarm', threshold=threshold,
                vmax=SURFACE_VMAX, colorbar=False,
                axes=ax, figure=fig,
            )
            title = f'{hemi.capitalize()} {view}'
            if col == 0:
                title = f"{run_label}\n{title}"
            ax.set_title(title, fontsize=10)

    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                               norm=plt.Normalize(vmin=-SURFACE_VMAX, vmax=SURFACE_VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.02, pad=0.05, shrink=0.5)
    cbar.set_label('t-value', fontsize=10)

    out = OUTPUT_DIR / f"GROUP_filmfest_segb_surface_{lock_mode}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# PLOTTING — Subject level
# ============================================================================

def plot_subject_timecourse(subject, run1_result, run2_result, lock_mode):
    """Plot per-run time courses for a single subject (2 rows x 2 cols)."""
    mode_label = 'Onset-locked' if lock_mode == 'onset' else 'Offset-locked'
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Filmfest SEG-B ({mode_label}): {subject}",
                 fontsize=TITLE_FS, fontweight='bold')

    for row, (result, run_key) in enumerate([
        (run1_result, 'filmfest1'),
        (run2_result, 'filmfest2'),
    ]):
        if result is None:
            continue
        tv = result['time_vec']
        run_label = RUN_LABELS[run_key]

        for col, (roi, roi_title) in enumerate([('pmc', 'PMC'), ('hipp', 'Hippocampus')]):
            ax = axes[row, col]
            for cond, label in [('boundary', 'Boundary'), ('nonboundary', 'Non-boundary')]:
                c = COLORS[cond]
                m = result[f'{roi}_{cond}_tc']
                se = result[f'{roi}_{cond}_sem']
                ax.plot(tv, m, color=c, lw=3, label=label)
                ax.fill_between(tv, m - se, m + se, color=c, alpha=0.3)
            ax.axvline(0, color='grey', ls='--', alpha=0.5)
            ax.axhline(0, color='k', ls='-', alpha=0.3)
            ax.axvspan(SURFACE_WINDOW_START, SURFACE_WINDOW_END, alpha=0.15, color='yellow')
            ax.set(xlabel='Time (s)', ylabel='BOLD (z-scored)',
                   title=f'{roi_title} — {run_label}',
                   xlim=(tv[0], tv[-1]), ylim=Y_LIM_SINGLE)
            ax.legend(loc='upper right')
            ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / f"{subject}_filmfest_segb_timecourse_{lock_mode}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def plot_subject_surface(subject, run1_result, run2_result, lock_mode,
                         threshold=SURFACE_THRESHOLD):
    """Plot per-run surface maps for a single subject (2 rows x 4 cols)."""
    mode_label = 'Onset-locked' if lock_mode == 'onset' else 'Offset-locked'
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})
    fig.suptitle(f"Filmfest SEG-B ({mode_label}): Boundary > Non-Boundary\n"
                 f"{subject}, window {SURFACE_WINDOW_START}–{SURFACE_WINDOW_END}s, "
                 f"display |t|>{threshold}",
                 fontsize=TITLE_FS, fontweight='bold', y=1.02)

    for row, (result, run_key) in enumerate([
        (run1_result, 'filmfest1'),
        (run2_result, 'filmfest2'),
    ]):
        if result is None:
            for col in range(4):
                axes[row, col].set_visible(False)
            continue
        run_label = RUN_LABELS[run_key]
        t_L = result['t_map_left']
        t_R = result['t_map_right']

        views = [
            (t_L, 'left',  'lateral'),
            (t_L, 'left',  'medial'),
            (t_R, 'right', 'lateral'),
            (t_R, 'right', 'medial'),
        ]
        for col, (data, hemi, view) in enumerate(views):
            ax = axes[row, col]
            plotting.plot_surf_stat_map(
                FSAVERAGE[f'infl_{hemi}'], data,
                hemi=hemi, view=view,
                bg_map=FSAVERAGE[f'sulc_{hemi}'],
                cmap='coolwarm', threshold=threshold,
                vmax=SURFACE_VMAX, colorbar=False,
                axes=ax, figure=fig,
            )
            title = f'{hemi.capitalize()} {view}'
            if col == 0:
                title = f"{run_label}\n{title}"
            ax.set_title(title, fontsize=10)

    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                               norm=plt.Normalize(vmin=-SURFACE_VMAX, vmax=SURFACE_VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.02, pad=0.05, shrink=0.5)
    cbar.set_label('t-value', fontsize=10)

    out = OUTPUT_DIR / f"{subject}_filmfest_segb_surface_{lock_mode}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_by_subject(results):
    by_sub = defaultdict(list)
    for r in results:
        by_sub[r['subject']].append(r)

    agg = {}
    for sub, runs in by_sub.items():
        n = len(runs)
        d = {'time_vec': runs[0]['time_vec'], 'n_runs': n}
        for roi in ('pmc', 'hipp'):
            for cond in ('boundary', 'nonboundary'):
                k = f'{roi}_{cond}_tc'
                stack = np.array([r[k] for r in runs])
                d[f'{roi}_{cond}_tc'] = stack.mean(0)
                d[f'{roi}_{cond}_sem'] = (
                    stack.std(0) / np.sqrt(n) if n > 1 else np.zeros_like(stack[0]))
        d['t_map_left'] = np.mean([r['t_map_left'] for r in runs], axis=0)
        d['t_map_right'] = np.mean([r['t_map_right'] for r in runs], axis=0)
        agg[sub] = d
    return agg


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("FILMFEST SEG-B BOUNDARY ANALYSIS")
    print(f"Surface contrast window: {SURFACE_WINDOW_START}–{SURFACE_WINDOW_END}s")
    print("=" * 60)

    for lock_mode in ('onset', 'offset'):
        print(f"\n--- Lock mode: {lock_mode} ---")
        for task in ('filmfest1', 'filmfest2'):
            bnd, nbnd = get_segb_events(task, lock_mode)
            print(f"  {task}: {len(bnd)} boundary, {len(nbnd)} non-boundary events")

    for lock_mode in ('onset',):
        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {lock_mode.upper()}-LOCKED")
        print(f"{'=' * 60}")

        all_results = []
        for subject, session in FILMFEST_SUBJECTS.items():
            for task in ('filmfest1', 'filmfest2'):
                result = process_run(subject, session, task, lock_mode)
                if result is not None:
                    all_results.append(result)

        print(f"\nProcessed {len(all_results)} runs ({lock_mode})")

        run1_results = [r for r in all_results if r['task'] == 'filmfest1']
        run2_results = [r for r in all_results if r['task'] == 'filmfest2']

        # --- Subject-level ---
        print(f"\n--- Subject-level plots ({lock_mode}) ---")
        # Index results by subject + task for per-run lookup
        by_sub_task = {}
        for r in all_results:
            by_sub_task[(r['subject'], r['task'])] = r
        for subject in sorted(FILMFEST_SUBJECTS.keys()):
            r1 = by_sub_task.get((subject, 'filmfest1'))
            r2 = by_sub_task.get((subject, 'filmfest2'))
            if r1 is not None or r2 is not None:
                plot_subject_timecourse(subject, r1, r2, lock_mode)
                plot_subject_surface(subject, r1, r2, lock_mode)

        # --- Per-run group plots ---
        if len(run1_results) >= 2 and len(run2_results) >= 2:
            print(f"\n--- Per-run group plots ({lock_mode}) ---")
            plot_group_timecourse_by_run(run1_results, run2_results, lock_mode)
            plot_group_surface_by_run(run1_results, run2_results, lock_mode)

        # --- Combined group plots ---
        if len(all_results) >= 2:
            print(f"\n--- Combined group plots ({lock_mode}) ---")
            plot_group_timecourse(all_results, lock_mode)
            plot_group_surface(all_results, lock_mode)

    print("\n" + "=" * 60)
    print(f"DONE. Figures in {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
