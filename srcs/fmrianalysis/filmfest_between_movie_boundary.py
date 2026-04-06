"""
Filmfest Movie Boundary Analysis

ROI time courses (PMC, Hippocampus, Angular Gyrus, Auditory Cortex)
locked to the offset of the previous movie (i.e., the boundary between
consecutive movies within each run).

4 boundaries per run × 2 runs × 6 subjects = 48 boundary events.
Plot: -30 to +60 s, individual subjects + group average.

Usage:
    python filmfest_movie_boundary.py
"""
from pathlib import Path

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from nilearn import datasets, surface
from nilearn.plotting import plot_surf

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, FILMFEST_SUBJECTS, MOVIE_INFO
from configs.schaefer_rois import POSTERIOR_MEDIAL, ANGULAR_GYRUS, EARLY_AUDITORY, EARLY_VISUAL
from fmrianalysis.utils import get_parcel_data, get_movie_boundary_offsets

ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
OUTPUT_DIR = FIGS_DIR / 'filmfest_movie_boundary'

# === PARAMETERS ===
TRS_BEFORE = 20   # 30s / 1.5s = 20 TRs
TRS_AFTER = 40    # 60s / 1.5s = 40 TRs

# === STYLE ===
SUBJECT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
TITLE_FS = 14
LABEL_FS = 12

# === ROI definitions (labels to average from cached parcel data) ===
ROI_SPEC = [
    ('pmc',  'Posterior Medial Cortex'),
    ('hipp', 'Hippocampus'),
    ('ag',   'Angular Gyrus'),
    ('eac',  'Auditory Cortex'),
    ('evc',  'Early Visual Cortex'),
]
PMC_LABELS = POSTERIOR_MEDIAL.get('left_labels', []) + POSTERIOR_MEDIAL.get('right_labels', [])
AG_LABELS = ANGULAR_GYRUS.get('left_labels', []) + ANGULAR_GYRUS.get('right_labels', [])
EAC_LABELS = EARLY_AUDITORY.get('left_labels', []) + EARLY_AUDITORY.get('right_labels', [])
EVC_LABELS = EARLY_VISUAL.get('left_labels', []) + EARLY_VISUAL.get('right_labels', [])
HIPP_KEYWORDS = ['hippocampus']

# === ROI surface visualization ===
# Parcel IDs (1-based) for projecting to inflated surface
# Best view for each ROI: 'medial' for PMC, 'lateral' for AG/EAC/EVC
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
}
ROI_COLOR = '#e74c3c'
ROI_CMAP = LinearSegmentedColormap.from_list('roi', ['#888888', ROI_COLOR], N=256)


def _build_surface_parcellation():
    """Project Schaefer atlas to fsaverage surface (cached across calls)."""
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=17, resolution_mm=2)
    fsavg = datasets.fetch_surf_fsaverage()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        surf_l = surface.vol_to_surf(
            atlas['maps'], fsavg['pial_left'],
            interpolation='nearest_most_frequent').astype(int)
        surf_r = surface.vol_to_surf(
            atlas['maps'], fsavg['pial_right'],
            interpolation='nearest_most_frequent').astype(int)
    return fsavg, surf_l, surf_r


def plot_roi_brain(ax, roi_key, fsavg, surf_l, surf_r):
    """Render an ROI on inflated brain surface (gray background, red ROI).

    For 'hipp' (subcortical), show a text label instead.
    """
    if roi_key == 'hipp':
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'Hippocampus', ha='center', va='center',
                fontsize=LABEL_FS, transform=ax.transAxes)
        return

    spec = ROI_SURFACE[roi_key]
    view = spec['view']

    # Build bilateral mask: ROI vertices = 1.0, non-ROI = NaN
    # NaN lets the sulcal background (inflated surface anatomy) show through
    mask_l = np.where(np.isin(surf_l, spec['left']), 1.0, np.nan)
    mask_r = np.where(np.isin(surf_r, spec['right']), 1.0, np.nan)

    # Medial views use right hemisphere, lateral views use left hemisphere
    # so the brain outline faces the same direction across all rows
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
# HELPERS
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
    """Load cached parcel data and return dict of ROI time series."""
    schaefer = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
    ho_sub = get_parcel_data(subject, session, task, atlas='HarvardOxford_sub')
    return {
        'pmc':  _avg_labels(schaefer, PMC_LABELS),
        'ag':   _avg_labels(schaefer, AG_LABELS),
        'eac':  _avg_labels(schaefer, EAC_LABELS),
        'evc':  _avg_labels(schaefer, EVC_LABELS),
        'hipp': _avg_keywords(ho_sub, HIPP_KEYWORDS),
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get movie boundary times
    boundaries = {}
    for task in ('filmfest1', 'filmfest2'):
        boundaries[task] = get_movie_boundary_offsets(task)
        print(f"{task} movie boundaries (s): {boundaries[task]}")

    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR

    # Collect per-subject averaged time courses for each ROI
    roi_keys = [k for k, _ in ROI_SPEC]
    subject_data = {k: [] for k in roi_keys}  # roi_key -> list of (n_timepoints,)
    subject_labels = []

    for subject, session in FILMFEST_SUBJECTS.items():
        all_epochs = {k: [] for k in roi_keys}

        for task in ('filmfest1', 'filmfest2'):
            print(f"  {subject} {session} {task}: ", end='', flush=True)
            roi_ts = extract_roi_timeseries(subject, session, task)
            print(f"{len(roi_ts['pmc'])} TRs loaded")

            bnd_times = boundaries[task]
            for k in roi_keys:
                epochs = extract_event_locked(roi_ts[k], bnd_times)
                if epochs is not None:
                    all_epochs[k].append(epochs)

        n_epochs = np.vstack(all_epochs['pmc']).shape[0]
        print(f"  -> {subject}: {n_epochs} movie boundary epochs")

        for k in roi_keys:
            stacked = np.vstack(all_epochs[k])
            subject_data[k].append(stacked.mean(axis=0))
        subject_labels.append(subject)

    for k in roi_keys:
        subject_data[k] = np.array(subject_data[k])  # (n_subjects, n_timepoints)

    # --- Build surface parcellation for brain insets ---
    print("\nPreparing surface parcellation for ROI insets...")
    fsavg, surf_l, surf_r = _build_surface_parcellation()

    # --- Plot: brain inset (left) + time course (right) for each ROI ---
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(5, 2, width_ratios=[1, 3], wspace=0.05, hspace=0.35,
                           top=0.95)
    fig.suptitle("Movie Boundary Response (offset-locked)\n"
                 f"N={len(subject_labels)} subjects, 8 boundaries each",
                 fontsize=TITLE_FS, fontweight='bold', y=0.97)

    tc_axes = []
    for row, (roi_key, roi_title) in enumerate(ROI_SPEC):
        # Left column: brain surface inset
        if roi_key == 'hipp':
            ax_brain = fig.add_subplot(gs[row, 0])
        else:
            ax_brain = fig.add_subplot(gs[row, 0], projection='3d')
        plot_roi_brain(ax_brain, roi_key, fsavg, surf_l, surf_r)
        if roi_key != 'hipp':
            ax_brain.set_title(roi_title, fontsize=LABEL_FS, pad=-10, y=0)

        # Right column: time course
        ax_tc = fig.add_subplot(gs[row, 1])
        tc_axes.append(ax_tc)
        data = subject_data[roi_key]

        # Individual subjects
        for i, (tc, label) in enumerate(zip(data, subject_labels)):
            ax_tc.plot(time_vec, tc, color=SUBJECT_COLORS[i], lw=1, alpha=0.5,
                       label=label)

        # Group average +/- SEM
        group_mean = data.mean(axis=0)
        group_sem = data.std(axis=0) / np.sqrt(len(data))
        ax_tc.plot(time_vec, group_mean, color='k', lw=3)
        ax_tc.fill_between(time_vec, group_mean - group_sem,
                           group_mean + group_sem, color='k', alpha=0.2)

        ax_tc.axvline(0, color='grey', ls='--', lw=1)
        ax_tc.axhline(0, color='k', ls='-', alpha=0.3)
        ax_tc.set(xlabel='Time from movie offset (s)', ylabel='BOLD (z-scored)',
                  xlim=(-30, 60))
        ax_tc.legend(loc='upper right', fontsize=8)
        ax_tc.spines[['top', 'right']].set_visible(False)

    # Unify y-axis limits across time course subplots
    all_ylims = [ax.get_ylim() for ax in tc_axes]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax in tc_axes:
        ax.set_ylim(ymin, ymax)

    out = OUTPUT_DIR / "movie_boundary_timecourse.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
