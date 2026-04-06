"""
Vertex-level boundary analysis: conjunction z-score surface map + ROI time courses.

Three tasks: SVF, AHC, movie (filmfest1 + filmfest2 combined).

Conjunction map: parcels where mean BOLD z-score (0-9 TR post-boundary) > threshold
in ALL three tasks. Displayed value = minimum z across tasks.

ROI time courses: PMC, dlPFC, dACC, mPFC, AG — bilateral vertex averages.

Alignment modes:
  --align offset  t=0 = end of trial / movie segment  [default]
  --align onset   t=0 = start of next trial / movie segment
  For filmfest the boundary time is identical under both modes (no inter-movie gap).

Usage:
    python task_boundary_vertex.py [--align offset|onset]
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import butter, filtfilt
from scipy.stats import zscore as sp_zscore
from nilearn import datasets, plotting
from nilearn.plotting import plot_surf

from configs.config import (
    DERIVATIVES_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR,
    FILMFEST_SUBJECTS, MOVIE_INFO, SUBJECT_IDS,
)
from configs.schaefer_rois import (
    POSTERIOR_MEDIAL, ANGULAR_GYRUS, DLPFC, DACC, MPFC,
)
from fmrianalysis.utils import load_surface_data, highpass_filter, mss_to_seconds

# ============================================================================
# CONFIG
# ============================================================================

PSYCHOPY_DIR     = Path('/home/datasets/stateswitch/psychopy')
ANNOTATIONS_DIR  = Path('/home/datasets/stateswitch/filmfest_annotations')
OUTPUT_DIR       = FIGS_DIR / 'task_boundary_vertex'
CACHE_DIR        = ANALYSIS_CACHE_DIR / 'task_boundary_vertex'

TRS_BEFORE  = 20   # 30 s baseline
TRS_AFTER   = 40   # 60 s after boundary
WIN_START   = 0    # TRs post-boundary (z-map window)
WIN_END     = 10   # exclusive → 0-9 TRs = 0-13.5 s
CONJ_THRESH = 0.1  # min z across tasks to enter conjunction

TASK_KEYS    = ('svf', 'ahc', 'movie')
TASK_DISPLAY = {
    'svf':   'Semantic Fluency',
    'ahc':   'Explanation Generation',
    'movie': 'FilmFest',
}
TASK_COLORS  = {'svf': '#1f77b4', 'ahc': '#ff7f0e', 'movie': '#2ca02c'}

# ROI definitions: (name, LH global parcel IDs, RH global parcel IDs)
ROI_DEFS = [
    ('PMC',   POSTERIOR_MEDIAL['left'], POSTERIOR_MEDIAL['right']),
    ('dlPFC', DLPFC['left'],            DLPFC['right']),
    ('dACC',  DACC['left'],             DACC['right']),
    ('mPFC',  MPFC['left'],             MPFC['right']),
    ('AG',    ANGULAR_GYRUS['left'],    ANGULAR_GYRUS['right']),
]

LABEL_FS = 11
TITLE_FS = 13


# ============================================================================
# SURFACE ATLAS SETUP (Schaefer 400 from fsaverage6 annot files)
# ============================================================================

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT  = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT  = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'


def _build_schaefer_surface():
    """Read Schaefer 400 17-net annot files to get vertex parcel IDs.

    LH annot index == global parcel ID (1–200; 0 = background).
    RH annot index + 200 == global parcel ID (201–400; 0 = background).
    """
    import nibabel.freesurfer as fs
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')
    lh_labels, _, _ = fs.read_annot(str(LH_ANNOT))
    rh_labels, _, _ = fs.read_annot(str(RH_ANNOT))
    surf_l = lh_labels.astype(int)                                    # 0–200
    surf_r = np.where(rh_labels > 0, rh_labels + 200, 0).astype(int) # 0 or 201–400
    return fsavg, surf_l, surf_r


def _roi_masks(surf_l, surf_r):
    """Return dict of roi_name -> (mask_L, mask_R) boolean arrays over vertices."""
    masks = {}
    for name, lh_ids, rh_ids in ROI_DEFS:
        masks[name] = (
            np.isin(surf_l, lh_ids),
            np.isin(surf_r, rh_ids),
        )
    return masks


# ============================================================================
# EVENT TIME HELPERS
# ============================================================================

def _parse_svf_ahc_events(subject, session, task, align):
    """Return event times (seconds, scan-relative) for SVF or AHC.

    align='offset': t=0 at trial stopped (response)
    align='onset':  t=0 at trial started (next trial onset)
    """
    ses_dir = PSYCHOPY_DIR / subject / session
    if not ses_dir.exists():
        return None
    stopped_col = f'{task}_trial.stopped'
    started_col = f'{task}_trial.started'
    csv_path = None
    for p in sorted(ses_dir.glob('*.csv')):
        df = pd.read_csv(p, nrows=0)
        if stopped_col in df.columns:
            csv_path = p
            break
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    mask = df[stopped_col].notna()
    if not mask.any():
        return None
    rows = df[mask]
    t0 = rows[started_col].iloc[0]
    if align == 'offset':
        times = (rows[stopped_col].values - t0).tolist()
        return times[:-1]   # drop last (no boundary after)
    else:
        times = (rows[started_col].values - t0).tolist()
        return times[1:]    # drop first (onset=0)


TITLE_SCENE_OFFSET = 4 * TR  # 6 s title scene at the start of each filmfest movie


def _get_movie_events(align):
    """Return {filmfest_task: [boundary_times_sec]} for filmfest1 and filmfest2.

    For filmfest, movies play back-to-back so offset == onset of next movie.
    When align='onset', shift boundaries forward by TITLE_SCENE_OFFSET (4 TRs = 6 s)
    to skip the title card and align to when the actual movie content begins.
    """
    result = {}
    for ff_task in ('filmfest1', 'filmfest2'):
        movies = [m for m in MOVIE_INFO if m['task'] == ff_task]
        boundaries = []
        for movie in movies[:-1]:
            df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
            segb = df.dropna(subset=['SEG-B_Number'])
            last_end = segb['End Time (m.ss)'].values[-1]
            boundaries.append(mss_to_seconds(last_end))
        if align == 'onset':
            boundaries = [t + TITLE_SCENE_OFFSET for t in boundaries]
        result[ff_task] = boundaries
    return result


# ============================================================================
# EPOCH EXTRACTION
# ============================================================================

def _extract_epochs_2d(data, event_times_sec, trs_before=TRS_BEFORE, trs_after=TRS_AFTER):
    """Extract event-locked epochs from 2D surface data.

    Parameters
    ----------
    data : (T, V) array
    event_times_sec : list of floats

    Returns
    -------
    epochs : (n_valid_events, n_trs, V) array  or None
    """
    n_time = data.shape[0]
    centers = np.round(np.array(event_times_sec) / TR).astype(int)
    offsets = np.arange(-trs_before, trs_after + 1)
    idx = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n_time), axis=1)
    if not valid.any():
        return None
    return data[idx[valid]]    # (n_events, n_trs, V)


def _extract_roi_tc(data_l, data_r, roi_masks):
    """Return dict of roi_name -> 1D time series (mean over bilateral vertices)."""
    out = {}
    for name, (mask_l, mask_r) in roi_masks.items():
        parts = []
        if mask_l.any():
            parts.append(data_l[:, mask_l].mean(axis=1))
        if mask_r.any():
            parts.append(data_r[:, mask_r].mean(axis=1))
        out[name] = np.stack(parts, axis=1).mean(axis=1) if parts else np.zeros(data_l.shape[0])
    return out


# ============================================================================
# SURFACE DATA LOADING
# ============================================================================

def _load_surf(subject, session, task):
    """Load, highpass filter, and z-score bilateral surface data.

    Returns (surf_L, surf_R), each (T, V), or (None, None) on failure.
    """
    try:
        raw_l = load_surface_data(subject, session, task, 'L', DERIVATIVES_DIR)
        raw_r = load_surface_data(subject, session, task, 'R', DERIVATIVES_DIR)
    except FileNotFoundError:
        return None, None
    surf_l = sp_zscore(highpass_filter(raw_l.astype(np.float64).T), axis=0, nan_policy='omit')
    surf_r = sp_zscore(highpass_filter(raw_r.astype(np.float64).T), axis=0, nan_policy='omit')
    return np.nan_to_num(surf_l), np.nan_to_num(surf_r)


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _parcel_to_surface_vertex(values_l, values_r):
    """Pass-through: vertex arrays are already in surface space."""
    return values_l, values_r


def _plot_conjunction_surface(conj_l, conj_r, fsavg, title, outpath, vmin, vmax):
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(title, fontsize=TITLE_FS, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 2, wspace=0.01, hspace=0.01,
                           top=0.92, bottom=0.05, left=0.02, right=0.90)
    panels = [
        (0, 0, fsavg['infl_left'],  conj_l, 'left',  'lateral', fsavg['sulc_left']),
        (0, 1, fsavg['infl_right'], conj_r, 'right', 'lateral', fsavg['sulc_right']),
        (1, 0, fsavg['infl_left'],  conj_l, 'left',  'medial',  fsavg['sulc_left']),
        (1, 1, fsavg['infl_right'], conj_r, 'right', 'medial',  fsavg['sulc_right']),
    ]
    for row, col, mesh, sdata, hemi, view, bg in panels:
        ax = fig.add_subplot(gs[row, col], projection='3d')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_surf(
                mesh, surf_map=sdata, hemi=hemi, view=view,
                bg_map=bg, axes=ax, colorbar=False,
                cmap='Reds', bg_on_data=True, darkness=0.5,
                vmin=vmin, vmax=vmax,
            )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap='Reds',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('min z-score across tasks (0–9 TR)', fontsize=LABEL_FS)
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {outpath}')


def _plot_task_surface(zmap_l, zmap_r, fsavg, title, outpath, vmin, vmax):
    """Plot a single task's mean z-map on inflated surface (4 panels)."""
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(title, fontsize=TITLE_FS, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 2, wspace=0.01, hspace=0.01,
                           top=0.92, bottom=0.05, left=0.02, right=0.90)
    panels = [
        (0, 0, fsavg['infl_left'],  zmap_l, 'left',  'lateral', fsavg['sulc_left']),
        (0, 1, fsavg['infl_right'], zmap_r, 'right', 'lateral', fsavg['sulc_right']),
        (1, 0, fsavg['infl_left'],  zmap_l, 'left',  'medial',  fsavg['sulc_left']),
        (1, 1, fsavg['infl_right'], zmap_r, 'right', 'medial',  fsavg['sulc_right']),
    ]
    for row, col, mesh, sdata, hemi, view, bg in panels:
        ax = fig.add_subplot(gs[row, col], projection='3d')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_surf(
                mesh, surf_map=sdata, hemi=hemi, view=view,
                bg_map=bg, axes=ax, colorbar=False,
                cmap='RdBu_r', bg_on_data=True, darkness=0.5,
                vmin=-vmax, vmax=vmax,
            )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                               norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('mean z-score (0–9 TR)', fontsize=LABEL_FS)
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {outpath}')


def _plot_roi_timecourses(group_tc, time_vec, align, outpath):
    """Plot mean ± SEM time courses per ROI per task.

    group_tc : dict task -> dict roi_name -> list of subject-mean 1D arrays
    """
    roi_names = [r[0] for r in ROI_DEFS]
    n_rois = len(roi_names)
    fig, axes = plt.subplots(1, n_rois, figsize=(4 * n_rois, 4), sharey=True)
    fig.suptitle(f'ROI time courses — boundary-locked (align={align})',
                 fontsize=TITLE_FS, fontweight='bold')

    for ax, roi in zip(axes, roi_names):
        for task in TASK_KEYS:
            tcs = group_tc[task][roi]
            if not tcs:
                continue
            arr = np.array(tcs)        # (n_subj, n_trs)
            mean = arr.mean(axis=0)
            sem  = arr.std(axis=0) / np.sqrt(len(arr))
            color = TASK_COLORS[task]
            ax.plot(time_vec, mean, color=color, lw=1.5, label=TASK_DISPLAY[task])
            ax.fill_between(time_vec, mean - sem, mean + sem,
                            color=color, alpha=0.2)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.axhline(0, color='k', lw=0.4, ls=':')
        if align == 'offset':
            ax.axvline(15, color='gray', lw=0.8, ls='--')  # next trial onset
        ax.set_ylim(-1, 1)
        ax.set_xlim(time_vec[0], time_vec[-1])
        ax.set_title(roi, fontsize=LABEL_FS, fontweight='bold')
        ax.set_xlabel('Time from boundary (s)', fontsize=9)
        ax.set_ylabel('BOLD (z-score)', fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].legend(fontsize=8, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {outpath}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--align', choices=['offset', 'onset'], default='offset',
                        help='Align epochs to trial offset or onset (default: offset)')
    args = parser.parse_args()
    align = args.align

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print('Building Schaefer surface parcellation (fsaverage6)...')
    fsavg, surf_l_atlas, surf_r_atlas = _build_schaefer_surface()
    roi_masks = _roi_masks(surf_l_atlas, surf_r_atlas)
    for name, (ml, mr) in roi_masks.items():
        print(f'  {name}: {ml.sum()} L + {mr.sum()} R vertices')

    # movie boundary times are subject-independent
    movie_events = _get_movie_events(align)
    for ff_task, times in movie_events.items():
        print(f'  {ff_task} movie boundaries: {times}')

    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR
    roi_names = [r[0] for r in ROI_DEFS]

    # Accumulators
    # group_zmaps[task] = list of per-subject mean z arrays over window
    #   each entry: {'L': (V_L,), 'R': (V_R,)}
    group_zmaps = {t: [] for t in TASK_KEYS}
    # group_tc[task][roi] = list of per-subject mean epoch 1D arrays
    group_tc = {t: {roi: [] for roi in roi_names} for t in TASK_KEYS}

    # ── SVF and AHC ──────────────────────────────────────────────────────────
    for subject in SUBJECT_IDS:
        print(f'\n{subject}')
        # Discover (session, task) pairs
        sub_dir = DERIVATIVES_DIR / subject
        if not sub_dir.exists():
            continue
        for ses_dir in sorted(sub_dir.glob('ses-*')):
            session = ses_dir.name
            func_dir = ses_dir / 'func'
            if not func_dir.exists():
                continue
            for task_key in ('svf', 'ahc'):
                bold_files = list(func_dir.glob(f'*_task-{task_key}_*_bold.nii.gz'))
                if not bold_files:
                    continue
                event_times = _parse_svf_ahc_events(subject, session, task_key, align)
                if not event_times:
                    print(f'  {session} {task_key}: no events found, skipping')
                    continue
                surf_l, surf_r = _load_surf(subject, session, task_key)
                if surf_l is None:
                    print(f'  {session} {task_key}: surface data missing, skipping')
                    continue
                print(f'  {session} {task_key}: {len(event_times)} events, '
                      f'{surf_l.shape[0]} TRs')
                # ROI time courses
                roi_tc = _extract_roi_tc(surf_l, surf_r, roi_masks)
                for roi in roi_names:
                    ep_l = _extract_epochs_2d(surf_l[:, roi_masks[roi][0]], event_times)
                    ep_r = _extract_epochs_2d(surf_r[:, roi_masks[roi][1]], event_times)
                    parts = []
                    if ep_l is not None:
                        parts.append(ep_l.mean(axis=2))  # (n_ev, n_trs)
                    if ep_r is not None:
                        parts.append(ep_r.mean(axis=2))
                    if parts:
                        combined = np.stack(parts, axis=0).mean(axis=0)  # (n_ev, n_trs)
                        group_tc[task_key][roi].append(combined.mean(axis=0))  # (n_trs,)
                # Vertex z-map over window
                ep_l_full = _extract_epochs_2d(surf_l, event_times)
                ep_r_full = _extract_epochs_2d(surf_r, event_times)
                if ep_l_full is not None and ep_r_full is not None:
                    zmap_l = ep_l_full[:, TRS_BEFORE + WIN_START:TRS_BEFORE + WIN_END, :].mean(axis=1).mean(axis=0)
                    zmap_r = ep_r_full[:, TRS_BEFORE + WIN_START:TRS_BEFORE + WIN_END, :].mean(axis=1).mean(axis=0)
                    group_zmaps[task_key].append({'L': zmap_l, 'R': zmap_r})

    # ── Filmfest ─────────────────────────────────────────────────────────────
    for subject, session in FILMFEST_SUBJECTS.items():
        print(f'\n{subject} (filmfest)')
        for ff_task in ('filmfest1', 'filmfest2'):
            event_times = movie_events[ff_task]
            if not event_times:
                continue
            surf_l, surf_r = _load_surf(subject, session, ff_task)
            if surf_l is None:
                print(f'  {session} {ff_task}: surface data missing, skipping')
                continue
            print(f'  {session} {ff_task}: {len(event_times)} events, '
                  f'{surf_l.shape[0]} TRs')
            for roi in roi_names:
                ep_l = _extract_epochs_2d(surf_l[:, roi_masks[roi][0]], event_times)
                ep_r = _extract_epochs_2d(surf_r[:, roi_masks[roi][1]], event_times)
                parts = []
                if ep_l is not None:
                    parts.append(ep_l.mean(axis=2))
                if ep_r is not None:
                    parts.append(ep_r.mean(axis=2))
                if parts:
                    combined = np.stack(parts, axis=0).mean(axis=0)
                    group_tc['movie'][roi].append(combined.mean(axis=0))
            ep_l_full = _extract_epochs_2d(surf_l, event_times)
            ep_r_full = _extract_epochs_2d(surf_r, event_times)
            if ep_l_full is not None and ep_r_full is not None:
                zmap_l = ep_l_full[:, TRS_BEFORE + WIN_START:TRS_BEFORE + WIN_END, :].mean(axis=1).mean(axis=0)
                zmap_r = ep_r_full[:, TRS_BEFORE + WIN_START:TRS_BEFORE + WIN_END, :].mean(axis=1).mean(axis=0)
                group_zmaps['movie'].append({'L': zmap_l, 'R': zmap_r})

    # ── Conjunction z-map ────────────────────────────────────────────────────
    print(f'\nBuilding conjunction z-map (align={align})...')
    task_mean_l = {}
    task_mean_r = {}
    for task in TASK_KEYS:
        maps = group_zmaps[task]
        if not maps:
            print(f'  {task}: no data, skipping')
            continue
        task_mean_l[task] = np.stack([m['L'] for m in maps], axis=0).mean(axis=0)
        task_mean_r[task] = np.stack([m['R'] for m in maps], axis=0).mean(axis=0)
        print(f'  {task}: {len(maps)} subjects, '
              f'max z L={task_mean_l[task].max():.3f} R={task_mean_r[task].max():.3f}')

    # ── Per-task surface maps ────────────────────────────────────────────────
    print(f'\nPlotting per-task z-maps (align={align})...')
    all_vals = [v for d in [task_mean_l, task_mean_r] for v in d.values()]
    shared_vmax = min(float(np.nanmax(np.abs(np.concatenate(all_vals)))), 0.6) if all_vals else 0.6
    for task in TASK_KEYS:
        if task not in task_mean_l:
            continue
        _plot_task_surface(
            task_mean_l[task], task_mean_r[task], fsavg,
            f'{TASK_DISPLAY[task]} — boundary z-map (0–9 TR, align={align})',
            OUTPUT_DIR / f'zmap_{task}_align-{align}.png',
            vmin=0, vmax=shared_vmax,
        )

    if len(task_mean_l) == 3:
        all_l = np.stack([task_mean_l[t] for t in TASK_KEYS], axis=0)  # (3, V)
        all_r = np.stack([task_mean_r[t] for t in TASK_KEYS], axis=0)
        conj_mask_l = np.all(all_l > CONJ_THRESH, axis=0)
        conj_mask_r = np.all(all_r > CONJ_THRESH, axis=0)
        conj_l = np.where(conj_mask_l, all_l.min(axis=0), np.nan)
        conj_r = np.where(conj_mask_r, all_r.min(axis=0), np.nan)
        print(f'  Conjunction: {conj_mask_l.sum()} L + {conj_mask_r.sum()} R vertices pass')
        _plot_conjunction_surface(
            conj_l, conj_r, fsavg,
            f'Conjunction z-map (0–9 TR, z>{CONJ_THRESH} in all tasks, align={align})',
            OUTPUT_DIR / f'conjunction_zmap_align-{align}.png',
            vmin=0.1, vmax=0.6,
        )
    else:
        print('  Not enough tasks for conjunction, skipping surface map.')

    # ── ROI time courses ──────────────────────────────────────────────────────
    print(f'\nPlotting ROI time courses (align={align})...')
    _plot_roi_timecourses(
        group_tc, time_vec, align,
        OUTPUT_DIR / f'roi_timecourses_align-{align}.png',
    )


if __name__ == '__main__':
    main()
