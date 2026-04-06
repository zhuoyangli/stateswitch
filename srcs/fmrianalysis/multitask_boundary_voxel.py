"""
Voxel-wise boundary response maps with surface projection.

For each task (SVF, AHC, FilmFest):
  1. Load MNI152NLin6Asym res-2 BOLD per subject/session
  2. Highpass filter + z-score (matching utils.py preprocessing)
  3. Extract post-boundary window epochs → per-subject mean 3D response
  4. One-sample t-test across subjects → voxel-wise t-map
  5. Project to fsaverage6 surface with vol_to_surf

Conjunction map: voxels passing t > T_THRESH in ALL 3 tasks; displays min-t.

Usage:
    python task_boundary_voxel.py [--align offset|onset]
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_surf
from nilearn import surface as surf_mod
from scipy.signal import butter, filtfilt
from scipy.stats import zscore as sp_zscore

from configs.config import (
    DERIVATIVES_DIR, FIGS_DIR, TR, SUBJECT_IDS,
    FILMFEST_SUBJECTS, MOVIE_INFO,
)

# ============================================================================
# CONFIG
# ============================================================================

PSYCHOPY_DIR    = Path('/home/datasets/stateswitch/psychopy')
ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
OUTPUT_DIR      = FIGS_DIR / 'task_boundary_voxel'

WIN_START_TR    = 0    # first TR of post-boundary response window (inclusive)
WIN_END_TR      = 10   # last TR of post-boundary response window (exclusive)
CONJ_THRESH     = 0.1  # min z across tasks to enter conjunction (matches vertex script)
TITLE_SCENE_OFFSET = 6.0  # seconds to shift filmfest boundaries for onset alignment

LABEL_FS = 11
TITLE_FS = 13

TASK_KEYS    = ('svf', 'ahc', 'movie')
TASK_DISPLAY = {'svf': 'Semantic Fluency', 'ahc': 'Explanation Generation', 'movie': 'FilmFest'}
TASK_COLORS  = {'svf': '#1f77b4', 'ahc': '#ff7f0e', 'movie': '#2ca02c'}

BOLD_PATTERN = (
    '{sub}_{ses}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
)
MASK_PATTERN = (
    '{sub}_{ses}_task-{task}_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz'
)


# ============================================================================
# PREPROCESSING
# ============================================================================

def _highpass_filter(data_2d, cutoff=0.01, order=5):
    """Zero-phase Butterworth highpass filter along axis 0 (T, N_voxels)."""
    nyq = 0.5 / TR
    b, a = butter(order, cutoff / nyq, btype='high', analog=False)
    return filtfilt(b, a, data_2d, axis=0)


def _load_bold(bold_path, common_masker):
    """Load BOLD using a pre-fitted common masker, highpass filter, z-score.

    Returns data_2d : (T, N_voxels) float64, z-scored
    """
    data_2d = common_masker.transform(str(bold_path)).astype(np.float64)  # (T, N_voxels)
    data_2d = _highpass_filter(data_2d)
    data_2d = sp_zscore(data_2d, axis=0, nan_policy='omit')
    data_2d = np.nan_to_num(data_2d)
    return data_2d


# ============================================================================
# EVENT PARSING
# ============================================================================

def _parse_svf_ahc_events(subject, session, task, align):
    ses_dir = PSYCHOPY_DIR / subject / session
    if not ses_dir.exists():
        return None
    stopped_col = f'{task}_trial.stopped'
    started_col  = f'{task}_trial.started'
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
        return (rows[stopped_col].values - t0).tolist()[:-1]
    else:
        return (rows[started_col].values - t0).tolist()[1:]


def _mss_to_seconds(mss):
    minutes = int(mss)
    seconds = round((mss - minutes) * 100)
    return minutes * 60 + seconds


def _get_movie_events(align):
    result = {}
    for ff_task in ('filmfest1', 'filmfest2'):
        movies = [m for m in MOVIE_INFO if m['task'] == ff_task]
        boundaries = []
        for movie in movies[:-1]:
            df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
            segb = df.dropna(subset=['SEG-B_Number'])
            last_end = segb['End Time (m.ss)'].values[-1]
            boundaries.append(_mss_to_seconds(last_end))
        if align == 'onset':
            boundaries = [t + TITLE_SCENE_OFFSET for t in boundaries]
        result[ff_task] = boundaries
    return result


# ============================================================================
# EPOCH EXTRACTION
# ============================================================================

def _mean_response(data_2d, event_times_sec, win_start, win_end):
    """Average BOLD within [win_start, win_end) TRs post-event across all events.

    Returns (N_voxels,) or None if no valid events.
    """
    T = data_2d.shape[0]
    centers = np.round(np.array(event_times_sec) / TR).astype(int)
    responses = []
    for c in centers:
        i0 = c + win_start
        i1 = c + win_end
        if i0 >= 0 and i1 <= T:
            responses.append(data_2d[i0:i1].mean(axis=0))
    if not responses:
        return None
    return np.stack(responses).mean(axis=0)  # (N_voxels,)


# ============================================================================
# BOLD FILE DISCOVERY
# ============================================================================

def _bold_mask_paths(subject, session, task):
    func_dir = DERIVATIVES_DIR / subject / session / 'func'
    bold = func_dir / BOLD_PATTERN.format(sub=subject, ses=session, task=task)
    mask = func_dir / MASK_PATTERN.format(sub=subject, ses=session, task=task)
    if bold.exists() and mask.exists():
        return bold, mask
    return None, None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--align', choices=['offset', 'onset'], default='offset')
    args = parser.parse_args()
    align = args.align

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Align: {align}')
    movie_events = _get_movie_events(align)

    # ── Collect all mask paths → compute common intersection mask ─────────────
    print('Collecting fmriprep brain masks...')
    all_mask_paths = []
    for subject in SUBJECT_IDS:
        sub_dir = DERIVATIVES_DIR / subject
        if not sub_dir.exists():
            continue
        for ses_dir in sorted(sub_dir.glob('ses-*')):
            session = ses_dir.name
            for task in ('svf', 'ahc'):
                bold_path, mask_path = _bold_mask_paths(subject, session, task)
                if bold_path is not None:
                    all_mask_paths.append(str(mask_path))
    for subject, session in FILMFEST_SUBJECTS.items():
        for ff_task in ('filmfest1', 'filmfest2'):
            bold_path, mask_path = _bold_mask_paths(subject, session, ff_task)
            if bold_path is not None:
                all_mask_paths.append(str(mask_path))

    if not all_mask_paths:
        print('No mask files found. Exiting.')
        return

    print(f'  Intersecting {len(all_mask_paths)} masks...')
    common_mask = intersect_masks(all_mask_paths, threshold=1.0)  # strict intersection
    common_masker = NiftiMasker(mask_img=common_mask, standardize=False, detrend=False)
    common_masker.fit()
    ref_affine = common_mask.affine
    ref_shape  = common_mask.shape[:3]
    n_voxels   = int(common_mask.get_fdata().sum())
    print(f'  Common mask: {n_voxels} voxels')

    group_vecs = {t: [] for t in TASK_KEYS}

    # ── SVF and AHC ──────────────────────────────────────────────────────────
    for subject in SUBJECT_IDS:
        print(f'\n{subject}')
        sub_dir = DERIVATIVES_DIR / subject
        if not sub_dir.exists():
            continue

        subj_vecs = {t: [] for t in ('svf', 'ahc')}

        for ses_dir in sorted(sub_dir.glob('ses-*')):
            session = ses_dir.name
            for task in ('svf', 'ahc'):
                bold_path, mask_path = _bold_mask_paths(subject, session, task)
                if bold_path is None:
                    continue
                events = _parse_svf_ahc_events(subject, session, task, align)
                if not events:
                    continue

                print(f'  {session} {task}: loading BOLD...')
                try:
                    data_2d = _load_bold(bold_path, common_masker)
                except Exception as e:
                    print(f'    FAILED: {e}')
                    continue

                resp = _mean_response(data_2d, events, WIN_START_TR, WIN_END_TR)
                if resp is not None:
                    print(f'    {len(events)} events → response shape {resp.shape}')
                    subj_vecs[task].append(resp)

        # Average across sessions within subject
        for task in ('svf', 'ahc'):
            if subj_vecs[task]:
                group_vecs[task].append(np.stack(subj_vecs[task]).mean(axis=0))

    # ── Filmfest ─────────────────────────────────────────────────────────────
    for subject, session in FILMFEST_SUBJECTS.items():
        print(f'\n{subject} (filmfest)')
        subj_ff_vecs = []

        for ff_task in ('filmfest1', 'filmfest2'):
            events = movie_events[ff_task]
            if not events:
                continue
            bold_path, mask_path = _bold_mask_paths(subject, session, ff_task)
            if bold_path is None:
                continue

            print(f'  {session} {ff_task}: loading BOLD...')
            try:
                data_2d = _load_bold(bold_path, common_masker)
            except Exception as e:
                print(f'    FAILED: {e}')
                continue

            resp = _mean_response(data_2d, events, WIN_START_TR, WIN_END_TR)
            if resp is not None:
                print(f'    {len(events)} events → response shape {resp.shape}')
                subj_ff_vecs.append(resp)

        if subj_ff_vecs:
            group_vecs['movie'].append(np.stack(subj_ff_vecs).mean(axis=0))

    if not any(group_vecs[t] for t in TASK_KEYS):
        print('No data loaded. Exiting.')
        return

    # ── Group mean z-maps (voxel space) ──────────────────────────────────────
    print('\nComputing group mean z-maps...')
    group_mean_img = {}
    for task in TASK_KEYS:
        vecs = group_vecs[task]
        n = len(vecs)
        print(f'  {task}: N={n} subjects')
        if n < 1:
            continue
        mean_z = np.stack(vecs).mean(axis=0)   # (N_voxels,)
        print(f'    max z = {mean_z.max():.3f}')
        group_mean_img[task] = common_masker.inverse_transform(mean_z)

    if not group_mean_img:
        print('No z-maps computed. Exiting.')
        return

    # ── Project to fsaverage6 surface ────────────────────────────────────────
    print('\nProjecting to fsaverage6 surface (inflated)...')
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    z_surf = {}
    for task, z_img in group_mean_img.items():
        z_surf[task] = {}
        for hemi, mesh_key in (('L', 'infl_left'), ('R', 'infl_right')):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                texture = surf_mod.vol_to_surf(
                    z_img, fsavg[mesh_key], radius=3.0, interpolation='nearest')
            z_surf[task][hemi] = texture

    # ── Per-task surface maps ─────────────────────────────────────────────────
    print('Plotting per-task z-maps...')
    all_vals = np.concatenate([
        np.concatenate([z_surf[t]['L'], z_surf[t]['R']])
        for t in group_mean_img
    ])
    shared_vmax = min(float(np.nanmax(np.abs(all_vals))), 0.6)

    for task in group_mean_img:
        zmap_l = z_surf[task]['L']
        zmap_r = z_surf[task]['R']
        fig = plt.figure(figsize=(12, 7))
        fig.suptitle(
            f'{TASK_DISPLAY[task]} — voxel z-map (0–{WIN_END_TR} TR, align={align})',
            fontsize=TITLE_FS, fontweight='bold', y=1.01)
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
                plot_surf(mesh, surf_map=sdata, hemi=hemi, view=view,
                          bg_map=bg, axes=ax, colorbar=False,
                          cmap='RdBu_r', bg_on_data=True, darkness=0.5,
                          vmin=-shared_vmax, vmax=shared_vmax)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
        sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                   norm=plt.Normalize(vmin=-shared_vmax, vmax=shared_vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label('mean z-score (0–9 TR)', fontsize=LABEL_FS)
        out = OUTPUT_DIR / f'zmap_{task}_align-{align}.png'
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  Saved → {out}')

    # ── Conjunction map ───────────────────────────────────────────────────────
    print('Building conjunction map...')
    tasks_available = [t for t in TASK_KEYS if t in z_surf]
    if len(tasks_available) < 2:
        print('  Need >=2 tasks for conjunction. Skipping.')
        return

    conj_l_all = np.stack([z_surf[t]['L'] for t in tasks_available])  # (n_tasks, n_vert)
    conj_r_all = np.stack([z_surf[t]['R'] for t in tasks_available])
    mask_l = np.all(conj_l_all > CONJ_THRESH, axis=0)
    mask_r = np.all(conj_r_all > CONJ_THRESH, axis=0)
    conj_l = np.where(mask_l, conj_l_all.min(axis=0), np.nan)
    conj_r = np.where(mask_r, conj_r_all.min(axis=0), np.nan)
    print(f'  Conjunction: {mask_l.sum()} L + {mask_r.sum()} R vertices pass (z>{CONJ_THRESH} in all tasks)')

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(
        f'Conjunction z-map — voxel-wise (0–{WIN_END_TR} TR, z>{CONJ_THRESH} in all tasks, align={align})',
        fontsize=TITLE_FS, fontweight='bold', y=1.01)
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
            plot_surf(mesh, surf_map=sdata, hemi=hemi, view=view,
                      bg_map=bg, axes=ax, colorbar=False,
                      cmap='Reds', bg_on_data=True, darkness=0.5,
                      vmin=CONJ_THRESH, vmax=0.6)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap='Reds',
                               norm=plt.Normalize(vmin=CONJ_THRESH, vmax=0.6))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('min z-score across tasks (0–9 TR)', fontsize=LABEL_FS)
    out = OUTPUT_DIR / f'conjunction_zmap_align-{align}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')


if __name__ == '__main__':
    main()
