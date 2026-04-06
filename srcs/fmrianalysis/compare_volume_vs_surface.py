"""
Compare ROI time courses extracted via two methods:

  vol  — MNI volumetric BOLD (NiftiLabelsMasker + Schaefer 400 atlas NIfTI)
  surf — fsaverage6 surface BOLD (.func.gii + Schaefer 400 .annot files)

Style matches task_boundary_vertex: 1 row × 5 ROI subplots, 3 task lines,
group mean ± SEM across subjects. Vol = solid, surf = dashed.

Usage:
    python compare_vol_surf_timecourses.py [--align offset|onset]
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel.freesurfer as fs
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import zscore as sp_zscore

from configs.config import (
    DERIVATIVES_DIR, FIGS_DIR, TR, SUBJECT_IDS,
    FILMFEST_SUBJECTS, MOVIE_INFO,
)
from configs.schaefer_rois import POSTERIOR_MEDIAL, ANGULAR_GYRUS, DLPFC, DACC, MPFC
from fmrianalysis.utils import load_surface_data, highpass_filter, get_parcel_data, mss_to_seconds

# ============================================================================
# CONFIG
# ============================================================================

PSYCHOPY_DIR    = Path('/home/datasets/stateswitch/psychopy')
ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
OUTPUT_DIR      = FIGS_DIR / 'compare_vol_surf'

TRS_BEFORE = 20
TRS_AFTER  = 40
TITLE_SCENE_OFFSET = 4 * TR  # 6 s title card shift for filmfest onset alignment

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT  = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT  = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'

ROI_DEFS = [
    ('PMC',   POSTERIOR_MEDIAL['left'], POSTERIOR_MEDIAL['right'],
              POSTERIOR_MEDIAL.get('left_labels', []) + POSTERIOR_MEDIAL.get('right_labels', [])),
    ('dlPFC', DLPFC['left'],            DLPFC['right'],
              DLPFC.get('left_labels', []) + DLPFC.get('right_labels', [])),
    ('dACC',  DACC['left'],             DACC['right'],
              DACC.get('left_labels', []) + DACC.get('right_labels', [])),
    ('mPFC',  MPFC['left'],             MPFC['right'],
              MPFC.get('left_labels', []) + MPFC.get('right_labels', [])),
    ('AG',    ANGULAR_GYRUS['left'],    ANGULAR_GYRUS['right'],
              ANGULAR_GYRUS.get('left_labels', []) + ANGULAR_GYRUS.get('right_labels', [])),
]

TASK_KEYS    = ('svf', 'ahc', 'movie')
TASK_DISPLAY = {'svf': 'Semantic Fluency', 'ahc': 'Explanation Generation', 'movie': 'FilmFest'}
TASK_COLORS  = {'svf': '#1f77b4', 'ahc': '#ff7f0e', 'movie': '#2ca02c'}


# ============================================================================
# ATLAS SETUP
# ============================================================================

def _build_surf_masks():
    lh_labels, _, _ = fs.read_annot(str(LH_ANNOT))
    rh_labels, _, _ = fs.read_annot(str(RH_ANNOT))
    surf_l = lh_labels.astype(int)
    surf_r = np.where(rh_labels > 0, rh_labels + 200, 0).astype(int)
    masks = {}
    for name, lh_ids, rh_ids, _ in ROI_DEFS:
        masks[name] = (np.isin(surf_l, lh_ids), np.isin(surf_r, rh_ids))
    return masks


# ============================================================================
# EVENT PARSING
# ============================================================================

def _parse_svf_ahc_events(subject, session, task, align):
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
        return (rows[stopped_col].values - t0).tolist()[:-1]
    else:
        return (rows[started_col].values - t0).tolist()[1:]


def _get_movie_events(align):
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

def _extract_epochs(signal, event_times):
    n = len(signal)
    centers = np.round(np.array(event_times) / TR).astype(int)
    offsets = np.arange(-TRS_BEFORE, TRS_AFTER + 1)
    idx = centers[:, None] + offsets[None, :]
    valid = np.all((idx >= 0) & (idx < n), axis=1)
    if not valid.any():
        return None
    return signal[idx[valid]]


# ============================================================================
# TIME SERIES EXTRACTION
# ============================================================================

def _vol_roi_ts(subject, session, task, vol_labels):
    try:
        parcel_dict = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
    except FileNotFoundError:
        return None
    ts_list = [parcel_dict[l] for l in vol_labels if l in parcel_dict]
    if not ts_list:
        return None
    return np.column_stack(ts_list).mean(axis=1)


def _surf_roi_ts(subject, session, task, mask_l, mask_r):
    try:
        raw_l = load_surface_data(subject, session, task, 'L', DERIVATIVES_DIR)
        raw_r = load_surface_data(subject, session, task, 'R', DERIVATIVES_DIR)
    except FileNotFoundError:
        return None
    data_l = np.nan_to_num(sp_zscore(highpass_filter(raw_l.astype(np.float64).T), axis=0, nan_policy='omit'))
    data_r = np.nan_to_num(sp_zscore(highpass_filter(raw_r.astype(np.float64).T), axis=0, nan_policy='omit'))
    parts = []
    if mask_l.any():
        parts.append(data_l[:, mask_l].mean(axis=1))
    if mask_r.any():
        parts.append(data_r[:, mask_r].mean(axis=1))
    return np.stack(parts, axis=1).mean(axis=1) if parts else None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--align', choices=['offset', 'onset'], default='offset')
    args = parser.parse_args()
    align = args.align

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Building surface masks from annot files...')
    surf_masks = _build_surf_masks()
    for name, (ml, mr) in surf_masks.items():
        print(f'  {name}: {ml.sum()} L + {mr.sum()} R vertices')

    movie_events = _get_movie_events(align)
    roi_names = [r[0] for r in ROI_DEFS]
    time_vec  = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR

    # group_tc[method][task][roi] = list of per-subject mean epoch arrays (n_trs,)
    # One entry per subject (averaged across sessions/runs within subject first)
    group_tc = {m: {t: {roi: [] for roi in roi_names} for t in TASK_KEYS}
                for m in ('vol', 'surf')}

    # ── SVF and AHC ──────────────────────────────────────────────────────────
    for subject in SUBJECT_IDS:
        print(f'\n{subject}')
        sub_dir = DERIVATIVES_DIR / subject
        if not sub_dir.exists():
            continue

        # Accumulate across sessions within subject, then average
        subj_epochs = {m: {t: {roi: [] for roi in roi_names}
                           for t in ('svf', 'ahc')} for m in ('vol', 'surf')}

        for ses_dir in sorted(sub_dir.glob('ses-*')):
            session = ses_dir.name
            func_dir = ses_dir / 'func'
            if not func_dir.exists():
                continue
            for task in ('svf', 'ahc'):
                if not list(func_dir.glob(f'*_task-{task}_*_bold.nii.gz')):
                    continue
                events = _parse_svf_ahc_events(subject, session, task, align)
                if not events:
                    continue
                print(f'  {session} {task}: {len(events)} events')

                # Load data once per session/task, then extract all ROIs
                try:
                    parcel_dict = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
                except FileNotFoundError:
                    parcel_dict = None

                surf_l_data = surf_r_data = None
                try:
                    raw_l = load_surface_data(subject, session, task, 'L', DERIVATIVES_DIR)
                    raw_r = load_surface_data(subject, session, task, 'R', DERIVATIVES_DIR)
                    surf_l_data = np.nan_to_num(sp_zscore(highpass_filter(raw_l.astype(np.float64).T), axis=0, nan_policy='omit'))
                    surf_r_data = np.nan_to_num(sp_zscore(highpass_filter(raw_r.astype(np.float64).T), axis=0, nan_policy='omit'))
                except FileNotFoundError:
                    pass

                for name, lh_ids, rh_ids, vol_labels in ROI_DEFS:
                    mask_l, mask_r = surf_masks[name]
                    if parcel_dict is not None:
                        ts_list = [parcel_dict[l] for l in vol_labels if l in parcel_dict]
                        if ts_list:
                            ts_vol = np.column_stack(ts_list).mean(axis=1)
                            ep = _extract_epochs(ts_vol, events)
                            if ep is not None:
                                subj_epochs['vol'][task][name].append(ep.mean(axis=0))
                    if surf_l_data is not None:
                        parts = []
                        if mask_l.any():
                            parts.append(surf_l_data[:, mask_l].mean(axis=1))
                        if mask_r.any():
                            parts.append(surf_r_data[:, mask_r].mean(axis=1))
                        if parts:
                            ts_surf = np.stack(parts, axis=1).mean(axis=1)
                            ep = _extract_epochs(ts_surf, events)
                            if ep is not None:
                                subj_epochs['surf'][task][name].append(ep.mean(axis=0))

        # Average within subject across sessions → one value per subject
        for m in ('vol', 'surf'):
            for task in ('svf', 'ahc'):
                for roi in roi_names:
                    runs = subj_epochs[m][task][roi]
                    if runs:
                        group_tc[m][task][roi].append(np.stack(runs).mean(axis=0))

    # ── Filmfest ─────────────────────────────────────────────────────────────
    for subject, session in FILMFEST_SUBJECTS.items():
        print(f'\n{subject} (filmfest)')
        subj_epochs = {m: {roi: [] for roi in roi_names} for m in ('vol', 'surf')}

        for ff_task in ('filmfest1', 'filmfest2'):
            events = movie_events[ff_task]
            if not events:
                continue
            print(f'  {session} {ff_task}: {len(events)} events')

            try:
                parcel_dict = get_parcel_data(subject, session, ff_task, atlas='Schaefer400_17Nets')
            except FileNotFoundError:
                parcel_dict = None

            surf_l_data = surf_r_data = None
            try:
                raw_l = load_surface_data(subject, session, ff_task, 'L', DERIVATIVES_DIR)
                raw_r = load_surface_data(subject, session, ff_task, 'R', DERIVATIVES_DIR)
                surf_l_data = np.nan_to_num(sp_zscore(highpass_filter(raw_l.astype(np.float64).T), axis=0, nan_policy='omit'))
                surf_r_data = np.nan_to_num(sp_zscore(highpass_filter(raw_r.astype(np.float64).T), axis=0, nan_policy='omit'))
            except FileNotFoundError:
                pass

            for name, lh_ids, rh_ids, vol_labels in ROI_DEFS:
                mask_l, mask_r = surf_masks[name]
                if parcel_dict is not None:
                    ts_list = [parcel_dict[l] for l in vol_labels if l in parcel_dict]
                    if ts_list:
                        ts_vol = np.column_stack(ts_list).mean(axis=1)
                        ep = _extract_epochs(ts_vol, events)
                        if ep is not None:
                            subj_epochs['vol'][name].append(ep.mean(axis=0))
                if surf_l_data is not None:
                    parts = []
                    if mask_l.any():
                        parts.append(surf_l_data[:, mask_l].mean(axis=1))
                    if mask_r.any():
                        parts.append(surf_r_data[:, mask_r].mean(axis=1))
                    if parts:
                        ts_surf = np.stack(parts, axis=1).mean(axis=1)
                        ep = _extract_epochs(ts_surf, events)
                        if ep is not None:
                            subj_epochs['surf'][name].append(ep.mean(axis=0))

        for m in ('vol', 'surf'):
            for roi in roi_names:
                runs = subj_epochs[m][roi]
                if runs:
                    group_tc[m]['movie'][roi].append(np.stack(runs).mean(axis=0))

    # ── Plot ─────────────────────────────────────────────────────────────────
    print('\nPlotting...')
    n_rois = len(roi_names)
    fig, axes = plt.subplots(1, n_rois, figsize=(4 * n_rois, 4), sharey=True)
    fig.suptitle(f'Vol vs Surf ROI time courses (align={align})',
                 fontsize=13, fontweight='bold')

    for ax, roi in zip(axes, roi_names):
        for task in TASK_KEYS:
            color = TASK_COLORS[task]
            for method, ls in (('vol', '-'), ('surf', '--')):
                tcs = group_tc[method][task][roi]
                if not tcs:
                    continue
                arr  = np.array(tcs)
                mean = arr.mean(axis=0)
                sem  = arr.std(axis=0) / np.sqrt(len(arr))
                label = f'{TASK_DISPLAY[task]} ({method})'
                ax.plot(time_vec, mean, color=color, lw=1.5, ls=ls, label=label)
                ax.fill_between(time_vec, mean - sem, mean + sem, color=color, alpha=0.15)

        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.axhline(0, color='k', lw=0.4, ls=':')
        if align == 'offset':
            ax.axvline(15, color='gray', lw=0.8, ls='--')
        ax.set_ylim(-1, 1)
        ax.set_xlim(time_vec[0], time_vec[-1])
        ax.set_title(roi, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time from boundary (s)', fontsize=9)
        ax.set_ylabel('BOLD (z-score)', fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].legend(fontsize=7, loc='upper left', frameon=False, ncol=1)
    plt.tight_layout()

    out = OUTPUT_DIR / f'vol_vs_surf_timecourses_align-{align}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
