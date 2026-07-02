"""
Hippocampus Boundary-Locked Time Courses (whole / anterior / posterior)

Group-average BOLD time courses in the hippocampus, locked to every type of
cognitive/narrative boundary in the study. Onset-locked and offset-locked time
courses are overlaid in the *same* subplot.

Boundary types (columns):
  svf     — Semantic verbal fluency trial boundary (category offset / next onset)
  ahc     — Ad-hoc categories trial boundary (prompt offset / next onset)
  movie   — FilmFest between-movie boundary (movie watching)
  recall  — FilmFest between-movie boundary during free/cued recall

ROIs (rows), all from the Harvard-Oxford subcortical hippocampus label, split
along the MNI anterior-posterior (y) axis at the per-hemisphere median:
  whole   — entire hippocampus (L+R)
  ant     — anterior hippocampus (y >= median)
  post    — posterior hippocampus (y <  median)

EFFICIENT EXTRACTION
--------------------
The three ROI time courses share a single hippocampus mask, so each BOLD run is
loaded from disk exactly once and yields all three ROIs together (whole / ant /
post), cached to one small .npz. Only masked voxels are materialised. Extraction
across runs can be parallelised with --n_jobs; once cached, plotting is instant.

Usage:
    uv run python srcs/fmrianalysis/hippocampus_boundary_timecourse.py
    uv run python srcs/fmrianalysis/hippocampus_boundary_timecourse.py --n_jobs 6
    uv run python srcs/fmrianalysis/hippocampus_boundary_timecourse.py \
        --boundary_types svf ahc movie
"""
import sys
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # fmrianalysis.*
sys.path.insert(0, str(_HERE.parent))   # configs.config

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine
from nilearn import datasets
from scipy.stats import zscore as sp_zscore
import matplotlib.pyplot as plt

from configs.config import (
    DATA_DIR, ANALYSIS_CACHE_DIR, FIGS_DIR, TR, SUBJECT_IDS, FILMFEST_SUBJECTS,
    MOVIE_INFO,
)
from fmrianalysis.utils import (
    get_bold_path, get_atlas_data, highpass_filter, get_trial_times,
    discover_svf_ahc_sessions, get_movie_boundary_offsets, extract_event_locked,
    mss_to_seconds, ANNOTATIONS_DIR,
)

# ============================================================================
# CONSTANTS
# ============================================================================

OUTPUT_DIR = FIGS_DIR / 'hippocampus_boundary_timecourse'
CACHE_DIR = ANALYSIS_CACHE_DIR / 'hipp_ap'
RECALL_DIR = DATA_DIR / 'filmfest_recall_timestamps'

# Harvard-Oxford subcortical (maxprob thr25 2mm) hippocampus label indices
# (verified: idx 9 = Left Hippocampus, idx 19 = Right Hippocampus)
HO_ATLAS = 'sub-maxprob-thr25-2mm'
HIPP_LABEL_L = 9
HIPP_LABEL_R = 19

# Peri-boundary window (seconds). t=0 = the boundary anchor.
PRE_S = 15
POST_S = 30
TRS_BEFORE = int(round(PRE_S / TR))    # 10 TRs
TRS_AFTER = int(round(POST_S / TR))    # 20 TRs

BOUNDARY_TYPES = ('svf', 'ahc', 'movie', 'recall')
BOUNDARY_TITLE = {
    'svf': 'Word Generation\n(SVF trial)',
    'ahc': 'Explanation Generation\n(AHC trial)',
    'movie': 'Movie Watching\n(between-movie)',
    'recall': 'Movie Recall\n(between-movie)',
}

# ROIs: (key, display_name)
ROI_SPEC = [
    ('whole', 'Hippocampus (whole)'),
    ('ant', 'Anterior hippocampus'),
    ('post', 'Posterior hippocampus'),
]

ALIGN_STYLE = {
    'offset': dict(color='#d62728', label='Offset-locked'),
    'onset': dict(color='#1f77b4', label='Onset-locked'),
}

# HRF onset shift for the filmfest onset alignment: skip the title card that
# opens each movie so the "onset" reflects the new movie's content, matching
# the convention used elsewhere in the project.
TITLE_SCENE_OFFSET = 6.0


# ============================================================================
# BOUNDARY EVENT COLLECTION
# ============================================================================

def get_movie_boundary_onsets(task):
    """Movie-start times (s) of movies 2..N for a filmfest task (onset complement
    of get_movie_boundary_offsets)."""
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    onsets = []
    for movie in movies[1:]:
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        first_start = segb['Start Time (m.ss)'].values[0]
        onsets.append(mss_to_seconds(first_start))
    return onsets


def _parse_recall_tsv_filename(stem):
    parts = stem.split('_')
    sub, ses = parts[0], parts[1]
    task = '_'.join(parts[2:]).replace('task-', '')
    return sub, ses, task


def get_recall_boundary_events(subject, align='offset'):
    """[(session, task, [event_times_sec]), ...] for each recall TSV.

    offset -> scanner_end of each contiguous movie block's last segment.
    onset  -> scanner_start of each contiguous movie block's first segment.
    """
    out = []
    for tsv in sorted(RECALL_DIR.glob(f'{subject}_*_desc-recallsegments.tsv')):
        stem = tsv.stem.replace('_desc-recallsegments', '')
        _, ses, task = _parse_recall_tsv_filename(stem)
        df = pd.read_csv(tsv, sep='\t')
        movies = df['movie'].values
        events = []
        if align == 'offset':
            sc_end = df['scanner_end'].values
            for i in range(len(df)):
                if i == len(df) - 1 or movies[i] != movies[i + 1]:
                    events.append(float(sc_end[i]))
        else:
            sc_start = df['scanner_start'].values
            for i in range(len(df)):
                if i == 0 or movies[i] != movies[i - 1]:
                    events.append(float(sc_start[i]))
        out.append((ses, task, events))
    return out


def collect_subject_events(subject, btype):
    """Return {'offset': [(ses,task,times)], 'onset': [(ses,task,times)]} for one
    subject and boundary type. `times` are scan-relative seconds."""
    events = {'offset': [], 'onset': []}

    if btype in ('svf', 'ahc'):
        for ses, task in discover_svf_ahc_sessions(subject):
            if task != btype:
                continue
            onsets, offsets = get_trial_times(subject, ses, task)
            if len(offsets) >= 2:
                # offset of trial i (exclude last: no trial follows)
                events['offset'].append((ses, task, list(offsets[:-1])))
                # onset of trial i+1 (exclude first: no boundary precedes it)
                events['onset'].append((ses, task, list(onsets[1:])))

    elif btype == 'movie':
        if subject in FILMFEST_SUBJECTS:
            ses = FILMFEST_SUBJECTS[subject]
            for task in ('filmfest1', 'filmfest2'):
                off = get_movie_boundary_offsets(task)
                on = [t + TITLE_SCENE_OFFSET for t in get_movie_boundary_onsets(task)]
                if off:
                    events['offset'].append((ses, task, off))
                if on:
                    events['onset'].append((ses, task, on))

    elif btype == 'recall':
        if subject in FILMFEST_SUBJECTS:
            for align in ('offset', 'onset'):
                for ses, task, times in get_recall_boundary_events(subject, align):
                    if times:
                        events[align].append((ses, task, times))

    else:
        raise ValueError(f'Unknown boundary type: {btype}')

    return events


def all_needed_runs(subjects, boundary_types):
    """Union of (subject, session, task) tuples referenced by any boundary type."""
    runs = set()
    for subject in subjects:
        for btype in boundary_types:
            ev = collect_subject_events(subject, btype)
            for align in ('offset', 'onset'):
                for ses, task, _ in ev[align]:
                    runs.add((subject, ses, task))
    return sorted(runs)


# ============================================================================
# EFFICIENT HIPPOCAMPUS EXTRACTION (one BOLD load -> whole / ant / post)
# ============================================================================

def _cache_path(subject, session, task):
    return CACHE_DIR / f'{subject}_{session}_task-{task}_hipp_ap_hp-0.01_z.npz'


def _region_timecourse(vox):
    """Voxels (n_vox, T) -> region mean, high-passed and temporally z-scored (T,)."""
    ts = vox.mean(axis=0).astype(np.float64)       # region mean signal
    ts = highpass_filter(ts, order=2)              # 0.01 Hz high-pass
    ts = sp_zscore(ts, nan_policy='omit')
    return np.nan_to_num(ts).astype(np.float32)


def extract_hipp_run(subject, session, task, force=False):
    """Extract whole/ant/post hippocampus time courses for one run (cached).

    Loads the BOLD volume exactly once and materialises only hippocampus voxels;
    all three ROIs are derived from the same mask and saved together.
    """
    cache_file = _cache_path(subject, session, task)
    if cache_file.exists() and not force:
        return True

    bold_path = get_bold_path(subject, session, task)
    if not bold_path.exists():
        print(f'  [skip] BOLD missing: {bold_path.name}')
        return False

    # Atlas resampled to this BOLD's voxel grid (fetch cached on disk).
    ho_maps = datasets.fetch_atlas_harvard_oxford(HO_ATLAS)['maps']
    atlas_data = get_atlas_data(bold_path, ho_maps)     # (X, Y, Z) int

    mask_l = atlas_data == HIPP_LABEL_L
    mask_r = atlas_data == HIPP_LABEL_R
    if mask_l.sum() < 2 or mask_r.sum() < 2:
        print(f'  [skip] hippocampus mask too small for {subject} {session} {task}')
        return False

    img = nib.load(str(bold_path))
    affine = img.affine
    data4d = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z, T)

    def _split(mask):
        """Return (anterior_voxels, posterior_voxels) for a hemisphere mask.

        (T, ) time series per voxel; split at the median MNI-y of the hemisphere's
        hippocampus voxels (anterior = larger y)."""
        ijk = np.argwhere(mask)                         # (n, 3), C-order
        world_y = apply_affine(affine, ijk)[:, 1]       # MNI y (mm)
        vox = data4d[mask]                              # (n, T), same C-order
        ant = world_y >= np.median(world_y)
        return vox[ant], vox[~ant]

    ant_l, post_l = _split(mask_l)
    ant_r, post_r = _split(mask_r)

    whole = np.concatenate([data4d[mask_l], data4d[mask_r]], axis=0)
    ant = np.concatenate([ant_l, ant_r], axis=0)
    post = np.concatenate([post_l, post_r], axis=0)

    del data4d, img

    out = {
        'whole': _region_timecourse(whole),
        'ant': _region_timecourse(ant),
        'post': _region_timecourse(post),
        'n_vox': np.array([whole.shape[0], ant.shape[0], post.shape[0]]),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **out)
    print(f'  cached {cache_file.name}  '
          f'(whole={out["n_vox"][0]}, ant={out["n_vox"][1]}, post={out["n_vox"][2]})')
    return True


def load_hipp_run(subject, session, task):
    """Return {'whole','ant','post'} time-course dict, or None if unavailable."""
    cache_file = _cache_path(subject, session, task)
    if not cache_file.exists():
        if not extract_hipp_run(subject, session, task):
            return None
    d = np.load(cache_file)
    return {'whole': d['whole'], 'ant': d['ant'], 'post': d['post']}


# ============================================================================
# EPOCH AVERAGING
# ============================================================================

def subject_mean_timecourse(subject, btype, roi_key, align):
    """Mean peri-boundary time course for one subject/boundary/ROI/alignment.

    Pools epochs across all of the subject's runs, then averages. Returns
    (mean_tc, n_epochs) or (None, 0)."""
    events = collect_subject_events(subject, btype)
    all_epochs = []
    for ses, task, times in events[align]:
        run = load_hipp_run(subject, ses, task)
        if run is None:
            continue
        ep = extract_event_locked(run[roi_key], times, TRS_BEFORE, TRS_AFTER,
                                  return_epochs=True)
        if ep is not None:
            all_epochs.append(ep)
    if not all_epochs:
        return None, 0
    stacked = np.vstack(all_epochs)
    return stacked.mean(axis=0), stacked.shape[0]


# ============================================================================
# FIGURE
# ============================================================================

def _time_axis():
    return (np.arange(-TRS_BEFORE, TRS_AFTER + 1)) * TR


def make_figure(subjects, boundary_types):
    n_rows = len(ROI_SPEC)
    n_cols = len(boundary_types)
    time = _time_axis()

    # group[roi][btype][align] = {'mean','sem','n_subj','n_epochs'}
    group = {rk: {bt: {} for bt in boundary_types} for rk, _ in ROI_SPEC}

    for rk, _ in ROI_SPEC:
        for bt in boundary_types:
            for align in ('offset', 'onset'):
                subj_means, n_ep_total = [], 0
                for subject in subjects:
                    tc, n_ep = subject_mean_timecourse(subject, bt, rk, align)
                    if tc is not None:
                        subj_means.append(tc)
                        n_ep_total += n_ep
                if subj_means:
                    arr = np.vstack(subj_means)
                    group[rk][bt][align] = {
                        'mean': arr.mean(axis=0),
                        'sem': arr.std(axis=0) / np.sqrt(arr.shape[0]),
                        'n_subj': arr.shape[0],
                        'n_epochs': n_ep_total,
                    }

    # Shared y-limits across all subplots (project convention).
    ylo, yhi = np.inf, -np.inf
    for rk, _ in ROI_SPEC:
        for bt in boundary_types:
            for align in ('offset', 'onset'):
                g = group[rk][bt].get(align)
                if g is None:
                    continue
                ylo = min(ylo, np.min(g['mean'] - g['sem']))
                yhi = max(yhi, np.max(g['mean'] + g['sem']))
    if not np.isfinite(ylo):
        ylo, yhi = -1, 1
    pad = 0.08 * (yhi - ylo)
    ylo, yhi = ylo - pad, yhi + pad

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.0 * n_rows),
                             squeeze=False, sharex=True, sharey=True)
    fig.suptitle('Hippocampus boundary-locked time courses (group mean ± SEM)',
                 fontsize=14, fontweight='bold', y=0.99)

    for r, (rk, rname) in enumerate(ROI_SPEC):
        for c, bt in enumerate(boundary_types):
            ax = axes[r][c]
            ax.axvline(0, color='k', lw=1.0, ls='--', alpha=0.7)
            ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
            for align in ('offset', 'onset'):
                g = group[rk][bt].get(align)
                if g is None:
                    continue
                st = ALIGN_STYLE[align]
                ax.plot(time, g['mean'], color=st['color'], lw=1.8,
                        label=f"{st['label']} (N={g['n_subj']}, {g['n_epochs']} ev)")
                ax.fill_between(time, g['mean'] - g['sem'], g['mean'] + g['sem'],
                                color=st['color'], alpha=0.2, lw=0)
            ax.set_ylim(ylo, yhi)
            ax.set_xlim(time[0], time[-1])
            if r == 0:
                ax.set_title(BOUNDARY_TITLE[bt], fontsize=10, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{rname}\nBOLD (z)', fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel('Time rel. boundary (s)', fontsize=9)
            ax.legend(fontsize=6, loc='upper right', framealpha=0.7)
            ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / 'hippocampus_boundary_timecourse.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved figure -> {out}')
    return group


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--boundary_types', nargs='+', default=list(BOUNDARY_TYPES),
                    choices=list(BOUNDARY_TYPES))
    ap.add_argument('--n_jobs', type=int, default=1,
                    help='Parallelism for the one-time extraction pre-pass')
    ap.add_argument('--force', action='store_true',
                    help='Re-extract hippocampus time courses even if cached')
    args = ap.parse_args()

    boundary_types = args.boundary_types
    subjects = list(SUBJECT_IDS)

    print('=' * 64)
    print('HIPPOCAMPUS BOUNDARY-LOCKED TIME COURSES')
    print(f'Boundary types : {boundary_types}')
    print(f'ROIs           : {[k for k, _ in ROI_SPEC]}')
    print(f'Window         : -{PRE_S}s .. +{POST_S}s  '
          f'({TRS_BEFORE} + 1 + {TRS_AFTER} TRs)')
    print('=' * 64)

    # ---- Extraction pre-pass (cached; one BOLD load per run -> 3 ROIs) ----
    runs = all_needed_runs(subjects, boundary_types)
    todo = [r for r in runs if args.force or not _cache_path(*r).exists()]
    print(f'\nRuns referenced: {len(runs)}   needing extraction: {len(todo)}')
    if todo:
        if args.n_jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=args.n_jobs)(
                delayed(extract_hipp_run)(s, ses, t, args.force) for s, ses, t in todo)
        else:
            for s, ses, t in todo:
                extract_hipp_run(s, ses, t, args.force)

    # ---- Figure ----
    make_figure(subjects, boundary_types)
    print('\nDONE.')


if __name__ == '__main__':
    main()
