"""
Hippocampus Boundary-Locked Time Courses (whole / anterior / posterior)

Group-average BOLD time courses in the hippocampus, locked to every type of
cognitive/narrative boundary in the study. Figure: 3 ROI rows x 6 columns.

Columns 1-4 — coarse task/narrative boundaries, ONSET-locked (t=0 = onset of the
next unit). Because onset and offset are separated by an essentially fixed delay,
the offset is drawn as a single dashed vertical marker rather than a second curve:
  svf     — SVF trial boundary  (next category onset; prev. trial offset marked)
  ahc     — AHC trial boundary  (next prompt onset;   prev. trial offset marked)
  movie   — FilmFest between-movie boundary, watching (next movie onset)
  recall  — FilmFest between-movie boundary, recall   (next recall onset)

Columns 5-6 — fine within-trial production boundaries, OFFSET-locked (t=0 = the
preceding unit's offset), two conditions overlaid:
  svf_switch    — SVF, switch vs cluster, locked to the previous WORD offset
  ahc_sentence  — AHC, across- vs within-explanation, locked to the previous
                  SENTENCE offset

ROIs (rows), all from the Harvard-Oxford subcortical hippocampus label, split
along the MNI anterior-posterior (y) axis at the per-hemisphere median:
  whole   — entire hippocampus (L+R)
  ant     — anterior hippocampus (y >= median)
  post    — posterior hippocampus (y <  median)

EFFICIENT EXTRACTION
--------------------
The three ROI time courses share a single hippocampus mask, so each BOLD run is
loaded from disk exactly once and yields all three ROIs together (whole / ant /
post), cached to one small .npz. Only masked voxels are materialised. The fine
within-trial columns reuse the same SVF/AHC caches. Extraction across runs can be
parallelised with --n_jobs; once cached, plotting is instant.

Usage:
    uv run python srcs/fmrianalysis/hippocampus_boundary_timecourse.py
    uv run python srcs/fmrianalysis/hippocampus_boundary_timecourse.py --n_jobs 8
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

OUTPUT_DIR = FIGS_DIR / 'boundary_timecourse'
CACHE_DIR = ANALYSIS_CACHE_DIR / 'hipp_ap'
RECALL_DIR = DATA_DIR / 'filmfest_recall_timestamps'
SVF_SWITCH_DIR = DATA_DIR / 'rec/svf_transition_ratings/source'
AHC_SENT_DIR = DATA_DIR / 'rec/ahc_sentences'
BOUNDARY_STRENGTH_CSV = ANNOTATIONS_DIR / 'filmfest_boundary_strength.csv'
FILMFEST_RUN1_LEN_TR = 996   # TRs in filmfest1 (concat offset for filmfest2)

# Harvard-Oxford subcortical (maxprob thr25 2mm) hippocampus label indices
# (verified: idx 9 = Left Hippocampus, idx 19 = Right Hippocampus)
HO_ATLAS = 'sub-maxprob-thr25-2mm'
HIPP_LABEL_L = 9
HIPP_LABEL_R = 19

# Peri-boundary windows (seconds), t=0 = anchor. The coarse task/narrative
# boundaries (columns 1-4) use a wider window than the fine within-trial
# boundaries (columns 5-6).
TRIAL_PRE_S, TRIAL_POST_S = 30, 45
COND2_PRE_S, COND2_POST_S = 18, 30
TRIAL_WIN = (int(round(TRIAL_PRE_S / TR)), int(round(TRIAL_POST_S / TR)))  # (20, 30)
COND2_WIN = (int(round(COND2_PRE_S / TR)), int(round(COND2_POST_S / TR)))  # (12, 20)

# WAV recordings start this many seconds before the fMRI scan; word/sentence
# timestamps are in recording time and must be shifted to scanner time.
SCANNER_START_OFFSET = 12.0

# HRF onset shift for the filmfest onset alignment: skip the title card that
# opens each movie so the "onset" reflects the new movie's content.
TITLE_SCENE_OFFSET = 6.0

# ROIs: (key, display_name)
ROI_SPEC = [
    ('whole', 'Hippocampus (whole)'),
    ('ant', 'Anterior hippocampus'),
    ('post', 'Posterior hippocampus'),
]

# Column specification.
#   kind='trial'  -> onset-locked single curve + dashed prev-offset marker
#   kind='cond2'  -> offset-locked, two condition curves
# yscale: which shared y-range a column uses — 'coarse' (subject-bounded, big
# trial responses) or 'fine' (fixed small range for subtle within-trial effects).
COLUMNS = [
    dict(key='svf', kind='trial', win=TRIAL_WIN, yscale='coarse',
         title='Word Generation\n(SVF trial)'),
    dict(key='ahc', kind='trial', win=TRIAL_WIN, yscale='coarse',
         title='Explanation Generation\n(AHC trial)'),
    dict(key='movie', kind='trial', win=TRIAL_WIN, yscale='coarse',
         title='Movie Watching\n(between-movie)'),
    dict(key='recall', kind='trial', win=TRIAL_WIN, yscale='coarse',
         title='Movie Recall\n(between-movie)'),
    dict(key='within_movie', kind='trial', win=TRIAL_WIN, yscale='fine',
         title='Movie Watching\n(within-movie\nevent boundary)'),
    dict(key='svf_switch', kind='cond2', win=COND2_WIN, yscale='fine',
         title='Word Generation\n(switch vs cluster,\nprev-word offset)',
         conds=[('switch', 'Switch', '#e74c3c'),
                ('cluster', 'Cluster', '#7f7f7f')]),
    dict(key='ahc_sentence', kind='cond2', win=COND2_WIN, yscale='fine',
         title='Explanation Generation\n(across vs within,\nprev-sentence offset)',
         conds=[('Across', 'Across-explanation', '#e74c3c'),
                ('Within', 'Within-explanation', '#7f7f7f')]),
]

ONSET_COLOR = '#1f77b4'
OFFSET_MARKER_COLOR = '#555555'
# Per-subject line colors for the coarse-boundary columns (group mean = black).
SUBJECT_COLORS = {s: c for s, c in zip(SUBJECT_IDS, plt.cm.tab10.colors)}


# ============================================================================
# TRIAL / NARRATIVE BOUNDARY EVENTS (columns 1-4)
# ============================================================================

def get_movie_boundary_onsets(task):
    """Movie-start times (s) of movies 2..N for a filmfest task."""
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    onsets = []
    for movie in movies[1:]:
        df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        first_start = segb['Start Time (m.ss)'].values[0]
        onsets.append(mss_to_seconds(first_start))
    return onsets


def get_within_movie_boundaries(task):
    """Strong, fMRI-retained within-movie event boundaries for a filmfest run.

    Returns run-relative seconds (RAW — no HRF pre-shift — so the boundary sits at
    t=0 and the BOLD response lags naturally, matching the other columns)."""
    df = pd.read_csv(BOUNDARY_STRENGTH_CSV)
    movie_ids = [1, 2, 3, 4, 5] if task == 'filmfest1' else [6, 7, 8, 9, 10]
    d = df[(df['movie'].isin(movie_ids)) &
           (df['retained_for_fmri'] == 1) &
           (df['boundary_type'] == 'strong')].copy()
    d['run_rel_TR'] = d['concat_TR']
    if task == 'filmfest2':
        d['run_rel_TR'] -= FILMFEST_RUN1_LEN_TR
    # Denoise each movie's run onset, then place the boundary at onset + timestamp.
    d['movie_onset_run_TR'] = d['run_rel_TR'] - d['timestamp_sec'] / TR
    onset_TR = d.groupby('movie')['movie_onset_run_TR'].mean()
    run_rel_sec = onset_TR[d['movie'].values].values * TR + d['timestamp_sec'].values
    return sorted(run_rel_sec)


def _parse_recall_tsv_filename(stem):
    parts = stem.split('_')
    return parts[0], parts[1], '_'.join(parts[2:]).replace('task-', '')


def _recall_blocks(subject):
    """[(session, task, block_end_times, block_start_times)] per recall TSV.

    A "block" is a contiguous run of segments for the same recalled movie; ends
    and starts are scanner-relative seconds, chronological."""
    out = []
    for tsv in sorted(RECALL_DIR.glob(f'{subject}_*_desc-recallsegments.tsv')):
        stem = tsv.stem.replace('_desc-recallsegments', '')
        _, ses, task = _parse_recall_tsv_filename(stem)
        df = pd.read_csv(tsv, sep='\t')
        movies = df['movie'].values
        sc_start = df['scanner_start'].values
        sc_end = df['scanner_end'].values
        ends, starts = [], []
        for i in range(len(df)):
            if i == 0 or movies[i] != movies[i - 1]:
                starts.append(float(sc_start[i]))
            if i == len(df) - 1 or movies[i] != movies[i + 1]:
                ends.append(float(sc_end[i]))
        out.append((ses, task, ends, starts))
    return out


def collect_trial_runs(subject, btype):
    """Return [(session, task, onset_times, delays)] for one trial-boundary type.

    onset_times : next-unit onset (scanner seconds), the anchor for the curve.
    delays      : onset - previous offset, index-aligned (used to place the
                  dashed offset marker)."""
    runs = []
    if btype in ('svf', 'ahc'):
        for ses, task in discover_svf_ahc_sessions(subject):
            if task != btype:
                continue
            onsets, offsets = get_trial_times(subject, ses, task)
            if len(offsets) >= 2:
                offs = np.asarray(offsets[:-1], float)   # end of trial i
                ons = np.asarray(onsets[1:], float)       # start of trial i+1
                runs.append((ses, task, ons, ons - offs))
    elif btype == 'movie':
        if subject in FILMFEST_SUBJECTS:
            ses = FILMFEST_SUBJECTS[subject]
            for task in ('filmfest1', 'filmfest2'):
                offs = np.asarray(get_movie_boundary_offsets(task), float)
                ons = np.asarray(get_movie_boundary_onsets(task), float) + TITLE_SCENE_OFFSET
                runs.append((ses, task, ons, ons - offs))
    elif btype == 'within_movie':
        if subject in FILMFEST_SUBJECTS:
            ses = FILMFEST_SUBJECTS[subject]
            for task in ('filmfest1', 'filmfest2'):
                bnd = np.asarray(get_within_movie_boundaries(task), float)
                # Single boundary events (no onset/offset pair) -> no offset marker.
                runs.append((ses, task, bnd, np.array([])))
    elif btype == 'recall':
        if subject in FILMFEST_SUBJECTS:
            for ses, task, ends, starts in _recall_blocks(subject):
                if len(ends) >= 2 and len(starts) >= 2:
                    offs = np.asarray(ends[:-1], float)      # end of block i
                    ons = np.asarray(starts[1:], float)      # start of block i+1
                    runs.append((ses, task, ons, ons - offs))
    else:
        raise ValueError(btype)
    return runs


# ============================================================================
# FINE WITHIN-TRIAL BOUNDARY EVENTS (columns 5-6), offset-locked
# ============================================================================

def parse_svf_switch(csv_path):
    """SVF word events locked to the PRECEDING word offset (scanner seconds).

    Mirrors svf_switch_boundary.get_events: drop 'next' words, drop depletion
    switches (switch immediately after a switch or after 'next'). Returns a frame
    with columns onset, trial_type in {'switch','cluster'}."""
    df = pd.read_csv(csv_path).sort_values('start').reset_index(drop=True)
    df['switch_flag'] = pd.to_numeric(df['switch_flag'], errors='coerce').fillna(0).astype(int)
    df['preceding_end'] = df['end'].shift(1)
    df['preceding_switch_flag'] = df['switch_flag'].shift(1)
    df['preceding_word'] = df['transcription'].shift(1).astype(str).str.lower()

    df = df[df['transcription'].astype(str).str.lower() != 'next'].copy()
    is_switch = df['switch_flag'] == 1
    prev_switch = df['preceding_switch_flag'] == 1
    prev_next = df['preceding_word'] == 'next'
    df = df[~(is_switch & (prev_switch | prev_next))].copy()

    df['onset'] = df['preceding_end'] - SCANNER_START_OFFSET       # t=0: prev word offset
    df['cur_onset_rel'] = df['start'] - df['preceding_end']        # current word onset
    df['trial_type'] = df['switch_flag'].map({1: 'switch', 0: 'cluster'})
    df = df.dropna(subset=['onset'])
    df = df[df['onset'] >= 0]
    return df[['onset', 'trial_type', 'cur_onset_rel']]


def parse_ahc_sentences(xlsx_path):
    """AHC sentence events locked to the PRECEDING sentence offset (End Time).

    Classify each sentence as Across- vs Within-Possibility relative to the
    previous sentence of the same prompt (as in ahc_across_vs_within_glm), then
    lock to that preceding sentence's offset so the anchor matches the SVF column
    (previous-unit offset). Returns onset, trial_type in {'Across','Within'}, and
    cur_onset_rel (current sentence onset relative to the anchor)."""
    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()
    df['Prompt Number'] = df['Prompt Number'].ffill()
    df = df.sort_values(['Prompt Number', 'Start Time']).reset_index(drop=True)
    df['Preceding_Possibility'] = df.groupby('Prompt Number')['Possibility Number'].shift(1)
    df['Preceding_End'] = df.groupby('Prompt Number')['End Time'].shift(1)
    df['is_switch'] = df['Possibility Number'] != df['Preceding_Possibility']
    df = df.dropna(subset=['Preceding_Possibility', 'Preceding_End']).copy()
    df['trial_type'] = df['is_switch'].map({True: 'Across', False: 'Within'})
    df['onset'] = df['Preceding_End'] - SCANNER_START_OFFSET       # t=0: prev sentence offset
    df['cur_onset_rel'] = df['Start Time'] - df['Preceding_End']   # current sentence onset
    df = df[df['onset'] >= 0]
    return df[['onset', 'trial_type', 'cur_onset_rel']]


def collect_cond2_runs(subject, btype):
    """Return [(session, task, {cond: onset_times}, cur_onset_rel_array)].

    cur_onset_rel_array holds, per kept event, the current unit's onset relative
    to the anchor (previous unit offset at t=0)."""
    runs = []
    if btype == 'svf_switch':
        pattern = f'{subject}_ses-*_task-svf_desc-wordtimestampswithswitch.csv'
        for csv in sorted(SVF_SWITCH_DIR.glob(pattern)):
            ses = csv.stem.split('_')[1]
            df = parse_svf_switch(csv)
            runs.append((ses, 'svf', {
                'switch': df.loc[df.trial_type == 'switch', 'onset'].values,
                'cluster': df.loc[df.trial_type == 'cluster', 'onset'].values,
            }, df['cur_onset_rel'].values))
    elif btype == 'ahc_sentence':
        for xlsx in sorted(AHC_SENT_DIR.glob(
                f'{subject}_ses-*_task-ahc_desc-sentences.xlsx')):
            ses = xlsx.stem.split('_')[1]
            df = parse_ahc_sentences(xlsx)
            runs.append((ses, 'ahc', {
                'Across': df.loc[df.trial_type == 'Across', 'onset'].values,
                'Within': df.loc[df.trial_type == 'Within', 'onset'].values,
            }, df['cur_onset_rel'].values))
    else:
        raise ValueError(btype)
    return runs


def all_needed_runs(subjects, columns):
    """Union of (subject, session, task) referenced by any column."""
    runs = set()
    for subject in subjects:
        for col in columns:
            if col['kind'] == 'trial':
                for ses, task, _, _ in collect_trial_runs(subject, col['key']):
                    runs.add((subject, ses, task))
            else:
                for ses, task, *_ in collect_cond2_runs(subject, col['key']):
                    runs.add((subject, ses, task))
    return sorted(runs)


# ============================================================================
# EFFICIENT HIPPOCAMPUS EXTRACTION (one BOLD load -> whole / ant / post)
# ============================================================================

def _cache_path(subject, session, task):
    return CACHE_DIR / f'{subject}_{session}_task-{task}_hipp_ap_hp-0.01_z.npz'


def _region_timecourse(vox):
    """Voxels (n_vox, T) -> region mean, high-passed and temporally z-scored (T,)."""
    ts = vox.mean(axis=0).astype(np.float64)
    ts = highpass_filter(ts, order=2)
    ts = sp_zscore(ts, nan_policy='omit')
    return np.nan_to_num(ts).astype(np.float32)


def extract_hipp_run(subject, session, task, force=False):
    """Extract whole/ant/post hippocampus time courses for one run (cached).

    Loads the BOLD volume exactly once and materialises only hippocampus voxels;
    all three ROIs are derived from the same mask and saved together."""
    cache_file = _cache_path(subject, session, task)
    if cache_file.exists() and not force:
        return True

    bold_path = get_bold_path(subject, session, task)
    if not bold_path.exists():
        print(f'  [skip] BOLD missing: {bold_path.name}')
        return False

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
        ijk = np.argwhere(mask)                          # (n, 3), C-order
        world_y = apply_affine(affine, ijk)[:, 1]        # MNI y (mm)
        vox = data4d[mask]                               # (n, T), same C-order
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

def subject_trial_mean(subject, btype, roi_key, win, load_run):
    """Onset-locked mean for one subject/trial-column/ROI -> (mean_tc, n_ev)."""
    tb, ta = win
    epochs = []
    for ses, task, onsets, _ in collect_trial_runs(subject, btype):
        run = load_run(subject, ses, task)
        if run is None:
            continue
        ep = extract_event_locked(run[roi_key], onsets, tb, ta,
                                  return_epochs=True)
        if ep is not None:
            epochs.append(ep)
    if not epochs:
        return None, 0
    stacked = np.vstack(epochs)
    return stacked.mean(axis=0), stacked.shape[0]


def subject_cond2_mean(subject, btype, roi_key, cond, win, load_run):
    """Offset-locked mean for one subject/fine-column/ROI/condition -> (tc, n)."""
    tb, ta = win
    epochs = []
    for ses, task, cond_times, _cur in collect_cond2_runs(subject, btype):
        run = load_run(subject, ses, task)
        if run is None:
            continue
        ep = extract_event_locked(run[roi_key], cond_times[cond], tb, ta,
                                  return_epochs=True)
        if ep is not None:
            epochs.append(ep)
    if not epochs:
        return None, 0
    stacked = np.vstack(epochs)
    return stacked.mean(axis=0), stacked.shape[0]


def _group(subj_means):
    """List of (subject, tc) -> {'mean','sem','n_subj'} or None."""
    if not subj_means:
        return None
    arr = np.vstack([m for _, m in subj_means])
    return dict(mean=arr.mean(0), sem=arr.std(0) / np.sqrt(arr.shape[0]),
                n_subj=arr.shape[0])


# ============================================================================
# FIGURE
# ============================================================================

def _time_axis(win):
    tb, ta = win
    return np.arange(-tb, ta + 1) * TR


def compute_column(subjects, col, roi_spec, load_run):
    """Return per-ROI curves for one column.

    trial: {roi: {'onset': grp, 'n_ev': int}}, plus col-level 'offset_marker'.
    cond2: {roi: {cond_key: grp, 'n_ev_<cond>': int}}."""
    win = col['win']
    out = {rk: {} for rk, _ in roi_spec}
    if col['kind'] == 'trial':
        # mean prev-offset delay across all subjects' events -> marker position
        delays = []
        for s in subjects:
            for _, _, _, d in collect_trial_runs(s, col['key']):
                delays.extend(list(d))
        offset_marker = -float(np.mean(delays)) if delays else None
        for rk, _ in roi_spec:
            subj_means, n_ev = [], 0
            for s in subjects:
                tc, n = subject_trial_mean(s, col['key'], rk, win, load_run)
                if tc is not None:
                    subj_means.append((s, tc))
                    n_ev += n
            out[rk]['subjects'] = subj_means          # [(subject, tc)] colored lines
            out[rk]['onset'] = _group(subj_means)     # group mean (black line)
            out[rk]['n_ev'] = n_ev
        out['_offset_marker'] = offset_marker
    else:
        # mean current-unit onset relative to the anchor (previous offset)
        cur_rels = []
        for s in subjects:
            for _, _, _cond, cur in collect_cond2_runs(s, col['key']):
                cur_rels.extend(list(cur))
        out['_onset_marker'] = float(np.nanmean(cur_rels)) if cur_rels else None
        for rk, _ in roi_spec:
            for cond_key, _, _ in col['conds']:
                subj_means, n_ev = [], 0
                for s in subjects:
                    tc, n = subject_cond2_mean(s, col['key'], rk, cond_key, win, load_run)
                    if tc is not None:
                        subj_means.append((s, tc))
                        n_ev += n
                out[rk][cond_key] = _group(subj_means)
                out[rk][f'n_ev_{cond_key}'] = n_ev
    return out


def make_figure(subjects, columns, roi_spec, load_run, *, title, out_path,
                fine_ylim=(-0.5, 0.5)):
    times = {col['key']: _time_axis(col['win']) for col in columns}
    data = {col['key']: compute_column(subjects, col, roi_spec, load_run)
            for col in columns}

    # Two shared y-scales: 'coarse' columns (individual-subject lines) share one
    # range bounded to those lines; 'fine' columns share a smaller fixed range for
    # readability. A column's yscale is independent of its kind (e.g. within-movie
    # is a trial column but plotted on the fine scale).
    ylo, yhi = np.inf, -np.inf
    for col in columns:
        if col.get('yscale', 'coarse') != 'coarse':
            continue
        for rk, _ in roi_spec:
            for _, tc in data[col['key']][rk].get('subjects', []):
                ylo = min(ylo, float(np.min(tc)))
                yhi = max(yhi, float(np.max(tc)))
    if not np.isfinite(ylo):
        ylo, yhi = -1, 1
    pad = 0.08 * (yhi - ylo)
    ylo, yhi = ylo - pad, yhi + pad
    FINE_YLIM = fine_ylim

    n_rows, n_cols = len(roi_spec), len(columns)
    first_fine = next((i for i, c in enumerate(columns)
                       if c.get('yscale', 'coarse') == 'fine'), None)
    # Independent x-axes (coarse boundaries use a wider window than the fine
    # within-trial ones); y shared within each column group, not across groups.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows),
                             squeeze=False, sharex=False, sharey=False)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    for r, (rk, rname) in enumerate(roi_spec):
        for c, col in enumerate(columns):
            ax = axes[r][c]
            time = times[col['key']]
            cell = data[col['key']][rk]
            ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
            ax.axvline(0, color='k', lw=1.0, ls='-', alpha=0.7)

            if col['kind'] == 'trial':
                for subj, tc in cell.get('subjects', []):
                    ax.plot(time, tc, color=SUBJECT_COLORS.get(subj, '#999999'),
                            lw=0.9, alpha=0.7)
                g = cell.get('onset')
                if g is not None:
                    # SEM across subjects (N), not pooled events.
                    ax.fill_between(time, g['mean'] - g['sem'], g['mean'] + g['sem'],
                                    color='k', alpha=0.18, lw=0)
                    ax.plot(time, g['mean'], color='k', lw=2.4,
                            label=f"Group mean ± SEM (N={g['n_subj']})")
                marker = data[col['key']].get('_offset_marker')
                if marker is not None and time[0] <= marker <= time[-1]:
                    ax.axvline(marker, color=OFFSET_MARKER_COLOR, lw=1.4, ls='--',
                               alpha=0.9, label=f'Prev. offset (≈{marker:.0f}s)')
            else:
                for cond_key, cond_label, cond_color in col['conds']:
                    g = cell.get(cond_key)
                    if g is None:
                        continue
                    n_ev = cell.get(f'n_ev_{cond_key}', 0)
                    ax.plot(time, g['mean'], color=cond_color, lw=1.8,
                            label=f"{cond_label} (N={g['n_subj']}, {n_ev} ev)")
                    ax.fill_between(time, g['mean'] - g['sem'], g['mean'] + g['sem'],
                                    color=cond_color, alpha=0.2, lw=0)
                marker = data[col['key']].get('_onset_marker')
                if marker is not None and time[0] <= marker <= time[-1]:
                    ax.axvline(marker, color=OFFSET_MARKER_COLOR, lw=1.4, ls='--',
                               alpha=0.9, label=f'Cur. onset (≈{marker:.0f}s)')

            if col.get('yscale', 'coarse') == 'coarse':
                ax.set_ylim(ylo, yhi)
            else:
                ax.set_ylim(*FINE_YLIM)
            ax.set_xlim(time[0], time[-1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if r == 0:
                ax.set_title(col['title'], fontsize=9, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{rname}\nBOLD (z)', fontsize=9)
            elif c == first_fine:
                ax.set_ylabel('BOLD (z)', fontsize=9)
            else:
                ax.tick_params(labelleft=False)
            if r == n_rows - 1:
                ax.set_xlabel('Time rel. boundary (s)', fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            ax.legend(fontsize=5.5, loc='upper right', framealpha=0.6)
            ax.tick_params(labelsize=8)

    # Figure-level legend mapping the per-subject line colors (trial columns).
    from matplotlib.lines import Line2D
    subj_handles = [Line2D([0], [0], color=SUBJECT_COLORS[s], lw=1.5, label=s)
                    for s in subjects]
    subj_handles.append(Line2D([0], [0], color='k', lw=2.4, label='Group mean'))
    fig.legend(handles=subj_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.965), ncol=len(subj_handles),
               fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved figure -> {out_path}')
    return data


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_jobs', type=int, default=1,
                    help='Parallelism for the one-time extraction pre-pass')
    ap.add_argument('--force', action='store_true',
                    help='Re-extract hippocampus time courses even if cached')
    args = ap.parse_args()

    subjects = list(SUBJECT_IDS)

    print('=' * 64)
    print('HIPPOCAMPUS BOUNDARY-LOCKED TIME COURSES')
    print(f'Columns : {[c["key"] for c in COLUMNS]}')
    print(f'ROIs    : {[k for k, _ in ROI_SPEC]}')
    print(f'Windows : trial -{TRIAL_PRE_S}..+{TRIAL_POST_S}s, '
          f'fine -{COND2_PRE_S}..+{COND2_POST_S}s')
    print('=' * 64)

    # ---- Extraction pre-pass (cached; one BOLD load per run -> 3 ROIs) ----
    runs = all_needed_runs(subjects, COLUMNS)
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
    make_figure(
        subjects, COLUMNS, ROI_SPEC, load_hipp_run,
        title='Hippocampus boundary-locked time courses '
              '(coarse: subjects + black group mean ± SEM; fine: mean ± SEM)',
        out_path=OUTPUT_DIR / 'hippocampus_boundary_timecourse.png')
    print('\nDONE.')


if __name__ == '__main__':
    main()
