"""
cross_task_wspc.py — Cross-Task Boundary Within-Subject Pattern Correlation

Computes within-subject pattern correlation (WSPC) across SVF, AHC, and filmfest
tasks. Analogous to the cross_task_ispc_matrix.py + cross_task_ispc_anova.py +
cross_task_ispc_cell_stats.py pipeline, but replacing LOO inter-subject correlation
with within-subject self-correlation.

For each subject, correlates their own activation pattern for condition i with their
own pattern for condition j, then Fisher-z averages across subjects.

Generates:
  1. Boundary WSPC cache: N_TOTAL×N_TOTAL instance matrix + collapsed 12×12
  2. Mid-event WSPC cache: N_INSTANCES×N_INSTANCES single-window matrix
  3. summary_bar_charts/ — per-window bar charts (3 comparison types per ROI)
  4. summary_bar_charts/combined_boundary_mid/ — boundary vs mid-event comparison

Usage:
    uv run python srcs/fmrianalysis/cross_task_wspc.py --hp --preset onset0 --onset
    uv run python srcs/fmrianalysis/cross_task_wspc.py --hp --preset onset0 --onset --roi pmc ag
    uv run python srcs/fmrianalysis/cross_task_wspc.py --hp --preset onset0 --onset --no-cache
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import signal as sp_signal
from scipy.stats import zscore as sp_zscore, ttest_1samp
from statsmodels.stats.multitest import multipletests
from nilearn import datasets, image
from nilearn.maskers import NiftiMasker

from configs.config import (
    DATA_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR,
    SUBJECT_IDS, FILMFEST_SUBJECTS, MOVIE_INFO,
)
from configs.schaefer_rois import (
    EARLY_AUDITORY, EARLY_VISUAL, POSTERIOR_MEDIAL, ANGULAR_GYRUS,
    DLPFC, DACC, MPFC, get_bilateral_ids,
)
from fmrianalysis.utils import (
    highpass_filter, get_bold_path, discover_svf_ahc_sessions, find_psychopy_csv,
)

# ============================================================================
# CONSTANTS
# ============================================================================

AHC_SENTENCES_DIR = DATA_DIR / 'rec' / 'ahc_sentences'
ANNOTATIONS_DIR   = DATA_DIR / 'filmfest_annotations'
VOXEL_CACHE_DIR   = ANALYSIS_CACHE_DIR / 'roi_voxels'
RESULT_CACHE_DIR  = ANALYSIS_CACHE_DIR / 'cross_task_wspc'
OUTPUT_DIR        = None  # set in main()

SMOOTH_FWHM  = 6.0
HP_CUTOFF    = 0.01
MID_HALF_WIN = 5   # ±5 TRs = 10-TR mid-event window

WINDOW_PRESETS = {
    'hrf': {
        'windows':   [('pre', -6, 3), ('onset', 4, 13), ('post', 14, 23), ('late', 24, 33)],
        'tr_labels': ['-10 to -1 TRs', '0 to 9 TRs', '10 to 19 TRs', '20 to 29 TRs'],
        'cache_tag': 'hrf',
    },
    'post4': {
        'windows':   [('onset', 4, 13), ('post', 14, 23), ('late', 24, 33), ('vlate', 34, 43)],
        'tr_labels': ['0 to 9 TRs', '10 to 19 TRs', '20 to 29 TRs', '30 to 39 TRs'],
        'cache_tag': 'post4',
    },
    'onset0': {
        'windows':   [('onset', 0, 9), ('post', 10, 19), ('late', 20, 29), ('vlate', 30, 39)],
        'tr_labels': ['0 to 9 TRs', '10 to 19 TRs', '20 to 29 TRs', '30 to 39 TRs'],
        'cache_tag': 'onset0',
    },
}

WINDOWS       = WINDOW_PRESETS['onset0']['windows']
N_WINDOWS     = 4
N_FILM        = 8
WINDOW_PRESET = 'onset0'
_WIN_TR_SHORT = WINDOW_PRESETS['onset0']['tr_labels']

# Set in main() after trial discovery
N_SVF        = 0
N_AHC        = 0
N_INSTANCES  = 0   # N_SVF + N_AHC + N_FILM
N_TOTAL      = 0   # N_INSTANCES * N_WINDOWS
N_CONDITIONS = 12  # 4 windows × 3 tasks
TASK_SLICES: dict = {}
USE_ONSET = False

ROI_SPEC = [
    ('hipp',  'Hippocampus',               None),
    ('pmc',   'Posterior Medial Cortex',   get_bilateral_ids(POSTERIOR_MEDIAL)),
    ('mpfc',  'Medial Prefrontal Cortex',  get_bilateral_ids(MPFC)),
    ('dacc',  'Dorsal Anterior Cingulate', get_bilateral_ids(DACC)),
    ('dlpfc', 'Dorsolateral PFC',          get_bilateral_ids(DLPFC)),
]

ROI_ABBREVS = {
    'eac': 'EAC', 'evc': 'EVC', 'pmc': 'PMC', 'ag': 'AG',
    'dlpfc': 'dlPFC', 'dacc': 'dACC', 'mpfc': 'mPFC', 'hipp': 'Hipp',
}

COMPARISON_LABELS       = ['within-task', 'cross-task\n(SVF × AHC)', 'cross-task\n(× FilmFest)']
COMPARISON_LABELS_SHORT = ['within-task', 'SVF × AHC', '× FilmFest']
COMPARISON_COLORS       = ['#4878CF', 'white', 'white']

# Split variant: within-task broken out per task
SPLIT_LABELS       = ['within-MW', 'within-SF', 'within-EG',
                      'cross-task\n(SF × EG)', 'cross-task\n(× MW)']
SPLIT_LABELS_SHORT = ['within-MW', 'within-SF', 'within-EG', 'SF × EG', '× MW']
SPLIT_COLORS       = ['#4878CF', '#D65F5F', '#E88C1D', 'white', 'white']

# ============================================================================
# ATLAS SETUP
# ============================================================================

print("Loading atlases...")
SCHAEFER = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)

_ho        = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
_ho_img    = nib.load(_ho['maps']) if isinstance(_ho['maps'], (str, Path)) else _ho['maps']
_hipp_ids  = [i for i, l in enumerate(_ho['labels']) if 'hippocampus' in l.lower()]
_hipp_data = np.isin(_ho_img.get_fdata().astype(int), _hipp_ids).astype(np.int8)
HIPP_MASK_IMG = nib.Nifti1Image(_hipp_data, _ho_img.affine, _ho_img.header)
print(f"  Hippocampus mask: {_hipp_data.sum()} voxels.")

_schaefer_resampled = {}


def _get_schaefer_atlas(bold_path):
    key = str(bold_path.parent)
    if key not in _schaefer_resampled:
        bold_ref  = image.index_img(str(bold_path), 0)
        atlas_img = image.resample_to_img(SCHAEFER['maps'], bold_ref, interpolation='nearest')
        _schaefer_resampled[key] = np.round(atlas_img.get_fdata()).astype(int)
    return _schaefer_resampled[key]


# ============================================================================
# STEP 1: TRIAL DISCOVERY
# ============================================================================

def get_svf_offsets_by_category(subject, session):
    csv_path = find_psychopy_csv(subject, session, 'svf')
    if csv_path is None:
        return {}
    df   = pd.read_csv(csv_path)
    mask = df['svf_trial.stopped'].notna() & df['category_name'].notna()
    rows = df[mask][['category_name', 'svf_trial.started', 'svf_trial.stopped']]
    if len(rows) < 2:
        return {}
    first_start = rows['svf_trial.started'].iloc[0]
    result = {}
    for _, row in rows.iterrows():
        cat = row['category_name']
        col = 'svf_trial.started' if USE_ONSET else 'svf_trial.stopped'
        tr  = int(round((row[col] - first_start) / TR))
        result.setdefault(cat, []).append(tr)
    return result


def get_svf_midpoints_by_category(subject, session):
    csv_path = find_psychopy_csv(subject, session, 'svf')
    if csv_path is None:
        return {}
    df   = pd.read_csv(csv_path)
    mask = df['svf_trial.stopped'].notna() & df['category_name'].notna()
    rows = df[mask][['category_name', 'svf_trial.started', 'svf_trial.stopped']]
    if len(rows) < 2:
        return {}
    first_start = rows['svf_trial.started'].iloc[0]
    result = {}
    for _, row in rows.iterrows():
        cat    = row['category_name']
        mid_tr = int(round(((row['svf_trial.started'] + row['svf_trial.stopped']) / 2.0
                             - first_start) / TR))
        result.setdefault(cat, []).append(mid_tr)
    return result


def discover_shared_categories(subjects, min_subjects=5):
    from collections import Counter
    cat_count: Counter = Counter()
    for subj in subjects:
        subj_cats: set = set()
        for session, t in discover_svf_ahc_sessions(subj):
            if t != 'svf':
                continue
            subj_cats.update(get_svf_offsets_by_category(subj, session).keys())
        for cat in subj_cats:
            cat_count[cat] += 1
    shared = [c for c, n in cat_count.items() if n >= min_subjects]
    shared.sort(key=lambda c: (-cat_count[c], c))
    return shared, dict(cat_count)


def _find_ahc_sentences(subject, session):
    p = AHC_SENTENCES_DIR / f'{subject}_{session}_task-ahc_desc-sentences.xlsx'
    return p if p.exists() else None


def get_ahc_offsets_by_prompt(subject, session):
    xlsx_path = _find_ahc_sentences(subject, session)
    csv_path  = find_psychopy_csv(subject, session, 'ahc')
    if xlsx_path is None or csv_path is None:
        return {}
    prompts = pd.read_excel(xlsx_path)['Prompt'].dropna().tolist()
    if len(prompts) < 2:
        return {}
    df   = pd.read_csv(csv_path)
    rows = df[df['ahc_trial.stopped'].notna()][['ahc_trial.started', 'ahc_trial.stopped']]
    if len(rows) < 2 or len(rows) != len(prompts):
        return {}
    first_start = rows['ahc_trial.started'].iloc[0]
    result = {}
    col = 'ahc_trial.started' if USE_ONSET else 'ahc_trial.stopped'
    for prompt, (_, row) in zip(prompts, rows.iterrows()):
        tr = int(round((row[col] - first_start) / TR))
        result.setdefault(prompt, []).append(tr)
    return result


def get_ahc_midpoints_by_prompt(subject, session):
    xlsx_path = _find_ahc_sentences(subject, session)
    csv_path  = find_psychopy_csv(subject, session, 'ahc')
    if xlsx_path is None or csv_path is None:
        return {}
    prompts = pd.read_excel(xlsx_path)['Prompt'].dropna().tolist()
    if len(prompts) < 2:
        return {}
    df   = pd.read_csv(csv_path)
    rows = df[df['ahc_trial.stopped'].notna()][['ahc_trial.started', 'ahc_trial.stopped']]
    if len(rows) < 2 or len(rows) != len(prompts):
        return {}
    first_start = rows['ahc_trial.started'].iloc[0]
    result = {}
    for prompt, (_, row) in zip(prompts, rows.iterrows()):
        mid_tr = int(round(((row['ahc_trial.started'] + row['ahc_trial.stopped']) / 2.0
                             - first_start) / TR))
        result.setdefault(prompt, []).append(mid_tr)
    return result


def discover_shared_prompts(subjects, min_subjects=5):
    from collections import Counter
    prompt_count: Counter = Counter()
    for subj in subjects:
        subj_prompts: set = set()
        for session, t in discover_svf_ahc_sessions(subj):
            if t != 'ahc':
                continue
            subj_prompts.update(get_ahc_offsets_by_prompt(subj, session).keys())
        for p in subj_prompts:
            prompt_count[p] += 1
    shared = [p for p, n in prompt_count.items() if n >= min_subjects]
    shared.sort(key=lambda p: (-prompt_count[p], p))
    return shared, dict(prompt_count)


def _mss_to_seconds(mss):
    minutes = int(mss)
    return minutes * 60 + round((float(mss) - minutes) * 100)


def get_all_movie_onsets(task):
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    onsets = []
    for movie in movies:
        df   = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        onsets.append(_mss_to_seconds(segb['Start Time (m.ss)'].values[0]))
    return onsets


def get_movie_midpoints(task):
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    mids   = []
    for movie in movies:
        df   = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        starts = segb['Start Time (m.ss)'].apply(_mss_to_seconds).values
        ends   = segb['End Time (m.ss)'].apply(_mss_to_seconds).values
        mids.append((starts[0] + ends[-1]) / 2.0)
    return mids


# ============================================================================
# STEP 2: VOXEL EXTRACTION & CACHING
# ============================================================================

def load_roi_voxels(subject, session, task, roi_key, parcel_ids,
                    fwhm=SMOOTH_FWHM, force=False):
    """Load smoothed ROI voxels (T, V) float32, with Level-1 caching."""
    VOXEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = VOXEL_CACHE_DIR / f'{subject}_{session}_task-{task}_roi-{roi_key}_sm{int(fwhm)}.npz'
    if not force and cache_path.exists():
        return np.load(cache_path)['voxels']

    bold_path = get_bold_path(subject, session, task)
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD not found: {bold_path}")

    print(f"  Smoothing {bold_path.name} ...")
    bold_smooth = image.smooth_img(nib.load(str(bold_path)), fwhm=fwhm)

    if roi_key == 'hipp':
        masker = NiftiMasker(mask_img=HIPP_MASK_IMG, standardize=False, verbose=0)
        voxels = masker.fit_transform(bold_smooth).astype(np.float32)
    else:
        bold_data  = bold_smooth.get_fdata(dtype=np.float32)
        atlas_data = _get_schaefer_atlas(bold_path)
        parts = [bold_data[atlas_data == pid].T
                 for pid in parcel_ids if (atlas_data == pid).sum() >= 2]
        del bold_data
        if not parts:
            raise ValueError(f"No usable parcels: {subject} {session} roi={roi_key}")
        voxels = np.concatenate(parts, axis=1).astype(np.float32)

    del bold_smooth
    np.savez_compressed(cache_path, voxels=voxels)
    print(f"  Cached → {cache_path.name}  shape={voxels.shape}")
    return voxels


def preprocess_voxels(voxels, do_hp=False):
    data = sp_signal.detrend(voxels.astype(np.float64), axis=0)
    if do_hp:
        data = highpass_filter(data, cutoff=HP_CUTOFF, tr=TR)
    data = sp_zscore(data, axis=0, nan_policy='omit')
    return np.nan_to_num(data)


# ============================================================================
# STEP 3: PATTERN EXTRACTION
# ============================================================================

def extract_pattern_at_tr(voxel_data, boundary_tr):
    T = voxel_data.shape[0]
    win_patterns = []
    for _, tr_start_rel, tr_end_rel in WINDOWS:
        start = boundary_tr + tr_start_rel
        end   = boundary_tr + tr_end_rel
        if start < 0 or end >= T:
            return None
        win_patterns.append(voxel_data[start:end + 1].mean(axis=0))
    return np.stack(win_patterns, axis=0)


def extract_boundary_patterns(voxel_data, boundary_trs):
    valid_patterns, valid_mask = [], []
    T = voxel_data.shape[0]
    for btr in boundary_trs:
        win_pats, ok = [], True
        for _, tr_start_rel, tr_end_rel in WINDOWS:
            s, e = btr + tr_start_rel, btr + tr_end_rel
            if s < 0 or e >= T:
                ok = False
                break
            win_pats.append(voxel_data[s:e + 1].mean(axis=0))
        valid_mask.append(ok)
        if ok:
            valid_patterns.append(np.stack(win_pats, axis=0))
    if not valid_patterns:
        return np.empty((0, N_WINDOWS, voxel_data.shape[1])), np.array(valid_mask)
    return np.stack(valid_patterns, axis=0), np.array(valid_mask)


def extract_mid_event_pattern(voxel_data, mid_tr):
    T = voxel_data.shape[0]
    s = mid_tr - MID_HALF_WIN
    e = mid_tr + MID_HALF_WIN - 1
    if s < 0 or e >= T:
        return None
    return voxel_data[s:e + 1].mean(axis=0)


# --- SVF ---

def get_subject_svf_patterns(subject, roi_key, parcel_ids, shared_cats, do_hp=False):
    """(N_SVF, N_WINDOWS, V) boundary patterns."""
    svf_sessions = [ses for ses, t in discover_svf_ahc_sessions(subject) if t == 'svf']
    cat_patterns: dict = {c: [] for c in shared_cats}
    for session in svf_sessions:
        offsets = get_svf_offsets_by_category(subject, session)
        if not any(c in offsets for c in shared_cats):
            continue
        try:
            raw  = load_roi_voxels(subject, session, 'svf', roi_key, parcel_ids)
            data = preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {session} svf: {e}"); continue
        for cat in shared_cats:
            for btr in offsets.get(cat, []):
                p = extract_pattern_at_tr(data, btr)
                if p is not None:
                    cat_patterns[cat].append(p)
    if not any(cat_patterns.values()):
        return None
    V    = next(p[0].shape[1] for p in cat_patterns.values() if p)
    full = np.full((N_SVF, N_WINDOWS, V), np.nan, dtype=np.float64)
    for ci, cat in enumerate(shared_cats):
        if cat_patterns[cat]:
            full[ci] = np.mean(cat_patterns[cat], axis=0)
    return full


def get_subject_svf_mid_patterns(subject, roi_key, parcel_ids, shared_cats, do_hp=False):
    """(N_SVF, V) mid-event patterns."""
    svf_sessions = [ses for ses, t in discover_svf_ahc_sessions(subject) if t == 'svf']
    cat_patterns: dict = {c: [] for c in shared_cats}
    for session in svf_sessions:
        mids = get_svf_midpoints_by_category(subject, session)
        if not any(c in mids for c in shared_cats):
            continue
        try:
            raw  = load_roi_voxels(subject, session, 'svf', roi_key, parcel_ids)
            data = preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {session} svf mid: {e}"); continue
        for cat in shared_cats:
            for mid_tr in mids.get(cat, []):
                p = extract_mid_event_pattern(data, mid_tr)
                if p is not None:
                    cat_patterns[cat].append(p)
    if not any(cat_patterns.values()):
        return None
    V    = next(p[0].shape[0] for p in cat_patterns.values() if p)
    full = np.full((N_SVF, V), np.nan, dtype=np.float64)
    for ci, cat in enumerate(shared_cats):
        if cat_patterns[cat]:
            full[ci] = np.mean(cat_patterns[cat], axis=0)
    return full


# --- AHC ---

def get_subject_ahc_patterns(subject, roi_key, parcel_ids, shared_prompts, do_hp=False):
    """(N_AHC, N_WINDOWS, V) boundary patterns."""
    ahc_sessions = [ses for ses, t in discover_svf_ahc_sessions(subject) if t == 'ahc']
    prompt_pats: dict = {p: [] for p in shared_prompts}
    for session in ahc_sessions:
        offsets = get_ahc_offsets_by_prompt(subject, session)
        if not any(p in offsets for p in shared_prompts):
            continue
        try:
            raw  = load_roi_voxels(subject, session, 'ahc', roi_key, parcel_ids)
            data = preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {session} ahc: {e}"); continue
        for prompt in shared_prompts:
            for btr in offsets.get(prompt, []):
                p = extract_pattern_at_tr(data, btr)
                if p is not None:
                    prompt_pats[prompt].append(p)
    if not any(prompt_pats.values()):
        return None
    V    = next(p[0].shape[1] for p in prompt_pats.values() if p)
    full = np.full((N_AHC, N_WINDOWS, V), np.nan, dtype=np.float64)
    for pi, prompt in enumerate(shared_prompts):
        if prompt_pats[prompt]:
            full[pi] = np.mean(prompt_pats[prompt], axis=0)
    return full


def get_subject_ahc_mid_patterns(subject, roi_key, parcel_ids, shared_prompts, do_hp=False):
    """(N_AHC, V) mid-event patterns."""
    ahc_sessions = [ses for ses, t in discover_svf_ahc_sessions(subject) if t == 'ahc']
    prompt_pats: dict = {p: [] for p in shared_prompts}
    for session in ahc_sessions:
        mids = get_ahc_midpoints_by_prompt(subject, session)
        if not any(p in mids for p in shared_prompts):
            continue
        try:
            raw  = load_roi_voxels(subject, session, 'ahc', roi_key, parcel_ids)
            data = preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {session} ahc mid: {e}"); continue
        for prompt in shared_prompts:
            for mid_tr in mids.get(prompt, []):
                p = extract_mid_event_pattern(data, mid_tr)
                if p is not None:
                    prompt_pats[prompt].append(p)
    if not any(prompt_pats.values()):
        return None
    V    = next(p[0].shape[0] for p in prompt_pats.values() if p)
    full = np.full((N_AHC, V), np.nan, dtype=np.float64)
    for pi, prompt in enumerate(shared_prompts):
        if prompt_pats[prompt]:
            full[pi] = np.mean(prompt_pats[prompt], axis=0)
    return full


# --- Filmfest ---

def _get_filmfest_patterns_core(subject, roi_key, parcel_ids, do_hp, tr_getter):
    """Shared logic for filmfest boundary/mid-event extraction.

    tr_getter(task) → list of N_PER_RUN=4 TRs (may be float or int).
    Returns (8, ...) list of patterns or Nones.
    """
    if subject not in FILMFEST_SUBJECTS:
        return None
    session   = FILMFEST_SUBJECTS[subject]
    N_PER_RUN = 4
    all_pats  = []
    for task in ('filmfest1', 'filmfest2'):
        try:
            raw  = load_roi_voxels(subject, session, task, roi_key, parcel_ids)
            data = preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {task}: {e}")
            all_pats.extend([None] * N_PER_RUN)
            continue
        trs = tr_getter(task)
        all_pats.extend(trs)
    return all_pats  # list of TRs/floats, length 8


def get_subject_filmfest_patterns(subject, roi_key, parcel_ids, do_hp=False):
    """(N_FILM=8, N_WINDOWS, V) boundary patterns."""
    if subject not in FILMFEST_SUBJECTS:
        return None
    session   = FILMFEST_SUBJECTS[subject]
    N_PER_RUN = 4
    all_patterns, all_valid = [], []
    for task in ('filmfest1', 'filmfest2'):
        try:
            raw  = load_roi_voxels(subject, session, task, roi_key, parcel_ids)
            data = preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {task}: {e}")
            all_patterns.append(None); all_valid.extend([False] * N_PER_RUN); continue
        onset_secs  = get_all_movie_onsets(task)[1:]
        if USE_ONSET:
            onset_secs = [s + 6.0 for s in onset_secs]
        btrs        = [int(round(s / TR)) for s in onset_secs]
        pats, vmask = extract_boundary_patterns(data, btrs)
        all_patterns.append(pats); all_valid.extend(vmask.tolist())

    valid_parts = [p for p in all_patterns if p is not None and p.shape[0] > 0]
    if not valid_parts:
        return None
    V    = valid_parts[0].shape[2]
    full = np.full((N_FILM, N_WINDOWS, V), np.nan, dtype=np.float64)
    run_vi = 0
    for ti, (task, pats) in enumerate(zip(('filmfest1', 'filmfest2'), all_patterns)):
        if pats is None or pats.shape[0] == 0:
            run_vi += N_PER_RUN; continue
        vf = all_valid[run_vi:run_vi + N_PER_RUN]
        vc = 0
        for bi, ok in enumerate(vf):
            if ok and vc < pats.shape[0]:
                full[ti * N_PER_RUN + bi] = pats[vc]; vc += 1
        run_vi += N_PER_RUN
    return full


def get_subject_filmfest_mid_patterns(subject, roi_key, parcel_ids, do_hp=False):
    """(N_FILM=8, V) mid-event patterns."""
    if subject not in FILMFEST_SUBJECTS:
        return None
    session   = FILMFEST_SUBJECTS[subject]
    N_PER_RUN = 4
    all_pats  = []
    for task in ('filmfest1', 'filmfest2'):
        try:
            raw  = load_roi_voxels(subject, session, task, roi_key, parcel_ids)
            data = preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {task}: {e}")
            all_pats.extend([None] * N_PER_RUN); continue
        for mid_sec in get_movie_midpoints(task)[1:]:
            p = extract_mid_event_pattern(data, int(round(mid_sec / TR)))
            all_pats.append(p)
    valid = [p for p in all_pats if p is not None]
    if not valid:
        return None
    V    = valid[0].shape[0]
    full = np.full((N_FILM, V), np.nan, dtype=np.float64)
    for i, p in enumerate(all_pats):
        if p is not None:
            full[i] = p
    return full


# --- Cross-task concatenation ---

def get_subject_cross_task_patterns(subject, roi_key, parcel_ids,
                                     shared_cats, shared_prompts, do_hp=False):
    """(N_INSTANCES, N_WINDOWS, V) boundary patterns across all tasks."""
    svf  = get_subject_svf_patterns(subject, roi_key, parcel_ids, shared_cats, do_hp)
    ahc  = get_subject_ahc_patterns(subject, roi_key, parcel_ids, shared_prompts, do_hp)
    film = get_subject_filmfest_patterns(subject, roi_key, parcel_ids, do_hp)
    Vs   = [p.shape[2] for p in (svf, ahc, film) if p is not None]
    if not Vs:
        return None
    V    = min(Vs)
    full = np.full((N_INSTANCES, N_WINDOWS, V), np.nan, dtype=np.float64)
    if svf  is not None: full[TASK_SLICES['svf']]  = svf[..., :V]
    if ahc  is not None: full[TASK_SLICES['ahc']]  = ahc[..., :V]
    if film is not None: full[TASK_SLICES['film']] = film[..., :V]
    return full


def get_subject_cross_task_mid_patterns(subject, roi_key, parcel_ids,
                                         shared_cats, shared_prompts, do_hp=False):
    """(N_INSTANCES, V) mid-event patterns across all tasks."""
    svf  = get_subject_svf_mid_patterns(subject, roi_key, parcel_ids, shared_cats, do_hp)
    ahc  = get_subject_ahc_mid_patterns(subject, roi_key, parcel_ids, shared_prompts, do_hp)
    film = get_subject_filmfest_mid_patterns(subject, roi_key, parcel_ids, do_hp)
    Vs   = [p.shape[1] for p in (svf, ahc, film) if p is not None]
    if not Vs:
        return None
    V    = min(Vs)
    full = np.full((N_INSTANCES, V), np.nan, dtype=np.float64)
    if svf  is not None: full[TASK_SLICES['svf']]  = svf[..., :V]
    if ahc  is not None: full[TASK_SLICES['ahc']]  = ahc[..., :V]
    if film is not None: full[TASK_SLICES['film']] = film[..., :V]
    return full


def flatten_cross_task_patterns(patterns):
    """(N_INSTANCES, N_WINDOWS, V) → (N_TOTAL, V), instances nested in windows."""
    return patterns.transpose(1, 0, 2).reshape(N_TOTAL, -1)


# ============================================================================
# STEP 4: WITHIN-SUBJECT PATTERN CORRELATION
# ============================================================================

def _compute_wspc_matrix(all_patterns):
    """Compute within-subject pattern correlation for arbitrary (N_subj, N_C, V) input.

    For each subject, correlates their own patterns across all condition pairs.
    Fisher-z averages across subjects.

    Returns group_wspc (N_C, N_C), per_subj_wspc (N_subj, N_C, N_C).
    """
    N, N_C, _ = all_patterns.shape
    per_subj = np.full((N, N_C, N_C), np.nan)

    for i in range(N):
        mat = np.full((N_C, N_C), np.nan)
        for row in range(N_C):
            a = all_patterns[i, row]
            if np.all(np.isnan(a)):
                continue
            for col in range(N_C):
                if col == row:
                    continue  # skip self-correlation (always 1)
                b = all_patterns[i, col]
                if np.all(np.isnan(b)):
                    continue
                valid = ~(np.isnan(a) | np.isnan(b))
                if valid.sum() < 2:
                    continue
                a_, b_ = a[valid] - a[valid].mean(), b[valid] - b[valid].mean()
                denom  = np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum())
                if denom == 0:
                    continue
                mat[row, col] = np.dot(a_, b_) / denom
        per_subj[i] = mat

    z_stack = np.arctanh(np.clip(per_subj, -0.999, 0.999))
    group   = np.tanh(np.nanmean(z_stack, axis=0))
    return group, per_subj


# ============================================================================
# STEP 5: COLLAPSE TO 12×12 CONDITION MATRIX
# ============================================================================

def _build_cond_instance_indices():
    task_starts = {'svf': 0, 'ahc': N_SVF, 'film': N_SVF + N_AHC}
    task_sizes  = {'svf': N_SVF, 'ahc': N_AHC, 'film': N_FILM}
    cond_indices = []
    for task in ('svf', 'ahc', 'film'):
        for win_idx in range(N_WINDOWS):
            start = win_idx * N_INSTANCES + task_starts[task]
            cond_indices.append(np.arange(start, start + task_sizes[task]))
    return cond_indices


def collapse_to_condition_matrix(instance_wspc):
    cond_indices = _build_cond_instance_indices()
    cond_matrix  = np.full((N_CONDITIONS, N_CONDITIONS), np.nan)
    for ci, idx_a in enumerate(cond_indices):
        for cj, idx_b in enumerate(cond_indices):
            vals = instance_wspc[np.ix_(idx_a, idx_b)].ravel()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                cond_matrix[ci, cj] = vals.mean()
    return cond_matrix


def collapse_per_subject_to_conditions(per_subj):
    """(N_subj, N_TOTAL, N_TOTAL) → (N_subj, 12, 12) Fisher-z."""
    N_subj = per_subj.shape[0]
    cond_indices = _build_cond_instance_indices()
    z_stack = np.arctanh(np.clip(per_subj, -0.999, 0.999))
    out = np.full((N_subj, N_CONDITIONS, N_CONDITIONS), np.nan)
    for ci, idx_a in enumerate(cond_indices):
        for cj, idx_b in enumerate(cond_indices):
            block = z_stack[:, idx_a, :][:, :, idx_b]
            with np.errstate(all='ignore'):
                out[:, ci, cj] = np.nanmean(block.reshape(N_subj, -1), axis=1)
    return out


def collapse_mid_per_subject(per_subj_mid):
    """(N_subj, N_INSTANCES, N_INSTANCES) → (N_subj, 3, 3) Fisher-z per task."""
    N_subj = per_subj_mid.shape[0]
    z_stack = np.arctanh(np.clip(per_subj_mid, -0.999, 0.999))
    task_slices = [slice(0, N_SVF), slice(N_SVF, N_SVF+N_AHC), slice(N_SVF+N_AHC, N_INSTANCES)]
    out = np.full((N_subj, 3, 3), np.nan)
    for ci, sl_a in enumerate(task_slices):
        for cj, sl_b in enumerate(task_slices):
            block = z_stack[:, sl_a, :][:, :, sl_b]
            with np.errstate(all='ignore'):
                out[:, ci, cj] = np.nanmean(block.reshape(N_subj, -1), axis=1)
    return out


# ============================================================================
# STEP 6: CONDITION VALUE EXTRACTION + STATISTICS
# ============================================================================

def _comparison_cell_pairs_12x12(win_idx, n_windows=N_WINDOWS):
    svf  = win_idx
    ahc  = 4 + win_idx
    film = 8 + win_idx
    within = [(svf, svf), (ahc, ahc), (film, film)]
    xsame  = [(svf, ahc), (ahc, svf)]
    xdiff  = [(svf, film), (film, svf), (ahc, film), (film, ahc)]
    return within, xsame, xdiff


def _per_subject_mean_from_12x12(per_subj_cond_z, cell_pairs):
    """Mean Fisher-z across cell_pairs for each subject. (N_subj,)"""
    N_subj = per_subj_cond_z.shape[0]
    vals   = np.stack([per_subj_cond_z[:, r, c] for r, c in cell_pairs], axis=1)
    return np.nanmean(vals, axis=1)


def _per_subject_mean_from_3x3(per_subj_3x3_z, cell_pairs):
    """Mean Fisher-z across cell_pairs for each subject from (N_subj, 3, 3)."""
    N_subj = per_subj_3x3_z.shape[0]
    vals   = np.stack([per_subj_3x3_z[:, r, c] for r, c in cell_pairs], axis=1)
    return np.nanmean(vals, axis=1)


def compute_condition_stats(sv):
    """(mean, sem, p) from per-subject Fisher-z values. NaN-safe.

    Returns mu in r-space (tanh of mean z), sem in z-space (for error bars).
    """
    v = sv[~np.isnan(sv)]
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    mu_z = float(np.nanmean(v))
    mu   = float(np.tanh(mu_z))   # convert back to r-space
    sem  = v.std(ddof=1) / np.sqrt(len(v))
    p    = ttest_1samp(v, 0.0).pvalue
    return mu, sem, p


def fdr_correct_ps(p_array):
    result = np.full_like(p_array, np.nan)
    valid  = ~np.isnan(p_array)
    if valid.sum() > 0:
        _, p_corr, _, _ = multipletests(p_array[valid], method='fdr_bh')
        result[valid] = p_corr
    return result


# ============================================================================
# STEP 7: FIGURES
# ============================================================================

def make_summary_bar_chart(roi_bar_data, win_label):
    """Grouped bar chart of mean WSPC with SEM error bars and per-subject dots."""
    n_rois  = len(roi_bar_data)
    bar_w   = 0.22
    offsets = np.array([-1, 0, 1]) * bar_w
    rng     = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.subplots_adjust(left=0.08, right=0.78, top=0.85, bottom=0.12)
    x = np.arange(n_rois)

    for ci, (label, color, offset) in enumerate(
            zip(COMPARISON_LABELS, COMPARISON_COLORS, offsets)):
        means = [d['mean'][ci] for d in roi_bar_data]
        sems  = [d['sem'][ci]  for d in roi_bar_data]
        sigs  = [d['sig'][ci]  for d in roi_bar_data]
        bars  = ax.bar(x + offset, means, bar_w, label=label,
                       color=color, alpha=0.75, edgecolor='black', linewidth=0.5,
                       yerr=sems, error_kw=dict(elinewidth=1.0, capsize=3, ecolor='black'))

        for xi, (bar, d) in enumerate(zip(bars, roi_bar_data)):
            sv = d['subj_vals'][ci]
            if sv is None or np.all(np.isnan(sv)):
                continue
            sv_v   = sv[~np.isnan(sv)]
            jitter = rng.uniform(-bar_w * 0.3, bar_w * 0.3, size=len(sv_v))
            ax.scatter(bar.get_x() + bar_w / 2 + jitter, sv_v,
                       s=18, color=color, edgecolors='black', linewidths=0.4,
                       zorder=4, alpha=0.9)

        for bar, sig, mu, sem in zip(bars, sigs, means, sems):
            if np.isnan(mu) or not sig:
                continue
            y_ast = max(mu, 0) + (sem if not np.isnan(sem) else 0) + 0.004
            ax.text(bar.get_x() + bar.get_width() / 2, y_ast, '*',
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([d['abbrev'] for d in roi_bar_data], fontsize=10)
    ax.set_ylabel('mean WSPC (± SEM)', fontsize=9)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)
    ax.set_title(
        f'Mean cross-task WSPC — {win_label}\n'
        '* = FDR-corrected t-test across subjects vs 0, p < 0.05 (corrected across ROIs × comparison types)',
        fontsize=9, fontweight='bold')
    return fig


def _draw_colored_segments(ax, xi, y_ax, segments, fontsize):
    from matplotlib.transforms import blended_transform_factory
    fig   = ax.get_figure()
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    texts = [ax.text(xi, y_ax, txt, fontsize=fontsize, color=color,
                     ha='left', va='top', transform=trans)
             for txt, color in segments]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths   = [t.get_window_extent(renderer).width for t in texts]
    cx, cy   = trans.transform((xi, y_ax))
    x_cur    = cx - sum(widths) / 2.0
    fig_inv  = fig.transFigure.inverted()
    for t, w in zip(texts, widths):
        xf, yf = fig_inv.transform((x_cur, cy))
        t.set_transform(fig.transFigure)
        t.set_position((xf, yf))
        x_cur += w


def make_combined_window_bar_chart(roi_name, bnd_data, mid_data):
    """6-bar chart per ROI: 3 comparison types × 2 windows (boundary, mid-event)."""
    import matplotlib.patches as mpatches
    rng   = np.random.default_rng(0)
    bar_w = 0.35
    x     = np.arange(3)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.subplots_adjust(left=0.12, right=0.72, top=0.85, bottom=0.14)

    for wi, (wdata, wlabel, hatch) in enumerate([
            (bnd_data, 'post-boundary', ''),
            (mid_data, 'mid-event',     '///'),
    ]):
        offset = (wi - 0.5) * bar_w
        for ci, color in enumerate(COMPARISON_COLORS):
            mu, sem, sig = wdata['mean'][ci], wdata['sem'][ci], wdata['sig'][ci]
            sv = wdata['subj_vals'][ci]
            ax.bar(x[ci] + offset, mu, bar_w,
                   color=color, alpha=0.75 if hatch == '' else 0.45,
                   edgecolor='black', linewidth=0.5, hatch=hatch,
                   yerr=sem, error_kw=dict(elinewidth=1.0, capsize=3, ecolor='black'))
            if sv is not None:
                sv_v   = sv[~np.isnan(sv)]
                jitter = rng.uniform(-bar_w * 0.25, bar_w * 0.25, size=len(sv_v))
                ax.scatter(x[ci] + offset + jitter, sv_v,
                           s=18, color=color, edgecolors='black',
                           linewidths=0.4, zorder=4, alpha=0.9)
            if sig and not np.isnan(mu):
                y_ast = max(mu, 0) + (sem if not np.isnan(sem) else 0) + 0.004
                ax.text(x[ci] + offset, y_ast, '*',
                        ha='center', va='bottom', fontsize=11,
                        fontweight='bold', color=color)

    leg_handles = [
        mpatches.Patch(facecolor=COMPARISON_COLORS[ci], edgecolor='black',
                       label=COMPARISON_LABELS_SHORT[ci])
        for ci in range(3)
    ] + [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='mid-event'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='post-boundary'),
    ]
    ax.legend(handles=leg_handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=7.5, framealpha=0.9)
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(COMPARISON_LABELS_SHORT, fontsize=9)
    ax.set_ylabel('mean WSPC (± SEM)', fontsize=9)
    ax.set_title(roi_name, fontsize=10, fontweight='bold')
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    return fig


def make_multi_roi_combined_bar_chart(roi_entries):
    """Combined figure: one subplot per ROI (shared y-axis)."""
    import matplotlib.patches as mpatches
    n     = len(roi_entries)
    rng   = np.random.default_rng(0)
    bar_w = 0.35
    x     = np.arange(3)

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.80, top=0.88, bottom=0.22, wspace=0.25)

    for col, (ax, (roi_name, bnd_data, mid_data)) in enumerate(zip(axes, roi_entries)):
        for wi, (wdata, facecolor) in enumerate([(bnd_data, 'white'), (mid_data, '#AAAAAA')]):
            offset = (wi - 0.5) * bar_w
            for ci in range(3):
                mu, sem, sig = wdata['mean'][ci], wdata['sem'][ci], wdata['sig'][ci]
                sv = wdata['subj_vals'][ci]
                ax.bar(x[ci] + offset, mu, bar_w,
                       color=facecolor, edgecolor='black', linewidth=0.8,
                       yerr=sem, error_kw=dict(elinewidth=1.0, capsize=3, ecolor='black'))
                if sv is not None:
                    sv_v   = sv[~np.isnan(sv)]
                    jitter = rng.uniform(-bar_w * 0.25, bar_w * 0.25, size=len(sv_v))
                    ax.scatter(x[ci] + offset + jitter, sv_v,
                               s=12, color=facecolor, edgecolors='black',
                               linewidths=0.4, zorder=4, alpha=0.9)
                if sig and not np.isnan(mu):
                    y_ast = max(mu, 0) + (sem if not np.isnan(sem) else 0) + 0.004
                    ax.text(x[ci] + offset, y_ast, '*',
                            ha='center', va='bottom', fontsize=11,
                            fontweight='bold', color='black')

        ax.axhline(0, color='black', linewidth=1.0, linestyle='-')
        from matplotlib.transforms import blended_transform_factory
        ax.set_xticks(x)
        ax.set_xticklabels([])
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        _main_labels = ['within-task', 'cross-task', 'cross-task']
        _sub_segs    = [
            None,
            [('SF', 'red'), (' × ', 'black'), ('EG', 'orange')],
            [('{', 'black'), ('SF', 'red'), (', ', 'black'), ('EG', 'orange'),
             ('} × ', 'black'), ('MW', 'blue')],
        ]
        for xi, main, sub in zip(x, _main_labels, _sub_segs):
            ax.text(xi, -0.04, main, fontsize=8, ha='center', va='top', transform=trans)
            if sub:
                _draw_colored_segments(ax, xi, -0.11, sub, fontsize=6.5)
        ax.tick_params(which='both', bottom=False)
        ax.set_title(roi_name, fontsize=10, fontweight='bold')
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        for sp in ['top', 'right', 'left', 'bottom']:
            ax.spines[sp].set_visible(False)
        if col == 0:
            ax.spines['left'].set_visible(True)
            ax.set_ylabel('within-subject pattern correlation (r)', fontsize=9)
        else:
            ax.tick_params(which='both', left=False)

    leg_handles = [
        mpatches.Patch(facecolor='white',   edgecolor='black', label='boundary'),
        mpatches.Patch(facecolor='#AAAAAA', edgecolor='black', label='mid-event'),
    ]
    axes[0].legend(handles=leg_handles, loc='upper right',
                   bbox_to_anchor=(1.0, 0.95), fontsize=7.5, frameon=False)
    return fig


# ============================================================================
# SPLIT-WITHIN FIGURE HELPERS
# ============================================================================

def _split_cell_pairs_12x12(win_idx):
    """Return 5 cell-pair lists for split-within comparison at a given window.

    Order: [within-MW, within-SF, within-EG, SF×EG, ×MW]
    Condition indices: SVF=win_idx, AHC=4+win_idx, Film=8+win_idx.
    """
    svf  = win_idx
    ahc  = 4 + win_idx
    film = 8 + win_idx
    return [
        [(film, film)],                                         # within-MW
        [(svf, svf)],                                           # within-SF
        [(ahc, ahc)],                                           # within-EG
        [(svf, ahc), (ahc, svf)],                               # SF × EG
        [(svf, film), (film, svf), (ahc, film), (film, ahc)],  # × MW
    ]


def _split_cell_pairs_3x3():
    """Return 5 cell-pair lists for split-within comparison from 3×3 mid-event matrix.

    Tasks: 0=SVF, 1=AHC, 2=Film.
    """
    return [
        [(2, 2)],                                     # within-MW
        [(0, 0)],                                     # within-SF
        [(1, 1)],                                     # within-EG
        [(0, 1), (1, 0)],                             # SF × EG
        [(0, 2), (2, 0), (1, 2), (2, 1)],            # × MW
    ]


def make_summary_bar_chart_split(roi_bar_data, win_label):
    """Summary bar chart with within-task split by task (5 comparison types)."""
    n_rois  = len(roi_bar_data)
    bar_w   = 0.14
    offsets = np.array([-2, -1, 0, 1, 2]) * bar_w
    rng     = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.subplots_adjust(left=0.08, right=0.75, top=0.85, bottom=0.12)
    x = np.arange(n_rois)

    for ci, (label, color, offset) in enumerate(
            zip(SPLIT_LABELS, SPLIT_COLORS, offsets)):
        means = [d['mean'][ci] for d in roi_bar_data]
        sems  = [d['sem'][ci]  for d in roi_bar_data]
        sigs  = [d['sig'][ci]  for d in roi_bar_data]
        bars  = ax.bar(x + offset, means, bar_w, label=label,
                       color=color, alpha=0.75, edgecolor='black', linewidth=0.5,
                       yerr=sems, error_kw=dict(elinewidth=0.8, capsize=2, ecolor='black'))

        for xi, (bar, d) in enumerate(zip(bars, roi_bar_data)):
            sv = d['subj_vals'][ci]
            if sv is None or np.all(np.isnan(sv)):
                continue
            sv_v   = sv[~np.isnan(sv)]
            jitter = rng.uniform(-bar_w * 0.3, bar_w * 0.3, size=len(sv_v))
            ax.scatter(bar.get_x() + bar_w / 2 + jitter, sv_v,
                       s=12, color=color, edgecolors='black', linewidths=0.3,
                       zorder=4, alpha=0.9)

        for bar, sig, mu, sem in zip(bars, sigs, means, sems):
            if np.isnan(mu) or not sig:
                continue
            y_ast = max(mu, 0) + (sem if not np.isnan(sem) else 0) + 0.004
            ax.text(bar.get_x() + bar.get_width() / 2, y_ast, '*',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([d['abbrev'] for d in roi_bar_data], fontsize=10)
    ax.set_ylabel('mean WSPC (± SEM)', fontsize=9)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)
    ax.set_title(
        f'Mean cross-task WSPC — {win_label} (within-task split by task)\n'
        '* = FDR-corrected t-test across subjects vs 0, p < 0.05',
        fontsize=9, fontweight='bold')
    return fig


def make_combined_window_bar_chart_split(roi_name, bnd_data, mid_data):
    """10-bar chart per ROI: 5 comparison types × 2 windows (boundary, mid-event)."""
    import matplotlib.patches as mpatches
    rng   = np.random.default_rng(0)
    bar_w = 0.28
    x     = np.arange(5)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(left=0.10, right=0.68, top=0.85, bottom=0.14)

    for wi, (wdata, wlabel, hatch) in enumerate([
            (bnd_data, 'post-boundary', ''),
            (mid_data, 'mid-event',     '///'),
    ]):
        offset = (wi - 0.5) * bar_w
        for ci, color in enumerate(SPLIT_COLORS):
            mu, sem, sig = wdata['mean'][ci], wdata['sem'][ci], wdata['sig'][ci]
            sv = wdata['subj_vals'][ci]
            ax.bar(x[ci] + offset, mu, bar_w,
                   color=color, alpha=0.75 if hatch == '' else 0.45,
                   edgecolor='black', linewidth=0.5, hatch=hatch,
                   yerr=sem, error_kw=dict(elinewidth=0.8, capsize=2, ecolor='black'))
            if sv is not None:
                sv_v   = sv[~np.isnan(sv)]
                jitter = rng.uniform(-bar_w * 0.25, bar_w * 0.25, size=len(sv_v))
                ax.scatter(x[ci] + offset + jitter, sv_v,
                           s=14, color=color, edgecolors='black',
                           linewidths=0.3, zorder=4, alpha=0.9)
            if sig and not np.isnan(mu):
                y_ast = max(mu, 0) + (sem if not np.isnan(sem) else 0) + 0.004
                ax.text(x[ci] + offset, y_ast, '*',
                        ha='center', va='bottom', fontsize=10,
                        fontweight='bold', color=color)

    leg_handles = [
        mpatches.Patch(facecolor=SPLIT_COLORS[ci], edgecolor='black',
                       label=SPLIT_LABELS_SHORT[ci])
        for ci in range(5)
    ] + [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='mid-event'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='post-boundary'),
    ]
    ax.legend(handles=leg_handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=7.5, framealpha=0.9)
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_LABELS_SHORT, fontsize=8)
    ax.set_ylabel('mean WSPC (± SEM)', fontsize=9)
    ax.set_title(roi_name, fontsize=10, fontweight='bold')
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    return fig


def make_multi_roi_combined_bar_chart_split(roi_entries):
    """Combined figure with split within-task: one subplot per ROI (shared y-axis)."""
    import matplotlib.patches as mpatches
    n     = len(roi_entries)
    rng   = np.random.default_rng(0)
    bar_w = 0.28
    x     = np.arange(5)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.6), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.82, top=0.88, bottom=0.22, wspace=0.22)

    for col, (ax, (roi_name, bnd_data, mid_data)) in enumerate(zip(axes, roi_entries)):
        for wi, (wdata, facecolor) in enumerate([(bnd_data, 'white'), (mid_data, '#AAAAAA')]):
            offset = (wi - 0.5) * bar_w
            for ci in range(5):
                mu, sem, sig = wdata['mean'][ci], wdata['sem'][ci], wdata['sig'][ci]
                sv = wdata['subj_vals'][ci]
                ax.bar(x[ci] + offset, mu, bar_w,
                       color=SPLIT_COLORS[ci] if facecolor == 'white' else facecolor,
                       edgecolor='black', linewidth=0.7,
                       alpha=0.75 if facecolor == 'white' else 0.5,
                       yerr=sem, error_kw=dict(elinewidth=0.8, capsize=2, ecolor='black'))
                if sv is not None:
                    sv_v   = sv[~np.isnan(sv)]
                    jitter = rng.uniform(-bar_w * 0.25, bar_w * 0.25, size=len(sv_v))
                    ax.scatter(x[ci] + offset + jitter, sv_v,
                               s=10, color=SPLIT_COLORS[ci] if facecolor == 'white' else facecolor,
                               edgecolors='black', linewidths=0.3, zorder=4, alpha=0.9)
                if sig and not np.isnan(mu):
                    y_ast = max(mu, 0) + (sem if not np.isnan(sem) else 0) + 0.004
                    ax.text(x[ci] + offset, y_ast, '*',
                            ha='center', va='bottom', fontsize=10,
                            fontweight='bold', color='black')

        ax.axhline(0, color='black', linewidth=1.0, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels(SPLIT_LABELS_SHORT, fontsize=7, rotation=30, ha='right')
        ax.tick_params(which='both', bottom=False)
        ax.set_title(roi_name, fontsize=10, fontweight='bold')
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        for sp in ['top', 'right', 'left', 'bottom']:
            ax.spines[sp].set_visible(False)
        if col == 0:
            ax.spines['left'].set_visible(True)
            ax.set_ylabel('within-subject pattern correlation (r)', fontsize=9)
        else:
            ax.tick_params(which='both', left=False)

    leg_handles = [
        mpatches.Patch(facecolor='white',   edgecolor='black', label='boundary'),
        mpatches.Patch(facecolor='#AAAAAA', edgecolor='black', label='mid-event'),
    ]
    axes[0].legend(handles=leg_handles, loc='upper right',
                   bbox_to_anchor=(1.0, 0.95), fontsize=7.5, frameon=False)
    return fig


# ============================================================================
# PER-ROI PIPELINE
# ============================================================================

def run_roi(roi_key, roi_name, parcel_ids, subjects,
            shared_cats, shared_prompts, do_hp=False, force=False):
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hp_tag    = '_hp' if do_hp else ''
    onset_tag = '_onset' if USE_ONSET else ''
    bnd_cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6{hp_tag}{onset_tag}_{WINDOW_PRESET}_wspc.npz'
    mid_cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6{hp_tag}_onset_mid_event_wspc.npz'

    print(f"\n{'='*60}\nROI: {roi_name} ({roi_key})")

    # ---- Boundary WSPC ----
    if not force and bnd_cache.exists():
        print(f"  [{roi_key}] Loading boundary WSPC from cache ...")
        bnd_data   = np.load(bnd_cache)
        per_subj   = bnd_data['per_subject']
    else:
        subject_patterns = []
        subj_list_used   = []
        for subj in subjects:
            print(f"  Subject: {subj}")
            pats = get_subject_cross_task_patterns(
                subj, roi_key, parcel_ids, shared_cats, shared_prompts, do_hp)
            if pats is None:
                print(f"  SKIP {subj}: no data"); continue
            flat    = flatten_cross_task_patterns(pats)
            n_valid = (~np.isnan(flat).all(axis=1)).sum()
            print(f"    {n_valid}/{N_TOTAL} conditions valid")
            subject_patterns.append(flat)
            subj_list_used.append(subj)

        if not subject_patterns:
            print(f"  [{roi_key}] No subjects, skipping.")
            return None, None

        all_pats = np.stack(subject_patterns, axis=0)
        print(f"  [{roi_key}] Computing boundary WSPC ({all_pats.shape[0]} subjects)...")
        group_wspc, per_subj = _compute_wspc_matrix(all_pats)
        cond_matrix = collapse_to_condition_matrix(group_wspc)

        np.savez_compressed(bnd_cache, instance_wspc=group_wspc,
                            cond_matrix=cond_matrix, per_subject=per_subj,
                            subjects=np.array(subj_list_used), do_hp=do_hp)
        print(f"  [{roi_key}] Cached → {bnd_cache.name}")

    # ---- Mid-event WSPC ----
    if not force and mid_cache.exists():
        print(f"  [{roi_key}] Loading mid-event WSPC from cache ...")
        per_subj_mid = np.load(mid_cache)['per_subject']
    else:
        mid_patterns = []
        for subj in subjects:
            pats = get_subject_cross_task_mid_patterns(
                subj, roi_key, parcel_ids, shared_cats, shared_prompts, do_hp)
            if pats is None:
                continue
            mid_patterns.append(pats)

        if not mid_patterns:
            print(f"  [{roi_key}] No mid-event data, skipping mid-event WSPC.")
            per_subj_mid = None
        else:
            all_mid = np.stack(mid_patterns, axis=0)
            print(f"  [{roi_key}] Computing mid-event WSPC ({all_mid.shape[0]} subjects)...")
            _, per_subj_mid = _compute_wspc_matrix(all_mid)
            np.savez_compressed(mid_cache, per_subject=per_subj_mid, do_hp=do_hp)
            print(f"  [{roi_key}] Cached → {mid_cache.name}")

    return per_subj, per_subj_mid


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-task boundary WSPC.')
    parser.add_argument('--roi', nargs='+', default=[k for k, _, _ in ROI_SPEC])
    parser.add_argument('--hp', action='store_true', help='HP filtering at 0.01 Hz')
    parser.add_argument('--preset', default='onset0', choices=['hrf', 'post4', 'onset0'])
    parser.add_argument('--onset', action='store_true',
                        help='Use trial onset (start) times for SVF/AHC; +6s for filmfest')
    parser.add_argument('--no-cache', action='store_true', help='Force recompute')
    parser.add_argument('--min-subjects', type=int, default=5)
    parser.add_argument('--split-within', action='store_true',
                        help='Also generate figures with within-task split by task '
                             '(within-MW, within-SF, within-EG as separate bars)')
    args = parser.parse_args()

    global OUTPUT_DIR, N_SVF, N_AHC, N_INSTANCES, N_TOTAL, TASK_SLICES
    global WINDOWS, WINDOW_PRESET, _WIN_TR_SHORT, USE_ONSET
    USE_ONSET = args.onset

    preset        = WINDOW_PRESETS[args.preset]
    WINDOWS       = preset['windows']
    WINDOW_PRESET = preset['cache_tag']
    _WIN_TR_SHORT = preset['tr_labels']

    hp_label    = 'hp' if args.hp else 'no_hp'
    onset_label = 'onset' if USE_ONSET else 'offset'
    OUTPUT_DIR  = FIGS_DIR / 'cross_task_wspc' / hp_label / WINDOW_PRESET / onset_label
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    svf_ahc_subjects = sorted(SUBJECT_IDS)
    all_subjects     = sorted(set(svf_ahc_subjects) | set(FILMFEST_SUBJECTS.keys()))

    shared_cats,    _ = discover_shared_categories(svf_ahc_subjects, args.min_subjects)
    shared_prompts, _ = discover_shared_prompts(svf_ahc_subjects, args.min_subjects)

    N_SVF       = len(shared_cats)
    N_AHC       = len(shared_prompts)
    N_INSTANCES = N_SVF + N_AHC + N_FILM
    N_TOTAL     = N_INSTANCES * N_WINDOWS
    TASK_SLICES = {
        'svf':  slice(0, N_SVF),
        'ahc':  slice(N_SVF, N_SVF + N_AHC),
        'film': slice(N_SVF + N_AHC, N_INSTANCES),
    }

    print(f"Subjects: {all_subjects}")
    print(f"SVF categories: {N_SVF}, AHC prompts: {N_AHC}, Filmfest: {N_FILM}")
    print(f"N_INSTANCES={N_INSTANCES}  N_WINDOWS={N_WINDOWS}  N_TOTAL={N_TOTAL}")
    print(f"Window preset: {WINDOW_PRESET}, HP: {args.hp}, Onset: {USE_ONSET}")

    roi_spec_filtered = [(k, n, p) for k, n, p in ROI_SPEC if k in args.roi]
    if not roi_spec_filtered:
        print("No valid ROIs."); return

    # ------------------------------------------------------------------ #
    # Per-ROI: compute WSPC, collect per-subject data                     #
    # ------------------------------------------------------------------ #
    all_per_subj_bnd = {}   # roi_key → (N_subj, N_TOTAL, N_TOTAL)
    all_per_subj_mid = {}   # roi_key → (N_subj, N_INSTANCES, N_INSTANCES) or None

    for roi_key, roi_name, parcel_ids in roi_spec_filtered:
        ps_bnd, ps_mid = run_roi(
            roi_key, roi_name, parcel_ids, all_subjects,
            shared_cats, shared_prompts,
            do_hp=args.hp, force=args.no_cache,
        )
        if ps_bnd is not None:
            all_per_subj_bnd[roi_key] = ps_bnd
        if ps_mid is not None:
            all_per_subj_mid[roi_key] = ps_mid

    if not all_per_subj_bnd:
        print("No ROIs with data — exiting."); return

    # ------------------------------------------------------------------ #
    # Summary bar charts (one per window)                                 #
    # ------------------------------------------------------------------ #
    bar_dir = OUTPUT_DIR / 'summary_bar_charts'
    bar_dir.mkdir(parents=True, exist_ok=True)

    for win_idx, win_label in enumerate(_WIN_TR_SHORT):
        within_pairs, xsame_pairs, xdiff_pairs = _comparison_cell_pairs_12x12(win_idx)

        roi_bar_data = []
        raw_p_by_comp = [[], [], []]   # per comparison type

        for roi_key, roi_name, _ in roi_spec_filtered:
            if roi_key not in all_per_subj_bnd:
                continue
            ps_cond_z = collapse_per_subject_to_conditions(all_per_subj_bnd[roi_key])

            means, sems, ttest_ps, subj_vals = [], [], [], []
            for ci, pairs in enumerate([within_pairs, xsame_pairs, xdiff_pairs]):
                sv         = _per_subject_mean_from_12x12(ps_cond_z, pairs)
                mu, sem, p = compute_condition_stats(sv)
                means.append(mu); sems.append(sem); ttest_ps.append(p)
                subj_vals.append(np.tanh(sv))
                raw_p_by_comp[ci].append((roi_key, p))

            roi_bar_data.append(dict(
                roi_key=roi_key,
                abbrev=ROI_ABBREVS.get(roi_key, roi_key.upper()),
                mean=means, sem=sems, p=ttest_ps,
                sig=[False, False, False],
                subj_vals=subj_vals,
            ))

        # FDR across all comparison types × ROIs
        all_p_tuples = [(rk, ci, p)
                        for ci, p_list in enumerate(raw_p_by_comp)
                        for rk, p in p_list]
        ps_all   = np.array([t[2] for t in all_p_tuples], dtype=float)
        p_fdr    = fdr_correct_ps(ps_all)
        for idx, (rk, ci, _) in enumerate(all_p_tuples):
            for d in roi_bar_data:
                if d['roi_key'] == rk:
                    d['sig'][ci] = bool(p_fdr[idx] < 0.05)

        fig = make_summary_bar_chart(roi_bar_data, win_label)
        bp  = bar_dir / f'wspc_{win_idx:02d}_{win_label.replace(" ", "_")}.png'
        fig.savefig(bp, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'\nBar chart ({win_label}) → {bp.name}')

    # ------------------------------------------------------------------ #
    # combined_boundary_mid — win_idx=0 (first window) vs mid-event      #
    # ------------------------------------------------------------------ #
    combined_dir = bar_dir / 'combined_boundary_mid'
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Boundary win0 data
    within0, xsame0, xdiff0 = _comparison_cell_pairs_12x12(0)
    bnd_win0_by_roi = {}
    raw_p_win0 = [[], [], []]

    for roi_key, roi_name, _ in roi_spec_filtered:
        if roi_key not in all_per_subj_bnd:
            continue
        ps_cond_z = collapse_per_subject_to_conditions(all_per_subj_bnd[roi_key])
        means, sems, ttest_ps, subj_vals = [], [], [], []
        for ci, pairs in enumerate([within0, xsame0, xdiff0]):
            sv         = _per_subject_mean_from_12x12(ps_cond_z, pairs)
            mu, sem, p = compute_condition_stats(sv)
            means.append(mu); sems.append(sem); ttest_ps.append(p)
            subj_vals.append(sv)
            raw_p_win0[ci].append((roi_key, p))
        bnd_win0_by_roi[roi_key] = dict(
            roi_name=roi_name, mean=means, sem=sems,
            sig=[False, False, False], subj_vals=subj_vals,
        )

    # FDR for boundary win0
    win0_all_p = [(rk, ci, p)
                  for ci, p_list in enumerate(raw_p_win0)
                  for rk, p in p_list]
    ps_win0  = np.array([t[2] for t in win0_all_p], dtype=float)
    p_fdr_win0 = fdr_correct_ps(ps_win0)
    for idx, (rk, ci, _) in enumerate(win0_all_p):
        bnd_win0_by_roi[rk]['sig'][ci] = bool(p_fdr_win0[idx] < 0.05)

    # Mid-event data
    mid_cell_pairs = [
        [(0, 0), (1, 1), (2, 2)],          # within-task
        [(0, 1), (1, 0)],                   # xsame
        [(0, 2), (2, 0), (1, 2), (2, 1)],  # xdiff
    ]
    mid_by_roi   = {}
    raw_p_mid    = [[], [], []]

    for roi_key, roi_name, _ in roi_spec_filtered:
        if roi_key not in all_per_subj_mid:
            continue
        ps_3x3_z = collapse_mid_per_subject(all_per_subj_mid[roi_key])
        means, sems, ttest_ps, subj_vals = [], [], [], []
        for ci, pairs in enumerate(mid_cell_pairs):
            sv         = _per_subject_mean_from_3x3(ps_3x3_z, pairs)
            mu, sem, p = compute_condition_stats(sv)
            means.append(mu); sems.append(sem); ttest_ps.append(p)
            subj_vals.append(sv)
            raw_p_mid[ci].append((roi_key, p))
        mid_by_roi[roi_key] = dict(
            roi_name=roi_name, mean=means, sem=sems,
            sig=[False, False, False], subj_vals=subj_vals,
        )

    # FDR for mid-event
    if mid_by_roi:
        mid_all_p = [(rk, ci, p)
                     for ci, p_list in enumerate(raw_p_mid)
                     for rk, p in p_list]
        ps_mid    = np.array([t[2] for t in mid_all_p], dtype=float)
        p_fdr_mid = fdr_correct_ps(ps_mid)
        for idx, (rk, ci, _) in enumerate(mid_all_p):
            mid_by_roi[rk]['sig'][ci] = bool(p_fdr_mid[idx] < 0.05)

        # Per-ROI combined figures
        for roi_key, roi_name, _ in roi_spec_filtered:
            if roi_key not in bnd_win0_by_roi or roi_key not in mid_by_roi:
                continue
            fig = make_combined_window_bar_chart(
                roi_name, bnd_win0_by_roi[roi_key], mid_by_roi[roi_key])
            fp = combined_dir / f'roi-{roi_key}_boundary_vs_mid.png'
            fig.savefig(fp, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Combined bar chart → {fp.name}')

        # Multi-ROI combined figure
        MULTI_ROI_KEYS   = ['hipp', 'pmc', 'mpfc', 'dacc', 'dlpfc']
        MULTI_ROI_LABELS = ['Hippocampus', 'PMC', 'mPFC', 'dACC', 'dlPFC']
        multi_entries = [
            (label, bnd_win0_by_roi[k], mid_by_roi[k])
            for k, label in zip(MULTI_ROI_KEYS, MULTI_ROI_LABELS)
            if k in bnd_win0_by_roi and k in mid_by_roi
        ]
        if multi_entries:
            fig = make_multi_roi_combined_bar_chart(multi_entries)
            fp  = combined_dir / 'multi_roi_boundary_vs_mid.png'
            fig.savefig(fp, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Multi-ROI combined → {fp.name}')
    else:
        print('\nNo mid-event WSPC data available — combined_boundary_mid figures skipped.')

    # ------------------------------------------------------------------ #
    # Split-within figures                                                 #
    # ------------------------------------------------------------------ #
    if args.split_within:
        print('\n--- Generating split-within figures ---')
        split_dir = OUTPUT_DIR / 'split_within'
        split_dir.mkdir(parents=True, exist_ok=True)

        # Summary bar charts (per window)
        split_bar_dir = split_dir / 'summary_bar_charts'
        split_bar_dir.mkdir(parents=True, exist_ok=True)
        for win_idx, win_label in enumerate(_WIN_TR_SHORT):
            split_pairs = _split_cell_pairs_12x12(win_idx)
            roi_bar_data_s = []
            raw_p_s = [[], [], [], [], []]
            for roi_key, roi_name, _ in roi_spec_filtered:
                if roi_key not in all_per_subj_bnd:
                    continue
                ps_cond_z = collapse_per_subject_to_conditions(all_per_subj_bnd[roi_key])
                means, sems, ttest_ps, subj_vals = [], [], [], []
                for ci, pairs in enumerate(split_pairs):
                    sv         = _per_subject_mean_from_12x12(ps_cond_z, pairs)
                    mu, sem, p = compute_condition_stats(sv)
                    means.append(mu); sems.append(sem); ttest_ps.append(p)
                    subj_vals.append(np.tanh(sv))
                    raw_p_s[ci].append((roi_key, p))
                roi_bar_data_s.append(dict(
                    roi_key=roi_key,
                    abbrev=ROI_ABBREVS.get(roi_key, roi_key.upper()),
                    mean=means, sem=sems, p=ttest_ps,
                    sig=[False] * 5, subj_vals=subj_vals,
                ))

            all_p_s = [(rk, ci, p)
                       for ci, p_list in enumerate(raw_p_s)
                       for rk, p in p_list]
            p_fdr_s = fdr_correct_ps(np.array([t[2] for t in all_p_s], dtype=float))
            for idx, (rk, ci, _) in enumerate(all_p_s):
                for d in roi_bar_data_s:
                    if d['roi_key'] == rk:
                        d['sig'][ci] = bool(p_fdr_s[idx] < 0.05)

            fig = make_summary_bar_chart_split(roi_bar_data_s, win_label)
            bp  = split_bar_dir / f'wspc_split_{win_idx:02d}_{win_label.replace(" ", "_")}.png'
            fig.savefig(bp, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Split bar chart ({win_label}) → {bp.name}')

        # combined_boundary_mid with split
        if mid_by_roi:
            split_combined_dir = split_dir / 'summary_bar_charts' / 'combined_boundary_mid'
            split_combined_dir.mkdir(parents=True, exist_ok=True)

            split_pairs_win0 = _split_cell_pairs_12x12(0)
            split_pairs_mid  = _split_cell_pairs_3x3()

            bnd_split_by_roi = {}
            raw_p_bnd_s = [[], [], [], [], []]
            for roi_key, roi_name, _ in roi_spec_filtered:
                if roi_key not in all_per_subj_bnd:
                    continue
                ps_cond_z = collapse_per_subject_to_conditions(all_per_subj_bnd[roi_key])
                means, sems, ttest_ps, subj_vals = [], [], [], []
                for ci, pairs in enumerate(split_pairs_win0):
                    sv         = _per_subject_mean_from_12x12(ps_cond_z, pairs)
                    mu, sem, p = compute_condition_stats(sv)
                    means.append(mu); sems.append(sem); ttest_ps.append(p)
                    subj_vals.append(np.tanh(sv))
                    raw_p_bnd_s[ci].append((roi_key, p))
                bnd_split_by_roi[roi_key] = dict(
                    roi_name=roi_name, mean=means, sem=sems,
                    sig=[False] * 5, subj_vals=subj_vals)

            all_p_bs = [(rk, ci, p)
                        for ci, p_list in enumerate(raw_p_bnd_s)
                        for rk, p in p_list]
            p_fdr_bs = fdr_correct_ps(np.array([t[2] for t in all_p_bs], dtype=float))
            for idx, (rk, ci, _) in enumerate(all_p_bs):
                bnd_split_by_roi[rk]['sig'][ci] = bool(p_fdr_bs[idx] < 0.05)

            mid_split_by_roi = {}
            raw_p_mid_s = [[], [], [], [], []]
            for roi_key, roi_name, _ in roi_spec_filtered:
                if roi_key not in all_per_subj_mid:
                    continue
                ps_3x3_z = collapse_mid_per_subject(all_per_subj_mid[roi_key])
                means, sems, ttest_ps, subj_vals = [], [], [], []
                for ci, pairs in enumerate(split_pairs_mid):
                    sv         = _per_subject_mean_from_3x3(ps_3x3_z, pairs)
                    mu, sem, p = compute_condition_stats(sv)
                    means.append(mu); sems.append(sem); ttest_ps.append(p)
                    subj_vals.append(np.tanh(sv))
                    raw_p_mid_s[ci].append((roi_key, p))
                mid_split_by_roi[roi_key] = dict(
                    roi_name=roi_name, mean=means, sem=sems,
                    sig=[False] * 5, subj_vals=subj_vals)

            all_p_ms = [(rk, ci, p)
                        for ci, p_list in enumerate(raw_p_mid_s)
                        for rk, p in p_list]
            p_fdr_ms = fdr_correct_ps(np.array([t[2] for t in all_p_ms], dtype=float))
            for idx, (rk, ci, _) in enumerate(all_p_ms):
                mid_split_by_roi[rk]['sig'][ci] = bool(p_fdr_ms[idx] < 0.05)

            for roi_key, roi_name, _ in roi_spec_filtered:
                if roi_key not in bnd_split_by_roi or roi_key not in mid_split_by_roi:
                    continue
                fig = make_combined_window_bar_chart_split(
                    roi_name, bnd_split_by_roi[roi_key], mid_split_by_roi[roi_key])
                fp = split_combined_dir / f'roi-{roi_key}_boundary_vs_mid.png'
                fig.savefig(fp, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f'Split combined → {fp.name}')

            MULTI_ROI_KEYS   = ['hipp', 'pmc', 'mpfc', 'dacc', 'dlpfc']
            MULTI_ROI_LABELS = ['Hippocampus', 'PMC', 'mPFC', 'dACC', 'dlPFC']
            multi_s = [
                (label, bnd_split_by_roi[k], mid_split_by_roi[k])
                for k, label in zip(MULTI_ROI_KEYS, MULTI_ROI_LABELS)
                if k in bnd_split_by_roi and k in mid_split_by_roi
            ]
            if multi_s:
                fig = make_multi_roi_combined_bar_chart_split(multi_s)
                fp  = split_combined_dir / 'multi_roi_boundary_vs_mid.png'
                fig.savefig(fp, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f'Split multi-ROI combined → {fp.name}')

    print(f"\nDone.\n  Figures: {OUTPUT_DIR}\n  Cache: {RESULT_CACHE_DIR}")


if __name__ == '__main__':
    main()
