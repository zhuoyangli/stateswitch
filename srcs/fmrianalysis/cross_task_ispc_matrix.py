"""
cross_task_boundary_ispc.py — Cross-Task Boundary Inter-Subject Pattern Correlation

Computes a 12×12 ISPC matrix (4 HRF windows × 3 tasks: SVF, AHC, filmfest).
Each cell is the mean LOO inter-subject pattern correlation across all instance
pairs from the two conditions (e.g., SVF-pre vs AHC-onset).

All 29 trial instances (13 SVF categories + 8 AHC prompts + 8 filmfest movies)
are pooled per subject → (29 × 4 = 116 instance-conditions) → 116×116 LOO ISPC
matrix → collapsed to 12×12 by block-averaging.

Sub-001 has no filmfest data; filmfest conditions are NaN for that subject.
NaN-aware LOO ISPC handles partial subjects throughout.

Windows (HRF preset only):
  pre   : -6 to  3 TRs
  onset :  4 to 13 TRs
  post  : 14 to 23 TRs
  late  : 24 to 33 TRs

Usage:
    uv run python srcs/fmrianalysis/cross_task_boundary_ispc.py --roi pmc --no-cache
    uv run python srcs/fmrianalysis/cross_task_boundary_ispc.py --full-matrix
    uv run python srcs/fmrianalysis/cross_task_boundary_ispc.py --roi pmc ag --hp
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.stats import zscore as sp_zscore
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
RESULT_CACHE_DIR  = ANALYSIS_CACHE_DIR / 'cross_task_ispc'
OUTPUT_DIR        = FIGS_DIR / 'cross_task_ispc'

SMOOTH_FWHM = 6.0
HP_CUTOFF   = 0.01

# Window presets — set globally in main() based on --post-windows flag
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

WINDOWS       = WINDOW_PRESETS['hrf']['windows']
N_WINDOWS     = 4
N_FILM        = 8   # fixed: 4 boundaries per filmfest run × 2 runs
WINDOW_PRESET = 'hrf'

# TR labels shifted -4 TRs to account for HRF delay (updated in main)
_WIN_TR_SHORT = WINDOW_PRESETS['hrf']['tr_labels']
WINDOW_LABELS = _WIN_TR_SHORT

# Set dynamically in main() after trial discovery
N_SVF        = 0
N_AHC        = 0
N_INSTANCES  = 0   # N_SVF + N_AHC + N_FILM
N_TOTAL      = 0   # N_INSTANCES * N_WINDOWS
N_CONDITIONS = 12  # always 4 windows × 3 tasks

# Task index slices into the N_INSTANCES-length axis (set in main)
TASK_SLICES: dict = {}

# Onset mode: use trial start times instead of end times (set in main)
USE_ONSET = False

ROI_SPEC = [
    ('eac',   'Early Auditory Cortex',     get_bilateral_ids(EARLY_AUDITORY)),
    ('evc',   'Early Visual Cortex',       get_bilateral_ids(EARLY_VISUAL)),
    ('pmc',   'Posterior Medial Cortex',   get_bilateral_ids(POSTERIOR_MEDIAL)),
    ('ag',    'Angular Gyrus',             get_bilateral_ids(ANGULAR_GYRUS)),
    ('dlpfc', 'Dorsolateral PFC',          get_bilateral_ids(DLPFC)),
    ('dacc',  'Dorsal Anterior Cingulate', get_bilateral_ids(DACC)),
    ('mpfc',  'Medial Prefrontal Cortex',  get_bilateral_ids(MPFC)),
    ('hipp',  'Hippocampus',               None),
]

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
# STEP 1: TRIAL OFFSET DISCOVERY (copied from per-task scripts)
# ============================================================================

# --- SVF ---

def get_svf_offsets_by_category(subject, session):
    """Return {category: [offset_TR, ...]} for all trials in session."""
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
        cat        = row['category_name']
        col        = 'svf_trial.started' if USE_ONSET else 'svf_trial.stopped'
        offset_sec = row[col] - first_start
        tr         = int(round(offset_sec / TR))
        result.setdefault(cat, []).append(tr)
    return result


def discover_shared_categories(subjects, min_subjects=5):
    """Return categories present in ≥ min_subjects subjects."""
    from collections import Counter
    cat_count: Counter = Counter()
    for subj in subjects:
        sessions_tasks = discover_svf_ahc_sessions(subj)
        subj_cats: set = set()
        for session, t in sessions_tasks:
            if t != 'svf':
                continue
            offsets = get_svf_offsets_by_category(subj, session)
            subj_cats.update(offsets.keys())
        for cat in subj_cats:
            cat_count[cat] += 1
    shared = [cat for cat, n in cat_count.items() if n >= min_subjects]
    shared.sort(key=lambda c: (-cat_count[c], c))
    return shared, dict(cat_count)


# --- AHC ---

def _find_ahc_sentences(subject, session):
    p = AHC_SENTENCES_DIR / f'{subject}_{session}_task-ahc_desc-sentences.xlsx'
    return p if p.exists() else None


def get_ahc_offsets_by_prompt(subject, session):
    """Return {prompt_text: [offset_TR, ...]} for all prompts in session."""
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
        if len(rows) != len(prompts):
            print(f"  WARNING {subject} {session}: "
                  f"{len(prompts)} prompts vs {len(rows)} trials — skipping")
        return {}

    first_start = rows['ahc_trial.started'].iloc[0]
    result = {}
    col = 'ahc_trial.started' if USE_ONSET else 'ahc_trial.stopped'
    for prompt, (_, row) in zip(prompts, rows.iterrows()):
        offset_sec = row[col] - first_start
        tr         = int(round(offset_sec / TR))
        result.setdefault(prompt, []).append(tr)
    return result


def discover_shared_prompts(subjects, min_subjects=5):
    """Return prompts present in ≥ min_subjects subjects."""
    from collections import Counter
    prompt_count: Counter = Counter()
    for subj in subjects:
        sessions_tasks = discover_svf_ahc_sessions(subj)
        subj_prompts: set = set()
        for session, t in sessions_tasks:
            if t != 'ahc':
                continue
            offsets = get_ahc_offsets_by_prompt(subj, session)
            subj_prompts.update(offsets.keys())
        for prompt in subj_prompts:
            prompt_count[prompt] += 1
    shared = [p for p, n in prompt_count.items() if n >= min_subjects]
    shared.sort(key=lambda p: (-prompt_count[p], p))
    return shared, dict(prompt_count)


# --- Filmfest ---

def _mss_to_seconds(mss):
    """Convert m.ss timestamp to total seconds."""
    minutes = int(mss)
    seconds = round((float(mss) - minutes) * 100)
    return minutes * 60 + seconds


def get_all_movie_onsets(task):
    """Return start times (seconds) for all 5 movies in one filmfest run."""
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    onsets = []
    for movie in movies:
        df   = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        first_start_raw = segb['Start Time (m.ss)'].values[0]
        onsets.append(_mss_to_seconds(first_start_raw))
    return onsets  # list of 5 floats


# ============================================================================
# STEP 2: VOXEL EXTRACTION & CACHING (shared logic)
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
# STEP 3: BOUNDARY-LOCKED PATTERN EXTRACTION
# ============================================================================

def extract_pattern_at_tr(voxel_data, boundary_tr):
    """Extract (N_WINDOWS, V) pattern around one boundary. Returns None if OOB."""
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
    """Extract (N_valid, N_WINDOWS, V) for a list of boundary TRs."""
    T = voxel_data.shape[0]
    valid_patterns = []
    valid_mask = []
    for btr in boundary_trs:
        win_patterns = []
        ok = True
        for _, tr_start_rel, tr_end_rel in WINDOWS:
            start = btr + tr_start_rel
            end   = btr + tr_end_rel
            if start < 0 or end >= T:
                ok = False
                break
            win_patterns.append(voxel_data[start:end + 1].mean(axis=0))
        valid_mask.append(ok)
        if ok:
            valid_patterns.append(np.stack(win_patterns, axis=0))
    if not valid_patterns:
        return np.empty((0, N_WINDOWS, voxel_data.shape[1])), np.array(valid_mask)
    return np.stack(valid_patterns, axis=0), np.array(valid_mask)


# ============================================================================
# STEP 4: PER-TASK PATTERN GETTERS
# ============================================================================

def get_subject_svf_patterns(subject, roi_key, parcel_ids, shared_cats, do_hp=False):
    """Build (N_SVF, N_WINDOWS, V) for shared SVF categories. NaN for missing."""
    sessions_tasks = discover_svf_ahc_sessions(subject)
    svf_sessions   = [ses for ses, t in sessions_tasks if t == 'svf']

    cat_patterns: dict[str, list] = {c: [] for c in shared_cats}

    for session in svf_sessions:
        offsets_by_cat    = get_svf_offsets_by_category(subject, session)
        shared_in_session = [c for c in shared_cats if c in offsets_by_cat]
        if not shared_in_session:
            continue
        try:
            raw        = load_roi_voxels(subject, session, 'svf', roi_key, parcel_ids)
            voxel_data = preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {session} svf roi={roi_key}: {e}")
            continue
        for cat in shared_in_session:
            for btr in offsets_by_cat[cat]:
                pats = extract_pattern_at_tr(voxel_data, btr)
                if pats is not None:
                    cat_patterns[cat].append(pats)

    if not any(cat_patterns.values()):
        return None

    V = next(p[0].shape[1] for p in cat_patterns.values() if p)
    full = np.full((N_SVF, N_WINDOWS, V), np.nan, dtype=np.float64)
    for ci, cat in enumerate(shared_cats):
        if cat_patterns[cat]:
            full[ci] = np.mean(cat_patterns[cat], axis=0)
    return full


def get_subject_ahc_patterns(subject, roi_key, parcel_ids, shared_prompts, do_hp=False):
    """Build (N_AHC, N_WINDOWS, V) for shared AHC prompts. NaN for missing."""
    sessions_tasks = discover_svf_ahc_sessions(subject)
    ahc_sessions   = [ses for ses, t in sessions_tasks if t == 'ahc']

    prompt_patterns: dict[str, list] = {p: [] for p in shared_prompts}

    for session in ahc_sessions:
        offsets_by_prompt = get_ahc_offsets_by_prompt(subject, session)
        shared_in_session = [p for p in shared_prompts if p in offsets_by_prompt]
        if not shared_in_session:
            continue
        try:
            raw        = load_roi_voxels(subject, session, 'ahc', roi_key, parcel_ids)
            voxel_data = preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {session} ahc roi={roi_key}: {e}")
            continue
        for prompt in shared_in_session:
            for btr in offsets_by_prompt[prompt]:
                pats = extract_pattern_at_tr(voxel_data, btr)
                if pats is not None:
                    prompt_patterns[prompt].append(pats)

    if not any(prompt_patterns.values()):
        return None

    V = next(p[0].shape[1] for p in prompt_patterns.values() if p)
    full = np.full((N_AHC, N_WINDOWS, V), np.nan, dtype=np.float64)
    for pi, prompt in enumerate(shared_prompts):
        if prompt_patterns[prompt]:
            full[pi] = np.mean(prompt_patterns[prompt], axis=0)
    return full


def get_subject_filmfest_patterns(subject, roi_key, parcel_ids, do_hp=False):
    """Build (N_FILM=8, N_WINDOWS, V) for filmfest movie boundaries. NaN for missing."""
    if subject not in FILMFEST_SUBJECTS:
        return None

    session      = FILMFEST_SUBJECTS[subject]
    N_PER_RUN    = 4
    all_patterns = []
    all_valid    = []

    for task in ('filmfest1', 'filmfest2'):
        try:
            raw = load_roi_voxels(subject, session, task, roi_key, parcel_ids)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {task} roi={roi_key}: {e}")
            all_patterns.append(None)
            all_valid.extend([False] * N_PER_RUN)
            continue
        voxel_data   = preprocess_voxels(raw, do_hp=do_hp)
        onset_secs   = get_all_movie_onsets(task)[1:]  # skip first movie
        if USE_ONSET:
            onset_secs = [s + 6.0 for s in onset_secs]  # skip 6s title scene
        boundary_trs = [int(round(s / TR)) for s in onset_secs]
        pats, vmask  = extract_boundary_patterns(voxel_data, boundary_trs)
        all_patterns.append(pats)
        all_valid.extend(vmask.tolist())

    valid_parts = [p for p in all_patterns if p is not None and p.shape[0] > 0]
    if not valid_parts:
        return None

    V = valid_parts[0].shape[2]
    full = np.full((N_FILM, N_WINDOWS, V), np.nan, dtype=np.float64)
    run_valid_idx = 0
    for task_idx, (task, pats) in enumerate(zip(('filmfest1', 'filmfest2'), all_patterns)):
        if pats is None or pats.shape[0] == 0:
            run_valid_idx += N_PER_RUN
            continue
        valid_flags = all_valid[run_valid_idx:run_valid_idx + N_PER_RUN]
        valid_count = 0
        for bi, is_valid in enumerate(valid_flags):
            movie_idx = task_idx * N_PER_RUN + bi
            if is_valid and valid_count < pats.shape[0]:
                full[movie_idx] = pats[valid_count]
                valid_count += 1
        run_valid_idx += N_PER_RUN

    return full


def get_subject_cross_task_patterns(subject, roi_key, parcel_ids,
                                     shared_cats, shared_prompts, do_hp=False):
    """Build (N_INSTANCES, N_WINDOWS, V) concatenating SVF + AHC + filmfest.

    Missing tasks → NaN rows. Returns None if no task has data.
    """
    svf_pats  = get_subject_svf_patterns(subject, roi_key, parcel_ids, shared_cats, do_hp)
    ahc_pats  = get_subject_ahc_patterns(subject, roi_key, parcel_ids, shared_prompts, do_hp)
    film_pats = get_subject_filmfest_patterns(subject, roi_key, parcel_ids, do_hp)

    # Determine V from first available task
    V = None
    for p in (svf_pats, ahc_pats, film_pats):
        if p is not None:
            V = p.shape[2]
            break
    if V is None:
        return None

    # Guard against V mismatch across tasks (use minimum)
    Vs = [p.shape[2] for p in (svf_pats, ahc_pats, film_pats) if p is not None]
    V = min(Vs)

    full = np.full((N_INSTANCES, N_WINDOWS, V), np.nan, dtype=np.float64)
    if svf_pats is not None:
        full[TASK_SLICES['svf']] = svf_pats[..., :V]
    if ahc_pats is not None:
        full[TASK_SLICES['ahc']] = ahc_pats[..., :V]
    if film_pats is not None:
        full[TASK_SLICES['film']] = film_pats[..., :V]

    return full


def flatten_cross_task_patterns(patterns):
    """(N_INSTANCES, N_WINDOWS, V) → (N_TOTAL, V), instances nested in windows."""
    return patterns.transpose(1, 0, 2).reshape(N_TOTAL, -1)


# ============================================================================
# STEP 5: NaN-AWARE LOO ISPC
# ============================================================================

def compute_group_ispc(all_patterns):
    """NaN-aware LOO ISPC. all_patterns: (N_subj, N_TOTAL, V).

    Returns group_ispc (N_TOTAL, N_TOTAL), per_subj (N_subj, N_TOTAL, N_TOTAL).
    """
    N = all_patterns.shape[0]

    valid_subj   = ~np.all(np.isnan(all_patterns), axis=2)  # (N_subj, N_TOTAL)
    valid_count  = valid_subj.sum(axis=0)                    # (N_TOTAL,)
    group_nansum = np.nansum(all_patterns, axis=0)           # (N_TOTAL, V)

    per_subj_ispc = np.full((N, N_TOTAL, N_TOTAL), np.nan)

    for i in range(N):
        loo_mean = np.full_like(group_nansum, np.nan)
        for c in range(N_TOTAL):
            n = valid_count[c]
            if valid_subj[i, c]:
                n_others = n - 1
                if n_others < 1:
                    continue
                loo_mean[c] = (group_nansum[c] - all_patterns[i, c]) / n_others
            else:
                if n < 1:
                    continue
                loo_mean[c] = group_nansum[c] / n

        corr_mat = np.full((N_TOTAL, N_TOTAL), np.nan)
        for row in range(N_TOTAL):
            a = all_patterns[i, row]
            if np.all(np.isnan(a)):
                continue
            for col in range(N_TOTAL):
                b = loo_mean[col]
                if np.all(np.isnan(b)):
                    continue
                valid = ~(np.isnan(a) | np.isnan(b))
                if valid.sum() < 2:
                    continue
                a_, b_ = a[valid] - a[valid].mean(), b[valid] - b[valid].mean()
                denom  = np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum())
                if denom == 0:
                    continue
                corr_mat[row, col] = np.dot(a_, b_) / denom
        per_subj_ispc[i] = corr_mat

    z_stack    = np.arctanh(np.clip(per_subj_ispc, -0.999, 0.999))
    group_ispc = np.tanh(np.nanmean(z_stack, axis=0))
    return group_ispc, per_subj_ispc


# ============================================================================
# STEP 6: COLLAPSE TO 12×12 CONDITION MATRIX
# ============================================================================

def _build_cond_instance_indices():
    """Return list of 12 index arrays mapping conditions → rows in N_TOTAL-vector.

    Condition order: SVF-pre, SVF-onset, SVF-post, SVF-late,
                     AHC-pre, AHC-onset, AHC-post, AHC-late,
                     Film-pre, Film-onset, Film-post, Film-late.

    In N_TOTAL-vector, instances are nested within windows:
      [win0: svf_0..svf_{N_SVF-1}, ahc_0..ahc_{N_AHC-1}, film_0..film_{N_FILM-1},
       win1: same order, ...]
    """
    task_sizes  = {'svf': N_SVF, 'ahc': N_AHC, 'film': N_FILM}
    task_starts = {'svf': 0, 'ahc': N_SVF, 'film': N_SVF + N_AHC}
    cond_indices = []
    for task in ('svf', 'ahc', 'film'):
        for win_idx in range(N_WINDOWS):
            start = win_idx * N_INSTANCES + task_starts[task]
            end   = start + task_sizes[task]
            cond_indices.append(np.arange(start, end))
    return cond_indices


def collapse_to_condition_matrix(instance_ispc):
    """Collapse N_TOTAL×N_TOTAL instance ISPC to 12×12 condition matrix.

    Each cell is the mean of all ISPC values in the corresponding block.
    """
    cond_indices = _build_cond_instance_indices()
    cond_matrix  = np.full((N_CONDITIONS, N_CONDITIONS), np.nan)

    for ci, idx_a in enumerate(cond_indices):
        for cj, idx_b in enumerate(cond_indices):
            block = instance_ispc[np.ix_(idx_a, idx_b)]
            vals  = block.ravel()
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                cond_matrix[ci, cj] = valid.mean()

    return cond_matrix


def collapse_per_subject_to_conditions(per_subj_ispc):
    """Collapse (N_subj, N_TOTAL, N_TOTAL) → (N_subj, 12, 12) Fisher-z values."""
    N_subj = per_subj_ispc.shape[0]
    cond_indices = _build_cond_instance_indices()
    z_stack = np.arctanh(np.clip(per_subj_ispc, -0.999, 0.999))
    out = np.full((N_subj, N_CONDITIONS, N_CONDITIONS), np.nan)
    for ci, idx_a in enumerate(cond_indices):
        for cj, idx_b in enumerate(cond_indices):
            block = z_stack[:, idx_a, :][:, :, idx_b]   # (N_subj, len_a, len_b)
            with np.errstate(all='ignore'):
                out[:, ci, cj] = np.nanmean(block.reshape(N_subj, -1), axis=1)
    return out


def compute_tvalue_matrix(per_subj_cond_z):
    """1-sample t-test against 0 on per-subject Fisher-z condition matrices.

    per_subj_cond_z: (N_subj, 12, 12) Fisher-z values.
    Returns tval_matrix (12, 12).
    """
    from scipy.stats import ttest_1samp
    tval = np.full((N_CONDITIONS, N_CONDITIONS), np.nan)
    for ci in range(N_CONDITIONS):
        for cj in range(N_CONDITIONS):
            vals = per_subj_cond_z[:, ci, cj]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 2:
                t, _ = ttest_1samp(valid, popmean=0)
                tval[ci, cj] = t
    return tval


# ============================================================================
# STEP 7: FIGURES
# ============================================================================

_COND_LABELS = [f'{task}\n{w}' for task in ('SVF', 'AHC', 'Film')
                for w in _WIN_TR_SHORT]


def make_condition_figure(cond_matrix, roi_name, vmax=0.3):
    """12×12 heatmap of cross-task condition ISPC."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(left=0.18, right=0.88, top=0.88, bottom=0.22)

    im = ax.imshow(cond_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal', origin='upper', interpolation='none')

    ax.set_xticks(range(N_CONDITIONS))
    ax.set_xticklabels(_COND_LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(N_CONDITIONS))
    ax.set_yticklabels(_COND_LABELS, fontsize=7)

    # Fine gridlines between cells
    for pos in np.arange(0.5, N_CONDITIONS - 1, 1):
        ax.axhline(pos, color='white', linewidth=0.4, alpha=0.6)
        ax.axvline(pos, color='white', linewidth=0.4, alpha=0.6)

    # Task block dividers
    for edge in [3.5, 7.5]:
        ax.axhline(edge, color='black', linewidth=2.0)
        ax.axvline(edge, color='black', linewidth=2.0)

    # pre/onset boundary within each task block (red bold) — only when first window is pre-boundary
    if WINDOWS[0][1] < 0:
        for task_start in [0, 4, 8]:
            edge = task_start + 0.5
            ax.axhline(edge, color='red', linewidth=1.5)
            ax.axvline(edge, color='red', linewidth=1.5)

    # Highlight all 0–9 TRs × 0–9 TRs cells (diagonal and cross-task) when all windows are post-boundary
    if WINDOWS[0][1] >= 0:
        from matplotlib.patches import Rectangle
        for row_start in [0, 4, 8]:
            for col_start in [0, 4, 8]:
                ax.add_patch(Rectangle((col_start - 0.5, row_start - 0.5), 1, 1,
                                       fill=False, edgecolor='red', linewidth=2.0, zorder=5))

    ax.set_title(roi_name, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04, shrink=0.7)
    cbar.set_label('inter-subject pattern correlation (r)', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    return fig


def make_condition_tvalue_figure(tval_matrix, roi_name, tval_vmax=5.0):
    """12×12 heatmap of 1-sample t-values (vs 0) for cross-task condition ISPC."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(left=0.18, right=0.88, top=0.88, bottom=0.22)

    im = ax.imshow(tval_matrix, cmap='RdBu_r', vmin=-tval_vmax, vmax=tval_vmax,
                   aspect='equal', origin='upper', interpolation='none')

    ax.set_xticks(range(N_CONDITIONS))
    ax.set_xticklabels(_COND_LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(N_CONDITIONS))
    ax.set_yticklabels(_COND_LABELS, fontsize=7)

    for pos in np.arange(0.5, N_CONDITIONS - 1, 1):
        ax.axhline(pos, color='white', linewidth=0.4, alpha=0.6)
        ax.axvline(pos, color='white', linewidth=0.4, alpha=0.6)

    for edge in [3.5, 7.5]:
        ax.axhline(edge, color='black', linewidth=2.0)
        ax.axvline(edge, color='black', linewidth=2.0)

    if WINDOWS[0][1] < 0:
        for task_start in [0, 4, 8]:
            edge = task_start + 0.5
            ax.axhline(edge, color='red', linewidth=1.5)
            ax.axvline(edge, color='red', linewidth=1.5)

    if WINDOWS[0][1] >= 0:
        from matplotlib.patches import Rectangle
        for row_start in [0, 4, 8]:
            for col_start in [0, 4, 8]:
                ax.add_patch(Rectangle((col_start - 0.5, row_start - 0.5), 1, 1,
                                       fill=False, edgecolor='red', linewidth=2.0, zorder=5))

    ax.set_title(roi_name, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04, shrink=0.7)
    cbar.set_label('t-value (1-sample vs 0)', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    return fig


def make_combined_tvalue_figure(tval_by_roi, tval_vmax=5.0):
    """All ROIs laid out horizontally, showing t-values."""
    from matplotlib.transforms import blended_transform_factory

    n_rois    = sum(1 for k, _, _ in ROI_SPEC if k in tval_by_roi)
    fig, axes = plt.subplots(1, n_rois, figsize=(3.5 * n_rois + 2.0, 5.5))
    fig.subplots_adjust(left=0.08, right=0.90, top=0.85, bottom=0.18, wspace=0.15)
    axes_flat = np.array(axes).flatten()

    task_labels  = ['SVF', 'AHC', 'Filmfest']
    task_centers = [1.5, 5.5, 9.5]
    win_short    = _WIN_TR_SHORT

    last_im  = None
    plot_idx = 0
    for roi_key, roi_name, _ in ROI_SPEC:
        mat = tval_by_roi.get(roi_key)
        if mat is None:
            continue
        ax = axes_flat[plot_idx]

        im = ax.imshow(mat, cmap='RdBu_r', vmin=-tval_vmax, vmax=tval_vmax,
                       aspect='equal', origin='upper', interpolation='none')
        last_im = im

        for edge in [3.5, 7.5]:
            ax.axhline(edge, color='black', linewidth=1.8)
            ax.axvline(edge, color='black', linewidth=1.8)

        if WINDOWS[0][1] < 0:
            for task_start in [0, 4, 8]:
                ax.axhline(task_start + 0.5, color='red', linewidth=1.2)
                ax.axvline(task_start + 0.5, color='red', linewidth=1.2)

        if WINDOWS[0][1] >= 0:
            from matplotlib.patches import Rectangle
            for row_start in [0, 4, 8]:
                for col_start in [0, 4, 8]:
                    ax.add_patch(Rectangle((col_start - 0.5, row_start - 0.5), 1, 1,
                                           fill=False, edgecolor='red', linewidth=1.5, zorder=5))

        ax.set_xticks(range(N_CONDITIONS))
        ax.set_xticklabels(win_short * 3, fontsize=4.5, rotation=90)
        ax.xaxis.set_tick_params(length=1)

        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks(task_centers)
        ax2.set_xticklabels(task_labels, fontsize=5.5, ha='center')
        ax2.tick_params(length=0)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        if plot_idx == 0:
            ax.set_yticks(range(N_CONDITIONS))
            ax.set_yticklabels(win_short * 3, fontsize=4.5)
        else:
            ax.set_yticks([])

        ax.set_title(roi_name, fontsize=8.5, fontweight='bold', pad=4)
        plot_idx += 1

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_flat[:plot_idx].tolist(),
                            fraction=0.012, pad=0.01, shrink=0.6)
        cbar.set_label('t-value (1-sample vs 0)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle('Inter-subject pattern correlation — t-values across tasks (SVF, AHC, filmfest)\n'
                 '(TR windows are shifted by 4 TRs to account for HRF delay)',
                 fontsize=10, fontweight='bold', y=0.98)
    return fig


def make_instance_figure(instance_ispc, roi_name, vmax=0.3):
    """N_TOTAL×N_TOTAL instance-level ISPC heatmap."""
    fig, ax = plt.subplots(figsize=(12, 11))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    im = ax.imshow(instance_ispc, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal', origin='upper', interpolation='none')

    ax.set_xticks([]); ax.set_yticks([])

    # Task block edges within each window block
    task_sizes = [N_SVF, N_AHC, N_FILM]
    for win_idx in range(N_WINDOWS):
        win_start = win_idx * N_INSTANCES
        cumulative = 0
        for ts in task_sizes[:-1]:
            cumulative += ts
            edge = win_start + cumulative - 0.5
            ax.axhline(edge, color='black', linewidth=0.8)
            ax.axvline(edge, color='black', linewidth=0.8)

    # Window block dividers (black thick) except pre/onset boundary (red bold)
    for win_idx in range(1, N_WINDOWS):
        edge  = win_idx * N_INSTANCES - 0.5
        color = 'red' if win_idx == 1 else 'black'
        ax.axhline(edge, color=color, linewidth=2.0)
        ax.axvline(edge, color=color, linewidth=2.0)

    # Window labels on top and left
    from matplotlib.transforms import blended_transform_factory
    win_centers = [(win_idx * N_INSTANCES + N_INSTANCES / 2 - 0.5) for win_idx in range(N_WINDOWS)]
    ax2_top = ax.secondary_xaxis('top')
    ax2_top.set_xticks(win_centers)
    ax2_top.set_xticklabels(WINDOW_LABELS, fontsize=7)
    ax2_top.tick_params(length=0)
    for spine in ax2_top.spines.values():
        spine.set_visible(False)

    blend = blended_transform_factory(ax.transAxes, ax.transData)
    for data_y, label in zip(win_centers, WINDOW_LABELS):
        ax.text(-0.02, data_y, label, ha='center', va='center',
                fontsize=6, rotation=90, rotation_mode='anchor',
                transform=blend, clip_on=False)

    ax.set_title(roi_name, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.03, shrink=0.6)
    cbar.set_label('inter-subject pattern correlation (r)', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    return fig


def _task_sort_permutation():
    """Return index permutation to reorder N_TOTAL vector from window-first to task-first.

    Window-first (current): [win0: svf0..svfN, ahc0..ahcN, film0..filmN, win1: ...]
    Task-first (new):        [svf: win0_cats, win1_cats, win2_cats, win3_cats,
                              ahc: win0_prompts, ...,
                              film: win0_movies, win1_movies, win2_movies, win3_movies]

    Within each task block, windows are outer and instances are inner — matching
    the flatten_patterns() convention in filmfest_boundary_ispc.py so the
    filmfest×filmfest sub-block is directly comparable.
    """
    task_offsets = [0, N_SVF, N_SVF + N_AHC]
    task_sizes   = [N_SVF, N_AHC, N_FILM]
    perm = []
    for t_off, t_size in zip(task_offsets, task_sizes):
        for win in range(N_WINDOWS):
            for inst in range(t_size):
                perm.append(win * N_INSTANCES + t_off + inst)
    return np.array(perm)


def make_instance_figure_by_task(instance_ispc, roi_name, vmax=0.3):
    """N_TOTAL×N_TOTAL instance-level ISPC heatmap, reordered task-first."""
    perm = _task_sort_permutation()
    mat  = instance_ispc[np.ix_(perm, perm)]

    fig, ax = plt.subplots(figsize=(12, 11))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal', origin='upper', interpolation='none')
    ax.set_xticks([]); ax.set_yticks([])

    # Task block sizes in task-first order
    task_sizes  = [N_SVF * N_WINDOWS, N_AHC * N_WINDOWS, N_FILM * N_WINDOWS]
    task_labels = ['SVF', 'AHC', 'Filmfest']

    # Task dividers (thickest)
    cumulative = 0
    for ts in task_sizes[:-1]:
        cumulative += ts
        ax.axhline(cumulative - 0.5, color='black', linewidth=4.0)
        ax.axvline(cumulative - 0.5, color='black', linewidth=4.0)

    # Window boundaries within each task; pre/onset is red, others thin black
    cumulative = 0
    for ts in task_sizes:
        n_inst = ts // N_WINDOWS
        for w in range(1, N_WINDOWS):
            edge  = cumulative + w * n_inst - 0.5
            if w == 1 and WINDOWS[0][1] < 0:
                ax.axhline(edge, color='red', linewidth=2.0)
                ax.axvline(edge, color='red', linewidth=2.0)
            else:
                ax.axhline(edge, color='black', linewidth=0.8)
                ax.axvline(edge, color='black', linewidth=0.8)
        cumulative += ts

    # Task labels on top and left
    from matplotlib.transforms import blended_transform_factory
    task_centers = []
    cumulative = 0
    for ts in task_sizes:
        task_centers.append(cumulative + ts / 2 - 0.5)
        cumulative += ts

    ax2_top = ax.secondary_xaxis('top')
    ax2_top.set_xticks(task_centers)
    ax2_top.set_xticklabels(task_labels, fontsize=9)
    ax2_top.tick_params(length=0)
    for spine in ax2_top.spines.values():
        spine.set_visible(False)

    blend = blended_transform_factory(ax.transAxes, ax.transData)
    for data_y, label in zip(task_centers, task_labels):
        ax.text(-0.02, data_y, label, ha='center', va='center',
                fontsize=8, rotation=90, rotation_mode='anchor',
                transform=blend, clip_on=False)

    ax.set_title(roi_name, fontsize=11, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.03, shrink=0.6)
    cbar.set_label('inter-subject pattern correlation (r)', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    return fig


def make_combined_figure(cond_by_roi, vmax=0.3):
    """All ROIs laid out horizontally, one panel per ROI."""
    from matplotlib.transforms import blended_transform_factory

    n_rois    = sum(1 for k, _, _ in ROI_SPEC if k in cond_by_roi)
    fig, axes = plt.subplots(1, n_rois, figsize=(3.5 * n_rois + 2.0, 5.5))
    fig.subplots_adjust(left=0.08, right=0.90, top=0.85, bottom=0.18, wspace=0.15)
    axes_flat = np.array(axes).flatten()

    # Task block centers and labels for secondary x-axis
    task_sizes  = [4, 4, 4]  # windows per task (SVF, AHC, Film)
    task_labels = ['SVF', 'AHC', 'Filmfest']
    task_centers = [1.5, 5.5, 9.5]

    win_short = _WIN_TR_SHORT

    last_im = None
    plot_idx = 0
    for roi_key, roi_name, _ in ROI_SPEC:
        mat = cond_by_roi.get(roi_key)
        if mat is None:
            continue
        ax = axes_flat[plot_idx]

        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       aspect='equal', origin='upper', interpolation='none')
        last_im = im

        # Task block dividers
        for edge in [3.5, 7.5]:
            ax.axhline(edge, color='black', linewidth=1.8)
            ax.axvline(edge, color='black', linewidth=1.8)

        # pre/onset boundary (red) — only when first window is pre-boundary
        if WINDOWS[0][1] < 0:
            for task_start in [0, 4, 8]:
                ax.axhline(task_start + 0.5, color='red', linewidth=1.2)
                ax.axvline(task_start + 0.5, color='red', linewidth=1.2)

        # Highlight all 0–9 TRs × 0–9 TRs cells (diagonal and cross-task) when all windows are post-boundary
        if WINDOWS[0][1] >= 0:
            from matplotlib.patches import Rectangle
            for row_start in [0, 4, 8]:
                for col_start in [0, 4, 8]:
                    ax.add_patch(Rectangle((col_start - 0.5, row_start - 0.5), 1, 1,
                                           fill=False, edgecolor='red', linewidth=1.5, zorder=5))

        # Bottom x-axis: window labels (rotated), every panel
        ax.set_xticks(range(N_CONDITIONS))
        ax.set_xticklabels(win_short * 3, fontsize=4.5, rotation=90)
        ax.xaxis.set_tick_params(length=1)

        # Top secondary axis: task block labels
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks(task_centers)
        ax2.set_xticklabels(task_labels, fontsize=5.5, ha='center')
        ax2.tick_params(length=0)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        # Y-axis: first panel only gets row labels
        if plot_idx == 0:
            ax.set_yticks(range(N_CONDITIONS))
            ax.set_yticklabels(win_short * 3, fontsize=4.5)
        else:
            ax.set_yticks([])

        ax.set_title(roi_name, fontsize=8.5, fontweight='bold', pad=4)
        plot_idx += 1

    # Single shared colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_flat[:plot_idx].tolist(),
                            fraction=0.012, pad=0.01, shrink=0.6)
        cbar.set_label('inter-subject pattern correlation (r)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle('Inter-subject pattern correlation between peri-boundary windows across tasks (SVF, AHC, filmfest)\n'
                 '(TR windows are shifted by 4 TRs to account for HRF delay)',
                 fontsize=10, fontweight='bold', y=0.98)
    return fig


# ============================================================================
# PER-ROI PIPELINE
# ============================================================================

def run_roi(roi_key, roi_name, parcel_ids, subjects,
            shared_cats, shared_prompts,
            do_hp=False, force_ispc=False, full_matrix=False, vmax=0.3,
            compute_tval=False, tval_vmax=5.0):
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hp_tag     = '_hp' if do_hp else ''
    onset_tag  = '_onset' if USE_ONSET else ''
    ispc_cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6{hp_tag}{onset_tag}_{WINDOW_PRESET}_ispc.npz'

    if not force_ispc and ispc_cache.exists():
        print(f"  [{roi_key}] Loading from cache ...")
        data          = np.load(ispc_cache)
        instance_ispc = data['instance_ispc']
        cond_matrix   = data['cond_matrix']
        per_subj      = data['per_subject'] if 'per_subject' in data else None
    else:
        print(f"\n{'='*60}\nROI: {roi_name} ({roi_key})")
        subject_patterns = []
        for subj in subjects:
            print(f"  Subject: {subj}")
            pats = get_subject_cross_task_patterns(
                subj, roi_key, parcel_ids, shared_cats, shared_prompts, do_hp)
            if pats is None:
                print(f"  SKIP {subj}: no data for any task")
                continue
            flat    = flatten_cross_task_patterns(pats)
            n_valid = (~np.isnan(flat).all(axis=1)).sum()
            print(f"    {n_valid}/{N_TOTAL} instance-conditions valid")
            subject_patterns.append(flat)

        if len(subject_patterns) < 2:
            print(f"  [{roi_key}] Not enough subjects, skipping.")
            return None, None

        all_patterns = np.stack(subject_patterns, axis=0)
        print(f"  [{roi_key}] Computing ISPC ({all_patterns.shape[0]} subjects, {N_TOTAL} conditions)...")
        instance_ispc, per_subj = compute_group_ispc(all_patterns)
        cond_matrix = collapse_to_condition_matrix(instance_ispc)

        np.savez_compressed(ispc_cache,
                            instance_ispc=instance_ispc,
                            cond_matrix=cond_matrix,
                            per_subject=per_subj,
                            subjects=np.array(subjects),
                            do_hp=do_hp)
        print(f"  [{roi_key}] Cached → {ispc_cache.name}")

    hp_label   = 'hp' if do_hp else 'no_hp'
    onset_label = 'onset' if USE_ONSET else 'offset'
    hp_sub     = OUTPUT_DIR / hp_label / WINDOW_PRESET / onset_label

    # Condition-level figure — window-sorted (original cache ordering)
    by_win_dir = hp_sub / 'collapsed_by_window'
    by_win_dir.mkdir(parents=True, exist_ok=True)
    fig = make_condition_figure(cond_matrix, roi_name, vmax=vmax)
    fig_path = by_win_dir / f'roi-{roi_key}_cross_task_ispc.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{roi_key}] By-window figure → {fig_path}")

    # Task-sorted condition figure — always recomputed fresh from instance_ispc
    cond_by_task = collapse_to_condition_matrix(instance_ispc)
    by_task_dir  = hp_sub / 'collapsed_by_task'
    by_task_dir.mkdir(parents=True, exist_ok=True)
    fig = make_condition_figure(cond_by_task, roi_name, vmax=vmax)
    fig_path_bt = by_task_dir / f'roi-{roi_key}_cross_task_ispc.png'
    fig.savefig(fig_path_bt, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{roi_key}] By-task figure → {fig_path_bt}")

    # Instance-level figures (only with --full-matrix)
    if full_matrix:
        full_win_dir = hp_sub / 'full_resolution_by_window'
        full_win_dir.mkdir(parents=True, exist_ok=True)
        fig = make_instance_figure(instance_ispc, roi_name, vmax=vmax)
        fig_path2 = full_win_dir / f'roi-{roi_key}_instance_ispc.png'
        fig.savefig(fig_path2, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [{roi_key}] Full-resolution (by window) → {fig_path2}")

        full_task_dir = hp_sub / 'full_resolution_by_task'
        full_task_dir.mkdir(parents=True, exist_ok=True)
        fig = make_instance_figure_by_task(instance_ispc, roi_name, vmax=vmax)
        fig_path3 = full_task_dir / f'roi-{roi_key}_instance_ispc.png'
        fig.savefig(fig_path3, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [{roi_key}] Full-resolution (by task)   → {fig_path3}")

    # T-value figure (only with --tvalue)
    tval_matrix = None
    if compute_tval:
        if per_subj is None:
            print(f"  [{roi_key}] WARNING: per_subject not in cache, cannot compute t-values.")
        else:
            per_subj_cond_z = collapse_per_subject_to_conditions(per_subj)
            tval_matrix = compute_tvalue_matrix(per_subj_cond_z)
            tval_dir = hp_sub / 'collapsed_by_task_tvalue'
            tval_dir.mkdir(parents=True, exist_ok=True)
            fig = make_condition_tvalue_figure(tval_matrix, roi_name, tval_vmax=tval_vmax)
            fig_path_tv = tval_dir / f'roi-{roi_key}_cross_task_ispc_tvalue.png'
            fig.savefig(fig_path_tv, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  [{roi_key}] T-value figure → {fig_path_tv}")

    return cond_matrix, cond_by_task, instance_ispc, tval_matrix


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-task boundary ISPC (HRF windows).')
    parser.add_argument('--roi', nargs='+', default=[k for k, _, _ in ROI_SPEC])
    parser.add_argument('--hp', action='store_true',
                        help='Enable HP filtering at 0.01 Hz')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force recompute ISPC')
    parser.add_argument('--full-matrix', action='store_true',
                        help='Also save/plot full N_TOTAL×N_TOTAL instance-level ISPC')
    parser.add_argument('--vmax', type=float, default=0.5)
    parser.add_argument('--min-subjects', type=int, default=5,
                        help='Min subjects for SVF/AHC trial inclusion (default: 5)')
    parser.add_argument('--post-windows', action='store_true',
                        help='Use post4 windows (4-13, 14-23, 24-33, 34-43 TRs) instead of hrf')
    parser.add_argument('--preset', default=None, choices=['hrf', 'post4', 'onset0'],
                        help='Window preset (overrides --post-windows if set)')
    parser.add_argument('--tvalue', action='store_true',
                        help='Also generate t-value figures (1-sample t-test vs 0 per cell)')
    parser.add_argument('--tvalue-vmax', type=float, default=5.0,
                        help='Color scale limit for t-value figures (default: 5.0)')
    parser.add_argument('--onset', action='store_true',
                        help='Use trial onset (start) times for SVF/AHC; +6s shift for filmfest')
    args = parser.parse_args()

    global OUTPUT_DIR, N_SVF, N_AHC, N_INSTANCES, N_TOTAL, TASK_SLICES
    global WINDOWS, WINDOW_PRESET, _WIN_TR_SHORT, WINDOW_LABELS, _COND_LABELS
    global USE_ONSET
    USE_ONSET = args.onset
    OUTPUT_DIR = FIGS_DIR / 'cross_task_ispc'

    # Apply window preset
    if args.preset is not None:
        preset_key = args.preset
    else:
        preset_key = 'post4' if args.post_windows else 'hrf'
    preset        = WINDOW_PRESETS[preset_key]
    WINDOWS       = preset['windows']
    WINDOW_PRESET = preset['cache_tag']
    _WIN_TR_SHORT = preset['tr_labels']
    WINDOW_LABELS = _WIN_TR_SHORT
    _COND_LABELS  = [f'{task}\n{w}' for task in ('SVF', 'AHC', 'Film') for w in _WIN_TR_SHORT]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    svf_ahc_subjects = sorted(SUBJECT_IDS)
    all_subjects     = sorted(set(svf_ahc_subjects) | set(FILMFEST_SUBJECTS.keys()))

    # Discover shared trials
    shared_cats,    cat_count    = discover_shared_categories(svf_ahc_subjects, args.min_subjects)
    shared_prompts, prompt_count = discover_shared_prompts(svf_ahc_subjects, args.min_subjects)

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
    print(f"SVF categories ({N_SVF}, ≥{args.min_subjects} subj): {shared_cats}")
    print(f"AHC prompts ({N_AHC}, ≥{args.min_subjects} subj)")
    print(f"Filmfest movies: {N_FILM}")
    print(f"Total instances: {N_INSTANCES}  Windows: {N_WINDOWS}  → {N_TOTAL} conditions")

    roi_spec_filtered = [(k, n, p) for k, n, p in ROI_SPEC if k in args.roi]
    if not roi_spec_filtered:
        print("No valid ROIs."); return

    cond_by_roi      = {}
    cond_by_roi_task = {}
    tval_by_roi      = {}
    for roi_key, roi_name, parcel_ids in roi_spec_filtered:
        cond_mat, cond_by_task, _, tval_mat = run_roi(
            roi_key, roi_name, parcel_ids, all_subjects,
            shared_cats, shared_prompts,
            do_hp=args.hp, force_ispc=args.no_cache,
            full_matrix=args.full_matrix, vmax=args.vmax,
            compute_tval=args.tvalue, tval_vmax=args.tvalue_vmax,
        )
        if cond_mat is not None:
            cond_by_roi[roi_key]      = cond_mat
        if cond_by_task is not None:
            cond_by_roi_task[roi_key] = cond_by_task
        if tval_mat is not None:
            tval_by_roi[roi_key] = tval_mat

    hp_sub = OUTPUT_DIR / ('hp' if args.hp else 'no_hp') / WINDOW_PRESET / ('onset' if USE_ONSET else 'offset')

    if sum(v is not None for v in cond_by_roi.values()) > 1:
        by_win_dir = hp_sub / 'collapsed_by_window'
        by_win_dir.mkdir(parents=True, exist_ok=True)
        fig = make_combined_figure(cond_by_roi, vmax=args.vmax)
        out = by_win_dir / 'all_rois_cross_task_ispc.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\nCombined by-window figure → {out}")

    if sum(v is not None for v in cond_by_roi_task.values()) > 1:
        by_task_dir = hp_sub / 'collapsed_by_task'
        by_task_dir.mkdir(parents=True, exist_ok=True)
        fig = make_combined_figure(cond_by_roi_task, vmax=args.vmax)
        out_bt = by_task_dir / 'all_rois_cross_task_ispc.png'
        fig.savefig(out_bt, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Combined by-task figure → {out_bt}")

    if sum(v is not None for v in tval_by_roi.values()) > 1:
        tval_dir = hp_sub / 'collapsed_by_task_tvalue'
        tval_dir.mkdir(parents=True, exist_ok=True)
        fig = make_combined_tvalue_figure(tval_by_roi, tval_vmax=args.tvalue_vmax)
        out_tv = tval_dir / 'all_rois_cross_task_ispc_tvalue.png'
        fig.savefig(out_tv, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Combined t-value figure → {out_tv}")

    print(f"\nDone.\n  Figures: {OUTPUT_DIR}\n  Cache: {RESULT_CACHE_DIR}")


if __name__ == '__main__':
    main()
