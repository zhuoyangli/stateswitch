"""
ahc_boundary_wspc.py — AHC Trial-Offset Within-Subject Pattern Correlation

For each shared AHC prompt (≥5 subjects by default), extracts activation
patterns in time windows around the trial offset (when the subject says
"next"). Computes within-subject pattern correlation across all
N_PROMPTS × N_WINDOWS conditions. Shared prompts are discovered at runtime.

Differs from ahc_boundary_ispc.py only in the correlation step:
  - ISPC: correlates each subject's pattern with the LOO mean across other subjects
  - WSPC: correlates each subject's pattern with their own pattern across conditions

Condition ordering (prompts nested in windows):
  [P0_pre, P1_pre, ..., PN_pre, P0_onset, ..., PN_late2]

Prompt-to-trial alignment: the ordered prompt list from the AHC sentences
xlsx (rec/ahc_sentences/) is aligned by index to the psychopy trial rows.

Preprocessing (per voxel):
  1. Spatial smoothing FWHM=6mm — cached in roi_voxels/
  2. Linear detrend
  3. Optional HP filter 0.01 Hz (--hp flag, default off)
  4. Z-score per voxel

Usage:
    uv run python srcs/fmrianalysis/ahc_boundary_wspc.py
    uv run python srcs/fmrianalysis/ahc_boundary_wspc.py --roi pmc ag --n_jobs 4
    uv run python srcs/fmrianalysis/ahc_boundary_wspc.py --hp --no-cache-wspc
    uv run python srcs/fmrianalysis/ahc_boundary_wspc.py --min-subjects 6
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
from joblib import Parallel, delayed

from configs.config import (
    DATA_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR, SUBJECT_IDS,
)
from configs.schaefer_rois import (
    EARLY_AUDITORY, EARLY_VISUAL, POSTERIOR_MEDIAL, ANGULAR_GYRUS,
    DLPFC, DACC, get_bilateral_ids,
)
from fmrianalysis.utils import (
    highpass_filter, get_bold_path, discover_svf_ahc_sessions, find_psychopy_csv,
)

# ============================================================================
# CONSTANTS
# ============================================================================

AHC_SENTENCES_DIR = DATA_DIR / 'rec' / 'ahc_sentences'
VOXEL_CACHE_DIR   = ANALYSIS_CACHE_DIR / 'roi_voxels'
RESULT_CACHE_DIR  = ANALYSIS_CACHE_DIR / 'ahc_boundary_wspc'
OUTPUT_DIR        = FIGS_DIR / 'ahc_boundary_wspc' / 'default'

SMOOTH_FWHM = 6.0
HP_CUTOFF   = 0.01

# Shared AHC prompts — discovered at runtime in main() via discover_shared_prompts()
SHARED_PROMPTS: list = []
N_TRIALS       = 0   # set after discovery

# Short tick labels — set dynamically in main() via _make_prompt_labels()
_PROMPT_LABELS: list = []

WINDOW_PRESETS = {
    'default': {
        'windows': [
            ('pre',   -5, -1),
            ('onset',  0,  4),
            ('post',   5,  9),
            ('late',  10, 14),
            ('late2', 15, 19),
        ],
        'labels': [
            'pre-boundary\n-5 to -1 TRs',
            'post-boundary\n0 to 4 TRs',
            'post-boundary\n5 to 9 TRs',
            'post-boundary\n10 to 14 TRs',
            'post-boundary\n15 to 19 TRs',
        ],
        'tag': '',
    },
    'hrf': {
        'windows': [
            ('pre',    -6,  3),
            ('onset',   4, 13),
            ('post',   14, 23),
            ('late',   24, 33),
        ],
        'labels': [
            'pre-boundary\n-6 to 3 TRs',
            'post-boundary\n4 to 13 TRs',
            'post-boundary\n14 to 23 TRs',
            'post-boundary\n24 to 33 TRs',
        ],
        'tag': '_hrf',
    },
    'post4': {
        'windows': [
            ('onset',   4, 13),
            ('post',   14, 23),
            ('late',   24, 33),
            ('vlate',  34, 43),
        ],
        'labels': [
            'post-boundary\n4 to 13 TRs',
            'post-boundary\n14 to 23 TRs',
            'post-boundary\n24 to 33 TRs',
            'post-boundary\n34 to 43 TRs',
        ],
        'tag': '_post4',
    },
}

WINDOWS      = WINDOW_PRESETS['default']['windows']
N_WINDOWS    = len(WINDOWS)
N_CONDITIONS = 0   # set after discovery (N_TRIALS * N_WINDOWS)
WIN_TAG      = ''


def apply_window_preset(name):
    global WINDOWS, N_WINDOWS, N_CONDITIONS, _WINDOW_LABELS, WIN_TAG
    preset         = WINDOW_PRESETS[name]
    WINDOWS        = preset['windows']
    N_WINDOWS      = len(WINDOWS)
    N_CONDITIONS   = N_TRIALS * N_WINDOWS
    _WINDOW_LABELS = preset['labels']
    WIN_TAG        = preset['tag']


ROI_SPEC = [
    ('hipp',  'Hippocampus', None),
    ('pmc',   'PMC',         get_bilateral_ids(POSTERIOR_MEDIAL)),
    ('ag',    'AG',          get_bilateral_ids(ANGULAR_GYRUS)),
    ('dacc',  'dACC',        get_bilateral_ids(DACC)),
    ('dlpfc', 'dlPFC',       get_bilateral_ids(DLPFC)),
    ('eac',   'EAC',         get_bilateral_ids(EARLY_AUDITORY)),
    ('evc',   'EVC',         get_bilateral_ids(EARLY_VISUAL)),
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
# STEP 1: TRIAL OFFSET DISCOVERY
# ============================================================================

def _find_ahc_sentences(subject, session):
    """Return path to AHC sentences xlsx for subject/session, or None."""
    p = AHC_SENTENCES_DIR / f'{subject}_{session}_task-ahc_desc-sentences.xlsx'
    return p if p.exists() else None


def get_ahc_offsets_by_prompt(subject, session):
    """Return {prompt_text: [offset_TR, ...]} for non-first prompts in session.

    Aligns sentences xlsx prompt list with psychopy trial rows by index order.
    Offset TR is relative to first trial start.
    """
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
    for prompt, (_, row) in zip(prompts, rows.iterrows()):
        offset_sec = row['ahc_trial.stopped'] - first_start
        tr         = int(round(offset_sec / TR))
        result.setdefault(prompt, []).append(tr)
    return result


def discover_shared_prompts(subjects, min_subjects=5):
    """Return prompts present in ≥ min_subjects subjects.

    Returns (sorted_prompt_list, {prompt: subject_count}) sorted by count desc
    then alphabetically.
    """
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


def _make_prompt_labels(prompts):
    """Generate short tick labels from prompt texts (≤8 chars each)."""
    stop_words = {'of', 'the', 'a', 'an', 'at', 'in', 'on'}
    seen: dict = {}
    labels = []
    for prompt in prompts:
        words = [w for w in prompt.split() if w.lower() not in stop_words] or prompt.split()
        short = ''.join(w[:3] for w in words)[:8]
        if short in seen:
            seen[short] += 1
            short = short[:6] + str(seen[short])
        else:
            seen[short] = 0
        labels.append(short)
    return labels


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
# STEP 3: BOUNDARY-LOCKED PATTERNS
# ============================================================================

def extract_pattern_at_tr(voxel_data, boundary_tr):
    """Extract (N_WINDOWS, V) pattern around one boundary. Returns None if out of bounds."""
    T = voxel_data.shape[0]
    win_patterns = []
    for _, tr_start_rel, tr_end_rel in WINDOWS:
        start = boundary_tr + tr_start_rel
        end   = boundary_tr + tr_end_rel
        if start < 0 or end >= T:
            return None
        win_patterns.append(voxel_data[start:end + 1].mean(axis=0))
    return np.stack(win_patterns, axis=0)


def get_subject_patterns(subject, roi_key, parcel_ids, do_hp=False, force_voxels=False):
    """Build (N_TRIALS, N_WINDOWS, V) pattern array for one subject.

    For each shared prompt, averages patterns across all valid instances.
    Returns NaN slices for prompts with no data.
    """
    sessions_tasks = discover_svf_ahc_sessions(subject)
    ahc_sessions   = [ses for ses, t in sessions_tasks if t == 'ahc']

    prompt_patterns: dict[str, list] = {p: [] for p in SHARED_PROMPTS}

    for session in ahc_sessions:
        offsets_by_prompt = get_ahc_offsets_by_prompt(subject, session)
        shared_in_session = [p for p in SHARED_PROMPTS if p in offsets_by_prompt]
        if not shared_in_session:
            continue
        try:
            raw        = load_roi_voxels(subject, session, 'ahc', roi_key,
                                         parcel_ids, force=force_voxels)
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
    full = np.full((N_TRIALS, N_WINDOWS, V), np.nan, dtype=np.float64)
    for pi, prompt in enumerate(SHARED_PROMPTS):
        if prompt_patterns[prompt]:
            full[pi] = np.mean(prompt_patterns[prompt], axis=0)

    return full


# ============================================================================
# STEP 4: WITHIN-SUBJECT PATTERN CORRELATION
# ============================================================================

def flatten_patterns(patterns):
    """(N_TRIALS, N_WINDOWS, V) → (N_CONDITIONS, V) in prompts-nested-in-windows order."""
    return patterns.transpose(1, 0, 2).reshape(N_CONDITIONS, -1)


def compute_group_wspc(all_patterns):
    """Compute within-subject pattern correlation matrix averaged across subjects.

    For each subject, correlates their own activation pattern for condition i
    with their own activation pattern for condition j. Then Fisher-z averages
    across subjects.

    Parameters
    ----------
    all_patterns : (N_subj, N_CONDITIONS, V) array — NaN indicates missing condition

    Returns
    -------
    group_wspc    : (N_CONDITIONS, N_CONDITIONS) Fisher-z averaged correlation matrix
    per_subj_wspc : (N_subj, N_CONDITIONS, N_CONDITIONS) per-subject correlation matrices
    """
    N = all_patterns.shape[0]
    per_subj_wspc = np.full((N, N_CONDITIONS, N_CONDITIONS), np.nan)

    for i in range(N):
        corr_mat = np.full((N_CONDITIONS, N_CONDITIONS), np.nan)
        for row in range(N_CONDITIONS):
            if np.all(np.isnan(all_patterns[i, row])):
                continue
            for col in range(N_CONDITIONS):
                if np.all(np.isnan(all_patterns[i, col])):
                    continue
                a = all_patterns[i, row]
                b = all_patterns[i, col]
                valid = ~(np.isnan(a) | np.isnan(b))
                if valid.sum() < 2:
                    continue
                a_, b_ = a[valid], b[valid]
                a_ -= a_.mean(); b_ -= b_.mean()
                denom = np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum())
                if denom == 0:
                    continue
                corr_mat[row, col] = np.dot(a_, b_) / denom

        per_subj_wspc[i] = corr_mat

    z_stack    = np.arctanh(np.clip(per_subj_wspc, -0.999, 0.999))
    group_wspc = np.tanh(np.nanmean(z_stack, axis=0))
    return group_wspc, per_subj_wspc


# ============================================================================
# STEP 5: FIGURES
# ============================================================================

_WINDOW_LABELS = WINDOW_PRESETS['default']['labels']
# Short prompt labels — set dynamically in main() via _make_prompt_labels()
_PROMPT_LABELS: list = []


def make_wspc_figure(wspc_matrix, roi_name, vmax=0.3):
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.subplots_adjust(left=0.28, right=0.88, top=0.88, bottom=0.12)

    im = ax.imshow(wspc_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal', origin='upper', interpolation='none')

    tick_pos    = np.arange(N_CONDITIONS)
    tick_labels = _PROMPT_LABELS * N_WINDOWS
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=5, rotation=90)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=5)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    for pos in np.arange(0.5, N_CONDITIONS - 1, 1):
        ax.axhline(pos, color='white', linewidth=0.3, alpha=0.5)
        ax.axvline(pos, color='white', linewidth=0.3, alpha=0.5)

    for block_edge in range(N_TRIALS, N_CONDITIONS, N_TRIALS):
        ax.axhline(block_edge - 0.5, color='black', linewidth=1.8)
        ax.axvline(block_edge - 0.5, color='black', linewidth=1.8)

    block_centers = np.arange(N_TRIALS / 2 - 0.5, N_CONDITIONS, N_TRIALS)
    ax2_top = ax.secondary_xaxis('top')
    ax2_top.set_xticks(block_centers)
    ax2_top.set_xticklabels(_WINDOW_LABELS, fontsize=7.5)
    ax2_top.tick_params(length=0)
    for spine in ax2_top.spines.values():
        spine.set_visible(False)

    from matplotlib.transforms import blended_transform_factory
    blend = blended_transform_factory(ax.transAxes, ax.transData)
    for data_y, label in zip(block_centers, _WINDOW_LABELS):
        ax.text(-0.035, data_y, label, ha='center', va='center',
                fontsize=7.5, rotation=90, rotation_mode='anchor',
                transform=blend, clip_on=False)

    ax.set_xlabel('Condition (prompt × window)', fontsize=10)
    ax.set_ylabel('Condition (prompt × window)', fontsize=10)
    ax.set_title(roi_name, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.12, shrink=0.4)
    cbar.set_label('within-subject pattern correlation (r)', fontsize=9)
    return fig


def make_combined_figure(wspc_by_roi, vmax=0.3, horizontal=False):
    """Plot all ROIs in a grid (2 columns) or single row (horizontal=True)."""
    n_rois = len(wspc_by_roi)

    if horizontal:
        ncols, nrows = n_rois, 1
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.2 * ncols + 1.5, 4.5))
        fig.subplots_adjust(left=0.10, right=0.91, top=0.90,
                            bottom=0.04, wspace=0.12)
        axes_flat  = np.array(axes).flatten()
        last_im    = None

        block_centers   = np.arange(N_TRIALS / 2 - 0.5, N_CONDITIONS, N_TRIALS)
        tr_range_labels = [f"{w[1]} - {w[2]} TRs" for w in WINDOWS]
        tick_pos    = np.arange(N_CONDITIONS)
        tick_labels = [f'P{i+1}' for i in range(N_TRIALS)] * N_WINDOWS

        from matplotlib.transforms import blended_transform_factory

        for idx, (roi_key, roi_name, _) in enumerate(ROI_SPEC):
            ax  = axes_flat[idx]
            mat = wspc_by_roi.get(roi_key)
            if mat is None:
                ax.set_visible(False)
                continue

            im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           aspect='equal', origin='upper', interpolation='none')
            last_im = im

            for block_edge in range(N_TRIALS, N_CONDITIONS, N_TRIALS):
                ax.axhline(block_edge - 0.5, color='black', linewidth=1.2)
                ax.axvline(block_edge - 0.5, color='black', linewidth=1.2)

            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=4, rotation=90)
            ax.xaxis.set_tick_params(length=1)
            ax.set_yticks([])

            if idx == 0:
                blend = blended_transform_factory(ax.transAxes, ax.transData)
                for data_y, label in zip(block_centers, _WINDOW_LABELS):
                    ax.text(-0.08, data_y, label, ha='center', va='center',
                            fontsize=5.5, rotation=90, rotation_mode='anchor',
                            transform=blend, clip_on=False)

            ax2 = ax.secondary_xaxis('top')
            ax2.set_xticks(block_centers)
            ax2.set_xticklabels(tr_range_labels, fontsize=5, rotation=0, ha='center')
            ax2.tick_params(length=0)
            for spine in ax2.spines.values():
                spine.set_visible(False)

            ax.set_title(roi_name, fontsize=8.5, fontweight='bold', pad=0)

        if last_im is not None:
            cbar = fig.colorbar(last_im,
                                ax=axes_flat[axes_flat != None].tolist(),
                                fraction=0.012, pad=0.01, shrink=0.6)
            cbar.set_label('within-subject pattern correlation (r)', fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        fig.suptitle(
            'WSPC between mean activation patterns of peri-boundary windows in each AHC prompt',
            fontsize=12, fontweight='bold', y=0.98)

    else:
        ncols  = 2
        nrows  = (n_rois + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(9 * ncols, 9 * nrows))
        axes_flat = np.array(axes).flatten()

        block_centers = np.arange(N_TRIALS / 2 - 0.5, N_CONDITIONS, N_TRIALS)

        for idx, (roi_key, roi_name, _) in enumerate(ROI_SPEC):
            ax  = axes_flat[idx]
            mat = wspc_by_roi.get(roi_key)
            if mat is None:
                ax.set_visible(False)
                continue

            im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           aspect='equal', origin='upper', interpolation='none')
            ax.set_xticks([]); ax.set_yticks([])
            for block_edge in range(N_TRIALS, N_CONDITIONS, N_TRIALS):
                ax.axhline(block_edge - 0.5, color='black', linewidth=1.5)
                ax.axvline(block_edge - 0.5, color='black', linewidth=1.5)

            for center, label in zip(block_centers, _WINDOW_LABELS):
                ax.text(center, -1.5, label.split('\n')[0], ha='center', va='bottom',
                        fontsize=7, transform=ax.get_xaxis_transform())
            ax.set_title(roi_name, fontsize=10, fontweight='bold')
            cb = fig.colorbar(im, ax=ax, fraction=0.023, pad=0.03, shrink=0.4)
            cb.set_label('within-subject pattern correlation (r)', fontsize=7)

        for idx in range(n_rois, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle('AHC Boundary WSPC — All ROIs', fontsize=13, fontweight='bold')
        plt.tight_layout()

    return fig


# ============================================================================
# PER-ROI PIPELINE
# ============================================================================

def run_roi(roi_key, roi_name, parcel_ids, subjects, do_hp=False,
            force_voxels=False, force_wspc=False, vmax=0.3):
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hp_tag     = '_hp' if do_hp else ''
    wspc_cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6{hp_tag}{WIN_TAG}_wspc.npz'

    if not force_wspc and wspc_cache.exists():
        print(f"  [{roi_key}] Loading from cache ...")
        group_wspc = np.load(wspc_cache)['group_wspc']
    else:
        print(f"\n{'='*60}\nROI: {roi_name} ({roi_key})")
        subject_patterns = []
        for subj in subjects:
            print(f"  Subject: {subj}")
            pats = get_subject_patterns(subj, roi_key, parcel_ids,
                                        do_hp=do_hp, force_voxels=force_voxels)
            if pats is None:
                print(f"  SKIP {subj}: no data")
                continue
            flat    = flatten_patterns(pats)
            n_valid = (~np.isnan(flat).all(axis=1)).sum()
            print(f"    {n_valid}/{N_CONDITIONS} conditions valid")
            subject_patterns.append(flat)

        if len(subject_patterns) < 1:
            print(f"  [{roi_key}] No subjects with data, skipping.")
            return None

        all_patterns = np.stack(subject_patterns, axis=0)
        print(f"  [{roi_key}] Computing WSPC ({all_patterns.shape[0]} subjects)...")
        group_wspc, per_subj = compute_group_wspc(all_patterns)

        np.savez_compressed(wspc_cache, group_wspc=group_wspc, per_subject=per_subj,
                            subjects=np.array(subjects), do_hp=do_hp)
        print(f"  [{roi_key}] Cached → {wspc_cache.name}")

    fig_path = OUTPUT_DIR / f'roi-{roi_key}_wspc.png'
    fig = make_wspc_figure(group_wspc, roi_name, vmax=vmax)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{roi_key}] Figure → {fig_path.name}")
    return group_wspc


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='AHC boundary WSPC.')
    parser.add_argument('--roi', nargs='+', default=[k for k, _, _ in ROI_SPEC])
    parser.add_argument('--hrf', action='store_true',
                        help='Use HRF-shifted windows')
    parser.add_argument('--post4', action='store_true',
                        help='Use post4 windows (4:13, 14:23, 24:33, 34:43 TRs); no pre-boundary')
    parser.add_argument('--hp', action='store_true',
                        help='Enable HP filtering at 0.01 Hz')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force recompute voxels and WSPC')
    parser.add_argument('--no-cache-wspc', action='store_true',
                        help='Force recompute WSPC only')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--vmax', type=float, default=0.5)
    parser.add_argument('--min-subjects', type=int, default=5,
                        help='Minimum subjects to include a prompt (default: 5)')
    args = parser.parse_args()

    global OUTPUT_DIR, SHARED_PROMPTS, N_TRIALS, N_CONDITIONS, _PROMPT_LABELS
    preset_name = 'post4' if args.post4 else ('hrf' if args.hrf else 'default')
    hp_suffix   = '_hp' if args.hp else ''
    apply_window_preset(preset_name)
    OUTPUT_DIR = FIGS_DIR / 'ahc_boundary_wspc' / f'{preset_name}{hp_suffix}'

    subjects = sorted(SUBJECT_IDS)
    print(f"Subjects: {subjects}")

    SHARED_PROMPTS, prompt_count = discover_shared_prompts(
        subjects, min_subjects=args.min_subjects)
    N_TRIALS     = len(SHARED_PROMPTS)
    N_CONDITIONS = N_TRIALS * N_WINDOWS
    _PROMPT_LABELS = _make_prompt_labels(SHARED_PROMPTS)
    print(f"Prompts ({N_TRIALS}, ≥{args.min_subjects} subjects):")
    for p in SHARED_PROMPTS:
        print(f"  {prompt_count[p]} subjects: {p}")
    print(f"Windows: {N_WINDOWS}  →  {N_CONDITIONS} conditions")

    roi_spec_filtered = [(k, n, p) for k, n, p in ROI_SPEC if k in args.roi]
    if not roi_spec_filtered:
        print("No valid ROIs."); return

    VOXEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.n_jobs > 1:
        def _extract_job(subj, roi_key, parcel_ids):
            sessions_tasks = discover_svf_ahc_sessions(subj)
            for session, t in sessions_tasks:
                if t != 'ahc': continue
                try:
                    load_roi_voxels(subj, session, 'ahc', roi_key, parcel_ids,
                                    force=args.no_cache)
                except Exception as e:
                    print(f"  ERROR {subj} {session}: {e}")
        Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_extract_job)(subj, roi_key, parcel_ids)
            for roi_key, _, parcel_ids in roi_spec_filtered
            for subj in subjects
        )

    wspc_by_roi = {}
    for roi_key, roi_name, parcel_ids in roi_spec_filtered:
        wspc_by_roi[roi_key] = run_roi(
            roi_key, roi_name, parcel_ids, subjects,
            do_hp=args.hp, force_voxels=args.no_cache,
            force_wspc=(args.no_cache or args.no_cache_wspc), vmax=args.vmax,
        )

    if sum(v is not None for v in wspc_by_roi.values()) > 1:
        fig = make_combined_figure(wspc_by_roi, vmax=args.vmax, horizontal=(args.hrf or args.post4))
        out = OUTPUT_DIR / 'all_rois_wspc.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\nCombined figure → {out.name}")

    print(f"\nDone.\n  Figures: {OUTPUT_DIR}\n  Cache: {RESULT_CACHE_DIR}")


if __name__ == '__main__':
    main()
