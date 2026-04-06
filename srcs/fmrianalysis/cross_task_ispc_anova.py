"""
cross_task_ispc_formal_test.py — Formal statistical test of cross-task ISPC

Compares inter-subject pattern correlation (ISPC) across 3 comparison types
(within-task, cross-task SVF×AHC, cross-task ×FilmFest) in 2 time windows:

  1. Post-boundary window (TRs 4–13 relative to onset boundary, HRF-shifted)
     → loaded from cached `roi-{key}_sm6_hp_onset_post4_ispc.npz`
  2. Mid-event baseline window (10 TRs centered on the temporal midpoint of the
     post-boundary event — new movie / next SVF category / next AHC prompt)
     → newly extracted from cached smoothed voxels

Statistical approach:
  • Per-ROI 2 (window) × 3 (condition) repeated-measures ANOVA
  • Permutation test (shuffle window labels within subjects, 5000 iterations)
  • Post-hoc paired t-tests (boundary vs mid-event per condition, Bonferroni)

Settings: HP filtering (0.01 Hz), onset boundaries.

Usage:
    uv run python srcs/fmrianalysis/cross_task_ispc_formal_test.py
    uv run python srcs/fmrianalysis/cross_task_ispc_formal_test.py --roi pmc ag
    uv run python srcs/fmrianalysis/cross_task_ispc_formal_test.py --n-perm 5000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

from configs.config import (
    DATA_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR,
    SUBJECT_IDS, FILMFEST_SUBJECTS, MOVIE_INFO,
)
from configs.schaefer_rois import (
    EARLY_AUDITORY, EARLY_VISUAL, POSTERIOR_MEDIAL, ANGULAR_GYRUS,
    DLPFC, DACC, MPFC, get_bilateral_ids,
)
from fmrianalysis.utils import (
    highpass_filter, discover_svf_ahc_sessions, find_psychopy_csv,
)

# Re-use functions from the boundary ISPC script
import fmrianalysis.cross_task_boundary_ispc as _bnd

# ============================================================================
# CONSTANTS
# ============================================================================

AHC_SENTENCES_DIR = DATA_DIR / 'rec' / 'ahc_sentences'
ANNOTATIONS_DIR   = DATA_DIR / 'filmfest_annotations'
VOXEL_CACHE_DIR   = ANALYSIS_CACHE_DIR / 'roi_voxels'
RESULT_CACHE_DIR  = ANALYSIS_CACHE_DIR / 'cross_task_ispc'
OUTPUT_DIR        = None  # set in main() based on --onset flag

SMOOTH_FWHM = 6.0
HP_CUTOFF   = 0.01
N_FILM      = 8   # 4 boundaries per filmfest run × 2 runs
MID_HALF_WIN = 5  # ±5 TRs around midpoint = 10-TR window

ROI_SPEC = [
    ('eac',   'Early Auditory Cortex',     get_bilateral_ids(EARLY_AUDITORY)),
    ('evc',   'Early Visual Cortex',       get_bilateral_ids(EARLY_VISUAL)),
    ('pmc',   'Posterior Medial Cortex',   get_bilateral_ids(POSTERIOR_MEDIAL)),
    ('ag',    'Angular Gyrus',             get_bilateral_ids(ANGULAR_GYRUS)),
    ('dlpfc', 'Dorsolateral PFC',          get_bilateral_ids(DLPFC)),
    ('dacc',  'Dorsal Anterior Cingulate', get_bilateral_ids(DACC)),
    ('mpfc',  'Medial Prefrontal Cortex',  get_bilateral_ids(MPFC)),
    ('hipp',  'Hippocampus',               None),  # uses NiftiMasker separately
]

ROI_ABBREVS = {
    'eac': 'EAC', 'evc': 'EVC', 'pmc': 'PMC', 'ag': 'AG',
    'dlpfc': 'dlPFC', 'dacc': 'dACC', 'mpfc': 'mPFC', 'hipp': 'Hipp',
}

CONDITION_LABELS = ['within-task', 'cross-task\n(SVF×AHC)', 'cross-task\n(×FilmFest)']
CONDITION_NAMES  = ['within', 'xsame', 'xdiff']
WINDOW_LABELS    = ['post-boundary', 'mid-event']
COLORS           = ['#4878CF', '#D65F5F', '#6ACC65']

# ============================================================================
# STEP 1: MID-EVENT TIMEPOINT GETTERS
# ============================================================================

def get_svf_midpoints_by_category(subject, session):
    """Return {category: [mid_TR, ...]} using (started + stopped) / 2."""
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
        cat     = row['category_name']
        mid_sec = (row['svf_trial.started'] + row['svf_trial.stopped']) / 2.0 - first_start
        tr      = int(round(mid_sec / TR))
        result.setdefault(cat, []).append(tr)
    return result


def get_ahc_midpoints_by_prompt(subject, session):
    """Return {prompt: [mid_TR, ...]} using (started + stopped) / 2."""
    xlsx_path = AHC_SENTENCES_DIR / f'{subject}_{session}_task-ahc_desc-sentences.xlsx'
    csv_path  = find_psychopy_csv(subject, session, 'ahc')
    if not xlsx_path.exists() or csv_path is None:
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
        mid_sec = (row['ahc_trial.started'] + row['ahc_trial.stopped']) / 2.0 - first_start
        tr      = int(round(mid_sec / TR))
        result.setdefault(prompt, []).append(tr)
    return result


def _mss_to_seconds(mss):
    """Convert m.ss timestamp to total seconds."""
    minutes = int(mss)
    seconds = round((float(mss) - minutes) * 100)
    return minutes * 60 + seconds


def get_movie_midpoints(task):
    """Midpoint (seconds) of each movie in a filmfest run — 5 values per run."""
    movies = [m for m in MOVIE_INFO if m['task'] == task]
    mids   = []
    for movie in movies:
        df   = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
        segb = df.dropna(subset=['SEG-B_Number'])
        starts = segb['Start Time (m.ss)'].apply(_mss_to_seconds).values
        ends   = segb['End Time (m.ss)'].apply(_mss_to_seconds).values
        mids.append((starts[0] + ends[-1]) / 2.0)
    return mids  # list of 5 floats


# ============================================================================
# STEP 2: MID-EVENT PATTERN EXTRACTION (single 10-TR window per instance)
# ============================================================================

def extract_mid_event_pattern(voxel_data, mid_tr):
    """Extract mean voxel pattern in [mid_tr - 5, mid_tr + 4] (10 TRs). Returns None if OOB."""
    T     = voxel_data.shape[0]
    start = mid_tr - MID_HALF_WIN
    end   = mid_tr + MID_HALF_WIN - 1
    if start < 0 or end >= T:
        return None
    return voxel_data[start:end + 1].mean(axis=0)


def get_subject_svf_mid_patterns(subject, shared_cats, do_hp=False):
    """Build (N_SVF, V) mid-event patterns for shared SVF categories."""
    sessions_tasks = discover_svf_ahc_sessions(subject)
    svf_sessions   = [ses for ses, t in sessions_tasks if t == 'svf']

    cat_patterns: dict = {c: [] for c in shared_cats}

    for session in svf_sessions:
        mids = get_svf_midpoints_by_category(subject, session)
        shared_in_session = [c for c in shared_cats if c in mids]
        if not shared_in_session:
            continue
        try:
            raw        = _bnd.load_roi_voxels(subject, session, 'svf', _ROI_KEY, _PARCEL_IDS)
            voxel_data = _bnd.preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {session} svf: {e}")
            continue
        for cat in shared_in_session:
            for mid_tr in mids[cat]:
                pat = extract_mid_event_pattern(voxel_data, mid_tr)
                if pat is not None:
                    cat_patterns[cat].append(pat)

    if not any(cat_patterns.values()):
        return None

    V    = next(p[0].shape[0] for p in cat_patterns.values() if p)
    full = np.full((N_SVF, V), np.nan)
    for ci, cat in enumerate(shared_cats):
        if cat_patterns[cat]:
            full[ci] = np.mean(cat_patterns[cat], axis=0)
    return full


def get_subject_ahc_mid_patterns(subject, shared_prompts, do_hp=False):
    """Build (N_AHC, V) mid-event patterns for shared AHC prompts."""
    sessions_tasks = discover_svf_ahc_sessions(subject)
    ahc_sessions   = [ses for ses, t in sessions_tasks if t == 'ahc']

    prompt_patterns: dict = {p: [] for p in shared_prompts}

    for session in ahc_sessions:
        mids = get_ahc_midpoints_by_prompt(subject, session)
        shared_in_session = [p for p in shared_prompts if p in mids]
        if not shared_in_session:
            continue
        try:
            raw        = _bnd.load_roi_voxels(subject, session, 'ahc', _ROI_KEY, _PARCEL_IDS)
            voxel_data = _bnd.preprocess_voxels(raw, do_hp=do_hp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {session} ahc: {e}")
            continue
        for prompt in shared_in_session:
            for mid_tr in mids[prompt]:
                pat = extract_mid_event_pattern(voxel_data, mid_tr)
                if pat is not None:
                    prompt_patterns[prompt].append(pat)

    if not any(prompt_patterns.values()):
        return None

    V    = next(p[0].shape[0] for p in prompt_patterns.values() if p)
    full = np.full((N_AHC, V), np.nan)
    for pi, prompt in enumerate(shared_prompts):
        if prompt_patterns[prompt]:
            full[pi] = np.mean(prompt_patterns[prompt], axis=0)
    return full


def get_subject_filmfest_mid_patterns(subject, do_hp=False):
    """Build (N_FILM=8, V) mid-event patterns — midpoint of the post-boundary movie."""
    if subject not in FILMFEST_SUBJECTS:
        return None

    session   = FILMFEST_SUBJECTS[subject]
    N_PER_RUN = 4
    all_pats  = []

    for task in ('filmfest1', 'filmfest2'):
        try:
            raw = _bnd.load_roi_voxels(subject, session, task, _ROI_KEY, _PARCEL_IDS)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {task}: {e}")
            all_pats.extend([None] * N_PER_RUN)
            continue
        voxel_data = _bnd.preprocess_voxels(raw, do_hp=do_hp)
        # midpoints[1:] = midpoints of movies 2-5 (post-boundary movies for the 4 boundaries)
        mid_secs = get_movie_midpoints(task)[1:]
        for mid_sec in mid_secs:
            mid_tr = int(round(mid_sec / TR))
            pat    = extract_mid_event_pattern(voxel_data, mid_tr)
            all_pats.append(pat)

    valid = [p for p in all_pats if p is not None]
    if not valid:
        return None

    V    = valid[0].shape[0]
    full = np.full((N_FILM, V), np.nan)
    for i, pat in enumerate(all_pats):
        if pat is not None:
            full[i] = pat
    return full


def get_subject_mid_event_patterns(subject, shared_cats, shared_prompts, do_hp=False):
    """Build (N_INSTANCES, V) mid-event patterns concatenating SVF + AHC + filmfest."""
    svf_pats  = get_subject_svf_mid_patterns(subject, shared_cats, do_hp)
    ahc_pats  = get_subject_ahc_mid_patterns(subject, shared_prompts, do_hp)
    film_pats = get_subject_filmfest_mid_patterns(subject, do_hp)

    V = None
    for p in (svf_pats, ahc_pats, film_pats):
        if p is not None:
            V = p.shape[1]
            break
    if V is None:
        return None

    full = np.full((N_INSTANCES, V), np.nan)
    if svf_pats is not None:
        full[:N_SVF]             = svf_pats[..., :V]
    if ahc_pats is not None:
        full[N_SVF:N_SVF+N_AHC] = ahc_pats[..., :V]
    if film_pats is not None:
        full[N_SVF+N_AHC:]       = film_pats[..., :V]
    return full


# ============================================================================
# STEP 3: LOO ISPC FOR MID-EVENT (N_INSTANCES × N_INSTANCES)
# ============================================================================

def compute_mid_event_ispc(all_patterns):
    """NaN-aware LOO ISPC for mid-event patterns.

    all_patterns: (N_subj, N_INSTANCES, V)
    Returns: group_ispc (N_INSTANCES, N_INSTANCES),
             per_subj   (N_subj, N_INSTANCES, N_INSTANCES)
    """
    N_subj, N_inst, _ = all_patterns.shape
    valid_subj   = ~np.all(np.isnan(all_patterns), axis=2)  # (N_subj, N_inst)
    valid_count  = valid_subj.sum(axis=0)                    # (N_inst,)
    group_nansum = np.nansum(all_patterns, axis=0)           # (N_inst, V)

    per_subj_ispc = np.full((N_subj, N_inst, N_inst), np.nan)

    for i in range(N_subj):
        loo_mean = np.full_like(group_nansum, np.nan)
        for c in range(N_inst):
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

        corr_mat = np.full((N_inst, N_inst), np.nan)
        for row in range(N_inst):
            a = all_patterns[i, row]
            if np.all(np.isnan(a)):
                continue
            for col in range(N_inst):
                b = loo_mean[col]
                if np.all(np.isnan(b)):
                    continue
                valid = ~(np.isnan(a) | np.isnan(b))
                if valid.sum() < 2:
                    continue
                a_ = a[valid] - a[valid].mean()
                b_ = b[valid] - b[valid].mean()
                denom = np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum())
                if denom == 0:
                    continue
                corr_mat[row, col] = np.dot(a_, b_) / denom
        per_subj_ispc[i] = corr_mat

    z_stack    = np.arctanh(np.clip(per_subj_ispc, -0.999, 0.999))
    group_ispc = np.tanh(np.nanmean(z_stack, axis=0))
    return group_ispc, per_subj_ispc


def collapse_mid_per_subject(per_subj_ispc):
    """Collapse (N_subj, N_INSTANCES, N_INSTANCES) → (N_subj, 3, 3) Fisher-z.

    Condition order: 0=SVF, 1=AHC, 2=Film.
    """
    N_subj  = per_subj_ispc.shape[0]
    z_stack = np.arctanh(np.clip(per_subj_ispc, -0.999, 0.999))
    task_slices = [slice(0, N_SVF), slice(N_SVF, N_SVF+N_AHC), slice(N_SVF+N_AHC, N_INSTANCES)]
    out = np.full((N_subj, 3, 3), np.nan)
    for ci, sl_a in enumerate(task_slices):
        for cj, sl_b in enumerate(task_slices):
            block = z_stack[:, sl_a, :][:, :, sl_b]  # (N_subj, len_a, len_b)
            with np.errstate(all='ignore'):
                out[:, ci, cj] = np.nanmean(block.reshape(N_subj, -1), axis=1)
    return out


# ============================================================================
# STEP 4: EXTRACT PER-SUBJECT CONDITION VALUES
# ============================================================================

def extract_boundary_condition_values(per_subj_12x12):
    """Extract per-subject mean Fisher-z for each comparison type at window 0.

    per_subj_12x12: (N_subj, 12, 12) — from collapse_per_subject_to_conditions()
    Returns: (N_subj, 3) for [within, xsame, xdiff] conditions.
    Condition indices at window 0: SVF=0, AHC=4, Film=8.
    """
    N_subj = per_subj_12x12.shape[0]
    out    = np.full((N_subj, 3), np.nan)

    # within-task: diagonal cells at window 0 for each task
    within_cells = [(0, 0), (4, 4), (8, 8)]
    xsame_cells  = [(0, 4), (4, 0)]
    xdiff_cells  = [(0, 8), (8, 0), (4, 8), (8, 4)]

    for ci, cells in enumerate([within_cells, xsame_cells, xdiff_cells]):
        vals = np.stack([per_subj_12x12[:, r, c] for r, c in cells], axis=1)
        out[:, ci] = np.nanmean(vals, axis=1)
    return out


def extract_mid_condition_values(per_subj_3x3):
    """Extract per-subject mean Fisher-z for each comparison type from mid-event.

    per_subj_3x3: (N_subj, 3, 3) — tasks: 0=SVF, 1=AHC, 2=Film.
    Returns: (N_subj, 3) for [within, xsame, xdiff].
    """
    N_subj = per_subj_3x3.shape[0]
    out    = np.full((N_subj, 3), np.nan)

    within_cells = [(0, 0), (1, 1), (2, 2)]
    xsame_cells  = [(0, 1), (1, 0)]
    xdiff_cells  = [(0, 2), (2, 0), (1, 2), (2, 1)]

    for ci, cells in enumerate([within_cells, xsame_cells, xdiff_cells]):
        vals = np.stack([per_subj_3x3[:, r, c] for r, c in cells], axis=1)
        out[:, ci] = np.nanmean(vals, axis=1)
    return out


# ============================================================================
# STEP 5: STATISTICS
# ============================================================================

def run_stats(bnd_vals, mid_vals, subjects, n_perm=5000, rng=None):
    """Run 2×3 RM ANOVA, permutation test, and post-hoc paired t-tests.

    bnd_vals: (N_subj, 3) boundary window per-subject condition values (Fisher-z)
    mid_vals: (N_subj, 3) mid-event window per-subject condition values (Fisher-z)
    subjects: list of subject IDs

    Returns dict with ANOVA table, permutation p-values, post-hoc results.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N_subj = len(subjects)

    # Build long-format DataFrame
    rows = []
    for si, subj in enumerate(subjects):
        for ci, cname in enumerate(CONDITION_NAMES):
            bval = bnd_vals[si, ci]
            mval = mid_vals[si, ci]
            if not np.isnan(bval):
                rows.append(dict(subject=subj, window='boundary', condition=cname, ispc=bval))
            if not np.isnan(mval):
                rows.append(dict(subject=subj, window='mid_event', condition=cname, ispc=mval))

    df_long = pd.DataFrame(rows)

    # Keep only subjects with complete data across all 6 cells (3 conditions × 2 windows)
    counts = df_long.groupby('subject').size()
    complete_subjects = counts[counts == 6].index.tolist()
    df_anova = df_long[df_long['subject'].isin(complete_subjects)].copy()
    print(f"  ANOVA: {len(complete_subjects)}/{N_subj} subjects with complete data: {complete_subjects}")

    results = {}

    # 2×3 repeated-measures ANOVA
    try:
        rm = AnovaRM(df_anova, depvar='ispc', subject='subject',
                     within=['window', 'condition'])
        anova_table = rm.fit().anova_table
        results['anova'] = anova_table
        print("\n  2×3 Repeated-measures ANOVA:")
        print(anova_table.to_string())
    except Exception as e:
        print(f"  ANOVA failed: {e}")
        results['anova'] = None

    # Permutation test for the window×condition interaction.
    # Statistic: variance of (boundary - mid-event) differences across conditions.
    # Permutation: for each subject × condition, randomly sign-flip the difference
    # (equivalent to swapping boundary/mid-event within that cell).
    # This breaks the window assignment while preserving the within-subject structure.

    def interaction_stat(b, m):
        """Variance of per-condition mean differences — measures interaction effect."""
        diffs = np.nanmean(b - m, axis=0)  # (3,) mean diff per condition
        return float(np.nanvar(diffs))

    # Only use complete subjects for permutation
    cs_idx = [list(subjects).index(s) for s in complete_subjects if s in subjects]
    b_comp = bnd_vals[cs_idx]  # (n_complete, 3)
    m_comp = mid_vals[cs_idx]  # (n_complete, 3)

    obs_stat  = interaction_stat(b_comp, m_comp)
    perm_stats = np.full(n_perm, np.nan)
    for pi in range(n_perm):
        # For each subject × condition, flip with 50% probability
        flip = rng.integers(0, 2, size=b_comp.shape).astype(float)  # 0 or 1
        b_p  = np.where(flip, m_comp, b_comp)
        m_p  = np.where(flip, b_comp, m_comp)
        perm_stats[pi] = interaction_stat(b_p, m_p)

    perm_p = float((perm_stats >= obs_stat).mean())
    results['perm_stat_obs']   = obs_stat
    results['perm_p_interact'] = perm_p
    print(f"\n  Permutation test (interaction, var of diffs): "
          f"stat_obs={obs_stat:.5f}, p_perm={perm_p:.4f} ({n_perm} permutations)")

    # Post-hoc: boundary vs mid-event within each condition (paired t-test)
    posthoc_rows = []
    bonf = 3  # 3 comparisons
    for ci, cname in enumerate(CONDITION_NAMES):
        b = bnd_vals[:, ci]
        m = mid_vals[:, ci]
        paired = [(b[si], m[si]) for si in range(N_subj)
                  if not np.isnan(b[si]) and not np.isnan(m[si])]
        if len(paired) < 2:
            posthoc_rows.append(dict(condition=cname, n=len(paired),
                                     mean_bnd=np.nan, mean_mid=np.nan,
                                     t=np.nan, p_raw=np.nan, p_bonf=np.nan))
            continue
        b_arr, m_arr = zip(*paired)
        t, p_raw = ttest_rel(np.array(b_arr), np.array(m_arr))
        posthoc_rows.append(dict(
            condition=cname,
            n=len(paired),
            mean_bnd=float(np.mean(b_arr)),
            mean_mid=float(np.mean(m_arr)),
            t=float(t),
            p_raw=float(p_raw),
            p_bonf=float(min(p_raw * bonf, 1.0)),
        ))

    df_posthoc = pd.DataFrame(posthoc_rows)
    results['posthoc'] = df_posthoc
    print("\n  Post-hoc paired t-tests (boundary vs mid-event, Bonferroni k=3):")
    print(df_posthoc.to_string(index=False))

    return results


# ============================================================================
# STEP 6: FIGURES
# ============================================================================

def make_interaction_plot(bnd_vals, mid_vals, subjects, roi_name):
    """Plot condition × window interaction: lines per condition, x = window,
    with individual subject dots at each timepoint."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    x = [0, 1]

    for ci, (label, color) in enumerate(zip(CONDITION_LABELS, COLORS)):
        b = bnd_vals[:, ci]
        m = mid_vals[:, ci]
        # Fisher-z → r for display
        b_r = np.tanh(b)
        m_r = np.tanh(m)

        b_valid = b_r[~np.isnan(b_r)]
        m_valid = m_r[~np.isnan(m_r)]
        means = [b_valid.mean(), m_valid.mean()]
        sems  = [b_valid.std(ddof=1) / np.sqrt(len(b_valid)),
                 m_valid.std(ddof=1) / np.sqrt(len(m_valid))]

        ax.plot(x, means, color=color, linewidth=2, label=label, zorder=3)
        ax.errorbar(x, means, yerr=sems, fmt='none', color=color,
                    capsize=4, elinewidth=1.5, zorder=3)

        # Per-subject dots with jitter — draw connecting lines per subject first
        for si in range(len(subjects)):
            bv = b_r[si] if not np.isnan(b_r[si]) else None
            mv = m_r[si] if not np.isnan(m_r[si]) else None
            if bv is not None and mv is not None:
                jit = rng.uniform(-0.04, 0.04)
                ax.plot([0 + jit, 1 + jit], [bv, mv],
                        color=color, linewidth=0.5, alpha=0.3, zorder=2)
                ax.scatter([0 + jit, 1 + jit], [bv, mv],
                           s=20, color=color, edgecolors='black',
                           linewidths=0.4, zorder=4, alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(WINDOW_LABELS, fontsize=10)
    ax.set_ylabel('mean ISPC (Pearson r)', fontsize=9)
    ax.set_title(roi_name, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    return fig


# ============================================================================
# GLOBALS (set in main)
# ============================================================================

N_SVF       = 0
N_AHC       = 0
N_INSTANCES = 0
_ROI_KEY    = None
_PARCEL_IDS = None


# ============================================================================
# PER-ROI PIPELINE
# ============================================================================

def run_roi(roi_key, roi_name, parcel_ids, all_subjects, shared_cats, shared_prompts,
            do_hp=True, n_perm=5000, force_mid=False, use_onset=True):
    global _ROI_KEY, _PARCEL_IDS
    _ROI_KEY    = roi_key
    _PARCEL_IDS = parcel_ids

    onset_tag = '_onset' if use_onset else ''

    # --- BOUNDARY: load from cache ---
    bnd_cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6_hp{onset_tag}_post4_ispc.npz'
    if not bnd_cache.exists():
        print(f"  [{roi_key}] Boundary cache not found: {bnd_cache.name} — skipping.")
        return None

    bnd_data       = np.load(bnd_cache, allow_pickle=True)
    per_subj_bnd   = bnd_data['per_subject']   # (N_subj, N_TOTAL, N_TOTAL)
    cached_subjects = bnd_data['subjects'].tolist()
    print(f"  [{roi_key}] Boundary cache loaded: {per_subj_bnd.shape[0]} subjects")

    # Set boundary-script globals needed for collapse_per_subject_to_conditions
    _bnd.N_SVF       = N_SVF
    _bnd.N_AHC       = N_AHC
    _bnd.N_INSTANCES = N_INSTANCES
    _bnd.N_TOTAL     = N_INSTANCES * 4
    _bnd.N_WINDOWS   = 4
    _bnd.TASK_SLICES = {
        'svf':  slice(0, N_SVF),
        'ahc':  slice(N_SVF, N_SVF + N_AHC),
        'film': slice(N_SVF + N_AHC, N_INSTANCES),
    }

    per_subj_12x12 = _bnd.collapse_per_subject_to_conditions(per_subj_bnd)
    bnd_vals       = extract_boundary_condition_values(per_subj_12x12)  # (N_subj, 3)

    # --- MID-EVENT: extract or load from cache ---
    # Mid-event patterns don't depend on onset vs offset — reuse onset cache if available
    mid_cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6_hp_onset_mid_event_ispc.npz'

    if not force_mid and mid_cache.exists():
        print(f"  [{roi_key}] Mid-event cache found, loading ...")
        mid_data       = np.load(mid_cache)
        per_subj_mid   = mid_data['per_subject']  # (N_subj, N_INSTANCES, N_INSTANCES)
    else:
        print(f"  [{roi_key}] Extracting mid-event patterns ...")
        subject_patterns = []
        for subj in cached_subjects:
            print(f"    Subject: {subj}")
            pats = get_subject_mid_event_patterns(subj, shared_cats, shared_prompts, do_hp)
            if pats is None:
                print(f"    SKIP {subj}: no mid-event data")
                pats = np.full((N_INSTANCES, 1), np.nan)
            subject_patterns.append(pats)

        # Align V dimensions
        Vs = [p.shape[1] for p in subject_patterns]
        V  = min(Vs)
        all_pats = np.stack([p[:, :V] for p in subject_patterns], axis=0)
        print(f"  [{roi_key}] Computing mid-event LOO ISPC ({all_pats.shape[0]} subjects) ...")

        _, per_subj_mid = compute_mid_event_ispc(all_pats)
        np.savez_compressed(mid_cache, per_subject=per_subj_mid,
                            subjects=np.array(cached_subjects))
        print(f"  [{roi_key}] Mid-event cache → {mid_cache.name}")

    per_subj_3x3 = collapse_mid_per_subject(per_subj_mid)
    mid_vals     = extract_mid_condition_values(per_subj_3x3)   # (N_subj, 3)

    # --- STATS ---
    print(f"\n  [{roi_key}] Running statistics ...")
    results = run_stats(bnd_vals, mid_vals, cached_subjects, n_perm=n_perm)
    results['roi_key']   = roi_key
    results['roi_name']  = roi_name
    results['bnd_vals']  = bnd_vals
    results['mid_vals']  = mid_vals
    results['subjects']  = cached_subjects

    # --- FIGURE ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    fig = make_interaction_plot(bnd_vals, mid_vals, cached_subjects, roi_name)
    fig_path = OUTPUT_DIR / f'roi-{roi_key}_interaction.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{roi_key}] Interaction plot → {fig_path.name}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Formal 2×3 RM ANOVA: boundary vs mid-event ISPC across conditions.')
    parser.add_argument('--roi', nargs='+', default=[k for k, _, _ in ROI_SPEC])
    parser.add_argument('--n-perm', type=int, default=5000,
                        help='Permutation iterations (default: 5000)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-extraction of mid-event patterns')
    parser.add_argument('--onset', action='store_true',
                        help='Use onset-locked boundary cache (default: offset)')
    args = parser.parse_args()

    global N_SVF, N_AHC, N_INSTANCES, OUTPUT_DIR

    _bnd.USE_ONSET     = args.onset
    _bnd.WINDOW_PRESET = 'post4'

    boundary_label = 'onset' if args.onset else 'offset'
    OUTPUT_DIR = FIGS_DIR / 'cross_task_ispc' / 'hp' / 'post4' / boundary_label / 'formal_test'

    svf_ahc_subjects = sorted(SUBJECT_IDS)
    all_subjects     = sorted(set(svf_ahc_subjects) | set(FILMFEST_SUBJECTS.keys()))

    shared_cats,    _ = _bnd.discover_shared_categories(svf_ahc_subjects, min_subjects=5)
    shared_prompts, _ = _bnd.discover_shared_prompts(svf_ahc_subjects, min_subjects=5)

    N_SVF       = len(shared_cats)
    N_AHC       = len(shared_prompts)
    N_INSTANCES = N_SVF + N_AHC + N_FILM

    print(f"SVF categories: {N_SVF}, AHC prompts: {N_AHC}, FilmFest: {N_FILM}")
    print(f"N_INSTANCES: {N_INSTANCES}")

    roi_spec_filtered = [(k, n, p) for k, n, p in ROI_SPEC if k in args.roi]
    if not roi_spec_filtered:
        print("No valid ROIs."); return

    all_results = []
    for roi_key, roi_name, parcel_ids in roi_spec_filtered:
        print(f"\n{'='*60}\nROI: {roi_name} ({roi_key})")
        res = run_roi(
            roi_key, roi_name, parcel_ids, all_subjects,
            shared_cats, shared_prompts,
            do_hp=True, n_perm=args.n_perm, force_mid=args.no_cache,
            use_onset=args.onset,
        )
        if res is not None:
            all_results.append(res)

    # Summary CSV
    if all_results:
        csv_rows = []
        for res in all_results:
            ph = res['posthoc']
            for _, row in ph.iterrows():
                csv_rows.append(dict(
                    roi=res['roi_key'],
                    roi_name=res['roi_name'],
                    condition=row['condition'],
                    n_subjects=row['n'],
                    mean_boundary=row['mean_bnd'],
                    mean_mid_event=row['mean_mid'],
                    t_boundary_vs_mid=row['t'],
                    p_raw=row['p_raw'],
                    p_bonferroni=row['p_bonf'],
                    perm_p_interaction=res.get('perm_p_interact', np.nan),
                ))
        df_csv = pd.DataFrame(csv_rows)
        csv_path = OUTPUT_DIR / 'formal_test_results.csv'
        df_csv.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"\nResults CSV → {csv_path}")
        print(df_csv.to_string(index=False))

    print("\nDone.")


if __name__ == '__main__':
    main()
