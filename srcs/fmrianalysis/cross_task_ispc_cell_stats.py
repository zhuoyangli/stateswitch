"""
cross_task_ispc_stats.py — Statistical annotations for cross-task ISPC matrices.

Loads cached cross-task ISPC data (instance_ispc matrices) and generates:
  Output 1: Annotated 12×12 heatmaps per ROI (two versions: bootstrap markers /
            proportion-positive text overlay)
  Output 2: Summary bar charts of proportion-positive by comparison type (two
            time windows)
  Output 3: Comprehensive statistics CSV

Statistical approaches:
  Approach 1 — Binomial test: for each cell, extract all instance-pair ISPC values,
    compute proportion > 0, run two-tailed binomial test vs 0.5. FDR-correct across
    all cells per ROI (Benjamini-Hochberg).
  Approach 2 — Bootstrap CI: resample instance-pair values (10,000 iterations), take
    2.5/97.5 percentiles as 95% CI, compute two-tailed bootstrap p-value. FDR-correct
    across all cells per ROI.

NOTE: Both tests are approximate because instance pairs share subject data via the
leave-one-out computation and are therefore not fully independent. However, they are
informative about generalizability across boundary events, which is the primary claim.

Usage:
    uv run python srcs/fmrianalysis/cross_task_ispc_stats.py --hp
    uv run python srcs/fmrianalysis/cross_task_ispc_stats.py --hp --vmax 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import binomtest, ttest_1samp
from statsmodels.stats.multitest import multipletests

from configs.config import FIGS_DIR, ANALYSIS_CACHE_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULT_CACHE_DIR = ANALYSIS_CACHE_DIR / 'cross_task_ispc'

N_WINDOWS    = 4
N_FILM       = 8
N_CONDITIONS = 12   # 3 tasks × 4 windows

# Default trial counts (override with --n-svf / --n-ahc if needed)
_DEFAULT_N_SVF = 13
_DEFAULT_N_AHC = 8

# HRF preset window labels (shifted by −4 TRs to account for HRF delay)
WIN_LABELS_HRF = ['-10 to -1 TRs', '0 to 9 TRs', '10 to 19 TRs', '20 to 29 TRs']

# "First" and "second" time windows for bar charts (onset and post in hrf preset)
WIN_ONSET_IDX = 1   # 0 to 9 TRs
WIN_POST_IDX  = 2   # 10 to 19 TRs

ROI_SPEC = [
    ('eac',   'Early Auditory Cortex'),
    ('evc',   'Early Visual Cortex'),
    ('pmc',   'Posterior Medial Cortex'),
    ('ag',    'Angular Gyrus'),
    ('dlpfc', 'Dorsolateral PFC'),
    ('dacc',  'Dorsal Anterior Cingulate'),
    ('mpfc',  'Medial Prefrontal Cortex'),
    ('hipp',  'Hippocampus'),
]

ROI_ABBREVS = {
    'eac':   'EAC',
    'evc':   'EVC',
    'pmc':   'PMC',
    'ag':    'AG',
    'dlpfc': 'dlPFC',
    'dacc':  'dACC',
    'mpfc':  'mPFC',
    'hipp':  'Hipp',
}

# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def build_cond_indices(n_svf, n_ahc, n_film=N_FILM, n_windows=N_WINDOWS):
    """Return list of 12 index arrays mapping 12×12 conditions → N_TOTAL-vector rows.

    N_TOTAL-vector ordering (window-first):
      [win0: svf_0..svf_{N_SVF-1}, ahc_0..ahc_{N_AHC-1}, film_0..film_{N_FILM-1},
       win1: same, win2: same, win3: same]

    Condition ordering: SVF-win0..win3, AHC-win0..win3, Film-win0..win3.
    """
    n_instances  = n_svf + n_ahc + n_film
    task_starts  = [0, n_svf, n_svf + n_ahc]
    task_sizes   = [n_svf, n_ahc, n_film]
    cond_indices = []
    for task_start, task_size in zip(task_starts, task_sizes):
        for win_idx in range(n_windows):
            start = win_idx * n_instances + task_start
            cond_indices.append(np.arange(start, start + task_size))
    return cond_indices   # list of 12 arrays


def cond_idx(task_idx, win_idx, n_windows=N_WINDOWS):
    """Flat condition index for task × window."""
    return task_idx * n_windows + win_idx


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def fdr_correct(p_flat):
    """BH FDR correction on a flat array (may contain NaN). Returns corrected array."""
    result = np.full_like(p_flat, np.nan)
    valid  = ~np.isnan(p_flat)
    if valid.sum() > 0:
        _, p_corr, _, _ = multipletests(p_flat[valid], method='fdr_bh')
        result[valid] = p_corr
    return result


def extract_block(instance_ispc, cond_indices, ci, cj):
    """Extract non-NaN instance-pair values for cell (ci, cj) in the 12×12 matrix."""
    vals = instance_ispc[np.ix_(cond_indices[ci], cond_indices[cj])].ravel()
    return vals[~np.isnan(vals)]


def binomial_test_cell(values):
    """Proportion positive + two-tailed binomial test vs 0.5."""
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    n_pos  = int((values > 0).sum())
    prop   = n_pos / n
    result = binomtest(n_pos, n, p=0.5, alternative='two-sided')
    return prop, result.pvalue


def bootstrap_test_cell(values, n_boot=10_000, rng=None):
    """Vectorised bootstrap: 95% CI + two-tailed p-value for mean vs 0."""
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    idx        = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    obs   = values.mean()
    p     = 2.0 * (float((boot_means <= 0).mean()) if obs > 0
                   else float((boot_means >= 0).mean()))
    return ci_lo, ci_hi, min(p, 1.0)


def compute_roi_stats(instance_ispc, cond_indices, n_boot=10_000, seed=42):
    """Compute Approach-1 and Approach-2 stats for every cell in a 12×12 matrix.

    Returns dict of (12,12) arrays + FDR-corrected variants.
    """
    rng    = np.random.default_rng(seed)
    nc     = N_CONDITIONS

    prop_pos   = np.full((nc, nc), np.nan)
    binom_p    = np.full((nc, nc), np.nan)
    boot_lo    = np.full((nc, nc), np.nan)
    boot_hi    = np.full((nc, nc), np.nan)
    boot_p     = np.full((nc, nc), np.nan)
    n_pairs    = np.zeros((nc, nc), dtype=int)
    mean_ispc  = np.full((nc, nc), np.nan)

    for ci in range(nc):
        for cj in range(nc):
            vals = extract_block(instance_ispc, cond_indices, ci, cj)
            n_pairs[ci, cj]   = len(vals)
            if len(vals) == 0:
                continue
            mean_ispc[ci, cj] = vals.mean()
            prop, bp           = binomial_test_cell(vals)
            prop_pos[ci, cj]  = prop
            binom_p[ci, cj]   = bp
            lo, hi, bootp      = bootstrap_test_cell(vals, n_boot=n_boot, rng=rng)
            boot_lo[ci, cj]   = lo
            boot_hi[ci, cj]   = hi
            boot_p[ci, cj]    = bootp

    binom_p_fdr = fdr_correct(binom_p.ravel()).reshape(nc, nc)
    boot_p_fdr  = fdr_correct(boot_p.ravel()).reshape(nc, nc)

    return dict(
        mean_ispc=mean_ispc,
        prop_pos=prop_pos,
        binom_p=binom_p,
        binom_p_fdr=binom_p_fdr,
        boot_lo=boot_lo,
        boot_hi=boot_hi,
        boot_p=boot_p,
        boot_p_fdr=boot_p_fdr,
        n_pairs=n_pairs,
    )


# ---------------------------------------------------------------------------
# Annotated heatmap figures (Output 1)
# ---------------------------------------------------------------------------

def _text_color(value, vmax, threshold=0.4):
    """White text on saturated (dark) cells, black on unsaturated (light) cells."""
    return 'white' if abs(value) / vmax > threshold else 'black'


def _add_heatmap_structure(ax, windows):
    """Add gridlines, task-block dividers, and optional pre/onset red line."""
    # Fine cell gridlines
    for pos in np.arange(0.5, N_CONDITIONS - 1, 1):
        ax.axhline(pos, color='white', linewidth=0.3, alpha=0.5)
        ax.axvline(pos, color='white', linewidth=0.3, alpha=0.5)
    # Task block dividers
    for edge in [3.5, 7.5]:
        ax.axhline(edge, color='black', linewidth=2.0)
        ax.axvline(edge, color='black', linewidth=2.0)
    # Pre/onset boundary (red) when first window is pre-boundary
    if windows[0][1] < 0:
        for task_start in [0, 4, 8]:
            ax.axhline(task_start + 0.5, color='red', linewidth=1.5)
            ax.axvline(task_start + 0.5, color='red', linewidth=1.5)


def make_annotated_matrix_figure(cond_matrix, stats, roi_name,
                                  cond_labels, windows,
                                  version='bootstrap', vmax=0.5):
    """12×12 heatmap with statistical annotations overlaid.

    version='bootstrap' : asterisk (*) in FDR-significant bootstrap cells
    version='proportion': proportion-positive text (e.g., ".72") in each cell
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.subplots_adjust(left=0.20, right=0.88, top=0.88, bottom=0.22)

    im = ax.imshow(cond_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal', origin='upper', interpolation='none')

    ax.set_xticks(range(N_CONDITIONS))
    ax.set_xticklabels(cond_labels, fontsize=6.5, rotation=45, ha='right')
    ax.set_yticks(range(N_CONDITIONS))
    ax.set_yticklabels(cond_labels, fontsize=6.5)

    _add_heatmap_structure(ax, windows)

    # Overlay annotations
    for ci in range(N_CONDITIONS):
        for cj in range(N_CONDITIONS):
            cell_val = cond_matrix[ci, cj]
            if np.isnan(cell_val):
                continue
            tc = _text_color(cell_val, vmax)

            if version == 'bootstrap':
                # Asterisk if FDR-corrected bootstrap p < 0.05 and CI excludes 0
                sig = (stats['boot_p_fdr'][ci, cj] < 0.05)
                if sig:
                    ax.text(cj, ci, '*', ha='center', va='center',
                            fontsize=11, fontweight='bold', color=tc)

            elif version == 'proportion':
                prop = stats['prop_pos'][ci, cj]
                if not np.isnan(prop):
                    label = f'{prop:.2f}'[1:]   # ".72"
                    ax.text(cj, ci, label, ha='center', va='center',
                            fontsize=4.5, color=tc)

    ax.set_title(roi_name, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04, shrink=0.7)
    cbar.set_label('inter-subject pattern correlation (r)', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    ann_label = ('* = FDR-corrected bootstrap p < 0.05'
                 if version == 'bootstrap'
                 else 'proportion of instance pairs with ISPC > 0')
    fig.text(0.5, 0.01, ann_label, ha='center', fontsize=7, style='italic')
    return fig


# ---------------------------------------------------------------------------
# Bar chart helpers (Output 2)
# ---------------------------------------------------------------------------

def _pooled_binom(instance_ispc, cond_indices, cell_pairs):
    """Pool instance-pair values from a list of (ci, cj) cell pairs and run binomial."""
    all_vals = np.concatenate([
        extract_block(instance_ispc, cond_indices, ci, cj)
        for ci, cj in cell_pairs
    ])
    if len(all_vals) == 0:
        return np.nan, np.nan, np.nan, 0
    prop, p = binomial_test_cell(all_vals)
    return prop, p, all_vals.mean(), len(all_vals)


def _pooled_mean_stats(instance_ispc, cond_indices, cell_pairs):
    """Pool instance-pair values, return (mean, sem, ttest_p, n)."""
    all_vals = np.concatenate([
        extract_block(instance_ispc, cond_indices, ci, cj)
        for ci, cj in cell_pairs
    ])
    if len(all_vals) == 0:
        return np.nan, np.nan, np.nan, 0
    mu  = all_vals.mean()
    sem = all_vals.std(ddof=1) / np.sqrt(len(all_vals))
    p   = ttest_1samp(all_vals, 0.0).pvalue
    return mu, sem, p, len(all_vals)


def _per_subject_means(per_subj_ispc, cond_indices, cell_pairs):
    """Per-subject mean ISPC across cell pairs. Returns (N_subj,) array."""
    N_subj = per_subj_ispc.shape[0]
    vals   = np.full(N_subj, np.nan)
    for s in range(N_subj):
        blocks = [extract_block(per_subj_ispc[s], cond_indices, ci, cj)
                  for ci, cj in cell_pairs]
        all_v  = np.concatenate(blocks)
        if len(all_v) > 0:
            vals[s] = np.nanmean(all_v)
    return vals


def _comparison_cell_pairs(win_idx, n_windows=N_WINDOWS):
    """Return cell-pair lists for the three comparison types at a given window index."""
    svf  = cond_idx(0, win_idx, n_windows)
    ahc  = cond_idx(1, win_idx, n_windows)
    film = cond_idx(2, win_idx, n_windows)

    within   = [(svf, svf), (ahc, ahc), (film, film)]
    xsame    = [(svf, ahc), (ahc, svf)]                      # SVF × AHC (both directions)
    xdiff    = [(svf, film), (film, svf), (ahc, film), (film, ahc)]  # ×Film
    return within, xsame, xdiff


COMPARISON_LABELS = ['within-task', 'cross-task\n(SVF × AHC)', 'cross-task\n(× FilmFest)']
COMPARISON_LABELS_SHORT = ['within-task', 'SVF × AHC', '× FilmFest']
COMPARISON_COLORS = ['#4878CF', '#D65F5F', '#6ACC65']


def make_summary_bar_chart(roi_bar_data, win_label, win_idx):
    """Grouped bar chart of mean ISPC with SEM error bars and per-subject dots.

    roi_bar_data: list of dicts, one per ROI, each containing:
      'abbrev', 'mean'=[within, xsame, xdiff], 'sem'=[...], 'sig'=[...],
      'subj_vals'=[(N_subj,), ...] per comparison type
    win_label: string describing the window (e.g. '0 to 9 TRs')
    win_idx: integer, used in title only
    """
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

        # Per-subject dots
        for xi, (bar, d) in enumerate(zip(bars, roi_bar_data)):
            sv = d['subj_vals'][ci]
            if sv is None or np.all(np.isnan(sv)):
                continue
            sv_valid = sv[~np.isnan(sv)]
            jitter   = rng.uniform(-bar_w * 0.3, bar_w * 0.3, size=len(sv_valid))
            ax.scatter(bar.get_x() + bar_w / 2 + jitter, sv_valid,
                       s=18, color=color, edgecolors='black', linewidths=0.4,
                       zorder=4, alpha=0.9)

        for bar, sig, mu, sem in zip(bars, sigs, means, sems):
            if np.isnan(mu):
                continue
            if sig:
                y_ast = max(mu, 0) + (sem if not np.isnan(sem) else 0) + 0.004
                ax.text(bar.get_x() + bar.get_width() / 2, y_ast, '*',
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        color=color)

    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([d['abbrev'] for d in roi_bar_data], fontsize=10)
    ax.set_ylabel('mean ISPC (± SEM)', fontsize=9)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=8, framealpha=0.9)
    ax.set_title(
        f'Mean cross-task ISPC — {win_label}\n'
        '* = FDR-corrected t-test across subjects vs 0, p < 0.05 (corrected across ROIs × comparison types)',
        fontsize=9, fontweight='bold')
    return fig


def _draw_colored_segments(ax, xi, y_ax, segments, fontsize):
    """Draw colored text segments centered at (xi data, y_ax axes-fraction).

    Places texts, measures actual rendered widths via the canvas renderer,
    then repositions so the full string is horizontally centered at xi.
    segments: list of (text_string, color) tuples.
    """
    from matplotlib.transforms import blended_transform_factory
    fig  = ax.get_figure()
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # Place all segments at the anchor point (positions corrected below)
    texts = []
    for txt, color in segments:
        t = ax.text(xi, y_ax, txt, fontsize=fontsize, color=color,
                    ha='left', va='top', transform=trans)
        texts.append(t)

    # Render to get accurate text extents
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer).width for t in texts]
    total_w = sum(widths)

    # Anchor in display (pixel) coords
    cx, cy = trans.transform((xi, y_ax))
    x_cur  = cx - total_w / 2.0

    # Reposition each segment using figure-fraction transform
    fig_inv = fig.transFigure.inverted()
    for t, w in zip(texts, widths):
        xf, yf = fig_inv.transform((x_cur, cy))
        t.set_transform(fig.transFigure)
        t.set_position((xf, yf))
        x_cur += w


def make_multi_roi_combined_bar_chart(roi_entries):
    """Combined figure: one subplot per ROI (shared y-axis).

    roi_entries: list of (roi_name, bnd_data, mid_data) tuples.
    """
    import matplotlib.patches as mpatches
    n = len(roi_entries)
    rng   = np.random.default_rng(0)
    bar_w = 0.35
    x     = np.arange(3)

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.80, top=0.88, bottom=0.22, wspace=0.25)

    for col, (ax, (roi_name, bnd_data, mid_data)) in enumerate(zip(axes, roi_entries)):
        for wi, (wdata, facecolor, hatch) in enumerate([
                (bnd_data, 'white',     ''),
                (mid_data, '#AAAAAA',   ''),
        ]):
            offset = (wi - 0.5) * bar_w
            for ci in range(3):
                mu  = wdata['mean'][ci]
                sem = wdata['sem'][ci]
                sig = wdata['sig'][ci]
                sv  = wdata['subj_vals'][ci]

                ax.bar(x[ci] + offset, mu, bar_w,
                       color=facecolor, edgecolor='black', linewidth=0.8,
                       hatch=hatch, yerr=sem,
                       error_kw=dict(elinewidth=1.0, capsize=3, ecolor='black'))

                if sv is not None:
                    sv_valid = sv[~np.isnan(sv)]
                    jitter   = rng.uniform(-bar_w * 0.25, bar_w * 0.25, size=len(sv_valid))
                    ax.scatter(x[ci] + offset + jitter, sv_valid,
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
        _sub_segments = [
            None,
            [('SF', 'red'), (' × ', 'black'), ('EG', 'orange')],
            [('{', 'black'), ('SF', 'red'), (', ', 'black'), ('EG', 'orange'),
             ('} × ', 'black'), ('MW', 'blue')],
        ]
        for xi, main, sub_segs in zip(x, _main_labels, _sub_segments):
            ax.text(xi, -0.04, main, fontsize=8, ha='center', va='top', transform=trans)
            if sub_segs:
                _draw_colored_segments(ax, xi, -0.11, sub_segs, fontsize=6.5)
        ax.tick_params(which='both', bottom=False)
        ax.set_title(roi_name, fontsize=10, fontweight='bold')
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if col == 0:
            ax.spines['left'].set_visible(True)
            ax.set_yticks([0, 0.1, 0.2, 0.3])
            ax.set_yticklabels(['0', '0.1', '0.2', '0.3'])
            ax.set_ylabel('inter-subject pattern correlation (r)', fontsize=9)
        else:
            ax.tick_params(which='both', left=False)

    leg_handles = [
        mpatches.Patch(facecolor='white',   edgecolor='black', label='boundary'),
        mpatches.Patch(facecolor='#AAAAAA', edgecolor='black', label='mid-event'),
    ]
    axes[0].legend(handles=leg_handles, loc='upper right',
                   bbox_to_anchor=(1.0, 0.95), fontsize=7.5, frameon=False)
    return fig


def make_combined_window_bar_chart(roi_name, bnd_data, mid_data):
    """6-bar chart per ROI: 3 comparison types × 2 windows (boundary, mid-event).

    bnd_data / mid_data: dicts with 'mean', 'sem', 'sig', 'subj_vals' lists (len 3).
    Boundary = solid fill; mid-event = hatched.
    """
    rng   = np.random.default_rng(0)
    bar_w = 0.35
    x     = np.arange(3)  # 3 comparison types

    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.subplots_adjust(left=0.12, right=0.72, top=0.85, bottom=0.14)

    for wi, (wdata, wlabel, hatch) in enumerate([
            (bnd_data,  'post-boundary', ''),
            (mid_data,  'mid-event',     '///'),
    ]):
        offset = (wi - 0.5) * bar_w
        for ci, color in enumerate(COMPARISON_COLORS):
            mu  = wdata['mean'][ci]
            sem = wdata['sem'][ci]
            sig = wdata['sig'][ci]
            sv  = wdata['subj_vals'][ci]

            bar = ax.bar(x[ci] + offset, mu, bar_w,
                         color=color, alpha=0.75 if hatch == '' else 0.45,
                         edgecolor='black', linewidth=0.5,
                         hatch=hatch,
                         yerr=sem,
                         error_kw=dict(elinewidth=1.0, capsize=3, ecolor='black'),
                         label=f'{COMPARISON_LABELS_SHORT[ci]} ({wlabel})' if ci == 0 else '_')

            # Subject dots
            if sv is not None:
                sv_valid = sv[~np.isnan(sv)]
                jitter   = rng.uniform(-bar_w * 0.25, bar_w * 0.25, size=len(sv_valid))
                ax.scatter(x[ci] + offset + jitter, sv_valid,
                           s=18, color=color, edgecolors='black',
                           linewidths=0.4, zorder=4, alpha=0.9)

            if sig and not np.isnan(mu):
                y_ast = max(mu, 0) + (sem if not np.isnan(sem) else 0) + 0.004
                ax.text(x[ci] + offset, y_ast, '*',
                        ha='center', va='bottom', fontsize=11,
                        fontweight='bold', color=color)

    # Custom legend: boundary (solid) vs mid-event (hatched) patches
    import matplotlib.patches as mpatches
    leg_handles = [
        mpatches.Patch(facecolor=COMPARISON_COLORS[ci], edgecolor='black',
                       label=COMPARISON_LABELS_SHORT[ci])
        for ci in range(3)
    ] + [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='///',
                       label='mid-event'),
        mpatches.Patch(facecolor='white', edgecolor='black',
                       label='post-boundary'),
    ]
    ax.legend(handles=leg_handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=7.5, framealpha=0.9)

    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(COMPARISON_LABELS_SHORT, fontsize=9)
    ax.set_ylabel('mean ISPC (± SEM)', fontsize=9)
    ax.set_title(roi_name, fontsize=10, fontweight='bold')
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    return fig


# ---------------------------------------------------------------------------
# Statistics CSV (Output 3)
# ---------------------------------------------------------------------------

COMP_NAMES = ['within_task', 'cross_task_same_modality', 'cross_task_diff_modality']


def build_stats_rows(roi_key, roi_name, instance_ispc, cond_indices, stats,
                     win_idx, win_label):
    """Return list of dicts (one per comparison type) for one ROI × window."""
    rows = []
    within_pairs, xsame_pairs, xdiff_pairs = _comparison_cell_pairs(win_idx)
    for comp_name, pairs in zip(COMP_NAMES,
                                [within_pairs, xsame_pairs, xdiff_pairs]):
        prop, binom_p, mean_ispc, n = _pooled_binom(instance_ispc, cond_indices, pairs)
        rows.append(dict(
            roi=roi_key,
            roi_name=roi_name,
            comparison_type=comp_name,
            time_window=win_label,
            mean_ispc=mean_ispc,
            proportion_positive=prop,
            n_instance_pairs=n,
            binomial_p_raw=binom_p,
            # FDR fields filled in later across ROIs
            binomial_p_fdr=np.nan,
            # Bootstrap stats from the per-cell data (average across relevant cells)
            bootstrap_CI_lower=np.nanmean([stats['boot_lo'][ci, cj] for ci, cj in pairs]),
            bootstrap_CI_upper=np.nanmean([stats['boot_hi'][ci, cj] for ci, cj in pairs]),
            bootstrap_p_raw=np.nanmean([stats['boot_p'][ci, cj] for ci, cj in pairs]),
            bootstrap_p_fdr=np.nan,
        ))
    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Statistical annotation of cross-task ISPC matrices.')
    parser.add_argument('--hp', action='store_true',
                        help='Use high-pass filtered ISPC cache')
    parser.add_argument('--preset', default='hrf', choices=['hrf', 'post4', 'onset0'],
                        help='Window preset (default: hrf)')
    parser.add_argument('--n-svf', type=int, default=_DEFAULT_N_SVF,
                        help=f'Number of SVF categories (default: {_DEFAULT_N_SVF})')
    parser.add_argument('--n-ahc', type=int, default=_DEFAULT_N_AHC,
                        help=f'Number of AHC prompts (default: {_DEFAULT_N_AHC})')
    parser.add_argument('--n-boot', type=int, default=10_000,
                        help='Bootstrap iterations (default: 10000)')
    parser.add_argument('--vmax', type=float, default=0.5)
    parser.add_argument('--roi', nargs='+', default=[k for k, _ in ROI_SPEC])
    parser.add_argument('--onset', action='store_true',
                        help='Use onset-boundary ISPC cache')
    parser.add_argument('--multi-roi-only', action='store_true',
                        help='Skip per-ROI figures; only regenerate multi_roi_boundary_vs_mid.png')
    args = parser.parse_args()

    hp_tag     = '_hp' if args.hp else ''
    onset_tag  = '_onset' if args.onset else ''
    hp_label   = 'hp' if args.hp else 'no_hp'
    onset_label = 'onset' if args.onset else 'offset'
    out_base = FIGS_DIR / 'cross_task_ispc' / hp_label / args.preset / onset_label

    # Window labels for the two bar-chart windows
    if args.preset == 'hrf':
        windows_raw = [('pre', -6, 3), ('onset', 4, 13), ('post', 14, 23), ('late', 24, 33)]
        win_labels  = WIN_LABELS_HRF
    elif args.preset == 'onset0':
        windows_raw = [('onset', 0, 9), ('post', 10, 19), ('late', 20, 29), ('vlate', 30, 39)]
        win_labels  = ['0 to 9 TRs', '10 to 19 TRs', '20 to 29 TRs', '30 to 39 TRs']
    else:
        windows_raw = [('onset', 4, 13), ('post', 14, 23), ('late', 24, 33), ('vlate', 34, 43)]
        win_labels  = ['0 to 9 TRs', '10 to 19 TRs', '20 to 29 TRs', '30 to 39 TRs']

    cond_labels = [f'{task}\n{w}'
                   for task in ('SVF', 'AHC', 'Film')
                   for w in win_labels]

    cond_indices = build_cond_indices(args.n_svf, args.n_ahc)
    roi_spec_filtered = [(k, n) for k, n in ROI_SPEC if k in args.roi]

    # ------------------------------------------------------------------ #
    # Per-ROI: load cache, compute stats, save annotated figures          #
    # ------------------------------------------------------------------ #
    all_stats          = {}   # roi_key → stats dict
    all_instance_ispc  = {}   # roi_key → (N_TOTAL, N_TOTAL)
    all_per_subj_ispc  = {}   # roi_key → (N_subj, N_TOTAL, N_TOTAL)

    for roi_key, roi_name in roi_spec_filtered:
        cache_path = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6{hp_tag}{onset_tag}_{args.preset}_ispc.npz'
        if not cache_path.exists():
            print(f'[{roi_key}] Cache not found: {cache_path.name} — skipping.')
            continue

        print(f'[{roi_key}] Loading cache and computing stats ...')
        data          = np.load(cache_path)
        instance_ispc = data['instance_ispc']
        cond_matrix   = data['cond_matrix']
        per_subj_ispc = data['per_subject'] if 'per_subject' in data else None

        stats = compute_roi_stats(instance_ispc, cond_indices,
                                  n_boot=args.n_boot, seed=42)
        all_stats[roi_key]         = stats
        all_instance_ispc[roi_key] = instance_ispc
        all_per_subj_ispc[roi_key] = per_subj_ispc

        # -- Annotated figures --
        for version in ('bootstrap', 'proportion'):
            subdir = out_base / f'annotated_{version}'
            subdir.mkdir(parents=True, exist_ok=True)
            fig = make_annotated_matrix_figure(
                cond_matrix, stats, roi_name,
                cond_labels, windows_raw,
                version=version, vmax=args.vmax)
            fig_path = subdir / f'roi-{roi_key}_annotated_{version}.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'  [{roi_key}] {version} figure → {fig_path.name}')

    if not all_stats:
        print('No ROIs processed — exiting.')
        return

    # ------------------------------------------------------------------ #
    # Bar charts and CSV: two windows (onset + post)                      #
    # ------------------------------------------------------------------ #
    bar_windows = [(i, win_labels[i]) for i in range(len(win_labels))]

    all_csv_rows = []   # accumulated for CSV

    for win_idx, win_label in bar_windows:
        # Collect per-ROI bar data and raw p-values (for FDR across ROIs)
        roi_bar_data = []
        raw_p_by_comp = [[], [], []]   # lists of (roi_key, p) for FDR

        for roi_key, roi_name in roi_spec_filtered:
            if roi_key not in all_stats:
                continue
            stats        = all_stats[roi_key]
            inst_ispc    = all_instance_ispc[roi_key]
            within_pairs, xsame_pairs, xdiff_pairs = _comparison_cell_pairs(win_idx)

            props, binom_ps, means, sems, ttest_ps, ns = [], [], [], [], [], []
            subj_vals = []
            per_subj  = all_per_subj_ispc.get(roi_key)
            for pairs in [within_pairs, xsame_pairs, xdiff_pairs]:
                prop, bp, mu, n = _pooled_binom(inst_ispc, cond_indices, pairs)
                props.append(prop)
                binom_ps.append(bp)
                ns.append(n)
                if per_subj is not None:
                    sv = _per_subject_means(per_subj, cond_indices, pairs)
                    sv_valid = sv[~np.isnan(sv)]
                    mu2  = float(np.nanmean(sv)) if len(sv_valid) > 0 else np.nan
                    sem  = sv_valid.std(ddof=1) / np.sqrt(len(sv_valid)) if len(sv_valid) >= 2 else np.nan
                    tp   = ttest_1samp(sv_valid, 0.0).pvalue if len(sv_valid) >= 2 else np.nan
                    subj_vals.append(sv)
                else:
                    mu2, sem, tp, _ = _pooled_mean_stats(inst_ispc, cond_indices, pairs)
                    subj_vals.append(None)
                means.append(mu2)
                sems.append(sem)
                ttest_ps.append(tp)

            roi_bar_data.append(dict(
                roi_key=roi_key,
                abbrev=ROI_ABBREVS.get(roi_key, roi_key.upper()),
                prop=props,
                mean=means,
                sem=sems,
                p=ttest_ps,
                sig=[False, False, False],  # filled after FDR
                subj_vals=subj_vals,
            ))

            for ci, p in enumerate(ttest_ps):
                raw_p_by_comp[ci].append((roi_key, p))

            # CSV rows (bootstrap CI averaged across cell pairs)
            rows = build_stats_rows(roi_key, roi_name, inst_ispc, cond_indices,
                                    stats, win_idx, win_label)
            all_csv_rows.extend(rows)

        # FDR correction across all comparison types × ROIs (3 × N_ROIs tests)
        all_p_tuples = [(roi_key, ci, p)
                        for ci, p_list in enumerate(raw_p_by_comp)
                        for roi_key, p in p_list]
        ps_all    = np.array([t[2] for t in all_p_tuples], dtype=float)
        valid_all = ~np.isnan(ps_all)
        p_fdr_all = np.full(len(ps_all), np.nan)
        if valid_all.sum() > 0:
            _, p_corr, _, _ = multipletests(ps_all[valid_all], method='fdr_bh')
            p_fdr_all[valid_all] = p_corr
        for idx, (roi_key, ci, _) in enumerate(all_p_tuples):
            for d in roi_bar_data:
                if d['roi_key'] == roi_key:
                    d['sig'][ci] = bool(p_fdr_all[idx] < 0.05)
            for row in all_csv_rows:
                if (row['roi'] == roi_key
                        and row['time_window'] == win_label
                        and row['comparison_type'] == COMP_NAMES[ci]):
                    row['binomial_p_fdr'] = p_fdr_all[idx]

        # -- Bar chart figure --
        fig = make_summary_bar_chart(roi_bar_data, win_label, win_idx)
        bar_dir = out_base / 'summary_bar_charts'
        bar_dir.mkdir(parents=True, exist_ok=True)
        bar_path = bar_dir / f'proportion_positive_{win_idx:02d}_{win_label.replace(" ", "_")}.png'
        fig.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'\nBar chart ({win_label}) → {bar_path.name}')

    # ------------------------------------------------------------------ #
    # Bootstrap FDR for CSV (across all cells per ROI — already done      #
    # per-cell; here we update the comparison-type rows)                  #
    # For CSV bootstrap p_fdr: re-compute across ROIs per (comp × window) #
    # ------------------------------------------------------------------ #
    for win_idx, win_label in bar_windows:
        for ci, comp_name in enumerate(COMP_NAMES):
            rkeys, boot_ps = [], []
            for row in all_csv_rows:
                if row['time_window'] == win_label and row['comparison_type'] == comp_name:
                    rkeys.append(row['roi'])
                    boot_ps.append(row['bootstrap_p_raw'])
            boot_ps = np.array(boot_ps, dtype=float)
            valid   = ~np.isnan(boot_ps)
            p_fdr   = np.full(len(boot_ps), np.nan)
            if valid.sum() > 0:
                _, p_corr, _, _ = multipletests(boot_ps[valid], method='fdr_bh')
                p_fdr[valid] = p_corr
            for ri, roi_key in enumerate(rkeys):
                for row in all_csv_rows:
                    if (row['roi'] == roi_key
                            and row['time_window'] == win_label
                            and row['comparison_type'] == comp_name):
                        row['bootstrap_p_fdr'] = p_fdr[ri]

    # ------------------------------------------------------------------ #
    # Mid-event bar chart (from formal_test mid-event ISPC cache)         #
    # ------------------------------------------------------------------ #
    n_instances = args.n_svf + args.n_ahc + N_FILM
    mid_cond_indices = [
        np.arange(0, args.n_svf),
        np.arange(args.n_svf, args.n_svf + args.n_ahc),
        np.arange(args.n_svf + args.n_ahc, n_instances),
    ]
    mid_cell_pairs = [
        [(0, 0), (1, 1), (2, 2)],          # within-task
        [(0, 1), (1, 0)],                   # xsame
        [(0, 2), (2, 0), (1, 2), (2, 1)],  # xdiff
    ]

    mid_roi_data = []
    mid_raw_p_by_comp = [[], [], []]

    for roi_key, roi_name in roi_spec_filtered:
        mid_cache = RESULT_CACHE_DIR / f'roi-{roi_key}_sm6{hp_tag}_onset_mid_event_ispc.npz'
        if not mid_cache.exists():
            continue
        mid_data     = np.load(mid_cache)
        per_subj_mid = mid_data['per_subject']   # (N_subj, N_INSTANCES, N_INSTANCES)
        z_stack      = np.arctanh(np.clip(per_subj_mid, -0.999, 0.999))
        inst_ispc_mid = np.tanh(np.nanmean(z_stack, axis=0))  # (N_INSTANCES, N_INSTANCES)

        means, sems, ttest_ps, subj_vals_mid = [], [], [], []
        for pairs in [mid_cell_pairs[0], mid_cell_pairs[1], mid_cell_pairs[2]]:
            sv = _per_subject_means(per_subj_mid, mid_cond_indices, pairs)
            sv_valid = sv[~np.isnan(sv)]
            mu  = float(np.nanmean(sv)) if len(sv_valid) > 0 else np.nan
            sem = sv_valid.std(ddof=1) / np.sqrt(len(sv_valid)) if len(sv_valid) >= 2 else np.nan
            tp  = ttest_1samp(sv_valid, 0.0).pvalue if len(sv_valid) >= 2 else np.nan
            means.append(mu)
            sems.append(sem)
            ttest_ps.append(tp)
            subj_vals_mid.append(sv)

        mid_roi_data.append(dict(
            roi_key=roi_key,
            abbrev=ROI_ABBREVS.get(roi_key, roi_key.upper()),
            mean=means, sem=sems, p=ttest_ps,
            sig=[False, False, False],
            subj_vals=subj_vals_mid,
        ))
        for ci, p in enumerate(ttest_ps):
            mid_raw_p_by_comp[ci].append((roi_key, p))

    if mid_roi_data:
        # FDR across all comparison types × ROIs (3 × N_ROIs tests)
        mid_all_p = [(roi_key, ci, p)
                     for ci, p_list in enumerate(mid_raw_p_by_comp)
                     for roi_key, p in p_list]
        ps_mid    = np.array([t[2] for t in mid_all_p], dtype=float)
        valid_mid = ~np.isnan(ps_mid)
        p_fdr_mid = np.full(len(ps_mid), np.nan)
        if valid_mid.sum() > 0:
            _, p_corr, _, _ = multipletests(ps_mid[valid_mid], method='fdr_bh')
            p_fdr_mid[valid_mid] = p_corr
        for idx, (roi_key, ci, _) in enumerate(mid_all_p):
            for d in mid_roi_data:
                if d['roi_key'] == roi_key:
                    d['sig'][ci] = bool(p_fdr_mid[idx] < 0.05)

        fig = make_summary_bar_chart(mid_roi_data, 'mid-event', -1)
        bar_dir = out_base / 'summary_bar_charts'
        bar_dir.mkdir(parents=True, exist_ok=True)
        bar_path = bar_dir / 'mean_ispc_mid_event.png'
        fig.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'\nBar chart (mid-event) → {bar_path.name}')

    # ------------------------------------------------------------------ #
    # Per-ROI combined figure: 0-9 TRs window + mid-event (6 conditions) #
    # ------------------------------------------------------------------ #
    bnd_win0_by_roi = {}
    raw_p_win0 = [[], [], []]
    within_p0, xsame_p0, xdiff_p0 = _comparison_cell_pairs(0)
    for roi_key, roi_name in roi_spec_filtered:
        if roi_key not in all_instance_ispc:
            continue
        inst_ispc = all_instance_ispc[roi_key]
        per_subj  = all_per_subj_ispc.get(roi_key)
        means2, sems2, ttest_ps2, subj_vals2 = [], [], [], []
        for ci, pairs in enumerate([within_p0, xsame_p0, xdiff_p0]):
            if per_subj is not None:
                sv2 = _per_subject_means(per_subj, cond_indices, pairs)
                sv2_valid = sv2[~np.isnan(sv2)]
                mu  = float(np.nanmean(sv2)) if len(sv2_valid) > 0 else np.nan
                sem = sv2_valid.std(ddof=1) / np.sqrt(len(sv2_valid)) if len(sv2_valid) >= 2 else np.nan
                tp  = ttest_1samp(sv2_valid, 0.0).pvalue if len(sv2_valid) >= 2 else np.nan
                subj_vals2.append(sv2)
            else:
                mu, sem, tp, _ = _pooled_mean_stats(inst_ispc, cond_indices, pairs)
                subj_vals2.append(None)
            means2.append(mu); sems2.append(sem); ttest_ps2.append(tp)
            raw_p_win0[ci].append((roi_key, tp))
        bnd_win0_by_roi[roi_key] = dict(
            roi_name=roi_name, mean=means2, sem=sems2,
            sig=[False]*3, subj_vals=subj_vals2,
        )

    win0_all_p  = [(roi_key, ci, p)
                   for ci, p_list in enumerate(raw_p_win0)
                   for roi_key, p in p_list]
    ps_win0    = np.array([t[2] for t in win0_all_p], dtype=float)
    valid_win0 = ~np.isnan(ps_win0)
    p_fdr_win0 = np.full(len(ps_win0), np.nan)
    if valid_win0.sum() > 0:
        _, p_corr2, _, _ = multipletests(ps_win0[valid_win0], method='fdr_bh')
        p_fdr_win0[valid_win0] = p_corr2
    for idx, (rk, ci, _) in enumerate(win0_all_p):
        bnd_win0_by_roi[rk]['sig'][ci] = bool(p_fdr_win0[idx] < 0.05)

    # Build mid-event dict per ROI (reuse mid_roi_data)
    mid_by_roi = {d['roi_key']: d for d in mid_roi_data} if mid_roi_data else {}

    combined_dir = out_base / 'summary_bar_charts' / 'combined_boundary_mid'
    combined_dir.mkdir(parents=True, exist_ok=True)
    if not args.multi_roi_only:
        for roi_key, roi_name in roi_spec_filtered:
            if roi_key not in bnd_win0_by_roi or roi_key not in mid_by_roi:
                continue
            fig = make_combined_window_bar_chart(
                roi_name,
                bnd_win0_by_roi[roi_key],
                mid_by_roi[roi_key],
            )
            fig_path = combined_dir / f'roi-{roi_key}_boundary_vs_mid.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Combined bar chart → {fig_path.name}')

    # -- Multi-ROI combined figure --
    MULTI_ROI_KEYS   = ['pmc',   'mpfc',  'dlpfc',  'hipp',          'dacc']
    MULTI_ROI_LABELS = ['PMC',   'mPFC',  'dlPFC',  'Hippocampus',   'dACC']
    multi_entries = [
        (label, bnd_win0_by_roi[k], mid_by_roi[k])
        for k, label in zip(MULTI_ROI_KEYS, MULTI_ROI_LABELS)
        if k in bnd_win0_by_roi and k in mid_by_roi
    ]
    if multi_entries:
        fig = make_multi_roi_combined_bar_chart(multi_entries)
        fig_path = combined_dir / 'multi_roi_boundary_vs_mid.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Multi-ROI combined bar chart → {fig_path.name}')

    # ------------------------------------------------------------------ #
    # Save CSV                                                             #
    # ------------------------------------------------------------------ #
    csv_dir = out_base / 'stats_csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_csv_rows, columns=[
        'roi', 'roi_name', 'comparison_type', 'time_window',
        'mean_ispc', 'proportion_positive', 'n_instance_pairs',
        'binomial_p_raw', 'binomial_p_fdr',
        'bootstrap_CI_lower', 'bootstrap_CI_upper',
        'bootstrap_p_raw', 'bootstrap_p_fdr',
    ])
    csv_path = csv_dir / 'cross_task_ispc_stats.csv'
    df.round(6).to_csv(csv_path, index=False)
    print(f'\nCSV saved → {csv_path}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
