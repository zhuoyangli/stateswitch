#!/usr/bin/env python3
"""
fig3_dynamics.py  (manuscript_transitions)

FIGURE 3 -- a dynamic DN-B -> DN-A hand-off across event boundaries.

Claim: DN-B exceeds DN-A toward the END of an event (offset-locked), and this
inverts to DN-A > DN-B at the ONSET of the next event/task.

Approach (reuse): the relu-weighted DN-A / DN-B PMC timecourses from
pmc_dna_dnb_boundary_timecourse.py, computed twice -- once OFFSET-locked
(t0 = end of the preceding event) and once ONSET-locked (t0 = start of the next
event) -- by toggling the module ALIGN constant (separate caches per align).

Panels
  Top:  pooled across the four between-event transitions (movies, recall, SVF,
        AHC): group DN-A (red) & DN-B (blue) timecourses, OFFSET-locked (left)
        and ONSET-locked (right). Post-boundary window shaded.
  Bot:  crossover summary -- per-subject DN-A - DN-B in the post-boundary window,
        OFFSET vs ONSET, showing the inversion (paired t); plus per-condition
        DN-A - DN-B for offset vs onset.

Honest reading recorded in EVALUATION.md.

Reuse only; writes ONLY into figs/manuscript_transitions/. uv env.
The offset cache is built on first run (surface extraction, slow); afterwards
both aligns load from cache.

Usage:
    PYTHONPATH=srcs uv run python \
        srcs/fmrianalysis/manuscript_transitions/fig3_dynamics.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from configs.config import FIGS_DIR, TR
import fmrianalysis.pmc_dna_dnb_boundary_timecourse as TC


OUTPUT_DIR = FIGS_DIR / 'manuscript_transitions'
STATS_DIR = OUTPUT_DIR / 'stats'

COND_KEYS = ['within', 'filmfest', 'recall', 'svf', 'ahc']
COND_LABEL = {'within': 'Movie within', 'filmfest': 'Movies (between)',
              'recall': 'Recall', 'svf': 'Words (SVF)', 'ahc': 'Scenarios (AHC)'}
BETWEEN = ['filmfest', 'recall', 'svf', 'ahc']
DNA_RED = '#d62728'
DNB_BLUE = '#1f77b4'
LABEL_FS = 12

# Crossover windows (TR offsets relative to the boundary), tested WITHIN the
# onset-locked timecourse: PRE = the ending previous event (DN-B expected to
# lead); POST = the beginning next event (DN-A expected to lead).
PRE_OFF = (-8, -2)     # -12 .. -3 s   ("end of event")
POST_OFF = (3, 13)     # +4.5 .. +19.5 s ("onset of next")


def _stars(p):
    if p is None or np.isnan(p):
        return 'n.s.'
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'


def _one_sample_t(vals):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    if v.size < 2:
        return dict(mean=float(np.nanmean(v)) if v.size else np.nan,
                    t=np.nan, p=np.nan, n=int(v.size))
    t, p = stats.ttest_1samp(v, 0.0)
    return dict(mean=float(v.mean()), t=float(t), p=float(p), n=int(v.size))


def _paired_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 2:
        return dict(mean_diff=np.nan, t=np.nan, p=np.nan, n=int(m.sum()))
    t, p = stats.ttest_rel(a[m], b[m])
    return dict(mean_diff=float((a[m] - b[m]).mean()), t=float(t), p=float(p),
                n=int(m.sum()))


# ============================================================================
# DATA
# ============================================================================

def collect_align(align):
    """Return per-subject DN-A / DN-B mean timecourses for every condition.

    out[subj][cond] = dict(A=(T,), B=(T,))  (mean epoch timecourses)
    """
    TC.ALIGN = align
    out = {}
    for s in TC.SUBJECTS:
        r = TC.compute_subject_boundary_timecourses(s, 'both', force=False)
        sub = {}
        for k in COND_KEYS:
            d = r.get(k, {})
            na = [n for n in d if n.startswith('DN-A')]
            nb = [n for n in d if n.startswith('DN-B')]
            if na and nb:
                sub[k] = dict(A=np.asarray(d[na[0]][0], float),
                              B=np.asarray(d[nb[0]][0], float))
        out[s] = sub
    return out


def group_tc(data, conds):
    """Group-mean DN-A, DN-B timecourses pooled across `conds` (mean per subject
    over available conds, then mean/sem across subjects). Returns A,B,(sem)."""
    per_subj_A, per_subj_B = [], []
    for s, sub in data.items():
        a = [sub[c]['A'] for c in conds if c in sub]
        b = [sub[c]['B'] for c in conds if c in sub]
        if a:
            per_subj_A.append(np.nanmean(a, axis=0))
            per_subj_B.append(np.nanmean(b, axis=0))
    A = np.stack(per_subj_A); B = np.stack(per_subj_B)
    return (A.mean(0), stats.sem(A, axis=0),
            B.mean(0), stats.sem(B, axis=0))


def _wd(off):
    lo = TC.TRS_BEFORE + off[0]
    hi = TC.TRS_BEFORE + off[1]
    return lo, hi + 1


def window_diff(data, conds, off):
    """Per-subject DN-A - DN-B averaged over a TR-offset window, pooled over
    conds."""
    lo, hi = _wd(off)
    out = []
    for s, sub in data.items():
        diffs = [np.nanmean(sub[c]['A'][lo:hi] - sub[c]['B'][lo:hi])
                 for c in conds if c in sub]
        out.append(np.nanmean(diffs) if diffs else np.nan)
    return np.asarray(out, float)


def per_cond_window_diff(data, cond, off):
    lo, hi = _wd(off)
    out = []
    for s, sub in data.items():
        out.append(float(np.nanmean(sub[cond]['A'][lo:hi] - sub[cond]['B'][lo:hi]))
                   if cond in sub else np.nan)
    return np.asarray(out, float)


# ============================================================================
# PLOT
# ============================================================================

def _tvec(n):
    return (np.arange(n) - TC.TRS_BEFORE) * TR


def plot_tc(ax, A, semA, B, semB, title, mark_pre=False):
    t = _tvec(len(A))
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=1.0, ls='--', alpha=0.7)
    if mark_pre:
        ax.axvspan(PRE_OFF[0] * TR, PRE_OFF[1] * TR, color=DNB_BLUE, alpha=0.12,
                   zorder=0)
        ax.axvspan(POST_OFF[0] * TR, POST_OFF[1] * TR, color=DNA_RED, alpha=0.12,
                   zorder=0)
    else:
        ax.axvspan(POST_OFF[0] * TR, POST_OFF[1] * TR, color='0.85', alpha=0.5,
                   zorder=0)
    ax.plot(t, A, color=DNA_RED, lw=2, label='DN-A (RSC-weighted)')
    ax.fill_between(t, A - semA, A + semA, color=DNA_RED, alpha=0.2)
    ax.plot(t, B, color=DNB_BLUE, lw=2, label='DN-B (TPJ-weighted)')
    ax.fill_between(t, B - semB, B + semB, color=DNB_BLUE, alpha=0.2)
    ax.set_xlabel('Time from boundary (s)', fontsize=LABEL_FS - 3)
    ax.set_ylabel('PMC response (z)', fontsize=LABEL_FS - 3)
    ax.set_title(title, fontsize=LABEL_FS - 1, fontweight='bold')


def make_figure(off, on):
    fig = plt.figure(figsize=(14, 8.6), facecolor='white')
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.9],
                           left=0.08, right=0.97, top=0.90, bottom=0.09,
                           hspace=0.42, wspace=0.28)

    # -- top: onset-locked pooled (with pre/post windows) | movies-only ------
    An, sAn, Bn, sBn = group_tc(on, BETWEEN)
    axN = fig.add_subplot(gs[0, 0])
    plot_tc(axN, An, sAn, Bn, sBn,
            'ONSET-locked, pooled between-event transitions', mark_pre=True)
    axN.legend(fontsize=LABEL_FS - 4, loc='upper right')
    axN.annotate('DN-B leads\n(end of prev event)', xy=(PRE_OFF[0] * TR + 3, 0),
                 xytext=(-27, 0.30), fontsize=LABEL_FS - 5, color=DNB_BLUE,
                 ha='center', fontweight='bold')
    axN.annotate('DN-A leads\n(onset of next)', xy=(POST_OFF[0] * TR, 0),
                 xytext=(12, 0.33), fontsize=LABEL_FS - 5, color=DNA_RED,
                 ha='center', fontweight='bold')

    Am, sAm, Bm, sBm = group_tc(on, ['filmfest'])
    axM = fig.add_subplot(gs[0, 1])
    plot_tc(axM, Am, sAm, Bm, sBm,
            'ONSET-locked, Movies (between) - the clean case', mark_pre=True)
    ylo = min(axN.get_ylim()[0], axM.get_ylim()[0])
    yhi = max(axN.get_ylim()[1], axM.get_ylim()[1])
    axN.set_ylim(ylo, yhi); axM.set_ylim(ylo, yhi)

    # -- bottom-left: crossover pre vs post WITHIN onset-locked (pooled) ------
    axC = fig.add_subplot(gs[1, 0])
    dpre = window_diff(on, BETWEEN, PRE_OFF)
    dpost = window_diff(on, BETWEEN, POST_OFF)
    for i, (v, col) in enumerate([(dpre, DNB_BLUE), (dpost, DNA_RED)]):
        vv = v[~np.isnan(v)]
        axC.bar(i, vv.mean(), yerr=stats.sem(vv), width=0.6, color=col,
                alpha=0.85, capsize=4)
    for a, b in zip(dpre, dpost):
        if not (np.isnan(a) or np.isnan(b)):
            axC.plot([0, 1], [a, b], color='0.4', lw=1, marker='o', ms=4,
                     alpha=0.7)
    axC.axhline(0, color='k', lw=0.8)
    axC.set_xticks([0, 1])
    axC.set_xticklabels(['pre-onset\n(event end)', 'post-onset\n(next start)'],
                        fontsize=LABEL_FS - 4)
    axC.set_ylabel('DN-A - DN-B', fontsize=LABEL_FS - 3)
    cross = _paired_t(dpost, dpre)
    axC.set_title(f'crossover pooled: post>pre {_stars(cross["p"])} '
                  f'(t={cross["t"]:.1f})',
                  fontsize=LABEL_FS - 1, fontweight='bold')

    # -- bottom-right: per-condition crossover magnitude (post - pre) --------
    axP = fig.add_subplot(gs[1, 1])
    x = np.arange(len(COND_KEYS)); w = 0.38
    pre_means = [np.nanmean(per_cond_window_diff(on, c, PRE_OFF)) for c in COND_KEYS]
    post_means = [np.nanmean(per_cond_window_diff(on, c, POST_OFF)) for c in COND_KEYS]
    def _sem(a): a = a[~np.isnan(a)]; return stats.sem(a) if a.size > 1 else 0
    pre_sem = [_sem(per_cond_window_diff(on, c, PRE_OFF)) for c in COND_KEYS]
    post_sem = [_sem(per_cond_window_diff(on, c, POST_OFF)) for c in COND_KEYS]
    axP.bar(x - w / 2, pre_means, w, yerr=pre_sem, color=DNB_BLUE, alpha=0.85,
            capsize=2, label='pre-onset (event end)')
    axP.bar(x + w / 2, post_means, w, yerr=post_sem, color=DNA_RED, alpha=0.85,
            capsize=2, label='post-onset (next start)')
    per_cond_cross = {}
    for i, c in enumerate(COND_KEYS):
        st = _paired_t(per_cond_window_diff(on, c, POST_OFF),
                       per_cond_window_diff(on, c, PRE_OFF))
        per_cond_cross[c] = st
        y = max(post_means[i] + post_sem[i], pre_means[i] + pre_sem[i]) + 0.01
        axP.text(i, y, _stars(st['p']), ha='center', va='bottom',
                 fontsize=LABEL_FS - 4)
    axP.axhline(0, color='k', lw=0.8)
    axP.set_xticks(x)
    axP.set_xticklabels([COND_LABEL[c].replace(' ', '\n') for c in COND_KEYS],
                        fontsize=LABEL_FS - 5)
    axP.set_ylabel('DN-A - DN-B', fontsize=LABEL_FS - 3)
    axP.legend(fontsize=LABEL_FS - 5, loc='lower right')
    axP.set_title('crossover per transition type (onset-locked)',
                  fontsize=LABEL_FS - 1, fontweight='bold')

    fig.suptitle('Figure 3  |  Dynamic DN-B -> DN-A hand-off across the boundary: '
                 'DN-B leads as an event ends, DN-A leads as the next begins',
                 fontsize=LABEL_FS + 1, fontweight='bold', y=0.965)
    return fig, dict(crossover_pooled=cross,
                     pre_onset_diff_vs0=_one_sample_t(dpre),
                     post_onset_diff_vs0=_one_sample_t(dpost),
                     per_cond_crossover=per_cond_cross,
                     per_cond_pre={c: _one_sample_t(per_cond_window_diff(on, c, PRE_OFF))
                                   for c in COND_KEYS},
                     per_cond_post={c: _one_sample_t(per_cond_window_diff(on, c, POST_OFF))
                                    for c in COND_KEYS})


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting OFFSET-locked timecourses ...")
    off = collect_align('offset')
    print("Collecting ONSET-locked timecourses ...")
    on = collect_align('onset')

    fig, S = make_figure(off, on)
    out = OUTPUT_DIR / 'fig3_dynamics.png'
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Figure -> {out}")

    S['window_post_TR'] = list(TC.POST_TRS)
    S['subjects'] = list(TC.SUBJECTS)
    with open(STATS_DIR / 'fig3_dynamics_stats.json', 'w') as f:
        json.dump(S, f, indent=2)
    print(f"Stats -> {STATS_DIR / 'fig3_dynamics_stats.json'}")
    c = S['crossover_pooled']
    print(f"\nCrossover (post>pre, onset-locked, pooled): "
          f"mean_diff={c['mean_diff']:+.3f} t={c['t']:.2f} p={c['p']:.4g} n={c['n']}")
    print(f"pre-onset  DN-A-DN-B (event end):  {S['pre_onset_diff_vs0']}")
    print(f"post-onset DN-A-DN-B (next start): {S['post_onset_diff_vs0']}")
    print("per-condition crossover (post-pre):")
    for c, st in S['per_cond_crossover'].items():
        print(f"  {c:10s} t={st['t']:+.2f} p={st['p']:.3f}")


if __name__ == '__main__':
    main()
