#!/usr/bin/env python3
"""
fig1_generalization.py  (manuscript_transitions)

FIGURE 1 -- the PMC transition pattern generalizes across four transition types.

Mirrors the Lee & Chen (2022, eLife e73693) boundary-template logic, applied to
this project's four trial-level transitions:

  enc : filmfest between-movie boundary (encoding; filmfest1 + filmfest2)
  rec : filmfest between-movie boundary (recall; free + cued recall scans)
  ahc : between-scenario (trial-offset) boundary in AHC
  svf : between-category  (trial-offset) boundary in SVF

Method (Lee-faithful):
  - Voxels  : Schaefer-400 17-net PMC (DefaultA_pCunPCC), bilateral, 6 mm smooth,
              MNI 2 mm -- identical voxel set across all tasks/runs.
  - Window  : boundaries are OFFSET-locked (movie/trial ENDs). The peri-boundary
              epoch spans -20..+40 TR (index 20 = t0). The boundary TEMPLATE is
              the spatial pattern averaged over +4.5..+19.5 s post-offset
              (TR +3..+13) -- Lee's "first 15 s after offset" + 3-TR HRF shift.
  - Baseline: a non-boundary template averaged over -25.5..-10.5 s pre-offset
              (TR -17..-7), a within-epoch non-transition control.
  - Similarity is Pearson r between spatial templates; group average is Fisher-z.

Panels
  A  4x4 template-correlation matrix (off-diag = cross-type generalization;
     diag = within-type cross-instance reliability).
  B  within-type reliability vs cross-type generalization vs non-boundary
     baseline, with one-sample / paired stats.
  C  dissociation: between-movie vs within-movie (event) boundary templates --
     the transition pattern is NOT the same as an ordinary within-event boundary.

This is a thin driver: it reuses the voxel loaders / event definitions from the
existing cross_boundary_macro_pattern_similarity_pmc pipeline and writes ONLY
into figs/manuscript_transitions/. It creates/overwrites no existing analysis.

Usage:
    PYTHONPATH=srcs uv run python \
        srcs/fmrianalysis/manuscript_transitions/fig1_generalization.py
    ... --no-cache        # recompute per-subject templates from voxels
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from configs.config import TR, FIGS_DIR, ANALYSIS_CACHE_DIR, FILMFEST_SUBJECTS
from configs.schaefer_rois import POSTERIOR_MEDIAL, get_bilateral_ids

from fmrianalysis.filmfest_boundary_wspc import (
    load_roi_voxels as filmfest_load_roi,
    preprocess_voxels as filmfest_preprocess,
)
from fmrianalysis.filmfest_macro_micro_pattern_similarity import (
    extract_patterns_per_tr,
    get_macro_boundary_trs as filmfest_get_macro_trs,
    get_micro_boundary_trs as filmfest_get_micro_trs,
    TRS_BEFORE, TRS_AFTER, N_TRS,
)
from fmrianalysis.svf_macro_micro_pattern_similarity import (
    get_macro_boundary_trs as svf_get_macro_trs,
)
from fmrianalysis.ahc_macro_micro_pattern_similarity import (
    get_macro_boundary_trs as ahc_get_macro_trs,
)
from fmrianalysis.cross_boundary_macro_pattern_similarity_pmc import (
    _recall_runs_for_subject, FILMFEST_TASKS,
)
from fmrianalysis.utils import discover_svf_ahc_sessions
from fmrianalysis.manuscript_transitions._style import apply_zero_format


# ============================================================================
# CONSTANTS
# ============================================================================

CONDS = ('enc', 'rec', 'ahc', 'svf')
COND_LABELS = {
    'enc': 'Movies\n(between)',
    'rec': 'Recall\n(between)',
    'ahc': 'Scenarios\n(AHC)',
    'svf': 'Words\n(SVF)',
}
DISSOC_COND = 'enc_within'   # within-movie event boundary (dissociation control)

# Template windows in TR offsets relative to the boundary (index TRS_BEFORE=t0).
# Lee & Chen: 15 s post-offset window + 3-TR (4.5 s) HRF shift.
POST_LO, POST_HI = 3, 13     # +4.5 .. +19.5 s   (boundary template)
BASE_LO, BASE_HI = -17, -7   # -25.5 .. -10.5 s  (non-boundary baseline)

OUTPUT_DIR = FIGS_DIR / 'manuscript_transitions'
STATS_DIR = OUTPUT_DIR / 'stats'
CACHE_PATH = (ANALYSIS_CACHE_DIR / 'manuscript_transitions'
              / 'fig1_pmc_templates.npz')

LABEL_FS = 12


# ============================================================================
# PER-SUBJECT TEMPLATE COLLECTION
# ============================================================================

def _win(idx_lo, idx_hi):
    """TR-offset window -> absolute index slice into the N_TRS epoch."""
    return slice(TRS_BEFORE + idx_lo, TRS_BEFORE + idx_hi + 1)


def _templates_from_patterns(pats):
    """(N_inst, N_TRS, V) -> (post (N_inst, V), base (N_inst, V)) templates."""
    post = pats[:, _win(POST_LO, POST_HI), :].mean(axis=1)
    base = pats[:, _win(BASE_LO, BASE_HI), :].mean(axis=1)
    return post, base


def collect_subject(subject, parcel_ids, force=False):
    """Return dict cond -> dict(post=(Ni,V), base=(Ni,V)) for the 4 transition
    types plus the within-movie dissociation control. None where absent."""
    ses_ff = FILMFEST_SUBJECTS[subject]
    out = {c: None for c in (*CONDS, DISSOC_COND)}

    # -- filmfest encoding: between-movie (macro) AND within-movie (micro) -----
    enc_post, enc_base, win_post, win_base = [], [], [], []
    for task in FILMFEST_TASKS:
        try:
            raw = filmfest_load_roi(subject, ses_ff, task, 'pmc',
                                    parcel_ids, force=force)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {task}: {e}")
            continue
        vox = filmfest_preprocess(raw, do_hp=True)
        p_macro, _ = extract_patterns_per_tr(vox, filmfest_get_macro_trs(task))
        p_micro, _ = extract_patterns_per_tr(vox, filmfest_get_micro_trs(task))
        if p_macro.shape[0] > 0:
            a, b = _templates_from_patterns(p_macro)
            enc_post.append(a); enc_base.append(b)
        if p_micro.shape[0] > 0:
            a, b = _templates_from_patterns(p_micro)
            win_post.append(a); win_base.append(b)
    if enc_post:
        out['enc'] = dict(post=np.concatenate(enc_post),
                          base=np.concatenate(enc_base))
    if win_post:
        out[DISSOC_COND] = dict(post=np.concatenate(win_post),
                                base=np.concatenate(win_base))

    # -- filmfest recall: between-movie ---------------------------------------
    rec_post, rec_base = [], []
    for ses, task_name, btrs in _recall_runs_for_subject(subject):
        try:
            raw = filmfest_load_roi(subject, ses, task_name, 'pmc',
                                    parcel_ids, force=force)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {subject} {ses} {task_name}: {e}")
            continue
        vox = filmfest_preprocess(raw, do_hp=True)
        p, _ = extract_patterns_per_tr(vox, btrs)
        if p.shape[0] > 0:
            a, b = _templates_from_patterns(p)
            rec_post.append(a); rec_base.append(b)
    if rec_post:
        out['rec'] = dict(post=np.concatenate(rec_post),
                          base=np.concatenate(rec_base))

    # -- ahc / svf: trial-offset boundaries -----------------------------------
    sessions_tasks = discover_svf_ahc_sessions(subject)
    for cond, task_name, get_trs in (('ahc', 'ahc', ahc_get_macro_trs),
                                      ('svf', 'svf', svf_get_macro_trs)):
        post, base = [], []
        for ses, t in sessions_tasks:
            if t != task_name:
                continue
            try:
                raw = filmfest_load_roi(subject, ses, task_name, 'pmc',
                                        parcel_ids, force=force)
            except (FileNotFoundError, ValueError) as e:
                print(f"  SKIP {subject} {ses} {task_name}: {e}")
                continue
            vox = filmfest_preprocess(raw, do_hp=True)
            p, _ = extract_patterns_per_tr(vox, get_trs(subject, ses))
            if p.shape[0] > 0:
                a, b = _templates_from_patterns(p)
                post.append(a); base.append(b)
        if post:
            out[cond] = dict(post=np.concatenate(post),
                             base=np.concatenate(base))

    return out


# ============================================================================
# SIMILARITY HELPERS
# ============================================================================

def _pearson(a, b):
    """Pearson r between two 1-D spatial patterns (nan-safe)."""
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if na == 0 or nb == 0:
        return np.nan
    return float((a @ b) / (na * nb))


def _within_reliability(post):
    """Mean off-diagonal pairwise Pearson r across instance templates.

    Instance-pairwise (single-event) reliability; noisier than the split-half
    reliability used on the matrix diagonal. Kept for the confound baseline.
    """
    n = post.shape[0]
    if n < 2:
        return np.nan
    rs = [_pearson(post[k], post[l])
          for k in range(n) for l in range(k + 1, n)]
    return float(np.nanmean(rs)) if rs else np.nan


def _split_half_reliability(post, n_splits=200, seed=0):
    """Split-half within-condition reliability, on the SAME (subject-mean)
    template scale as the cross-type off-diagonal cells.

    Randomly halve the instances, average each half into a template, correlate
    the two halves; Fisher-z average over random splits. Comparable to the
    mean-template correlations used off-diagonal, so the matrix reads as
    within >= cross rather than being biased by instance-level noise.
    """
    n = post.shape[0]
    if n < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    zs = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        h = n // 2
        a = post[idx[:h]].mean(0)
        b = post[idx[h:2 * h]].mean(0)
        r = _pearson(a, b)
        if not np.isnan(r):
            zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
    return float(np.tanh(np.mean(zs))) if zs else np.nan


def _fisher_mean(vals):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.tanh(np.nanmean(np.arctanh(np.clip(vals, -0.999, 0.999)))))


# ============================================================================
# COMPUTE
# ============================================================================

def load_or_collect(subjects, force=False):
    """Collect (or load cached) per-subject templates. Voxel loading is the slow
    step; caching the small templates makes stats/plot iterations instant."""
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached templates: {CACHE_PATH}")
        blob = np.load(CACHE_PATH, allow_pickle=True)
        return blob['tems'].item()
    tems = {}
    for subj in subjects:
        print(f"\n== {subj} ==")
        tem = collect_subject(subj, get_bilateral_ids(POSTERIOR_MEDIAL),
                              force=force)
        present = [c for c in (*CONDS, DISSOC_COND) if tem[c] is not None]
        print(f"   conditions with data: {present}")
        tems[subj] = tem
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, tems=np.array(tems, dtype=object))
    print(f"Cached templates -> {CACHE_PATH}")
    return tems


def compute(subjects, tems):
    """Per subject: subject-mean post/base templates per condition, 4x4 matrix,
    within-reliability, and dissociation pairs. Returns a results dict.

    Matrix diagonal = split-half within-condition reliability (mean-template
    scale, comparable to the off-diagonal cross-type cells)."""
    n = len(CONDS)
    mats_post = []            # (S, 4, 4) cross-type corr of subject-mean posts
    mats_base = []            # (S, 4, 4) baseline control
    within = {c: [] for c in CONDS}          # per-subject within reliability
    dissoc = {'between_x_between': [], 'between_x_within': []}
    have = {c: 0 for c in (*CONDS, DISSOC_COND)}

    for subj in subjects:
        tem = tems[subj]
        for c in (*CONDS, DISSOC_COND):
            if tem[c] is not None:
                have[c] += 1

        mpost = {c: (tem[c]['post'].mean(0) if tem[c] is not None else None)
                 for c in CONDS}
        mbase = {c: (tem[c]['base'].mean(0) if tem[c] is not None else None)
                 for c in CONDS}

        Mp = np.full((n, n), np.nan)
        Mb = np.full((n, n), np.nan)
        for i, ci in enumerate(CONDS):
            for j, cj in enumerate(CONDS):
                if i == j:
                    if tem[ci] is not None:
                        Mp[i, j] = _split_half_reliability(tem[ci]['post'])
                        Mb[i, j] = _split_half_reliability(tem[ci]['base'])
                elif mpost[ci] is not None and mpost[cj] is not None:
                    Mp[i, j] = _pearson(mpost[ci], mpost[cj])
                    Mb[i, j] = _pearson(mbase[ci], mbase[cj])
        mats_post.append(Mp)
        mats_base.append(Mb)

        for c in CONDS:
            within[c].append(_split_half_reliability(tem[c]['post'])
                             if tem[c] is not None else np.nan)

        # dissociation: enc(between) vs rec(between) [same transition class]
        #           vs  enc(between) vs enc(within)  [different: event boundary]
        if mpost['enc'] is not None and mpost['rec'] is not None:
            dissoc['between_x_between'].append(_pearson(mpost['enc'],
                                                        mpost['rec']))
        else:
            dissoc['between_x_between'].append(np.nan)
        if tem['enc'] is not None and tem[DISSOC_COND] is not None:
            dissoc['between_x_within'].append(
                _pearson(mpost['enc'], tem[DISSOC_COND]['post'].mean(0)))
        else:
            dissoc['between_x_within'].append(np.nan)

    mats_post = np.stack(mats_post)
    mats_base = np.stack(mats_base)

    group_post = np.array([[_fisher_mean(mats_post[:, i, j])
                            for j in range(n)] for i in range(n)])
    group_base = np.array([[_fisher_mean(mats_base[:, i, j])
                            for j in range(n)] for i in range(n)])

    return dict(subjects=np.array(subjects),
                mats_post=mats_post, mats_base=mats_base,
                group_post=group_post, group_base=group_base,
                within=within, dissoc=dissoc, have=have)


# ============================================================================
# STATS
# ============================================================================

def _one_sample_t(vals):
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return dict(mean=float(np.nanmean(vals)) if vals.size else np.nan,
                    t=np.nan, p=np.nan, n=int(vals.size))
    t, p = stats.ttest_1samp(vals, 0.0)
    return dict(mean=float(vals.mean()), t=float(t), p=float(p), n=int(vals.size))


def _paired_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 2:
        return dict(mean_diff=np.nan, t=np.nan, p=np.nan, n=int(m.sum()))
    t, p = stats.ttest_rel(a[m], b[m])
    return dict(mean_diff=float((a[m] - b[m]).mean()),
                t=float(t), p=float(p), n=int(m.sum()))


def build_stats(res):
    n = len(CONDS)
    # per-subject mean cross-type generalization (off-diagonal, boundary vs base)
    off = ~np.eye(n, dtype=bool)
    cross_post = [ _fisher_mean(res['mats_post'][s][off])
                   for s in range(res['mats_post'].shape[0]) ]
    cross_base = [ _fisher_mean(res['mats_base'][s][off])
                   for s in range(res['mats_base'].shape[0]) ]
    within_diag = [ _fisher_mean([res['within'][c][s] for c in CONDS])
                    for s in range(len(res['subjects'])) ]

    S = {
        'window': dict(post_TR=[POST_LO, POST_HI], base_TR=[BASE_LO, BASE_HI],
                       post_sec=[POST_LO * TR, POST_HI * TR],
                       base_sec=[BASE_LO * TR, BASE_HI * TR]),
        'n_subjects_per_condition': res['have'],
        'group_template_matrix': res['group_post'].tolist(),
        'group_baseline_matrix': res['group_base'].tolist(),
        'cross_type_generalization_vs0': _one_sample_t(cross_post),
        'within_type_reliability_vs0': _one_sample_t(within_diag),
        'cross_type_boundary_vs_baseline': _paired_t(cross_post, cross_base),
        'per_condition_within_reliability_vs0':
            {c: _one_sample_t(res['within'][c]) for c in CONDS},
        'dissociation_between_x_between_vs0':
            _one_sample_t(res['dissoc']['between_x_between']),
        'dissociation_between_x_within_vs0':
            _one_sample_t(res['dissoc']['between_x_within']),
        'dissociation_paired':
            _paired_t(res['dissoc']['between_x_between'],
                      res['dissoc']['between_x_within']),
    }
    S['_cross_post_per_subject'] = [float(x) for x in cross_post]
    S['_cross_base_per_subject'] = [float(x) for x in cross_base]
    S['_within_per_subject'] = [float(x) for x in within_diag]
    return S


# ============================================================================
# PLOT
# ============================================================================

def _stars(p):
    if p is None or np.isnan(p):
        return 'n.s.'
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'


def make_figure(res, S, vmax=0.4):
    labels = [COND_LABELS[c] for c in CONDS]
    fig = plt.figure(figsize=(15, 5.0), facecolor='white')
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.0, 0.9],
                           left=0.07, right=0.97, top=0.82, bottom=0.16,
                           wspace=0.42)

    # -- Panel A: 4x4 template correlation matrix ----------------------------
    axA = fig.add_subplot(gs[0, 0])
    M = res['group_post']
    im = axA.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axA.set_xticks(range(len(CONDS))); axA.set_yticks(range(len(CONDS)))
    axA.set_xticklabels(labels, fontsize=LABEL_FS - 3)
    axA.set_yticklabels(labels, fontsize=LABEL_FS - 3)
    for i in range(len(CONDS)):
        for j in range(len(CONDS)):
            if not np.isnan(M[i, j]):
                axA.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                         fontsize=LABEL_FS - 3,
                         color='white' if abs(M[i, j]) > vmax * 0.6 else 'k')
    axA.set_title('A  PMC boundary-template correlation\n'
                  '(diag = within-type reliability)',
                  fontsize=LABEL_FS - 1, fontweight='bold')
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cb.set_label('Pattern r', fontsize=LABEL_FS - 3)

    # -- Panel B: within vs across vs baseline -------------------------------
    axB = fig.add_subplot(gs[0, 1])
    within_ps = np.array(S['_within_per_subject'])
    cross_ps = np.array(S['_cross_post_per_subject'])
    base_ps = np.array(S['_cross_base_per_subject'])
    groups = [('Within-type\nreliability', within_ps, '#4C72B0'),
              ('Cross-type\ngeneralization', cross_ps, '#C44E52'),
              ('Non-boundary\nbaseline', base_ps, '#999999')]
    rng = np.random.default_rng(0)
    for k, (lab, vals, col) in enumerate(groups):
        v = vals[~np.isnan(vals)]
        axB.bar(k, np.nanmean(v), width=0.62, color=col, alpha=0.85,
                yerr=stats.sem(v) if v.size > 1 else 0, capsize=4)
        axB.scatter(np.full(v.size, k) + rng.uniform(-0.12, 0.12, v.size),
                    v, s=18, color='k', alpha=0.6, zorder=3)
    axB.axhline(0, color='k', lw=0.8)
    axB.set_xticks(range(3))
    axB.set_xticklabels([g[0] for g in groups], fontsize=LABEL_FS - 4)
    axB.set_ylabel('PMC pattern r', fontsize=LABEL_FS - 2)
    gen = S['cross_type_generalization_vs0']
    pvb = S['cross_type_boundary_vs_baseline']
    axB.set_title(f"B  cross-type r>0: {_stars(gen['p'])} "
                  f"(t={gen['t']:.1f})\nvs baseline: {_stars(pvb['p'])}",
                  fontsize=LABEL_FS - 1, fontweight='bold')

    # -- Panel C: dissociation -----------------------------------------------
    axC = fig.add_subplot(gs[0, 2])
    bb = np.array(res['dissoc']['between_x_between'], float)
    bw = np.array(res['dissoc']['between_x_within'], float)
    for k, (lab, vals, col) in enumerate(
            [('Movies x Recall\n(both between-event)', bb, '#C44E52'),
             ('Movies x within-movie\n(event boundary)', bw, '#DD8452')]):
        v = vals[~np.isnan(vals)]
        axC.bar(k, np.nanmean(v), width=0.6, color=col, alpha=0.85,
                yerr=stats.sem(v) if v.size > 1 else 0, capsize=4)
        axC.scatter(np.full(v.size, k) + rng.uniform(-0.1, 0.1, v.size),
                    v, s=18, color='k', alpha=0.6, zorder=3)
    axC.axhline(0, color='k', lw=0.8)
    axC.set_xticks([0, 1])
    axC.set_xticklabels(['between x\nbetween', 'between x\nwithin'],
                        fontsize=LABEL_FS - 4)
    axC.set_ylabel('PMC pattern r', fontsize=LABEL_FS - 2)
    dp = S['dissociation_paired']
    axC.set_title(f"C  dissociation: {_stars(dp['p'])}\n(paired t={dp['t']:.1f})",
                  fontsize=LABEL_FS - 1, fontweight='bold')

    apply_zero_format(axB, axC)
    fig.suptitle('Figure 1  |  The PMC transition pattern generalizes across '
                 'movie, recall, word (SVF) and scenario (AHC) transitions',
                 fontsize=LABEL_FS + 1, fontweight='bold', y=0.98)
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-cache', action='store_true',
                    help='recompute per-subject templates from voxels')
    ap.add_argument('--vmax', type=float, default=0.6)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    subjects = sorted(FILMFEST_SUBJECTS.keys())
    print(f"Subjects: {subjects}")
    print(f"Boundary template window: TR +{POST_LO}..+{POST_HI} "
          f"({POST_LO*TR:.1f}..{POST_HI*TR:.1f} s post-offset)")

    tems = load_or_collect(subjects, force=args.no_cache)
    res = compute(subjects, tems)
    S = build_stats(res)

    with open(STATS_DIR / 'fig1_generalization_stats.json', 'w') as f:
        json.dump(S, f, indent=2)
    print(f"\nStats -> {STATS_DIR / 'fig1_generalization_stats.json'}")

    fig = make_figure(res, S, vmax=args.vmax)
    out = OUTPUT_DIR / 'fig1_generalization.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure -> {out}")

    # brief console summary
    gen = S['cross_type_generalization_vs0']
    print(f"\nCross-type generalization r = {gen['mean']:.3f} "
          f"(t={gen['t']:.2f}, p={gen['p']:.4g}, n={gen['n']})")
    dp = S['dissociation_paired']
    print(f"Dissociation (between-between vs between-within) paired "
          f"t={dp['t']:.2f}, p={dp['p']:.4g}")


if __name__ == '__main__':
    main()
