#!/usr/bin/env python3
"""
fig2_networks.py  (manuscript_transitions)

FIGURE 2 -- the PMC transition signature is the local read-out of large-scale
DMN subsystems: at transitions, DN-A (RSC / parahippocampal-retrosplenial)
engages more than DN-B (angular-gyrus / TPJ).

Networks (final, corrected seeds; see figs/dnab_seed_refine/final_dnab_mapping):
  DN-A = RSC_schaefer  (DefaultC_Rsp)          -- a canonical DN-A node ("PHG etc")
  DN-B = TPJ_hybrid    (RH TempPar_9,10 + DefaultA_IPL_1 ~ Saxe RTPJ)
Preference per PMC vertex (from story-listening FC, cached):
  pref = z(z(r_DN-A)) - z(z(r_DN-B))   (>0 = DN-A-preferring)
NB: the pmc_dna_dnb_contrast cache stores DN-A under key z_phc and DN-B under
z_tpj for backward compatibility; it has been regenerated with RSC / TPJ_hybrid
(regen_pmc_cache_final.py), verified against the *_phctpj_backup cache.

Panels
  A  Interdigitation: across PMC vertices, DN-A vs DN-B connectivity are
     negatively related -- the two subsystems coexist in PMC (group scatter).
  B  The peri-boundary PMC response spatially resembles the DN-A>DN-B preference
     map (spatial r), per transition type -- reuses corr_matrix().
  C  DN-A minus DN-B response magnitude in the post-onset window, per transition
     type + pooled across the four between-event transitions -- reuses the
     relu-weighted DN-A/DN-B timecourses.

Honest reading (see EVALUATION.md): the DN-A>DN-B signature is strong for
movie-between boundaries (the canonical Lee transition), a trend for SVF, and
weak/null for recall and AHC at the group level with these seeds.

Reuse only; writes ONLY into figs/manuscript_transitions/. uv env.

Usage:
    PYTHONPATH=srcs uv run python \
        srcs/fmrianalysis/manuscript_transitions/fig2_networks.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from configs.config import ANALYSIS_CACHE_DIR, FIGS_DIR
from fmrianalysis.pmc_dna_dnb_boundary_corr import corr_matrix, SUBJECTS as CORR_SUBS
import fmrianalysis.pmc_dna_dnb_boundary_timecourse as TC
from fmrianalysis.manuscript_transitions._style import apply_zero_format


OUTPUT_DIR = FIGS_DIR / 'manuscript_transitions'
STATS_DIR = OUTPUT_DIR / 'stats'
PMC_GROUP = ANALYSIS_CACHE_DIR / 'pmc_dna_dnb_contrast' / 'group.npz'

# transition conditions, shared order (drop within-movie for the pooled test)
COND_KEYS = ['within', 'filmfest', 'recall', 'svf', 'ahc']
COND_LABEL = {'within': 'Movie\nwithin', 'filmfest': 'Movies\n(between)',
              'recall': 'Recall\n(between)', 'svf': 'Words\n(SVF)',
              'ahc': 'Scenarios\n(AHC)'}
# corr_matrix column order (within, enc, rec, svf, ahc)
CORR_KEYS = ['within', 'enc', 'rec', 'svf', 'ahc']
BETWEEN = ['filmfest', 'recall', 'svf', 'ahc']   # the four between-event transitions
LABEL_FS = 12


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


# ============================================================================
# DATA
# ============================================================================

def panelA_interdigitation():
    d = np.load(PMC_GROUP)
    za = d['z_phc'].astype(float)   # DN-A (RSC) connectivity per PMC vertex
    zb = d['z_tpj'].astype(float)   # DN-B (TPJ_hybrid) connectivity per PMC vertex
    r = float(d['vertexwise_r']); p = float(d['vertexwise_p'])
    return za, zb, r, p


def panelB_spatial_corr():
    """Mean (over hemispheres) spatial r between DN-A>DN-B preference and the
    peri-boundary PMC response map, per subject x condition. corr columns are
    CORR_KEYS order."""
    Ml = corr_matrix('left')    # (n_subj, 5)
    Mr = corr_matrix('right')
    M = np.nanmean(np.stack([Ml, Mr]), axis=0)   # avg hemis -> (n_subj, 5)
    per_cond = {k: M[:, i] for i, k in enumerate(CORR_KEYS)}
    return per_cond


def generalized_spatial_corr():
    """The KEY framing test: does the GENERALIZED transition response (the
    per-subject response map averaged across the four between-event transitions,
    which denoises) spatially load onto the DN-A>DN-B preference map?
    Averaging maps first is better-powered than averaging per-condition rs.
    Returns per-subject spatial r (hemisphere-averaged)."""
    from scipy.stats import pearsonr
    BND = ANALYSIS_CACHE_DIR / 'five_boundary_pattern_per_subject'
    between = ['enc', 'rec', 'svf', 'ahc']
    out = []
    for s in CORR_SUBS:
        rs = []
        for hemi in ('left', 'right'):
            vk = 'lh_pmc_verts' if hemi == 'left' else 'rh_pmc_verts'
            pk = 'z_phc_lh' if hemi == 'left' else 'z_phc_rh'
            tk = 'z_tpj_lh' if hemi == 'left' else 'z_tpj_rh'
            stem = f'pmc_{hemi}_tb10_ta20_post3-13_onset'
            d = np.load(ANALYSIS_CACHE_DIR / 'pmc_dna_dnb_contrast' / f'{s}.npz',
                        allow_pickle=True)
            verts = np.asarray(d[vk], int)
            pref = stats.zscore(stats.zscore(d[pk]) - stats.zscore(d[tk]))
            maps = [pickle.load(open(BND / f'{p}_{stem}.pkl', 'rb'))[s][verts]
                    for p in between]
            gen = np.nanmean(np.vstack(maps), axis=0)
            ok = ~np.isnan(gen)
            if ok.sum() > 2:
                rs.append(pearsonr(pref[ok], gen[ok])[0])
        out.append(float(np.nanmean(rs)) if rs else np.nan)
    return np.asarray(out, float)


def panelC_magnitude():
    """DN-A minus DN-B post-onset magnitude per subject x condition, from the
    relu-weighted PMC timecourses (onset-locked, cached)."""
    lo = TC.TRS_BEFORE + TC.POST_TRS[0]
    hi = TC.TRS_BEFORE + TC.POST_TRS[1]
    subs = list(CORR_SUBS)
    per_cond = {k: [] for k in COND_KEYS}
    for s in subs:
        r = TC.compute_subject_boundary_timecourses(s, 'both', force=False)
        for k in COND_KEYS:
            d = r.get(k, {})
            na = [n for n in d if n.startswith('DN-A')]
            nb = [n for n in d if n.startswith('DN-B')]
            if not na or not nb:
                per_cond[k].append(np.nan); continue
            a = d[na[0]][0]; b = d[nb[0]][0]
            per_cond[k].append(float(np.nanmean(a[lo:hi + 1])
                                     - np.nanmean(b[lo:hi + 1])))
    return {k: np.asarray(v, float) for k, v in per_cond.items()}, subs


# ============================================================================
# PLOT
# ============================================================================

def make_figure(za, zb, vr, vp, corrB, magC, genB, subs):
    fig = plt.figure(figsize=(15.5, 5.0), facecolor='white')
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.85, 1.05, 1.15],
                           left=0.06, right=0.98, top=0.82, bottom=0.17,
                           wspace=0.36)

    # -- A: DN-A/DN-B preference distribution within PMC ----------------------
    # NB raw coupling to the two seeds shares a positive DMN gradient (r=+vr);
    # the DN-A/DN-B distinction lives in the *preference* contrast, which spans
    # both signs -> PMC contains DN-A- and DN-B-preferring vertices.
    axA = fig.add_subplot(gs[0, 0])
    pref = stats.zscore(za) - stats.zscore(zb)
    fA = float(np.mean(pref > 0)); fB = 1 - fA
    axA.hist(pref[pref > 0], bins=30, color='#d62728', alpha=0.8,
             label=f'DN-A-pref ({fA*100:.0f}%)')
    axA.hist(pref[pref <= 0], bins=30, color='#1f77b4', alpha=0.8,
             label=f'DN-B-pref ({fB*100:.0f}%)')
    axA.axvline(0, color='k', lw=1.0)
    axA.set_xlabel('DN-A vs DN-B preference\nz(r$_{RSC}$) - z(r$_{TPJ}$)',
                   fontsize=LABEL_FS - 3)
    axA.set_ylabel('PMC vertices', fontsize=LABEL_FS - 3)
    axA.legend(fontsize=LABEL_FS - 5, loc='upper right')
    axA.set_title('A  PMC holds both subsystems\n(preference spans both signs; '
                  f'raw coupling r=+{vr:.2f})',
                  fontsize=LABEL_FS - 1, fontweight='bold')

    # -- B: spatial corr of transition response with DN-A>DN-B preference ----
    axB = fig.add_subplot(gs[0, 1])
    keysB = CORR_KEYS
    means = [np.nanmean(corrB[k]) for k in keysB]
    sems = [stats.sem(corrB[k][~np.isnan(corrB[k])]) for k in keysB]
    stats_B = {}
    rng = np.random.default_rng(0)
    for i, k in enumerate(keysB):
        col = '#08306b' if k == 'within' else '#C44E52'
        axB.bar(i, means[i], yerr=sems[i], width=0.66, color=col, alpha=0.85,
                capsize=3)
        v = corrB[k][~np.isnan(corrB[k])]
        axB.scatter(np.full(v.size, i) + rng.uniform(-0.12, 0.12, v.size), v,
                    s=14, color='k', alpha=0.55, zorder=3)
        st = _one_sample_t(corrB[k]); stats_B[k] = st
        axB.text(i, max(means[i] + sems[i], 0) + 0.02, _stars(st['p']),
                 ha='center', va='bottom', fontsize=LABEL_FS - 4)
    # generalized (pooled across the 4 between-event transitions) — the key test
    ig = len(keysB)
    gv = genB[~np.isnan(genB)]
    axB.bar(ig, gv.mean(), yerr=stats.sem(gv), width=0.66, color='#8B0000',
            alpha=0.9, capsize=3)
    axB.scatter(np.full(gv.size, ig) + rng.uniform(-0.12, 0.12, gv.size), gv,
                s=14, color='k', alpha=0.55, zorder=3)
    st_gen = _one_sample_t(genB); stats_B['generalized'] = st_gen
    axB.text(ig, gv.mean() + stats.sem(gv) + 0.02, _stars(st_gen['p']),
             ha='center', va='bottom', fontsize=LABEL_FS - 4)
    axB.axhline(0, color='k', lw=0.8)
    axB.set_xticks(range(len(keysB) + 1))
    lblmap = {'within': 'Movie\nwithin', 'enc': 'Movies\n(between)',
              'rec': 'Recall', 'svf': 'Words', 'ahc': 'Scenarios'}
    axB.set_xticklabels([lblmap[k] for k in keysB] + ['Generalized\n(pooled)'],
                        fontsize=LABEL_FS - 4)
    axB.set_ylabel('spatial r(response, DN-A>DN-B pref)', fontsize=LABEL_FS - 3)
    axB.set_title('B  transition response is DN-A-like\n(response x preference map)',
                  fontsize=LABEL_FS - 1, fontweight='bold')

    # -- C: DN-A minus DN-B post-onset magnitude + pooled --------------------
    axC = fig.add_subplot(gs[0, 2])
    keysC = COND_KEYS
    stats_C = {}
    for i, k in enumerate(keysC):
        v = magC[k][~np.isnan(magC[k])]
        col = '#08306b' if k == 'within' else '#C44E52'
        axC.bar(i, v.mean(), yerr=stats.sem(v), width=0.66, color=col,
                alpha=0.85, capsize=3)
        axC.scatter(np.full(v.size, i) + rng.uniform(-0.12, 0.12, v.size), v,
                    s=14, color='k', alpha=0.55, zorder=3)
        st = _one_sample_t(magC[k]); stats_C[k] = st
        axC.text(i, v.mean() + stats.sem(v) + 0.006, _stars(st['p']),
                 ha='center', va='bottom', fontsize=LABEL_FS - 4)
    # pooled across the four between-event transitions
    pooled = np.nanmean(np.stack([magC[k] for k in BETWEEN]), axis=0)
    ip = len(keysC)
    vp_ = pooled[~np.isnan(pooled)]
    axC.bar(ip, vp_.mean(), yerr=stats.sem(vp_), width=0.66, color='#8B0000',
            alpha=0.9, capsize=3)
    axC.scatter(np.full(vp_.size, ip) + rng.uniform(-0.12, 0.12, vp_.size),
                vp_, s=14, color='k', alpha=0.55, zorder=3)
    st_pool = _one_sample_t(pooled); stats_C['pooled_between'] = st_pool
    axC.text(ip, vp_.mean() + stats.sem(vp_) + 0.006, _stars(st_pool['p']),
             ha='center', va='bottom', fontsize=LABEL_FS - 4)
    axC.axhline(0, color='k', lw=0.8)
    axC.set_xticks(range(len(keysC) + 1))
    axC.set_xticklabels([COND_LABEL[k] for k in keysC] + ['Pooled\nbetween'],
                        fontsize=LABEL_FS - 4)
    axC.set_ylabel('DN-A - DN-B response (post-onset)', fontsize=LABEL_FS - 3)
    axC.set_title('C  DN-A > DN-B at transitions\n(relu-weighted PMC magnitude)',
                  fontsize=LABEL_FS - 1, fontweight='bold')

    apply_zero_format(axA, which='x')
    apply_zero_format(axB, axC)
    fig.suptitle('Figure 2  |  The PMC transition signature reflects DN-A '
                 '(RSC/parahippocampal) > DN-B (TPJ) engagement',
                 fontsize=LABEL_FS + 1, fontweight='bold', y=0.98)
    return fig, stats_B, stats_C


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    za, zb, vr, vp = panelA_interdigitation()
    corrB = panelB_spatial_corr()
    genB = generalized_spatial_corr()
    magC, subs = panelC_magnitude()

    fig, stats_B, stats_C = make_figure(za, zb, vr, vp, corrB, magC, genB, subs)
    out = OUTPUT_DIR / 'fig2_networks.png'
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Figure -> {out}")

    S = {
        'subjects': list(subs),
        'dn_a_seed': 'RSC_schaefer', 'dn_b_seed': 'TPJ_hybrid',
        'pmc_interdigitation_vertexwise_r': vr,
        'pmc_interdigitation_vertexwise_p': vp,
        'panelB_spatial_corr_vs0': stats_B,
        'panelC_dnA_minus_dnB_vs0': stats_C,
        'window_post_TR': list(TC.POST_TRS), 'align': TC.ALIGN,
    }
    with open(STATS_DIR / 'fig2_networks_stats.json', 'w') as f:
        json.dump(S, f, indent=2)
    print(f"Stats -> {STATS_DIR / 'fig2_networks_stats.json'}")
    print("\nPanel C DN-A-DN-B (post-onset):")
    for k, st in stats_C.items():
        print(f"  {k:16s} mean={st['mean']:+.3f} t={st['t']:+.2f} "
              f"p={st['p']:.3f} n={st['n']}")


if __name__ == '__main__':
    main()
