# manuscript_transitions — neural signatures of mental transitions

Standalone figure set for the manuscript framing:

> What are the neural signatures of mental transitions? A posterior-medial (PMC)
> signature identified by **Lee & Chen (2022, eLife e73693)** generalized across
> movie watching and free recall. Here we ask (1) how *general* this default-mode
> transition signature is across cognitive/perceptual/task transitions, and
> (2) whether it is the local read-out of distributed DMN subsystems **DN-A** and
> **DN-B**.

Everything here is **new and self-contained**: thin drivers that reuse existing
StateSwitch compute and write ONLY into `figs/manuscript_transitions/`. No existing
script or figure is modified.

## Figures

| Script | Figure | Claim | Env |
|---|---|---|---|
| `fig1_generalization.py` | `fig1_generalization.png` | the PMC transition *pattern* generalizes across movies / recall / SVF words / AHC scenarios | uv |
| `fig2_networks.py` | `fig2_networks.png` | at transitions, DN-A (RSC/parahippocampal) > DN-B (TPJ) | uv |
| `fig3_dynamics.py` | `fig3_dynamics.png` | dynamic hand-off: DN-B leads as an event ends, DN-A leads as the next begins | uv |

Stats for each are written to `figs/manuscript_transitions/stats/*.json`.
The honest, iteration-by-iteration assessment against the framing (including where
the data do NOT support the claims) is in `figs/manuscript_transitions/EVALUATION.md`.

## Method (Lee & Chen faithful)

- Boundaries are **offset-locked** (movie/trial ends). PMC = Schaefer-400/17-net
  `DefaultA_pCunPCC`.
- **Fig 1**: boundary *template* = spatial pattern averaged +4.5..+19.5 s post-offset
  (Lee's 15 s window + 3-TR HRF shift); generalization = spatial pattern correlation
  (Fisher-z group mean); within-type reliability via split-half; dissociation vs
  within-movie event boundaries.
- **Figs 2–3**: DN-A/DN-B defined from story-listening FC (final corrected seeds
  DN-A=`RSC_schaefer`, DN-B=`TPJ_hybrid`; the `pmc_dna_dnb_contrast` cache was
  regenerated with these — the old PHC/RIGHT_TPJ version is kept as
  `pmc_dna_dnb_contrast_phctpj_backup`). PMC vertices relu-weighted by the
  DN-A/DN-B preference; peri-boundary timecourses onset- and offset-locked.

## Run

```bash
# Fig 1 (first run reloads voxels ~25 min, then caches templates -> instant)
PYTHONPATH=srcs uv run python srcs/fmrianalysis/manuscript_transitions/fig1_generalization.py
# Fig 2 (reads cached preference + boundary maps + onset timecourses)
PYTHONPATH=srcs uv run python srcs/fmrianalysis/manuscript_transitions/fig2_networks.py
# Fig 3 (first run computes offset-locked timecourses, then caches)
PYTHONPATH=srcs uv run python srcs/fmrianalysis/manuscript_transitions/fig3_dynamics.py
```

Subjects: Fig 1 uses the 6 filmfest subjects; Figs 2–3 use the 5 with story-FC +
full task set (sub-003/004/007/008/009).

## Headline findings (see EVALUATION.md for the full, honest scorecard)

1. **Fig 1 — SUPPORTED.** PMC transition pattern generalizes across all four
   transition types (cross-type r=0.35, p=1e-6; > baseline p=9e-4; dissociated from
   within-event boundaries p=3.6e-3). Nuance: narrative vs task-subtask block
   structure (SVF↔narrative weakest).
2. **Fig 2 — PARTIAL.** DN-A>DN-B robust for movie-between (both measures); pooled
   trend (p=0.07); not individually significant for recall/word/scenario (n=5).
3. **Fig 3 — SUPPORTED for movies, trend pooled.** DN-B→DN-A crossover clean for
   movie-between (p=0.014).

Overall: the *generalized PMC pattern* is the robust headline; the *DN-A/DN-B
network mechanism* is established for the canonical movie transition and not yet
shown to generalize to all transition types with this sample.
