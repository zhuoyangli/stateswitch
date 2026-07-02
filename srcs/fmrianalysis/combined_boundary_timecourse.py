"""
Combined Boundary-Locked Time Courses across all ROIs.

One figure, 5 ROI rows x 7 columns, in the shared style (row order below):
  row 1 — PMC (Schaefer DefaultA pCunPCC parcels)
  row 2 — Hippocampus (whole, Harvard-Oxford subcortical)
  row 3 — Parahippocampal gyrus (Neurosynth "parahippocampal gyrus" volume ROI)
  row 4 — dACC (Neurosynth "dacc" volume ROI)
  row 5 — TPJ (Neurosynth "tpj" volume ROI)

Reuses the column definitions and plotting machinery from
`hippocampus_boundary_timecourse`. A single combined loader returns all four ROI
time courses per run (each from its own already-built cache), so plotting is
instant once the per-ROI caches exist (built by the four sibling scripts). Run
those first if any cache is missing; this script will also extract on demand.

Usage:
    uv run python srcs/fmrianalysis/combined_boundary_timecourse.py --n_jobs 8
"""
import sys
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # sibling modules
sys.path.insert(0, str(_HERE.parent))   # configs.config

import numpy as np

from configs.config import FIGS_DIR, SUBJECT_IDS

import hippocampus_boundary_timecourse as B
import dacc_boundary_timecourse as D
import tpj_boundary_timecourse as T
import phg_boundary_timecourse as P
import pmc_boundary_timecourse as M

OUTPUT_DIR = FIGS_DIR / 'boundary_timecourse'

# (key, display_name) — one row per ROI, in the requested order:
# PMC, hippocampus, parahippocampal gyrus, dACC, TPJ.
ROI_SPEC = [
    ('pmc', 'PMC (Schaefer)'),
    ('whole', 'Hippocampus'),
    ('phg', 'Parahippocampal g.\n(Neurosynth)'),
    ('dacc', 'dACC (Neurosynth)'),
    ('tpj', 'TPJ (Neurosynth)'),
]

# Each ROI's per-run loader and the key it returns.
_LOADERS = [
    (M.load_pmc_run, 'pmc'),
    (B.load_hipp_run, 'whole'),
    (P.load_phg_run, 'phg'),
    (D.load_dacc_run, 'dacc'),
    (T.load_tpj_run, 'tpj'),
]


def load_all(subject, session, task):
    """Return {'whole','dacc','tpj','phg'} for one run, or None if none load."""
    out = {}
    for loader, key in _LOADERS:
        run = loader(subject, session, task)
        if run is not None and key in run:
            out[key] = run[key]
    return out or None


def _extract_all(subjects, n_jobs, force):
    """Ensure every per-ROI cache exists for all referenced runs."""
    runs = B.all_needed_runs(subjects, B.COLUMNS)
    jobs = [
        (M.extract_pmc_run, M._cache_path),
        (B.extract_hipp_run, B._cache_path),
        (D.extract_dacc_run, D._cache_path),
        (T.extract_tpj_run, T._cache_path),
        (P.extract_phg_run, P._cache_path),
    ]
    for extract, cache_path in jobs:
        todo = [r for r in runs if force or not cache_path(*r).exists()]
        if not todo:
            continue
        print(f'  {extract.__name__}: extracting {len(todo)} runs')
        if n_jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_jobs)(
                delayed(extract)(s, ses, t, force) for s, ses, t in todo)
        else:
            for s, ses, t in todo:
                extract(s, ses, t, force)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_jobs', type=int, default=1,
                    help='Parallelism for any on-demand extraction')
    ap.add_argument('--force', action='store_true',
                    help='Re-extract all ROI time courses even if cached')
    args = ap.parse_args()

    subjects = list(SUBJECT_IDS)

    print('=' * 64)
    print(f'COMBINED BOUNDARY-LOCKED TIME COURSES ({len(ROI_SPEC)} ROIs)')
    print(f'ROIs    : {[k for k, _ in ROI_SPEC]}')
    print(f'Columns : {[c["key"] for c in B.COLUMNS]}')
    print('=' * 64)

    runs = _extract_all(subjects, args.n_jobs, args.force)
    print(f'\nRuns referenced: {len(runs)}')

    B.make_figure(
        subjects, B.COLUMNS, ROI_SPEC, load_all,
        title='Boundary-locked BOLD time courses across ROIs '
              '(coarse: subjects + black group mean ± SEM; fine: mean ± SEM)',
        out_path=OUTPUT_DIR / 'combined_boundary_timecourse.png')
    print('\nDONE.')


if __name__ == '__main__':
    main()
