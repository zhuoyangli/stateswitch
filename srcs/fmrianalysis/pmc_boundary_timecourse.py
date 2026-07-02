"""
PMC (Posterior Medial Cortex) Boundary-Locked Time Courses
(same style as the hippocampus/dACC/TPJ/PHG figures)

Reuses the column/plotting machinery from `hippocampus_boundary_timecourse` but
swaps in the PMC ROI: the project's Schaefer-based Posterior Medial Cortex
definition (Schaefer 400 / 17-network, the bilateral DefaultA pCunPCC parcels —
IDs [154..160, 363..367], variable POSTERIOR_MEDIAL in configs.schaefer_rois).
The Schaefer 2mm atlas is resampled to each run's BOLD grid and the PMC parcel
voxels are averaged -> one ROI time course (high-pass + temporal z-score), cached
to a small .npz — the same volumetric extraction path as the hippocampus.

Figure: 1 ROI row x 7 columns (SVF trial, AHC trial, movie watching between-movie,
movie recall between-movie, within-movie event boundary, SVF switch/cluster, AHC
across/within), identical styling to the sibling figures.

Usage:
    uv run python srcs/fmrianalysis/pmc_boundary_timecourse.py --n_jobs 8
"""
import sys
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # fmrianalysis.* / sibling modules
sys.path.insert(0, str(_HERE.parent))   # configs.config

import numpy as np
import nibabel as nib
from nilearn import datasets
from scipy.stats import zscore as sp_zscore

from configs.config import (
    ANALYSIS_CACHE_DIR, FIGS_DIR, SUBJECT_IDS,
)
from configs.schaefer_rois import POSTERIOR_MEDIAL, get_bilateral_ids
from fmrianalysis.utils import get_bold_path, get_atlas_data, highpass_filter

# Shared column definitions, event collectors, compute/plot machinery.
import hippocampus_boundary_timecourse as B

# ============================================================================
# PMC ROI (Schaefer 400/17Net bilateral DefaultA pCunPCC parcels)
# ============================================================================

PMC_IDS = get_bilateral_ids(POSTERIOR_MEDIAL)   # [154..160, 363..367]

OUTPUT_DIR = FIGS_DIR / 'boundary_timecourse'
CACHE_DIR = ANALYSIS_CACHE_DIR / 'pmc_roi'
ROI_SPEC = [('pmc', 'PMC (Schaefer)')]


def _cache_path(subject, session, task):
    return CACHE_DIR / f'{subject}_{session}_task-{task}_pmcvol_hp-0.01_z.npz'


def extract_pmc_run(subject, session, task, force=False):
    """PMC ROI time course for one run (cached), extracted from volume BOLD.

    Resamples the Schaefer atlas to the BOLD grid (nearest, cached), selects the
    PMC parcel voxels (value in PMC_IDS), averages them, then high-passes
    (0.01 Hz) and temporally z-scores. Returns True on success."""
    cache_file = _cache_path(subject, session, task)
    if cache_file.exists() and not force:
        return True

    bold_path = get_bold_path(subject, session, task)
    if not bold_path.exists():
        print(f'  [skip] BOLD missing: {bold_path.name}')
        return False

    schaefer_maps = datasets.fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=17, resolution_mm=2)['maps']
    atlas_data = get_atlas_data(bold_path, schaefer_maps)   # (X,Y,Z) 1-based IDs
    mask = np.isin(atlas_data, PMC_IDS)
    if mask.sum() < 2:
        print(f'  [skip] PMC mask empty in BOLD space for {subject} {session} {task}')
        return False

    img = nib.load(str(bold_path))
    data4d = np.asarray(img.dataobj, dtype=np.float32)   # (X, Y, Z, T)
    vox = data4d[mask]                                   # (n_vox, T)
    del data4d, img

    ts = vox.mean(axis=0).astype(np.float64)
    ts = highpass_filter(ts, order=2)
    ts = np.nan_to_num(sp_zscore(ts, nan_policy='omit')).astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, pmc=ts, n_vox=np.array([int(mask.sum())]))
    print(f'  cached {cache_file.name}  (n_vox={int(mask.sum())}, T={ts.shape[0]})')
    return True


def load_pmc_run(subject, session, task):
    """Return {'pmc': time_course} or None if the BOLD run is unavailable."""
    cache_file = _cache_path(subject, session, task)
    if not cache_file.exists():
        if not extract_pmc_run(subject, session, task):
            return None
    return {'pmc': np.load(cache_file)['pmc']}


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_jobs', type=int, default=1,
                    help='Parallelism for the one-time extraction pre-pass')
    ap.add_argument('--force', action='store_true',
                    help='Re-extract PMC time courses even if cached')
    args = ap.parse_args()

    subjects = list(SUBJECT_IDS)

    print('=' * 64)
    print('PMC BOUNDARY-LOCKED TIME COURSES')
    print(f'ROI     : Schaefer PMC (DefaultA pCunPCC), parcel IDs {PMC_IDS}')
    print(f'Columns : {[c["key"] for c in B.COLUMNS]}')
    print('=' * 64)

    # ---- Extraction pre-pass (cached; one BOLD load per run) ----
    runs = B.all_needed_runs(subjects, B.COLUMNS)
    todo = [r for r in runs if args.force or not _cache_path(*r).exists()]
    print(f'\nRuns referenced: {len(runs)}   needing extraction: {len(todo)}')
    if todo:
        if args.n_jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=args.n_jobs)(
                delayed(extract_pmc_run)(s, ses, t, args.force) for s, ses, t in todo)
        else:
            for s, ses, t in todo:
                extract_pmc_run(s, ses, t, args.force)

    # ---- Figure (shared machinery, PMC ROI) ----
    B.make_figure(
        subjects, B.COLUMNS, ROI_SPEC, load_pmc_run,
        title='PMC (Schaefer) boundary-locked time courses '
              '(coarse: subjects + black group mean ± SEM; fine: mean ± SEM)',
        out_path=OUTPUT_DIR / 'pmc_boundary_timecourse.png')
    print('\nDONE.')


if __name__ == '__main__':
    main()
