"""
Parahippocampal-gyrus (PHG) Boundary-Locked Time Courses
(same style as the hippocampus/dACC/TPJ figures)

Reuses the column/plotting machinery from `hippocampus_boundary_timecourse` but
swaps in the PHG ROI: the curated Neurosynth "parahippocampal gyrus" ROI (whole
bilateral medial-temporal cluster), a VOLUME ROI in MNI152 2mm
(data/neurosynth/phg_roi_mask.nii.gz), built by build_phg_roi_neurosynth.py. Here
we extract it volumetrically from the same MNI BOLD as the hippocampus/dACC/TPJ.

Figure: 1 ROI row x 7 columns (SVF trial, AHC trial, movie watching between-movie,
movie recall between-movie, within-movie event boundary, SVF switch/cluster, AHC
across/within), identical styling to the hippocampus/dACC/TPJ figures.

Usage:
    uv run python srcs/fmrianalysis/phg_boundary_timecourse.py --n_jobs 8
"""
import sys
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # fmrianalysis.* / sibling modules
sys.path.insert(0, str(_HERE.parent))   # configs.config

import numpy as np
import nibabel as nib
from nilearn import image as nli_image
from scipy.stats import zscore as sp_zscore

from configs.config import (
    ANALYSIS_CACHE_DIR, FIGS_DIR, SUBJECT_IDS,
)
from fmrianalysis.utils import get_bold_path, highpass_filter

# Shared column definitions, event collectors, compute/plot machinery.
import hippocampus_boundary_timecourse as B

# ============================================================================
# PHG ROI (curated Neurosynth "parahippocampal gyrus" VOLUME mask, MNI152 2mm)
# ============================================================================

ROI_MASK_NII = FIGS_DIR.parent / 'data' / 'neurosynth' / 'phg_roi_mask.nii.gz'
PHG_MASK_IMG = nib.load(str(ROI_MASK_NII))

OUTPUT_DIR = FIGS_DIR / 'boundary_timecourse'
CACHE_DIR = ANALYSIS_CACHE_DIR / 'phg_roi'
ROI_SPEC = [('phg', 'Parahippocampal gyrus (Neurosynth)')]


def _cache_path(subject, session, task):
    return CACHE_DIR / f'{subject}_{session}_task-{task}_phgvol_hp-0.01_z.npz'


def extract_phg_run(subject, session, task, force=False):
    """PHG ROI time course for one run (cached), extracted from volume BOLD.

    Resamples the PHG mask to the BOLD grid (nearest), averages the masked
    voxels, then high-passes (0.01 Hz) and temporally z-scores. Returns True on
    success."""
    cache_file = _cache_path(subject, session, task)
    if cache_file.exists() and not force:
        return True

    bold_path = get_bold_path(subject, session, task)
    if not bold_path.exists():
        print(f'  [skip] BOLD missing: {bold_path.name}')
        return False

    bold_ref = nli_image.index_img(str(bold_path), 0)
    mask_res = nli_image.resample_to_img(PHG_MASK_IMG, bold_ref, interpolation='nearest')
    mask = np.asarray(mask_res.get_fdata()) > 0.5
    if mask.sum() < 2:
        print(f'  [skip] PHG mask empty in BOLD space for {subject} {session} {task}')
        return False

    img = nib.load(str(bold_path))
    data4d = np.asarray(img.dataobj, dtype=np.float32)   # (X, Y, Z, T)
    vox = data4d[mask]                                   # (n_vox, T)
    del data4d, img

    ts = vox.mean(axis=0).astype(np.float64)
    ts = highpass_filter(ts, order=2)
    ts = np.nan_to_num(sp_zscore(ts, nan_policy='omit')).astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, phg=ts, n_vox=np.array([int(mask.sum())]))
    print(f'  cached {cache_file.name}  (n_vox={int(mask.sum())}, T={ts.shape[0]})')
    return True


def load_phg_run(subject, session, task):
    """Return {'phg': time_course} or None if the BOLD run is unavailable."""
    cache_file = _cache_path(subject, session, task)
    if not cache_file.exists():
        if not extract_phg_run(subject, session, task):
            return None
    return {'phg': np.load(cache_file)['phg']}


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_jobs', type=int, default=1,
                    help='Parallelism for the one-time extraction pre-pass')
    ap.add_argument('--force', action='store_true',
                    help='Re-extract PHG time courses even if cached')
    args = ap.parse_args()

    subjects = list(SUBJECT_IDS)

    print('=' * 64)
    print('PHG BOUNDARY-LOCKED TIME COURSES')
    print(f'ROI     : Neurosynth parahippocampal-gyrus volume mask '
          f'({int(np.asarray(PHG_MASK_IMG.get_fdata() > 0).sum())} voxels, MNI 2mm)')
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
                delayed(extract_phg_run)(s, ses, t, args.force) for s, ses, t in todo)
        else:
            for s, ses, t in todo:
                extract_phg_run(s, ses, t, args.force)

    # ---- Figure (shared machinery, PHG ROI) ----
    B.make_figure(
        subjects, B.COLUMNS, ROI_SPEC, load_phg_run,
        title='Parahippocampal gyrus (Neurosynth) boundary-locked time courses '
              '(coarse: subjects + black group mean ± SEM; fine: mean ± SEM)',
        out_path=OUTPUT_DIR / 'phg_boundary_timecourse.png')
    print('\nDONE.')


if __name__ == '__main__':
    main()
