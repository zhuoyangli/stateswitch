"""
filmfest_pISC.py — Parcel-wise Inter-Subject Spatial Pattern Correlation

For each of the 400 Schaefer parcels, compute spatial pattern ISC:
  pISC = mean Pearson r between one subject's spatial pattern and the
         LOO group mean spatial pattern, averaged over timepoints.
Fisher-z average across subjects gives one pISC value per parcel.

Output:
  figs/filmfest_pisc/GROUP_pisc_filmfest.png
  figs/filmfest_pisc/{subject}_{session}_pisc_filmfest.png  (6 subjects)
  figs/filmfest_pisc/cache/pISC_task-{task}_persubject.npz  (n_subjects, 400)

  With --roi:
  figs/filmfest_pisc/GROUP_pisc_filmfest_roi.png     (ROI parcels only)
  figs/filmfest_pisc/GROUP_lag_pisc_filmfest.png     (lag figure, 2×4)
  figs/filmfest_pisc/cache/pISC_task-{task}_roi_persubject.npz
  figs/filmfest_pisc/cache/lag_pISC_task-{task}_roi-{key}_persubject.npz

  With --searchlight:
  figs/filmfest_pisc/cache/pISC_searchlight_task-{task}_group.nii.gz
  figs/filmfest_pisc/GROUP_pisc_searchlight_filmfest.png
  figs/filmfest_pisc/GROUP_pisc_searchlight_filmfest1.nii.gz
  figs/filmfest_pisc/GROUP_pisc_searchlight_filmfest2.nii.gz

  With --surface:
  figs/filmfest_pisc/GROUP_pisc_filmfest_surface.png
  figs/filmfest_pisc/cache/pISC_surface_task-{task}_persubject.npz

Usage:
    uv run python srcs/fmrianalysis/filmfest_pISC.py
    uv run python srcs/fmrianalysis/filmfest_pISC.py --roi
    uv run python srcs/fmrianalysis/filmfest_pISC.py --surface
    uv run python srcs/fmrianalysis/filmfest_pISC.py --surface --roi
    uv run python srcs/fmrianalysis/filmfest_pISC.py --searchlight --n-jobs 8
"""
import io
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.freesurfer as fs
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from nilearn import datasets, image, plotting
from nilearn.surface import load_surf_data
from scipy.stats import zscore as sp_zscore

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR, FILMFEST_SUBJECTS
from configs.schaefer_rois import (
    EARLY_VISUAL, EARLY_AUDITORY, POSTERIOR_MEDIAL, ANGULAR_GYRUS,
    get_bilateral_ids,
)
from fmrianalysis.utils import highpass_filter, get_bold_path, cross_corrcoef

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'

OUTPUT_DIR = FIGS_DIR / 'filmfest_pisc'
CACHE_DIR  = ANALYSIS_CACHE_DIR / 'filmfest_pisc'

# 4 ROIs used in --roi mode
ROI_SPEC = [
    ('evc', 'Early Visual',      get_bilateral_ids(EARLY_VISUAL)),
    ('eac', 'Early Auditory',    get_bilateral_ids(EARLY_AUDITORY)),
    ('pmc', 'Posterior Medial',  get_bilateral_ids(POSTERIOR_MEDIAL)),
    ('ag',  'Angular Gyrus',     get_bilateral_ids(ANGULAR_GYRUS)),
]
ROI_PARCEL_IDS = sorted({pid for _, _, ids in ROI_SPEC for pid in ids})

# Load atlas at module level
print("Loading Schaefer atlas...")
SCHAEFER = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
_atlas_resampled = None       # cached at 2mm (native BOLD space)
_atlas_resampled_3mm = None   # cached at 3mm
print("  Schaefer atlas loaded.")


# ============================================================================
# CORE UTILITIES
# ============================================================================

def _get_mask_path(subject, session, task):
    return (DERIVATIVES_DIR / subject / session / 'func' /
            f'{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz')


def _get_atlas_data(bold_path):
    """Return Schaefer atlas resampled to 2mm BOLD voxel space (cached on first call)."""
    global _atlas_resampled
    if _atlas_resampled is None:
        bold_ref = image.index_img(str(bold_path), 0)
        atlas_resampled = image.resample_to_img(
            SCHAEFER['maps'], bold_ref, interpolation='nearest'
        )
        _atlas_resampled = np.round(atlas_resampled.get_fdata()).astype(int)
        print(f"  Atlas resampled to BOLD space: {_atlas_resampled.shape}")
    return _atlas_resampled


def _get_atlas_data_3mm(bold_path):
    """Return Schaefer atlas resampled to 3mm isotropic (cached on first call)."""
    global _atlas_resampled_3mm
    if _atlas_resampled_3mm is None:
        bold_ref = image.index_img(str(bold_path), 0)
        # Build 3mm target affine preserving origin from the 2mm BOLD affine
        aff = bold_ref.affine.copy()
        target_aff = aff.copy()
        target_aff[:3, :3] = np.diag([3, 3, 3]) * np.sign(np.diag(aff[:3, :3]))
        bold_ref_3mm = image.resample_img(bold_ref, target_affine=target_aff,
                                          interpolation='nearest')
        atlas_resampled = image.resample_to_img(
            SCHAEFER['maps'], bold_ref_3mm, interpolation='nearest'
        )
        _atlas_resampled_3mm = np.round(atlas_resampled.get_fdata()).astype(int)
        print(f"  Atlas resampled to 3mm space: {_atlas_resampled_3mm.shape}")
    return _atlas_resampled_3mm


def _resample_bold_to_3mm(bold_img):
    """Resample a 4D BOLD NIfTI image to 3mm isotropic, preserving origin."""
    aff = bold_img.affine.copy()
    target_aff = aff.copy()
    target_aff[:3, :3] = np.diag([3, 3, 3]) * np.sign(np.diag(aff[:3, :3]))
    return image.resample_img(bold_img, target_affine=target_aff,
                               interpolation='linear')


# ============================================================================
# pISC COMPUTATION (whole-brain or ROI-subset)
# ============================================================================

def compute_pisc(task, parcel_ids=None, resolution_mm=2, smooth_fwhm=None, do_hp=True):
    """Compute per-subject pISC for the given parcels.

    Parameters
    ----------
    task : str
    parcel_ids : iterable of int, optional
        1-based Schaefer parcel IDs to compute. Default: all 400.
    resolution_mm : int
        Voxel resolution for BOLD and atlas. 2 = native (default), 3 = resampled.
    smooth_fwhm : float or None
        If set, apply spatial smoothing with this FWHM (mm) before extracting
        parcel voxels. Default: None (no smoothing).
    do_hp : bool
        If True (default), apply 0.01 Hz Butterworth high-pass before z-scoring.

    Returns
    -------
    pisc : np.ndarray, shape (n_subjects, 400)
        Non-zero only for parcels in parcel_ids.
    """
    if parcel_ids is None:
        parcel_ids = range(1, 401)
    parcel_ids = list(parcel_ids)

    subjects = list(FILMFEST_SUBJECTS.keys())
    n_subjects = len(subjects)

    print(f"  Loading BOLD volumes for {task} (res={resolution_mm}mm, "
          f"smooth={'none' if smooth_fwhm is None else f'{smooth_fwhm}mm'}, "
          f"hp={'yes' if do_hp else 'no'})...")
    bold_volumes = {}
    T_list = []
    for subj, ses in FILMFEST_SUBJECTS.items():
        bold_path = get_bold_path(subj, ses, task)
        bold_img = nib.load(str(bold_path))
        if resolution_mm == 3:
            bold_img = _resample_bold_to_3mm(bold_img)
        if smooth_fwhm is not None:
            bold_img = image.smooth_img(bold_img, fwhm=smooth_fwhm)
        bold = bold_img.get_fdata()  # (X, Y, Z, T)
        bold_volumes[subj] = bold
        T_list.append(bold.shape[3])
        print(f"    {subj} {ses}: T={bold.shape[3]}, shape={bold.shape[:3]}")

    T_common = min(T_list)
    print(f"  T_common={T_common}")

    first_subj, first_ses = next(iter(FILMFEST_SUBJECTS.items()))
    first_bold_path = get_bold_path(first_subj, first_ses, task)
    if resolution_mm == 3:
        atlas_data = _get_atlas_data_3mm(first_bold_path)
    else:
        atlas_data = _get_atlas_data(first_bold_path)

    pisc = np.zeros((n_subjects, 400))
    n_valid = 0

    for parcel_id in parcel_ids:
        mask = atlas_data == parcel_id
        if mask.sum() < 2:
            continue
        n_valid += 1

        all_pdata = []
        for subj in subjects:
            pdata = bold_volumes[subj][mask].T[:T_common]  # (T, N_k)
            if do_hp:
                pdata = highpass_filter(pdata, order=2)
            pdata = sp_zscore(pdata, axis=0, nan_policy='omit')
            pdata = np.nan_to_num(pdata)
            all_pdata.append(pdata)

        all_pdata = np.stack(all_pdata, axis=0)  # (n_subj, T, N_k)
        group_mean = all_pdata.mean(axis=0)       # (T, N_k)

        for i in range(n_subjects):
            loo_mean = (group_mean * n_subjects - all_pdata[i]) / (n_subjects - 1)
            corr_diag = np.diag(cross_corrcoef(all_pdata[i], loo_mean))
            pisc[i, parcel_id - 1] = corr_diag.mean()

        if parcel_id % 50 == 0:
            print(f"    Processed parcel {parcel_id}")

    print(f"  {n_valid}/{len(parcel_ids)} parcels with ≥2 voxels")
    del bold_volumes
    return pisc


def load_or_compute_pisc(task, parcel_ids=None, resolution_mm=2, smooth_fwhm=None, do_hp=True):
    """Load per-subject pISC from cache or compute and save."""
    res_suffix    = '_res-3mm' if resolution_mm == 3 else ''
    smooth_suffix = f'_smooth-{int(smooth_fwhm)}mm' if smooth_fwhm is not None else ''
    hp_suffix     = '_hp' if do_hp else '_nohp'
    roi_suffix    = '_roi' if parcel_ids is not None else ''
    cache_file = CACHE_DIR / f'pISC_task-{task}{res_suffix}{smooth_suffix}{hp_suffix}{roi_suffix}_persubject.npz'
    if cache_file.exists():
        print(f"  {task}: loading pISC cache → {cache_file.name}")
        return np.load(cache_file)['pisc']
    print(f"  {task}: computing pISC{res_suffix}{smooth_suffix}{hp_suffix}{roi_suffix}...")
    pisc = compute_pisc(task, parcel_ids=parcel_ids, resolution_mm=resolution_mm,
                        smooth_fwhm=smooth_fwhm, do_hp=do_hp)
    np.savez_compressed(cache_file, pisc=pisc)
    print(f"  Cached → {cache_file.name}")
    return pisc


# ============================================================================
# SEARCHLIGHT pISC
# ============================================================================

def load_bold_volumes_for_searchlight(task):
    """Load, z-score, and intersect-mask all subjects' BOLD for searchlight.

    Returns
    -------
    bold_4d : dict  subject -> np.ndarray float32, shape (X, Y, Z, T_common)
    T_common : int
    affine : np.ndarray (4, 4)
    header : nibabel header
    intersection_mask : np.ndarray bool, shape (X, Y, Z)
    """
    subjects = list(FILMFEST_SUBJECTS.keys())
    bold_4d = {}
    masks = []
    T_list = []
    affine = header = None

    for subj, ses in FILMFEST_SUBJECTS.items():
        bold_path = get_bold_path(subj, ses, task)
        bold_img = nib.load(str(bold_path))
        if affine is None:
            affine = bold_img.affine
            header = bold_img.header
        bold = bold_img.get_fdata(dtype=np.float32)  # (X, Y, Z, T)
        T_list.append(bold.shape[3])
        bold_4d[subj] = bold

        mask_img = nib.load(str(_get_mask_path(subj, ses, task)))
        masks.append(mask_img.get_fdata().astype(bool))
        print(f"    {subj} {ses}: T={bold.shape[3]}, shape={bold.shape[:3]}")

    T_common = min(T_list)
    print(f"  T_common={T_common}")

    # Truncate and z-score voxel-wise along time axis
    for subj in subjects:
        bold = bold_4d[subj][..., :T_common]  # (X, Y, Z, T_common)
        # z-score each voxel time series independently
        mean = bold.mean(axis=3, keepdims=True)
        std = bold.std(axis=3, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        bold_4d[subj] = ((bold - mean) / std).astype(np.float32)

    intersection_mask = np.ones(masks[0].shape, dtype=bool)
    for m in masks:
        intersection_mask &= m
    n_brain = intersection_mask.sum()
    print(f"  Intersection mask: {n_brain} in-brain voxels")

    return bold_4d, T_common, affine, header, intersection_mask


def _compute_pisc_for_voxel(cx, cy, cz, all_bold_5d, intersection_mask, n_subjects, T_common):
    """Compute group-averaged pISC for one searchlight center voxel.

    Parameters
    ----------
    cx, cy, cz : int  — center voxel indices
    all_bold_5d : np.ndarray, shape (n_subjects, X, Y, Z, T_common), float32
    intersection_mask : np.ndarray bool, shape (X, Y, Z)
    n_subjects : int
    T_common : int

    Returns
    -------
    pisc_mean : float  — Fisher-z-averaged pISC across subjects, or np.nan if
                         fewer than 50% of the 5×5×5 cube is in-brain.
    """
    X, Y, Z = intersection_mask.shape
    x_lo, x_hi = max(0, cx - 2), min(X, cx + 3)
    y_lo, y_hi = max(0, cy - 2), min(Y, cy + 3)
    z_lo, z_hi = max(0, cz - 2), min(Z, cz + 3)

    # Check that at least 50% of the full 5×5×5 = 125 voxels are in-brain
    n_inbrain = intersection_mask[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi].sum()
    if n_inbrain < 63:
        return np.nan

    # Extract neighborhood: (n_subjects, nx, ny, nz, T) -> (n_subjects, T, N_k)
    nbhd = all_bold_5d[:, x_lo:x_hi, y_lo:y_hi, z_lo:z_hi, :]  # (n_subj, nx, ny, nz, T)
    nx, ny, nz = nbhd.shape[1], nbhd.shape[2], nbhd.shape[3]
    N_k = nx * ny * nz
    if N_k < 2:
        return np.nan

    all_pdata = nbhd.reshape(n_subjects, N_k, T_common).transpose(0, 2, 1)  # (n_subj, T, N_k)
    group_mean = all_pdata.mean(axis=0)  # (T, N_k)

    pisc_values = np.empty(n_subjects)
    for i in range(n_subjects):
        loo_mean = (group_mean * n_subjects - all_pdata[i]) / (n_subjects - 1)
        pisc_values[i] = np.diag(cross_corrcoef(all_pdata[i], loo_mean)).mean()

    return float(np.tanh(np.arctanh(np.clip(pisc_values, -0.999, 0.999)).mean()))


def compute_pisc_searchlight(task, n_jobs=1):
    """Compute searchlight pISC over the whole brain.

    For every in-brain voxel (intersection of all subject masks), extracts the
    5×5×5 neighborhood and computes the LOO pISC. Cubes with fewer than 50%
    in-brain voxels are set to NaN.

    Parameters
    ----------
    task : str
    n_jobs : int  — number of parallel jobs (joblib)

    Returns
    -------
    pisc_vol : np.ndarray, shape (X, Y, Z)  — group-averaged Fisher-z pISC
    affine : np.ndarray (4, 4)
    header : nibabel header
    """
    print(f"  Loading BOLD volumes for {task}...")
    bold_4d, T_common, affine, header, intersection_mask = load_bold_volumes_for_searchlight(task)

    subjects = list(FILMFEST_SUBJECTS.keys())
    n_subjects = len(subjects)

    # Stack into (n_subjects, X, Y, Z, T_common) — joblib will mmap-share this
    all_bold_5d = np.stack([bold_4d[s] for s in subjects], axis=0)
    del bold_4d

    center_voxels = np.argwhere(intersection_mask)  # (N_brain, 3)
    N_brain = len(center_voxels)
    print(f"  Running searchlight over {N_brain} center voxels (n_jobs={n_jobs})...")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_compute_pisc_for_voxel)(
            cx, cy, cz, all_bold_5d, intersection_mask, n_subjects, T_common
        )
        for cx, cy, cz in center_voxels
    )

    pisc_vol = np.full(intersection_mask.shape, np.nan, dtype=np.float32)
    for (cx, cy, cz), val in zip(center_voxels, results):
        pisc_vol[cx, cy, cz] = val

    return pisc_vol, affine, header


def load_or_compute_pisc_searchlight(task, n_jobs=1):
    """Load or compute searchlight pISC NIfTI for one task.

    Returns
    -------
    pisc_img : nibabel.Nifti1Image, shape (X, Y, Z)
    """
    cache_nii = CACHE_DIR / f'pISC_searchlight_task-{task}_group.nii.gz'
    if cache_nii.exists():
        print(f"  {task}: loading searchlight cache → {cache_nii.name}")
        return nib.load(str(cache_nii))

    print(f"  {task}: computing searchlight pISC (n_jobs={n_jobs})...")
    pisc_vol, affine, header = compute_pisc_searchlight(task, n_jobs=n_jobs)
    pisc_img = nib.Nifti1Image(pisc_vol, affine, header)
    nib.save(pisc_img, str(cache_nii))
    print(f"  Cached → {cache_nii.name}")
    return pisc_img


# ============================================================================
# SURFACE-BASED PARCEL pISC
# ============================================================================

def load_surface_bold_for_task(task):
    """Load and z-score fsaverage6 surface BOLD for all subjects.

    Returns
    -------
    surf_bold : dict  subject -> {'lh': arr, 'rh': arr}
                each arr shape (T_common, N_verts), float64, z-scored
    T_common : int
    lh_labels : np.ndarray int, shape (N_lh_verts,)  — annot label per vertex
    rh_labels : np.ndarray int, shape (N_rh_verts,)
    """
    lh_labels, _, _ = fs.read_annot(str(LH_ANNOT))
    rh_labels, _, _ = fs.read_annot(str(RH_ANNOT))

    surf_bold = {}
    T_list = []

    for subj, ses in FILMFEST_SUBJECTS.items():
        def _load(hemi):
            fname = (f'{subj}_{ses}_task-{task}_hemi-{hemi}'
                     f'_space-fsaverage6_bold.func.gii')
            fpath = DERIVATIVES_DIR / subj / ses / 'func' / fname
            # load_surf_data returns (N_verts, T) with dtype=object
            return load_surf_data(str(fpath)).astype(np.float64).T  # (T, N_verts)

        lh = _load('L')
        rh = _load('R')
        T_list.append(min(lh.shape[0], rh.shape[0]))
        surf_bold[subj] = {'lh': lh, 'rh': rh}
        print(f"    {subj} {ses}: T={lh.shape[0]}, "
              f"LH={lh.shape[1]} RH={rh.shape[1]} vertices")

    T_common = min(T_list)
    print(f"  T_common={T_common}")

    for subj in surf_bold:
        for hemi in ('lh', 'rh'):
            d = surf_bold[subj][hemi][:T_common]       # (T, N_verts)
            d = sp_zscore(d, axis=0, nan_policy='omit')
            surf_bold[subj][hemi] = np.nan_to_num(d)

    return surf_bold, T_common, lh_labels, rh_labels


def compute_pisc_surface(task, parcel_ids=None):
    """Compute per-subject pISC using surface vertices as spatial units.

    For each Schaefer parcel, the spatial pattern is defined by the vertices
    in fsaverage6 surface space (via .annot files) rather than volumetric voxels.

    Parameters
    ----------
    task : str
    parcel_ids : iterable of int, optional
        1-based Schaefer parcel IDs (1–400). Default: all 400.

    Returns
    -------
    pisc : np.ndarray, shape (n_subjects, 400)
    """
    if parcel_ids is None:
        parcel_ids = range(1, 401)
    parcel_ids = list(parcel_ids)

    subjects = list(FILMFEST_SUBJECTS.keys())
    n_subjects = len(subjects)

    print(f"  Loading surface BOLD for {task}...")
    surf_bold, T_common, lh_labels, rh_labels = load_surface_bold_for_task(task)

    pisc = np.zeros((n_subjects, 400))
    n_valid = 0

    for parcel_id in parcel_ids:
        if 1 <= parcel_id <= 200:
            vert_mask = lh_labels == parcel_id
            hemi = 'lh'
        else:
            vert_mask = rh_labels == (parcel_id - 200)
            hemi = 'rh'

        if vert_mask.sum() < 2:
            continue
        n_valid += 1

        all_pdata = np.stack(
            [surf_bold[s][hemi][:, vert_mask] for s in subjects], axis=0
        )  # (n_subj, T, N_verts)
        group_mean = all_pdata.mean(axis=0)  # (T, N_verts)

        for i in range(n_subjects):
            loo_mean = (group_mean * n_subjects - all_pdata[i]) / (n_subjects - 1)
            pisc[i, parcel_id - 1] = np.diag(cross_corrcoef(all_pdata[i], loo_mean)).mean()

        if parcel_id % 50 == 0:
            print(f"    Processed parcel {parcel_id}")

    print(f"  {n_valid}/{len(parcel_ids)} parcels with ≥2 vertices")
    del surf_bold
    return pisc


def load_or_compute_pisc_surface(task, parcel_ids=None):
    """Load per-subject surface pISC from cache or compute and save."""
    suffix = '_roi' if parcel_ids is not None else ''
    cache_file = CACHE_DIR / f'pISC_surface_task-{task}{suffix}_persubject.npz'
    if cache_file.exists():
        print(f"  {task}: loading surface pISC cache → {cache_file.name}")
        return np.load(cache_file)['pisc']
    print(f"  {task}: computing surface pISC{suffix}...")
    pisc = compute_pisc_surface(task, parcel_ids=parcel_ids)
    np.savez_compressed(cache_file, pisc=pisc)
    print(f"  Cached → {cache_file.name}")
    return pisc


# ============================================================================
# ROI LAG CURVES
# ============================================================================

def compute_roi_isc_ttc(task, roi_parcel_ids):
    """Compute per-subject T×T ISC corrmap for a multi-parcel ROI.

    Concatenates all voxels from the given parcels, temporally z-scores,
    and computes LOO cross-corrcoef between each subject and the group mean.

    Returns
    -------
    maps : dict subject -> np.ndarray, shape (T, T)
    """
    subjects = list(FILMFEST_SUBJECTS.keys())
    n_subjects = len(subjects)

    first_subj, first_ses = next(iter(FILMFEST_SUBJECTS.items()))
    atlas_data = _get_atlas_data(get_bold_path(first_subj, first_ses, task))

    T_list = []
    all_roi_data = {s: [] for s in subjects}
    for subj, ses in FILMFEST_SUBJECTS.items():
        bold_data = nib.load(str(get_bold_path(subj, ses, task))).get_fdata()
        T_list.append(bold_data.shape[3])
        for parcel_id in roi_parcel_ids:
            mask = atlas_data == parcel_id
            if mask.sum() < 2:
                continue
            all_roi_data[subj].append(bold_data[mask].T)  # (T, N_vox)
        del bold_data

    T = min(T_list)

    roi_data = {}
    for subj in subjects:
        pdata = np.concatenate([p[:T] for p in all_roi_data[subj]], axis=1)
        pdata = sp_zscore(pdata, axis=0, nan_policy='omit')
        pdata = np.nan_to_num(pdata)
        roi_data[subj] = pdata

    all_pdata = np.stack([roi_data[s] for s in subjects], axis=0)  # (n_subj, T, N_vox)
    group_mean = all_pdata.mean(axis=0)                            # (T, N_vox)

    result = {}
    for i, subj in enumerate(subjects):
        pdata = roi_data[subj]
        loo_mean = (group_mean * n_subjects - pdata) / (n_subjects - 1)
        result[subj] = cross_corrcoef(pdata, loo_mean)  # (T, T)

    return result


def compute_lag_curves_from_ttc(isc_maps, max_lag=30):
    """Extract per-subject lag curves from T×T ISC maps.

    For lag k, averages the k-th diagonal of the T×T matrix.

    Returns
    -------
    lag_curves : np.ndarray, shape (n_subjects, 2*max_lag+1)
    lags : np.ndarray — lag values in TRs, from -max_lag to max_lag
    """
    subjects = list(isc_maps.keys())
    lags = np.arange(-max_lag, max_lag + 1)
    lag_curves = np.zeros((len(subjects), len(lags)))

    for i, subj in enumerate(subjects):
        ttc = isc_maps[subj]
        for j, k in enumerate(lags):
            diag = np.diag(ttc, k)
            lag_curves[i, j] = diag.mean() if len(diag) > 0 else 0.0

    return lag_curves, lags


def load_or_compute_roi_lag_curves(task, roi_key, roi_parcel_ids, max_lag=30):
    """Load or compute lag curves for one ROI and task.

    Returns
    -------
    lag_curves : np.ndarray, shape (n_subjects, 2*max_lag+1)
    lags : np.ndarray
    """
    cache_file = CACHE_DIR / f'lag_pISC_task-{task}_roi-{roi_key}_persubject.npz'
    if cache_file.exists():
        print(f"    {task} {roi_key}: loading lag cache")
        data = np.load(cache_file)
        return data['lag_curves'], data['lags']

    print(f"    {task} {roi_key}: computing T×T ISC map...")
    isc_maps = compute_roi_isc_ttc(task, roi_parcel_ids)
    lag_curves, lags = compute_lag_curves_from_ttc(isc_maps, max_lag=max_lag)
    np.savez_compressed(cache_file, lag_curves=lag_curves, lags=lags)
    print(f"    Cached → {cache_file.name}")
    return lag_curves, lags


# ============================================================================
# SURFACE MAPPING
# ============================================================================

def pisc_to_textures(pisc_values):
    """Map 400-element pISC array to per-vertex textures for fsaverage6.

    LH parcels: volumetric IDs 1..200  → annot label indices 1..200
    RH parcels: volumetric IDs 201..400 → annot label indices 1..200

    Parameters
    ----------
    pisc_values : np.ndarray, shape (400,)

    Returns
    -------
    lh_texture, rh_texture : np.ndarray, shapes (n_lh_verts,), (n_rh_verts,)
    """
    lh_labels, _, _ = fs.read_annot(str(LH_ANNOT))
    rh_labels, _, _ = fs.read_annot(str(RH_ANNOT))

    lh_texture = np.zeros(len(lh_labels))
    for i in range(1, 201):
        lh_texture[lh_labels == i] = pisc_values[i - 1]

    rh_texture = np.zeros(len(rh_labels))
    for i in range(1, 201):
        rh_texture[rh_labels == i] = pisc_values[200 + i - 1]

    return lh_texture, rh_texture


# ============================================================================
# FIGURE: surface brain map
# ============================================================================

def make_pisc_figure(pisc_ff1, pisc_ff2, title, out_path, vmax):
    """Create 2 rows × 4 cols surface figure.

    Rows: filmfest1 (top), filmfest2 (bottom).
    Cols: left lateral | left medial | right lateral | right medial.

    Parameters
    ----------
    pisc_ff1, pisc_ff2 : np.ndarray, shape (400,)
    title : str — figure suptitle
    out_path : Path
    vmax : float — symmetric colorbar limit (vmin = -vmax)
    """
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    lh1, rh1 = pisc_to_textures(pisc_ff1)
    lh2, rh2 = pisc_to_textures(pisc_ff2)

    col_spec = [
        ('left',  'lateral',  fsavg.infl_left,  fsavg.sulc_left,  lh1, lh2),
        ('left',  'medial',   fsavg.infl_left,  fsavg.sulc_left,  lh1, lh2),
        ('right', 'lateral',  fsavg.infl_right, fsavg.sulc_right, rh1, rh2),
        ('right', 'medial',   fsavg.infl_right, fsavg.sulc_right, rh1, rh2),
    ]
    col_titles = ['Left Lateral', 'Left Medial', 'Right Lateral', 'Right Medial']
    row_titles = ['Filmfest 1', 'Filmfest 2']

    # Render each brain view in its own independent figure to avoid matplotlib
    # 3D state accumulation across subplots (causes wrong colormap for all-
    # positive textures when vmin=-vmax is passed to nilearn).
    brain_imgs = {}
    for col, (hemi, view, mesh, sulc, tex0, tex1) in enumerate(col_spec):
        for row, tex in enumerate([tex0, tex1]):
            fig_tmp = plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=tex,
                hemi=hemi,
                view=view,
                bg_map=sulc,
                colorbar=False,
                cmap='RdBu_r',
                vmin=-vmax,
                vmax=vmax,
                threshold=None,
                bg_on_data=True,
                darkness=0.5,
            )
            buf = io.BytesIO()
            fig_tmp.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                            facecolor='white')
            buf.seek(0)
            brain_imgs[(row, col)] = plt.imread(buf)
            plt.close(fig_tmp)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for col in range(4):
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(brain_imgs[(row, col)])
            ax.set_axis_off()
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold', pad=2)

    for row, label in enumerate(row_titles):
        y_pos = 0.75 - row * 0.5
        fig.text(0.01, y_pos, label, ha='left', va='center',
                 fontsize=11, fontweight='bold', rotation=90)

    sm = plt.cm.ScalarMappable(
        cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical',
                        fraction=0.015, pad=0.03, shrink=0.6)
    cbar.set_label('pISC (Pearson r)', fontsize=10)

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved → {out_path.name}")


# ============================================================================
# FIGURE: searchlight pISC projected to surface
# ============================================================================

def make_searchlight_surface_figure(lh1, rh1, lh2, rh2, out_path, vmax=0.15):
    """Plot searchlight pISC projected to fsaverage6 surface (2 rows × 4 cols).

    Rows: filmfest1 (top), filmfest2 (bottom).
    Cols: left lateral | left medial | right lateral | right medial.

    Parameters
    ----------
    lh1, rh1 : np.ndarray — per-vertex pISC for filmfest1, LH and RH
    lh2, rh2 : np.ndarray — per-vertex pISC for filmfest2, LH and RH
    out_path : Path
    vmax : float — symmetric colorbar limit
    """
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    col_spec = [
        ('left',  'lateral',  fsavg.infl_left,  fsavg.sulc_left,  lh1, lh2),
        ('left',  'medial',   fsavg.infl_left,  fsavg.sulc_left,  lh1, lh2),
        ('right', 'lateral',  fsavg.infl_right, fsavg.sulc_right, rh1, rh2),
        ('right', 'medial',   fsavg.infl_right, fsavg.sulc_right, rh1, rh2),
    ]
    col_titles = ['Left Lateral', 'Left Medial', 'Right Lateral', 'Right Medial']
    row_titles = ['Filmfest 1', 'Filmfest 2']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})
    fig.suptitle('Searchlight pISC — Filmfest (5×5×5 voxels, LOO, projected to surface)',
                 fontsize=13, fontweight='bold', y=1.01)

    for col, (hemi, view, mesh, sulc, tex0, tex1) in enumerate(col_spec):
        for row, tex in enumerate([tex0, tex1]):
            ax = axes[row, col]
            plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=tex,
                hemi=hemi,
                view=view,
                bg_map=sulc,
                axes=ax,
                colorbar=False,
                cmap='RdBu_r',
                vmin=-vmax,
                vmax=vmax,
                threshold=None,
                bg_on_data=True,
                darkness=0.5,
            )
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold', pad=2)

    for row, label in enumerate(row_titles):
        y_pos = 0.75 - row * 0.5
        fig.text(0.01, y_pos, label, ha='left', va='center',
                 fontsize=11, fontweight='bold', rotation=90)

    sm = plt.cm.ScalarMappable(
        cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical',
                        fraction=0.015, pad=0.03, shrink=0.6)
    cbar.set_label('pISC (Pearson r)', fontsize=10)

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved → {out_path.name}")


# ============================================================================
# FIGURE: lag curves (2 rows × 4 cols)
# ============================================================================

def make_lag_figure(all_lag_data, out_path):
    """Plot pISC vs. time lag for each ROI and task.

    Parameters
    ----------
    all_lag_data : dict  task -> dict roi_key -> (lag_curves, lags)
        lag_curves shape (n_subjects, n_lags); lags shape (n_lags,)
    out_path : Path
    """
    tasks = ['filmfest1', 'filmfest2']
    row_titles = ['Filmfest 1', 'Filmfest 2']
    col_titles = [name for _, name, _ in ROI_SPEC]
    roi_keys   = [key  for key, _, _ in ROI_SPEC]

    n_subjects = len(FILMFEST_SUBJECTS)
    subj_colors = plt.cm.tab10(np.linspace(0, 1, n_subjects))

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
    fig.suptitle('pISC by Time Lag — ROIs (LOO)', fontsize=13, fontweight='bold')

    for row, task in enumerate(tasks):
        for col, roi_key in enumerate(roi_keys):
            ax = axes[row, col]
            lag_curves, lags = all_lag_data[task][roi_key]

            # Fisher-z group average
            z = np.arctanh(np.clip(lag_curves, -0.999, 0.999))
            group_curve = np.tanh(z.mean(axis=0))

            # Individual subjects
            for i in range(n_subjects):
                ax.plot(lags, lag_curves[i], color=subj_colors[i],
                        alpha=0.35, linewidth=0.8)

            # Group mean
            ax.plot(lags, group_curve, color='k', linewidth=2.0)

            ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
            ax.axvline(0, color='gray', linewidth=0.6, linestyle='--')
            ax.set_xlim(lags[0], lags[-1])

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{row_titles[row]}\npISC (Pearson r)', fontsize=9)
            if row == 1:
                ax.set_xlabel('Lag (TRs)', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved → {out_path.name}")


# ============================================================================
# SEARCHLIGHT → SURFACE PROJECTION
# ============================================================================

def project_searchlight_to_surface(pisc_img, radius=6.0):
    """Project a volumetric searchlight pISC NIfTI image onto fsaverage6 surfaces.

    Uses nilearn's vol_to_surf to sample the MNI-space volume at each fsaverage6
    vertex within a sphere of `radius` mm, averaging valid voxels.

    Parameters
    ----------
    pisc_img : nib.Nifti1Image, shape (X, Y, Z)
    radius : float
        Sampling radius in mm (default 6 mm ≈ 3 voxels at 2mm iso).

    Returns
    -------
    lh_texture, rh_texture : np.ndarray, each shape (n_vertices,)
        NaN where no in-brain voxels fell within the sphere.
    """
    from nilearn.surface import vol_to_surf

    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    lh_texture = vol_to_surf(
        pisc_img,
        surf_mesh=fsavg.pial_left,
        radius=radius,
        interpolation='linear',
    )
    rh_texture = vol_to_surf(
        pisc_img,
        surf_mesh=fsavg.pial_right,
        radius=radius,
        interpolation='linear',
    )
    return lh_texture, rh_texture


# ============================================================================
# FIGURE: searchlight volumetric stat maps
# ============================================================================

def make_searchlight_figure(pisc_ff1_img, pisc_ff2_img, out_path, vmax=0.15):
    """Create 2-row figure with axial slices for searchlight pISC NIfTI maps.

    Parameters
    ----------
    pisc_ff1_img, pisc_ff2_img : nib.Nifti1Image, shape (X, Y, Z)
    out_path : Path
    vmax : float
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    row_titles = ['Filmfest 1', 'Filmfest 2']

    for ax, img, title in zip(axes, [pisc_ff1_img, pisc_ff2_img], row_titles):
        display = plotting.plot_stat_map(
            img,
            display_mode='z',
            cut_coords=8,
            axes=ax,
            colorbar=True,
            cmap='RdBu_r',
            vmax=vmax,
            threshold=None,
            title=title,
        )

    plt.suptitle('Searchlight pISC — Filmfest (5×5×5 voxels, LOO)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved → {out_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main(roi_only=False, resolution_mm=2, smooth_fwhm=None, do_hp=True):
    for d in (OUTPUT_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    res_label    = f'{resolution_mm}mm'
    res_suffix   = f'_res-{resolution_mm}mm' if resolution_mm != 2 else ''
    smooth_label  = f'_smooth-{int(smooth_fwhm)}mm' if smooth_fwhm is not None else ''
    hp_label     = '_hp' if do_hp else '_nohp'

    print("=" * 60)
    print("FILMFEST PARCEL-WISE ISC (pISC)")
    print(f"Mode: {'ROI only' if roi_only else 'whole-brain'}, resolution: {res_label}")
    print(f"Temporal filter: {'0.01 Hz high-pass' if do_hp else 'none'}")
    print(f"Spatial smoothing: {'none' if smooth_fwhm is None else f'{smooth_fwhm}mm FWHM'}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    parcel_ids = ROI_PARCEL_IDS if roi_only else None
    roi_suffix = '_roi' if roi_only else ''
    suffix = f'{res_suffix}{smooth_label}{hp_label}{roi_suffix}'

    # Load or compute per-subject pISC
    pisc = {}
    for task in ('filmfest1', 'filmfest2'):
        print(f"\n--- {task} ---")
        pisc[task] = load_or_compute_pisc(task, parcel_ids=parcel_ids,
                                          resolution_mm=resolution_mm,
                                          smooth_fwhm=smooth_fwhm,
                                          do_hp=do_hp)

    subjects = list(FILMFEST_SUBJECTS.keys())

    # Group pISC via Fisher-z averaging
    group_pisc = {}
    for task in ('filmfest1', 'filmfest2'):
        z = np.arctanh(np.clip(pisc[task], -0.999, 0.999))
        group_pisc[task] = np.tanh(z.mean(axis=0))
        gp = group_pisc[task]
        print(f"\n{task} group pISC: "
              f"min={gp.min():.3f}, max={gp.max():.3f}, mean={gp.mean():.3f}")

    vmax = 0.15

    # Group brain map
    print("\n--- Group figure ---")
    make_pisc_figure(
        pisc_ff1=group_pisc['filmfest1'],
        pisc_ff2=group_pisc['filmfest2'],
        title=f'Group pISC — Filmfest (Schaefer 400, {res_label}, LOO)',
        out_path=OUTPUT_DIR / f'GROUP_pisc_filmfest{suffix}.png',
        vmax=vmax,
    )

    # Per-subject brain maps
    print("\n--- Per-subject figures ---")
    for i, (subject, session) in enumerate(FILMFEST_SUBJECTS.items()):
        print(f"  {subject} {session}")
        make_pisc_figure(
            pisc_ff1=pisc['filmfest1'][i],
            pisc_ff2=pisc['filmfest2'][i],
            title=f'pISC ({res_label}) — {subject} {session}',
            out_path=OUTPUT_DIR / f'{subject}_{session}_pisc_filmfest{suffix}.png',
            vmax=vmax,
        )

    # Lag figure (ROI mode only)
    if roi_only:
        print("\n--- ROI lag curves ---")
        all_lag_data = {}
        for task in ('filmfest1', 'filmfest2'):
            all_lag_data[task] = {}
            for roi_key, _, roi_parcel_ids in ROI_SPEC:
                lag_curves, lags = load_or_compute_roi_lag_curves(
                    task, roi_key, roi_parcel_ids, max_lag=30
                )
                all_lag_data[task][roi_key] = (lag_curves, lags)

        make_lag_figure(
            all_lag_data=all_lag_data,
            out_path=OUTPUT_DIR / 'GROUP_lag_pisc_filmfest.png',
        )

    print("\n" + "=" * 60)
    print(f"DONE. Figures saved to {OUTPUT_DIR}")
    print("=" * 60)


def main_surface(roi_only=False):
    """Run surface-based parcel pISC for filmfest1 and filmfest2."""
    for d in (OUTPUT_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FILMFEST SURFACE-BASED PARCEL pISC")
    print(f"Mode: {'ROI only' if roi_only else 'whole-brain'}")
    print(f"Units: fsaverage6 surface vertices (via Schaefer .annot)")
    print("=" * 60)

    parcel_ids = ROI_PARCEL_IDS if roi_only else None
    suffix = '_surface_roi' if roi_only else '_surface'

    pisc = {}
    for task in ('filmfest1', 'filmfest2'):
        print(f"\n--- {task} ---")
        pisc[task] = load_or_compute_pisc_surface(task, parcel_ids=parcel_ids)

    # Group pISC via Fisher-z averaging
    group_pisc = {}
    for task in ('filmfest1', 'filmfest2'):
        z = np.arctanh(np.clip(pisc[task], -0.999, 0.999))
        group_pisc[task] = np.tanh(z.mean(axis=0))
        gp = group_pisc[task]
        print(f"\n{task} group surface pISC: "
              f"min={gp.min():.3f}, max={gp.max():.3f}, mean={gp.mean():.3f}")

    vmax = 0.15

    print("\n--- Group figure ---")
    make_pisc_figure(
        pisc_ff1=group_pisc['filmfest1'],
        pisc_ff2=group_pisc['filmfest2'],
        title='Group pISC — Filmfest (Surface vertices, Schaefer 400, LOO)',
        out_path=OUTPUT_DIR / f'GROUP_pisc_filmfest{suffix}.png',
        vmax=vmax,
    )

    print("\n--- Per-subject figures ---")
    for i, (subject, session) in enumerate(FILMFEST_SUBJECTS.items()):
        print(f"  {subject} {session}")
        make_pisc_figure(
            pisc_ff1=pisc['filmfest1'][i],
            pisc_ff2=pisc['filmfest2'][i],
            title=f'pISC (surface) — {subject} {session}',
            out_path=OUTPUT_DIR / f'{subject}_{session}_pisc_filmfest{suffix}.png',
            vmax=vmax,
        )

    print("\n" + "=" * 60)
    print(f"DONE. Figures saved to {OUTPUT_DIR}")
    print("=" * 60)


def main_searchlight(n_jobs=1):
    """Run searchlight pISC analysis for filmfest1 and filmfest2."""
    for d in (OUTPUT_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FILMFEST SEARCHLIGHT pISC")
    print(f"Searchlight: 5×5×5 voxel cubes (2mm iso → 10mm FWHM equiv.)")
    print(f"n_jobs: {n_jobs}")
    print("=" * 60)

    imgs = {}
    for task in ('filmfest1', 'filmfest2'):
        print(f"\n--- {task} ---")
        imgs[task] = load_or_compute_pisc_searchlight(task, n_jobs=n_jobs)

    make_searchlight_figure(
        pisc_ff1_img=imgs['filmfest1'],
        pisc_ff2_img=imgs['filmfest2'],
        out_path=OUTPUT_DIR / 'GROUP_pisc_searchlight_filmfest.png',
        vmax=0.15,
    )

    for task, img in imgs.items():
        out_nii = OUTPUT_DIR / f'GROUP_pisc_searchlight_{task}.nii.gz'
        nib.save(img, str(out_nii))
        print(f"  Saved NIfTI → {out_nii.name}")

    print("\n" + "=" * 60)
    print(f"DONE. Figures saved to {OUTPUT_DIR}")
    print("=" * 60)


def main_searchlight_surface(n_jobs=1, radius=6.0, vmax=0.15):
    """Load (or compute) searchlight pISC, project to surface, and plot."""
    for d in (OUTPUT_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FILMFEST SEARCHLIGHT pISC → SURFACE PROJECTION")
    print(f"Sampling radius: {radius} mm")
    print("=" * 60)

    imgs = {}
    for task in ('filmfest1', 'filmfest2'):
        print(f"\n--- {task} ---")
        imgs[task] = load_or_compute_pisc_searchlight(task, n_jobs=n_jobs)

    print("\n--- Projecting to fsaverage6 surface ---")
    lh1, rh1 = project_searchlight_to_surface(imgs['filmfest1'], radius=radius)
    lh2, rh2 = project_searchlight_to_surface(imgs['filmfest2'], radius=radius)
    print(f"  LH texture: {lh1.shape}, RH texture: {rh1.shape}")

    out_path = OUTPUT_DIR / 'GROUP_pisc_searchlight_filmfest_surface.png'
    make_searchlight_surface_figure(lh1, rh1, lh2, rh2, out_path=out_path, vmax=vmax)

    print("\n" + "=" * 60)
    print(f"DONE. Figures saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', action='store_true',
                        help='Compute pISC only for EVC/EAC/PMC/AG parcels '
                             'and produce ROI lag-pISC figure')
    parser.add_argument('--surface', action='store_true',
                        help='Use fsaverage6 surface vertices as spatial units '
                             'instead of volumetric voxels (still parcel-wise)')
    parser.add_argument('--3mm', dest='res3mm', action='store_true',
                        help='Resample BOLD and atlas to 3mm isotropic before '
                             'computing parcel-wise pISC (volumetric only)')
    parser.add_argument('--searchlight', action='store_true',
                        help='Run searchlight pISC with 5×5×5 voxel cubes '
                             'instead of Schaefer parcels. Outputs NIfTI maps.')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs for searchlight (default: 1)')
    parser.add_argument('--searchlight-surface', action='store_true',
                        help='Project searchlight pISC NIfTI maps onto fsaverage6 surface '
                             'using vol_to_surf and plot. Loads cached NIfTIs if available.')
    parser.add_argument('--radius', type=float, default=6.0,
                        help='vol_to_surf sampling radius in mm (default: 6.0)')
    parser.add_argument('--vmax', type=float, default=0.15,
                        help='Colorbar max for surface searchlight figure (default: 0.15)')
    parser.add_argument('--smooth', type=float, default=None,
                        help='Apply spatial smoothing with this FWHM in mm before '
                             'extracting parcel data (e.g. --smooth 6)')
    parser.add_argument('--no-hp', dest='no_hp', action='store_true',
                        help='Skip 0.01 Hz high-pass filtering (z-score only)')
    args = parser.parse_args()

    if args.searchlight_surface:
        main_searchlight_surface(n_jobs=args.n_jobs, radius=args.radius, vmax=args.vmax)
    elif args.searchlight:
        main_searchlight(n_jobs=args.n_jobs)
    elif args.surface:
        main_surface(roi_only=args.roi)
    else:
        main(roi_only=args.roi, resolution_mm=3 if args.res3mm else 2,
             smooth_fwhm=args.smooth, do_hp=not args.no_hp)
