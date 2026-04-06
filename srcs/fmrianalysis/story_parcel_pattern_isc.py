#!/usr/bin/env python3
"""
story_pisc.py — Per-story parcel-wise Inter-Subject Spatial Pattern Correlation

For each story-listening task, computes pISC across 5 preprocessing variants
(all z-scored per voxel over time):
  raw      : z-score only
  sm4      : 4mm spatial smoothing
  sm6      : 6mm spatial smoothing
  sm4hp    : 4mm smoothing + 0.01 Hz Butterworth high-pass
  sm4clean : 4mm smoothing + nilearn.signal.clean (detrend + DCT HP + confounds)

Confounds for sm4clean: 6 motion params + 6 derivatives + top-5 WM aCompCor
+ top-5 CSF aCompCor (22 regressors total).

For each Schaefer 400 parcel, pISC = mean of the diagonal of the LOO Pearson r
matrix between each subject and the LOO group mean, Fisher-z averaged across
subjects. Diagonal Pearson is computed efficiently as a row-wise dot product
(O(T×V)) rather than the full T×T matrix.

Outputs:
  CACHE_DIR/analyses/story_pisc/{story}/{variant}_pisc.npz
      (keys: 'group' (400,), 'per_subject' (n_subjects, 400),
             'subjects', 'sessions')
  FIGS_DIR/story_pisc/pisc_{story}.png   (5 rows × 4 cols surface map)

Usage:
    uv run python srcs/fmrianalysis/story_pisc.py
    uv run python srcs/fmrianalysis/story_pisc.py --story adollshouse buck
    uv run python srcs/fmrianalysis/story_pisc.py --no-cache
    uv run python srcs/fmrianalysis/story_pisc.py --no-fig
    uv run python srcs/fmrianalysis/story_pisc.py --vmax 0.3
"""
import io
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.freesurfer as fs
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import datasets, image, plotting
from nilearn.signal import clean as signal_clean
from scipy.stats import zscore as sp_zscore

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, ANALYSIS_CACHE_DIR
from fmrianalysis.utils import highpass_filter, get_bold_path

# ============================================================================
# CONSTANTS
# ============================================================================

STORIES = [
    'adollshouse', 'adventuresinsayingyes', 'beneaththemushroomcloud',
    'breakingupintheageofgoogle', 'buck', 'christmas1940', 'howtodraw',
    'inamoment', 'notontheusualtour', 'odetostepfather', 'penpal',
    'shoppinginchina', 'swimming', 'theclosetthatateeverything',
    'treasureisland', 'undertheinfluence', 'vixen', 'wheretheressmoke',
]

VARIANTS = ['raw', 'sm4', 'sm6', 'sm4hp', 'sm4clean']

SMOOTH_FWHM_4 = 4.0
SMOOTH_FWHM_6 = 6.0
HP_CUTOFF = 0.01
HP_ORDER = 2

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'

OUTPUT_DIR = FIGS_DIR / 'story_pisc'
CACHE_BASE  = ANALYSIS_CACHE_DIR / 'story_pisc'

# Load Schaefer atlas once at import time
print("Loading Schaefer atlas...")
SCHAEFER = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
_atlas_resampled = None  # lazily resampled to BOLD voxel space
print("  Schaefer atlas loaded.")


# ============================================================================
# DISCOVERY
# ============================================================================

def discover_story_subjects(story):
    """Find (subject, session) pairs with BOLD data for this story.

    Returns one pair per subject, using the earliest session if duplicates exist.
    """
    pattern = (f"sub-*/ses-*/func/sub-*_ses-*_task-{story}_space-MNI152NLin6Asym"
               f"_res-2_desc-preproc_bold.nii.gz")
    bold_files = sorted(DERIVATIVES_DIR.glob(pattern))
    seen = {}
    for f in bold_files:
        parts = f.name.split('_')
        subj = parts[0]
        ses = parts[1]
        if subj not in seen:
            seen[subj] = ses
    return list(seen.items())


# ============================================================================
# ATLAS
# ============================================================================

def _get_atlas_data(bold_path):
    """Schaefer atlas resampled to BOLD voxel space (cached globally)."""
    global _atlas_resampled
    if _atlas_resampled is None:
        bold_ref = image.index_img(str(bold_path), 0)
        atlas_img = image.resample_to_img(
            SCHAEFER['maps'], bold_ref, interpolation='nearest'
        )
        _atlas_resampled = np.round(atlas_img.get_fdata()).astype(int)
        print(f"  Atlas resampled to BOLD space: {_atlas_resampled.shape}")
    return _atlas_resampled


# ============================================================================
# EFFICIENT DIAGONAL pISC
# ============================================================================

def _diag_pisc(A, B):
    """Mean diagonal Pearson r between matching rows of A and B.

    Efficient O(T×V) computation — avoids building the full T×T matrix.

    Parameters
    ----------
    A, B : np.ndarray, shape (T, V)

    Returns
    -------
    float — mean of diagonal Pearson r values
    """
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)
    A_n = np.linalg.norm(A_c, axis=1)
    B_n = np.linalg.norm(B_c, axis=1)
    denom = A_n * B_n
    safe_denom = np.where(denom > 0, denom, 1.0)
    diag_r = np.where(denom > 0, (A_c * B_c).sum(axis=1) / safe_denom, 0.0)
    return float(diag_r.mean())


def _loo_group_pisc(all_pdata):
    """Per-subject LOO pISC from stacked (n_subjects, T, V) array.

    Returns np.ndarray, shape (n_subjects,).
    """
    n_subjects = all_pdata.shape[0]
    group_mean = all_pdata.mean(axis=0)  # (T, V)
    per_subj = np.empty(n_subjects)
    for i in range(n_subjects):
        loo_mean = (group_mean * n_subjects - all_pdata[i]) / (n_subjects - 1)
        per_subj[i] = _diag_pisc(all_pdata[i], loo_mean)
    return per_subj


# ============================================================================
# PREPROCESSING
# ============================================================================

def _preprocess(vox, do_hp):
    """Apply optional highpass filter then z-score.

    Parameters
    ----------
    vox   : np.ndarray, shape (T, V)
    do_hp : bool

    Returns
    -------
    np.ndarray, shape (T, V)
    """
    if do_hp:
        vox = highpass_filter(vox, cutoff=HP_CUTOFF, tr=TR, order=HP_ORDER)
    vox = sp_zscore(vox, axis=0, nan_policy='omit')
    return np.nan_to_num(vox)


# ============================================================================
# CONFOUND LOADING
# ============================================================================

MOTION_COLS = [
    'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
]
ACOMPCOR_COLS = (
    [f'w_comp_cor_{i:02d}' for i in range(5)] +
    [f'c_comp_cor_{i:02d}' for i in range(5)]
)
CONFOUND_COLS = MOTION_COLS + ACOMPCOR_COLS  # 22 total


def load_confounds(subject, session, task):
    """Load motion + aCompCor confounds from fMRIPrep TSV.

    Returns np.ndarray (T, 22) with NaN derivatives replaced by 0.
    """
    tsv = (DERIVATIVES_DIR / subject / session / 'func' /
           f'{subject}_{session}_task-{task}_desc-confounds_timeseries.tsv')
    df = pd.read_csv(tsv, sep='\t')
    return np.nan_to_num(df[CONFOUND_COLS].values.astype(np.float64))


def _preprocess_clean(vox, conf):
    """nilearn.signal.clean (detrend + DCT high-pass + confound regression) → z-score.

    Parameters
    ----------
    vox  : np.ndarray, shape (T, V)
    conf : np.ndarray, shape (T_full, 22) — trimmed to T inside

    Returns
    -------
    np.ndarray, shape (T, V)
    """
    T = vox.shape[0]
    cleaned = signal_clean(
        vox,
        detrend=True,
        standardize=False,
        high_pass=HP_CUTOFF,
        t_r=TR,
        confounds=conf[:T],
    )
    return np.nan_to_num(sp_zscore(cleaned, axis=0, nan_policy='omit'))


# ============================================================================
# CACHE PATHS
# ============================================================================

def _cache_file(story, variant):
    return CACHE_BASE / story / f'{variant}_pisc.npz'


def _all_cached(story):
    return all(_cache_file(story, v).exists() for v in VARIANTS)


# ============================================================================
# pISC COMPUTATION
# ============================================================================

def compute_story_pisc(story, subjects_sessions):
    """Compute group pISC for all 5 variants for one story.

    Memory-efficient: processes one subject at a time, stores only parcel
    voxel extracts (not full-brain volumes) before computing pISC.

    Parameters
    ----------
    story : str
    subjects_sessions : list of (subject, session) tuples

    Returns
    -------
    dict : variant -> group pISC np.ndarray (400,)
    """
    n_subjects = len(subjects_sessions)
    subjects = [s for s, _ in subjects_sessions]

    # Get atlas and T_common (header-only reads for T)
    first_path = get_bold_path(subjects_sessions[0][0], subjects_sessions[0][1], story)
    atlas_data = _get_atlas_data(first_path)

    T_list = [nib.load(str(get_bold_path(s, ses, story))).shape[3]
              for s, ses in subjects_sessions]
    T_common = min(T_list)
    print(f"  [{story}] T_common={T_common}, n_subjects={n_subjects}")

    valid_parcels = [pid for pid in range(1, 401)
                     if (atlas_data == pid).sum() >= 2]
    print(f"  [{story}] {len(valid_parcels)}/400 valid parcels (≥2 voxels)")

    # Pre-extract parcel voxels: one subject at a time to avoid holding
    # full-brain volumes for all subjects simultaneously.
    raw_pdata  = {s: {} for s in subjects}  # subj -> parcel_id -> (T, V) float32
    sm4_pdata  = {s: {} for s in subjects}
    sm6_pdata  = {s: {} for s in subjects}
    confounds  = {}  # subj -> (T_common, 22)

    for subj, ses in subjects_sessions:
        bold_path = get_bold_path(subj, ses, story)
        bold_img = nib.load(str(bold_path))
        bold = bold_img.get_fdata(dtype=np.float32)  # (X, Y, Z, T)

        for pid in valid_parcels:
            mask = atlas_data == pid
            raw_pdata[subj][pid] = bold[mask].T[:T_common]  # (T, V)

        bold_sm4 = image.smooth_img(bold_img, fwhm=SMOOTH_FWHM_4).get_fdata(dtype=np.float32)
        bold_sm6 = image.smooth_img(bold_img, fwhm=SMOOTH_FWHM_6).get_fdata(dtype=np.float32)
        del bold, bold_img

        for pid in valid_parcels:
            mask = atlas_data == pid
            sm4_pdata[subj][pid] = bold_sm4[mask].T[:T_common]
            sm6_pdata[subj][pid] = bold_sm6[mask].T[:T_common]

        del bold_sm4, bold_sm6

        confounds[subj] = load_confounds(subj, ses, story)[:T_common]
        print(f"    {subj} ({ses}) extracted")

    # Compute pISC parcel by parcel for all 5 variants simultaneously
    pisc_sub = {v: np.zeros((n_subjects, 400)) for v in VARIANTS}

    for pid in valid_parcels:
        arrays = {
            'raw':      np.stack([_preprocess(raw_pdata[s][pid].astype(np.float64),  False) for s in subjects], 0),
            'sm4':      np.stack([_preprocess(sm4_pdata[s][pid].astype(np.float64),  False) for s in subjects], 0),
            'sm6':      np.stack([_preprocess(sm6_pdata[s][pid].astype(np.float64),  False) for s in subjects], 0),
            'sm4hp':    np.stack([_preprocess(sm4_pdata[s][pid].astype(np.float64),  True)  for s in subjects], 0),
            'sm4clean': np.stack([_preprocess_clean(sm4_pdata[s][pid].astype(np.float64), confounds[s]) for s in subjects], 0),
        }
        for v, arr in arrays.items():
            pisc_sub[v][:, pid - 1] = _loo_group_pisc(arr)

        if pid % 100 == 0:
            print(f"    parcel {pid}/400")

    del raw_pdata, sm4_pdata, sm6_pdata

    # Fisher-z average across subjects
    def fisher_avg(ps):
        return np.tanh(np.arctanh(np.clip(ps, -0.999, 0.999)).mean(axis=0))

    # Save to cache
    group_pisc = {}
    cache_dir = CACHE_BASE / story
    cache_dir.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        group = fisher_avg(pisc_sub[v])
        np.savez_compressed(
            _cache_file(story, v),
            group=group,
            per_subject=pisc_sub[v],
            subjects=np.array(subjects),
            sessions=np.array([ses for _, ses in subjects_sessions]),
        )
        group_pisc[v] = group
        print(f"    Cached {v} → {_cache_file(story, v).name}")

    return group_pisc


def load_or_compute(story, subjects_sessions, force=False):
    """Load group pISC from cache or compute from scratch."""
    if not force and _all_cached(story):
        print(f"  [{story}] loading all variants from cache")
        return {v: np.load(_cache_file(story, v))['group'] for v in VARIANTS}
    return compute_story_pisc(story, subjects_sessions)


# ============================================================================
# SURFACE MAPPING
# ============================================================================

def pisc_to_textures(pisc_values):
    """Map 400-element pISC array to per-vertex textures for fsaverage6.

    LH parcels: IDs 1–200  → annot labels 1–200
    RH parcels: IDs 201–400 → annot labels 1–200
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
# FIGURE
# ============================================================================

VARIANT_LABELS = {
    'raw':      'Raw',
    'sm4':      f'Smooth {int(SMOOTH_FWHM_4)}mm',
    'sm6':      f'Smooth {int(SMOOTH_FWHM_6)}mm',
    'sm4hp':    f'Smooth {int(SMOOTH_FWHM_4)}mm + HP {HP_CUTOFF}Hz',
    'sm4clean': f'Smooth {int(SMOOTH_FWHM_4)}mm + Clean (detrend + HP 0.01Hz + confounds)',
}


def make_story_figure(story, pisc_by_variant, vmax, out_path):
    """5 rows (variants) × 4 cols (views) surface figure for one story.

    Rows: raw, sm4, sm6, sm4hp, sm4clean.
    Cols: left lateral, left medial, right lateral, right medial.
    """
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    col_spec = [
        ('left',  'lateral', fsavg.infl_left,  fsavg.sulc_left),
        ('left',  'medial',  fsavg.infl_left,  fsavg.sulc_left),
        ('right', 'lateral', fsavg.infl_right, fsavg.sulc_right),
        ('right', 'medial',  fsavg.infl_right, fsavg.sulc_right),
    ]
    col_titles = ['Left Lateral', 'Left Medial', 'Right Lateral', 'Right Medial']

    # Convert pISC values to surface textures
    textures = {v: pisc_to_textures(pisc_by_variant[v]) for v in VARIANTS}

    # Render each brain view to a buffer (avoids 3D state accumulation in subplots)
    brain_imgs = {}
    for row, variant in enumerate(VARIANTS):
        lh_tex, rh_tex = textures[variant]
        for col, (hemi, view, mesh, sulc) in enumerate(col_spec):
            tex = lh_tex if hemi == 'left' else rh_tex
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
            fig_tmp.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                            facecolor='white')
            buf.seek(0)
            brain_imgs[(row, col)] = plt.imread(buf)
            plt.close(fig_tmp)

    n_rows = len(VARIANTS)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 22))
    # Reserve left margin for row labels, right for colorbar, top for title
    fig.subplots_adjust(left=0.08, right=0.90, top=0.95, bottom=0.01,
                        hspace=0.02, wspace=0.01)
    fig.suptitle(f'Story pISC: {story}', fontsize=14, fontweight='bold')

    for col in range(4):
        axes[0, col].set_title(col_titles[col], fontsize=10, fontweight='bold', pad=2)

    for row, variant in enumerate(VARIANTS):
        for col in range(4):
            axes[row, col].imshow(brain_imgs[(row, col)])
            axes[row, col].set_axis_off()
        # Position row label at the actual vertical centre of the axes row
        bbox = axes[row, 0].get_position()
        fig.text(0.01, bbox.y0 + bbox.height / 2, VARIANT_LABELS[variant],
                 ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)

    # Colorbar in a dedicated axes on the right
    cbar_ax = fig.add_axes([0.91, 0.40, 0.015, 0.20])
    sm = plt.cm.ScalarMappable(
        cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('pISC (Pearson r)', fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved → {out_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Story pISC analysis across 4 preprocessing variants."
    )
    parser.add_argument(
        '--story', nargs='+', default=None,
        help="Stories to process (default: all 18)",
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help="Recompute even if cache exists",
    )
    parser.add_argument(
        '--no-fig', action='store_true',
        help="Skip figure generation (cache only mode)",
    )
    parser.add_argument(
        '--vmax', type=float, default=0.15,
        help="Symmetric colorbar limit for surface plots (default: 0.15)",
    )
    args = parser.parse_args()

    stories = args.story or STORIES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_BASE.mkdir(parents=True, exist_ok=True)

    for story in stories:
        print(f"\n{'='*60}")
        print(f"Story: {story}")
        pairs = discover_story_subjects(story)
        if not pairs:
            print(f"  WARNING: no BOLD data found for '{story}', skipping")
            continue
        print(f"  {len(pairs)} subjects: " +
              ", ".join(f"{s}({ses})" for s, ses in pairs))

        pisc_by_variant = load_or_compute(story, pairs, force=args.no_cache)

        if not args.no_fig:
            out_path = OUTPUT_DIR / f'pisc_{story}.png'
            if out_path.exists() and not args.no_cache:
                print(f"  Figure already exists: {out_path.name}")
            else:
                make_story_figure(story, pisc_by_variant,
                                  vmax=args.vmax, out_path=out_path)

    print("\nDone.")
    print(f"  Figures: {OUTPUT_DIR}")
    print(f"  Cache:   {CACHE_BASE}")


if __name__ == '__main__':
    main()
