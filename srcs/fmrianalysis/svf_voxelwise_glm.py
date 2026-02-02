#!/usr/bin/env python3
"""
Voxel-wise GLM: Switch vs Cluster.
Visualization: Surface (Cortical) AND Volumetric (Subcortical) Maps.
Features:
  - Parallel Processing (use --n-jobs N)
  - Fixed Color Scale (vmax=8) for consistent comparison
  - Less Conservative Threshold (Z > 1.96, p < 0.05)
  - Volumetric maps automatically center on the strongest positive activation.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from nilearn import datasets, surface, plotting, image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds_strategy

# === CONFIG SETUP ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DATA_DIR, DERIVATIVES_DIR, FIGS_DIR, TR
except ImportError:
    print("Error: Could not import 'configs.config'.")
    sys.exit(1)

# === PATHS ===
ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
FIG_OUT_DIR = FIGS_DIR / "fmri_voxelwise_maps"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HIGH_PASS_HZ = 0.01
FWHM_SMOOTHING = 6.0
PLOT_THRESHOLD = 1.96  # Z-score corresponding to p < 0.05 (uncorrected)
PLOT_VMAX = 8.0        # Fixed max scale for consistent cross-subject color

# === 1. DATA & CONFOUNDS ===
def get_functional_data(subject, session, task="svf"):
    """
    Locates the preprocessed BOLD file and generates confounds.
    """
    bold_path = DERIVATIVES_DIR / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    mask_path = DERIVATIVES_DIR / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz"
    
    if not bold_path.exists():
        raise FileNotFoundError(f"No BOLD found: {bold_path}")
    
    # print(f"  [Data] Found BOLD: {bold_path.name}") # Commented out to reduce spam in parallel mode
    
    confounds, _ = load_confounds_strategy(
        str(bold_path), denoise_strategy="simple", motion="basic", wm_csf="basic", global_signal="basic"
    )
    
    return str(bold_path), str(mask_path), confounds

# === 2. EVENT GENERATION ===
def get_events(subject, session):
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*wordtimestamps*.csv"))
        csv_path = candidates[0] if candidates else None
        if not csv_path: raise FileNotFoundError(f"No events file found")

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    
    # 1. Calculate Duration
    df["duration"] = df["end"] - df["start"]

    # 2. Shift for previous trial
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)
    df["prev_start"] = df["start"].shift(1)
    df["prev_duration"] = df["duration"].shift(1) 
    df["prev_switch"] = df["switch_flag"].shift(1)
    df["prev_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    # Exclude depletion switches
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["prev_switch"] == 1
    prev_was_next = df["prev_word"] == "next"
    df = df[~(is_switch & (prev_was_switch | prev_was_next))].copy()
    
    df["condition"] = df["switch_flag"].map({1: "Switch", 0: "Cluster"})
    
    events_curr = pd.DataFrame({
        "onset": df["start"] - SCANNER_START_OFFSET,
        "duration": df["duration"],
        "trial_type": df["condition"]
    })
    
    events_prev = pd.DataFrame({
        "onset": df["prev_start"] - SCANNER_START_OFFSET,
        "duration": df["prev_duration"],
        "trial_type": df["condition"]
    })
    
    return events_curr[events_curr["onset"] >= 0], events_prev[events_prev["onset"] >= 0]

# === 3. PLOTTING FUNCTIONS ===

def plot_surface_map(stat_map_img, subject, session, model_name, threshold=PLOT_THRESHOLD, vmax=PLOT_VMAX):
    """Projects 3D volume onto fsaverage surface (Cortical view)."""
    # print(f"  [Plotting] Surface Map ({model_name})...")
    
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    
    texture_left = surface.vol_to_surf(stat_map_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(stat_map_img, fsaverage.pial_right)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), subplot_kw={'projection': '3d'})
    views = [
        (texture_left, fsaverage.infl_left, fsaverage.sulc_left, 'left', 'lateral'),
        (texture_left, fsaverage.infl_left, fsaverage.sulc_left, 'left', 'medial'),
        (texture_right, fsaverage.infl_right, fsaverage.sulc_right, 'right', 'lateral'),
        (texture_right, fsaverage.infl_right, fsaverage.sulc_right, 'right', 'medial'),
    ]
    
    for ax, (tex, mesh, bg, hemi, view) in zip(axes, views):
        plotting.plot_surf_stat_map(
            mesh, tex, hemi=hemi, bg_map=bg, view=view,
            cmap='seismic', threshold=threshold, vmax=vmax,
            bg_on_data=True, darkness=None, axes=ax, colorbar=True
        )
        ax.set_title(f"{hemi.upper()} {view}", fontsize=16)
        
    fig.suptitle(f"{subject} {session}: {model_name} (Surface)", fontsize=20, y=1.05)
    
    out_name = f"{subject}_{session}_surf_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(FIG_OUT_DIR / out_name, dpi=300, bbox_inches='tight')
    plt.close()

def plot_volumetric_map(stat_map_img, subject, session, model_name, threshold=PLOT_THRESHOLD, vmax=PLOT_VMAX):
    """
    Plots orthogonal slices centered on the HIGHEST POSITIVE (Algebraic) value.
    """
    # print(f"  [Plotting] Volumetric Map ({model_name})...")
    
    out_name = f"{subject}_{session}_vol_{model_name.replace(' ', '_').lower()}.png"
    
    # 1. Find the coordinates of the max POSITIVE value
    data = stat_map_img.get_fdata()
    
    # Only try to find max if data isn't empty
    if np.nanmax(data) > 0: 
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        max_coords = image.coord_transform(max_idx[0], max_idx[1], max_idx[2], stat_map_img.affine)
    else:
        # Fallback if map has no positive values (unlikely but possible)
        max_coords = (0, 0, 0)

    # 2. Plot
    fig = plt.figure(figsize=(12, 6))
    
    plotting.plot_stat_map(
        stat_map_img, 
        bg_img=datasets.load_mni152_template(),
        threshold=threshold,
        vmax=vmax,
        display_mode='ortho', # Orthogonal slices
        cut_coords=max_coords, # Force center on max positive value
        draw_cross=True,      
        cmap='seismic',
        figure=fig,
        title=f"{subject} {session}: {model_name} (Volumetric)"
    )
    
    plt.savefig(FIG_OUT_DIR / out_name, dpi=300, bbox_inches='tight')
    plt.close()
    # print(f"  [Saved] {out_name}")

# === 4. GLM PIPELINE ===
def run_voxelwise_glm(subject, session):
    print(f"Starting: {subject} {session}")
    
    try:
        bold_img, mask_img, confounds = get_functional_data(subject, session)
        ev_curr, ev_prev = get_events(subject, session)
        
        fmri_glm = FirstLevelModel(
            t_r=TR,
            noise_model='ar1',
            standardize=True,
            hrf_model='spm',
            mask_img=mask_img,
            smoothing_fwhm=FWHM_SMOOTHING,
            verbose=0, # Reduce verbosity for parallel
            n_jobs=1   # Inner n_jobs=1 because we parallelize the outer loop
        )
        
        # --- MODEL A: CURRENT WORD ---
        fmri_glm = fmri_glm.fit(bold_img, events=ev_curr, confounds=confounds)
        z_map_curr = fmri_glm.compute_contrast("Switch - Cluster", stat_type='t', output_type='z_score')
        
        plot_surface_map(z_map_curr, subject, session, "Locked to Current Word")
        plot_volumetric_map(z_map_curr, subject, session, "Locked to Current Word")
        
        # --- MODEL B: PREVIOUS WORD ---
        fmri_glm = fmri_glm.fit(bold_img, events=ev_prev, confounds=confounds)
        z_map_prev = fmri_glm.compute_contrast("Switch - Cluster", stat_type='t', output_type='z_score')
        
        plot_surface_map(z_map_prev, subject, session, "Locked to Previous Word")
        plot_volumetric_map(z_map_prev, subject, session, "Locked to Previous Word")
        
        print(f"Finished: {subject} {session}")

    except FileNotFoundError as e:
        print(f"  [SKIP] {subject} {session}: {e}")
    except Exception as e:
        print(f"  [ERROR] {subject} {session}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help="Subject ID")
    parser.add_argument("--ses", help="Session ID")
    parser.add_argument("--all", action="store_true", help="Run for all subjects")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    args = parser.parse_args()
    
    if args.all:
        tasks = []
        for sub_dir in sorted(DERIVATIVES_DIR.glob("sub-*")):
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                tasks.append((sub_dir.name, ses_dir.name))
        
        print(f"Found {len(tasks)} sessions. Running with {args.n_jobs} parallel jobs...")
        
        Parallel(n_jobs=args.n_jobs)(
            delayed(run_voxelwise_glm)(sub, ses) for sub, ses in tasks
        )
            
    elif args.sub and args.ses:
        run_voxelwise_glm(args.sub, args.ses)
    else:
        print("Error: Use --sub/--ses OR --all")