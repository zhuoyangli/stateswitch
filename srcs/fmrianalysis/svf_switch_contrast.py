#!/usr/bin/env python3
"""
First-level GLM analysis for fMRI data using Schaefer parcellation.
Runs a contrast analysis for Switching > Clustering.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from nilearn import datasets, image, plotting, surface
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import make_first_level_design_matrix
import statsmodels.api as sm

# === IMPORT PROJECT CONFIG ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DATA_DIR, DERIVATIVES_DIR, FIGS_DIR, TR
except ImportError:
    print("Error: Could not import 'configs.config'. Ensure your directory structure is correct.")
    sys.exit(1)

# === PATH DEFINITIONS ===
ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
GLM_FIGS_DIR = FIGS_DIR / "fmri_glm"
GLM_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0 
HIGH_PASS_HZ = 0.01

def get_events_dataframe(subject, session):
    """Loads behavioral CSV and formats it for nilearn design matrix."""
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_transcription.csv"
    
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*.csv"))
        if candidates:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(f"No behavioral file found at: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # 1. Ensure chronological order
    df = df.sort_values("start").reset_index(drop=True)
    
    # 2. Clean switch_flag (fill NaNs with 0, ensure int)
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)
    
    # 3. Capture Preceding Info (Look-Back)
    # We do this BEFORE filtering "next" to accurately characterize the transition
    df["preceding_start"] = df["start"].shift(1)
    df["preceding_switch_flag"] = df["switch_flag"].shift(1)
    df["preceding_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    # 4. Filter out "next" from the targets (current words)
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    # 5. Drop the first item of the session (no preceding verbal response)
    df = df.dropna(subset=["preceding_start"])
    
    # 6. Apply Strict Filter for Switches
    # Implementation: A Switch is valid ONLY if the preceding word was a Clustering word (0)
    # AND the preceding word was not "next" (which is not a clustering word).
    
    is_switch = df["switch_flag"] == 1
    
    # Check 1: Preceding item was a switch (Cluster Size = 1) -> Invalid
    prev_was_switch = df["preceding_switch_flag"] == 1
    
    # Check 2: Preceding item was "next" (Category Boundary) -> Invalid
    # (Since "next" is a command, not a clustering word)
    prev_was_next = df["preceding_word"] == "next"
    
    # Identify invalid switches
    invalid_switch_mask = is_switch & (prev_was_switch | prev_was_next)
    
    # Filter dataset
    if invalid_switch_mask.sum() > 0:
        print(f"  Excluding {invalid_switch_mask.sum()} switch events (preceded by switch or 'next')")
        df = df[~invalid_switch_mask].copy()
    
    # 7. Map Conditions
    condition_map = {1: "Switch", 0: "Cluster"}
    df["trial_type"] = df["switch_flag"].map(condition_map)
    
    # 8. Calculate scanner-aligned onset
    df["onset"] = df["preceding_start"] - SCANNER_START_OFFSET
    df["duration"] = df["end"] - df["start"]
    
    # 9. Remove events before scanner start
    df = df[df["onset"] >= 0]
    
    return df[["onset", "duration", "trial_type"]]

def extract_roi_time_series(bold_path, n_rois=400, yeo_networks=17):
    """Extract time series from Schaefer atlas ROIs"""
    print(f"Extracting ROI signals from: {bold_path.name}")
    
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, 
        yeo_networks=yeo_networks,
        resolution_mm=2
    )
    
    masker = NiftiLabelsMasker(
        labels_img=schaefer_atlas['maps'],
        labels=schaefer_atlas['labels'],
        standardize='zscore_sample', 
        high_pass=HIGH_PASS_HZ,
        t_r=TR,
        memory='nilearn_cache',
        verbose=1
    )
    
    clean_strategy = ["motion", "wm_csf", "global_signal"]
    confounds, sample_mask = load_confounds(
        str(bold_path),
        strategy=clean_strategy,
        motion="basic", 
        global_signal="basic"
    )

    time_series = masker.fit_transform(bold_path, confounds=confounds)
    roi_labels = [label for label in schaefer_atlas['labels']]
    
    return time_series, roi_labels, schaefer_atlas

def plot_contrast_on_fsaverage(t_map_img, subject, session, output_path):
    """
    Projects volumetric t-map to fsaverage6 and plots 4 views (L/R x Lat/Med).
    """
    print("Projecting to fsaverage6 surface...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage6')
    
    # Project volume to surface
    texture_left = surface.vol_to_surf(t_map_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(t_map_img, fsaverage.pial_right)
    
    # Setup Figure (1x4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'projection': '3d'})
    
    # Define the 4 views
    views = [
        (texture_left, fsaverage.infl_left, fsaverage.sulc_left, 'left', 'lateral'),
        (texture_left, fsaverage.infl_left, fsaverage.sulc_left, 'left', 'medial'),
        (texture_right, fsaverage.infl_right, fsaverage.sulc_right, 'right', 'lateral'),
        (texture_right, fsaverage.infl_right, fsaverage.sulc_right, 'right', 'medial'),
    ]
    
    # Common plotting parameters
    vmax = np.nanmax(np.abs([texture_left, texture_right])) # Auto-scale
    if vmax < 2: vmax = 3 
    
    for ax, (tex, mesh, bg, hemi, view) in zip(axes, views):
        plotting.plot_surf_stat_map(
            mesh, tex, 
            hemi=hemi, 
            bg_map=bg,
            view=view,
            cmap='cold_hot', 
            threshold=1.96, # p < 0.05 uncorrected
            axes=ax,
            colorbar=False, 
            vmax=vmax,
            bg_on_data=True,
            darkness=0.5
        )
        ax.set_title(f"{hemi.upper()} {view.capitalize()}", fontsize=10)
    
    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap='cold_hot', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label('t-statistic')
    
    fig.suptitle(f"{subject} {session}: Switching > Clustering", fontsize=14, y=1.05)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_path}")

def run_subject_level_glm(subject, session, task_name="svf"):
    """
    Main function to run ROI-based GLM
    """
    print(f"\n=== Processing {subject} {session} ===")
    
    # 1. Locate BOLD file with specific MNI152NLin6Asym_res-2 space
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    
    search_pattern = f"*{task_name}*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(search_pattern))
    
    if not bold_files:
        print(f"No BOLD file found matching '{search_pattern}' in {func_dir}")
        return
    bold_path = bold_files[0]
    
    # 2. Extract Data
    try:
        roi_signals, roi_labels, atlas = extract_roi_time_series(bold_path)
    except Exception as e:
        print(f"Error extracting ROIs: {e}")
        return

    n_scans = roi_signals.shape[0]
    frame_times = np.arange(n_scans) * TR
    
    # 3. Get Events
    try:
        events_df = get_events_dataframe(subject, session)
        print(f"Loaded {len(events_df)} valid events")
    except FileNotFoundError:
        print(f"Skipping {subject} {session}: No annotation file found.")
        return
    except Exception as e:
        print(f"Error loading events: {e}")
        return
    
    # 4. Create Design Matrix
    try:
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events_df,
            hrf_model='spm',
            drift_model=None
        )
    except Exception as e:
        print(f"Error creating design matrix: {e}")
        return
    
    if "Switch" not in design_matrix.columns or "Cluster" not in design_matrix.columns:
        print(f"Missing conditions. Found: {design_matrix.columns}")
        return

    # 5. Fit GLM
    print("Fitting GLM on 400 ROIs...")
    t_values = []
    contrast_vector = np.zeros(design_matrix.shape[1])
    
    try:
        contrast_vector[design_matrix.columns.get_loc("Switch")] = 1
        contrast_vector[design_matrix.columns.get_loc("Cluster")] = -1
    except KeyError:
        return
    
    X = design_matrix.values
    for i in range(roi_signals.shape[1]):
        Y = roi_signals[:, i]
        model = sm.OLS(Y, X).fit()
        t_test = model.t_test(contrast_vector)
        t_values.append(t_test.tvalue.item())
        
    t_values = np.array(t_values)
    
    # 6. Visualization on fsaverage
    masker = NiftiLabelsMasker(labels_img=atlas['maps'])
    masker.fit()
    t_map_img = masker.inverse_transform(t_values.reshape(1, -1))
    
    out_file = GLM_FIGS_DIR / f"{subject}_{session}_contrast_fsaverage.png"
    plot_contrast_on_fsaverage(t_map_img, subject, session, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ROI GLM")
    parser.add_argument("--sub", help="Subject ID (e.g., sub-008)")
    parser.add_argument("--ses", help="Session ID (e.g., ses-01)")
    parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions found in derivatives")
    
    args = parser.parse_args()
    
    if args.all:
        print("Running batch analysis for ALL subjects...")
        sub_dirs = sorted(list(DERIVATIVES_DIR.glob("sub-*")))
        for sub_dir in sub_dirs:
            sub_id = sub_dir.name
            ses_dirs = sorted(list(sub_dir.glob("ses-*")))
            for ses_dir in ses_dirs:
                ses_id = ses_dir.name
                run_subject_level_glm(sub_id, ses_id)
    else:
        if not args.sub or not args.ses:
            print("Error: Must provide --sub and --ses OR use --all")
            sys.exit(1)
        run_subject_level_glm(args.sub, args.ses)