#!/usr/bin/env python3
"""
First-level GLM analysis for fMRI data using Schaefer parcellation.
Runs a contrast analysis for Switching > Clustering.
"""

import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import datasets, image, plotting
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import make_first_level_design_matrix
import statsmodels.api as sm

# === SETUP PATHS ===
# Update these paths to match your project structure
current_dir = os.getcwd()
PROJECT_ROOT = Path(current_dir).parent
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
DERIVATIVES_DIR = DATA_DIR / "derivatives/fmriprep" # Assuming standard fmriprep output
FIGS_DIR = PROJECT_ROOT / "figs/fmri_glm"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
TR = 1.0  # Set your Repetition Time here (e.g., 2.0, 0.72, etc.)
SCANNER_START_OFFSET = 12.0 # The behavioral t=12.0s matches fMRI volume 0

def get_events_dataframe(subject, session):
    """
    Loads behavioral CSV and formats it for nilearn design matrix.
    Aligns timestamps by subtracting SCANNER_START_OFFSET.
    """
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_transcription.csv"
    
    if not csv_path.exists():
        # Fallback to verify file naming pattern
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*.csv"))
        if candidates:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(f"No behavioral file found for {subject} {session}")

    df = pd.read_csv(csv_path)
    
    # Filter out "next" commands
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    # Define conditions based on switch_flag
    # 1 = Switch, 0 = Cluster
    condition_map = {1: "Switch", 0: "Cluster"}
    df["trial_type"] = df["switch_flag"].map(condition_map)
    
    # === CRITICAL ALIGNMENT STEP ===
    # Subtract 12s offset so behavioral t=12 becomes scanner t=0
    df["onset"] = df["start"] - SCANNER_START_OFFSET
    
    # Define duration. 
    # Option A: Instantaneous events (duration = 0)
    # Option B: Duration of utterance (end - start)
    # Using Option B gives slightly more power if utterances vary in length
    df["duration"] = df["end"] - df["start"]
    
    # Keep only valid onsets (events that happened after scanner started)
    df = df[df["onset"] >= 0]
    
    return df[["onset", "duration", "trial_type"]]

def extract_roi_time_series(bold_path, n_rois=400, yeo_networks=17):
    """Extract time series from Schaefer atlas ROIs"""
    print(f"Extracting ROI signals from: {bold_path.name}")
    
    # Load Schaefer atlas
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, 
        yeo_networks=yeo_networks,
        resolution_mm=2
    )
    
    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=schaefer_atlas['maps'],
        labels=schaefer_atlas['labels'],
        standardize='zscore_sample', # Z-score time series
        memory='nilearn_cache',
        verbose=1
    )
    
    # Load confounds (clean motion, CSF, etc.)
    # Note: 'scrub' strategy removes high-motion volumes, which complicates GLM design matrices
    # We stick to regressors that keep time series length intact.
    clean_strategy = ["high_pass", "motion", "wm_csf", "global_signal"]
    confounds, sample_mask = load_confounds(
        str(bold_path),
        strategy=clean_strategy,
        motion="basic", 
        global_signal="basic"
    )

    # Extract cleaned time series
    # nilearn handles the regression of confounds during extraction
    time_series = masker.fit_transform(bold_path, confounds=confounds)
    
    # Labels (skip background)
    roi_labels = [label.decode() for label in schaefer_atlas['labels']]
    
    return time_series, roi_labels, schaefer_atlas

def run_subject_level_glm(subject, session, task_name="svf"):
    """
    Main function to run ROI-based GLM
    """
    # 1. Locate BOLD file
    # Assuming BIDS structure: sub-XX/ses-YY/func/sub-XX_ses-YY_task-svf_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    # Adjust wildcard as necessary
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    bold_files = list(func_dir.glob(f"*{task_name}*MNI152*preproc_bold.nii.gz"))
    
    if not bold_files:
        print(f"No BOLD file found for {subject} {session}. Skipping.")
        return
    bold_path = bold_files[0]
    
    # 2. Extract Data
    roi_signals, roi_labels, atlas = extract_roi_time_series(bold_path)
    n_scans = roi_signals.shape[0]
    frame_times = np.arange(n_scans) * TR
    
    # 3. Get Events
    events_df = get_events_dataframe(subject, session)
    print(f"Loaded {len(events_df)} valid events (Start offset: {SCANNER_START_OFFSET}s)")
    
    # 4. Create Design Matrix
    # We disable drift model here because 'load_confounds' usually handles high-pass filtering
    # If you want to include drift in GLM, set drift_model='cosine'
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events_df,
        hrf_model='spm',
        drift_model=None  # Assuming data was high-passed during extraction
    )
    
    # Check if we have both conditions
    if "Switch" not in design_matrix.columns or "Cluster" not in design_matrix.columns:
        print("Error: Design matrix missing 'Switch' or 'Cluster' columns. Check event logs.")
        return

    # 5. Fit GLM (Mass Univariate on ROIs)
    print("Fitting GLM on 400 ROIs...")
    
    t_values = []
    p_values = []
    
    # Contrast: Switch (1) - Cluster (-1)
    contrast_vector = np.zeros(design_matrix.shape[1])
    contrast_vector[design_matrix.columns.get_loc("Switch")] = 1
    contrast_vector[design_matrix.columns.get_loc("Cluster")] = -1
    
    for i in range(roi_signals.shape[1]):
        Y = roi_signals[:, i]
        X = design_matrix.values
        
        # OLS Model
        model = sm.OLS(Y, X).fit()
        
        # Calculate Contrast
        t_test = model.t_test(contrast_vector)
        
        t_val = t_test.tvalue.item()
        p_val = t_test.pvalue.item()
        
        t_values.append(t_val)
        p_values.append(p_val)
        
    t_values = np.array(t_values)
    
    # 6. Visualization
    print("Generating surface map...")
    
    # Map t-values back to the atlas image
    # nilearn.maskers.inverse_transform can map 2D data back to 3D image
    # But for Schaefer, we often want to just assign values to the label map.
    
    # Easy way: Create a Nifti image where voxels have the t-value of their ROI
    masker = NiftiLabelsMasker(labels_img=atlas['maps'])
    masker.fit() # Initialize
    t_map_img = masker.inverse_transform(t_values.reshape(1, -1))
    
    # Plot on surface
    fig = plt.figure(figsize=(12, 5))
    plotting.plot_img_on_surf(
        t_map_img,
        views=['lateral', 'medial'],
        hemispheres=['left', 'right'],
        colorbar=True,
        cmap='cold_hot',
        threshold=1.96, # ~p<0.05 uncorrected
        title=f"{subject} {session}: Switching > Clustering"
    )
    
    out_file = FIGS_DIR / f"{subject}_{session}_contrast_surf.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved figure to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ROI GLM")
    parser.add_argument("--sub", required=True, help="Subject ID (e.g., sub-008)")
    parser.add_argument("--ses", required=True, help="Session ID (e.g., ses-01)")
    args = parser.parse_args()
    
    run_subject_level_glm(args.sub, args.ses)