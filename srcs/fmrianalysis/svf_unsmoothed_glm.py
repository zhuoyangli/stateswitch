#!/usr/bin/env python3
"""
01_unsmoothed_glm_glover.py
Run voxel-wise GLM without spatial smoothing using GLOVER HRF.
Saves raw Z-maps for both Previous and Current word models.
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
from joblib import Parallel, delayed

from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds_strategy

# === CONFIG SETUP ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DATA_DIR, DERIVATIVES_DIR, TR
except ImportError:
    print("Error: Could not import 'configs.config'.")
    sys.exit(1)

# === PATHS ===
ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
MAPS_OUT_DIR = DERIVATIVES_DIR / "contrast_maps_unsmoothed"
MAPS_OUT_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
FWHM_SMOOTHING = None  # Keep None for distribution analysis
HIGH_PASS_HZ = 0.01    # Frequency for drift model

def get_data_and_confounds(subject, session, task="svf"):
    bold_path = DERIVATIVES_DIR / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    mask_path = DERIVATIVES_DIR / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz"
    
    if not bold_path.exists(): raise FileNotFoundError(f"Missing BOLD: {bold_path}")
    
    # NOTE: No high_pass here. Filtering is handled by the GLM drift model.
    confounds, _ = load_confounds_strategy(
        str(bold_path), denoise_strategy="simple", motion="basic", wm_csf="basic", global_signal="basic"
    )
    return str(bold_path), str(mask_path), confounds

def get_events(subject, session):
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*wordtimestamps*.csv"))
        if candidates: csv_path = candidates[0]
        else: raise FileNotFoundError(f"No events file found")

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    df["duration"] = df["end"] - df["start"]
    
    # Lag logic
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)
    df["prev_start"] = df["start"].shift(1)
    df["prev_duration"] = df["duration"].shift(1) 
    df["prev_switch"] = df["switch_flag"].shift(1)
    df["prev_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    # Filter valid trials
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["prev_switch"] == 1
    prev_was_next = df["prev_word"] == "next"
    df = df[~(is_switch & (prev_was_switch | prev_was_next))].copy()
    
    df["condition"] = df["switch_flag"].map({1: "Switch", 0: "Cluster"})
    
    # 1. Current Word Events
    events_curr = pd.DataFrame({
        "onset": df["start"] - SCANNER_START_OFFSET,
        "duration": df["duration"],
        "trial_type": df["condition"]
    })
    
    # 2. Previous Word Events
    events_prev = pd.DataFrame({
        "onset": df["prev_start"] - SCANNER_START_OFFSET,
        "duration": df["prev_duration"],
        "trial_type": df["condition"]
    })
    
    # Return both as a dictionary
    return {
        "curr": events_curr[events_curr["onset"] >= 0],
        "prev": events_prev[events_prev["onset"] >= 0]
    }

def run_glm_and_save(subject, session):
    print(f"Processing: {subject} {session}...")
    try:
        bold_img, mask_img, confounds = get_data_and_confounds(subject, session)
        events_dict = get_events(subject, session)
        
        # Loop through both models
        for model_name, events in events_dict.items():
            
            # Fit Model with GLOVER HRF
            fmri_glm = FirstLevelModel(
                t_r=TR, 
                noise_model='ar1', 
                standardize=True, 
                hrf_model='glover',
                drift_model='cosine',     # <--- Added drift model
                high_pass=HIGH_PASS_HZ,   # <--- Added high pass filter
                mask_img=mask_img, 
                smoothing_fwhm=FWHM_SMOOTHING, 
                verbose=0, 
                n_jobs=1
            )
            
            fmri_glm = fmri_glm.fit(bold_img, events=events, confounds=confounds)
            
            # Compute Contrast (T-test converted to Z-score)
            z_map = fmri_glm.compute_contrast("Switch - Cluster", stat_type='t', output_type='z_score')
            
            # Save Map
            out_name = f"{subject}_{session}_model-{model_name}_desc-unsmoothed_zmap.nii.gz"
            z_map.to_filename(MAPS_OUT_DIR / out_name)
            
    except Exception as e:
        print(f"  [ERROR] {subject} {session}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    
    tasks = []
    for sub_dir in sorted(DERIVATIVES_DIR.glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            tasks.append((sub_dir.name, ses_dir.name))
            
    Parallel(n_jobs=args.n_jobs)(delayed(run_glm_and_save)(s, ses) for s, ses in tasks)