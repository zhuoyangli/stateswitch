#!/usr/bin/env python3
"""
Schaefer + Subcortical GLM: Switch vs Cluster.
Visualization: Top 10 ROI Bar Plots (Mixed Atlases) ONLY.
Mode: Instant Events (Delta Functions).
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.contrasts import compute_contrast
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
FIG_OUT_DIR = FIGS_DIR / "fmri_combined_barplots"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HIGH_PASS_HZ = 0.01

# === 1. DATA LOADING ===
def get_atlas_data(subject, session, atlas_name, task="svf"):
    """
    Generic function to load/compute parcel data for any atlas.
    """
    cache_dir = DATA_DIR / "derivatives" / "timeseries" / "nilearn_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{subject}_{session}_task-{task}_atlas-{atlas_name}_desc-clean_timeseries.npz"
    
    # --- ATLAS SELECTION ---
    if atlas_name == "Schaefer400_17Nets":
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    elif atlas_name == "HarvardOxford_sub":
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    else:
        raise ValueError(f"Unknown atlas: {atlas_name}")
        
    # Standardize labels
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    roi_labels = [l for l in all_labels if l != 'Background']
    
    # Initialize Masker (We need it strictly for transformation logic here)
    masker = NiftiLabelsMasker(
        atlas['maps'], labels=all_labels, 
        standardize='zscore_sample', high_pass=HIGH_PASS_HZ, t_r=TR, 
        verbose=0
    )
    masker.fit() 

    # --- PATH A: CACHE ---
    if cache_file.exists():
        print(f"  [{atlas_name}] Cache found.")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            parcel_dict = loaded['parcel_data'].item()
            # Reconstruct matrix
            data_matrix = np.array([parcel_dict[l] for l in roi_labels]).T
            return data_matrix, roi_labels
        except Exception as e:
            print(f"  [{atlas_name}] Cache corrupted ({e}). Re-computing...")

    # --- PATH B: COMPUTE ---
    print(f"  [{atlas_name}] Computing signal...")
    bold_path = DERIVATIVES_DIR / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    
    if not bold_path.exists(): 
        raise FileNotFoundError(f"No BOLD found: {bold_path}")
    
    confounds, sample_mask = load_confounds_strategy(
        str(bold_path), denoise_strategy="simple", motion="basic", wm_csf="basic", global_signal="basic"
    )
    
    data_matrix = masker.transform(str(bold_path), confounds=confounds, sample_mask=sample_mask)
    
    # Save
    parcel_data = {label: d for label, d in zip(roi_labels, data_matrix.T)}
    np.savez_compressed(cache_file, parcel_data=parcel_data)
    
    return data_matrix, roi_labels

# === 2. EVENT GENERATION ===
def get_glm_events(subject, session):
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*wordtimestamps*.csv"))
        csv_path = candidates[0] if candidates else None
        if not csv_path: raise FileNotFoundError(f"No events file found")

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)
    df["prev_start"] = df["start"].shift(1)
    df["prev_switch"] = df["switch_flag"].shift(1)
    df["prev_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    # Exclude depletion switches
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["prev_switch"] == 1
    prev_was_next = df["prev_word"] == "next"
    df = df[~(is_switch & (prev_was_switch | prev_was_next))].copy()
    
    df["condition"] = df["switch_flag"].map({1: "Switch", 0: "Cluster"})
    
    # === CHANGED: Duration is now 0 (Delta Function) ===
    events_curr = pd.DataFrame({
        "onset": df["start"] - SCANNER_START_OFFSET,
        "duration": 0,  # <--- Instant event
        "trial_type": df["condition"]
    })
    
    events_prev = pd.DataFrame({
        "onset": df["prev_start"] - SCANNER_START_OFFSET,
        "duration": 0,  # <--- Instant event
        "trial_type": df["condition"]
    })
    
    return events_curr[events_curr["onset"] >= 0], events_prev[events_prev["onset"] >= 0]

# === 3. GLM & PLOTTING ===
def run_glm_contrast(Y, events):
    n_scans = Y.shape[0]
    frame_times = np.arange(n_scans) * TR
    design_matrix = make_first_level_design_matrix(
        frame_times, events, hrf_model='spm', drift_model='cosine', high_pass=HIGH_PASS_HZ
    )
    
    if "Switch" not in design_matrix.columns: return np.zeros(Y.shape[1])

    labels, estimates = run_glm(Y, design_matrix.values, noise_model='ar1')
    
    con_vec = np.zeros(design_matrix.shape[1])
    con_vec[design_matrix.columns.get_loc("Switch")] = 1
    con_vec[design_matrix.columns.get_loc("Cluster")] = -1
    
    contrast = compute_contrast(labels, estimates, con_vec, stat_type='t')
    return contrast.stat()

def plot_top_rois(t_stats, labels_list, subject, session, model_name, top_n=10):
    """Plots top N ROIs from the combined list."""
    print(f"  Plotting Top {top_n} ROIs: {model_name}")
    
    # Create DataFrame
    df_stats = pd.DataFrame({'ROI': labels_list, 't_stat': t_stats})
    
    # Sort and take top N
    top_rois = df_stats.sort_values(by='t_stat', ascending=False).head(top_n).iloc[::-1]
    
    plt.figure(figsize=(12, 7))
    bars = plt.barh(top_rois['ROI'], top_rois['t_stat'], color='#2a9d8f')
    
    # Color H-O regions differently
    for bar, label in zip(bars, top_rois['ROI']):
        if "17Networks" not in label: # Simple check for H-O vs Schaefer
            bar.set_color('#e76f51') # Make Subcortical Orange
            
    plt.xlabel('T-statistic (Switch > Cluster)')
    plt.title(f"{subject} {session}: Top {top_n} ROIs (Instant Events)\n{model_name}")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    out_name = f"{subject}_{session}_top{top_n}_combined_delta_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(FIG_OUT_DIR / out_name, dpi=150)
    plt.close()

def run_analysis(subject, session):
    print(f"\n=== Combined GLM (Delta): {subject} {session} ===")
    try:
        # 1. Load Data (Cortical & Subcortical)
        Y_schaefer, labels_schaefer = get_atlas_data(subject, session, "Schaefer400_17Nets")
        Y_ho, labels_ho = get_atlas_data(subject, session, "HarvardOxford_sub")
        
        # 2. Combine Data
        Y_combined = np.hstack([Y_schaefer, Y_ho])
        labels_combined = labels_schaefer + labels_ho
        
        # 3. Load Events
        ev_curr, ev_prev = get_glm_events(subject, session)
        
        # 4. Analysis A: Current Word
        t_stats_curr = run_glm_contrast(Y_combined, ev_curr)
        plot_top_rois(t_stats_curr, labels_combined, subject, session, "Locked to Current Word")
        
        # 5. Analysis B: Previous Word
        t_stats_prev = run_glm_contrast(Y_combined, ev_prev)
        plot_top_rois(t_stats_prev, labels_combined, subject, session, "Locked to Previous Word")
        
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help="Subject ID")
    parser.add_argument("--ses", help="Session ID")
    parser.add_argument("--all", action="store_true", help="Run for all subjects")
    args = parser.parse_args()
    
    if args.all:
        for sub_dir in sorted(DERIVATIVES_DIR.glob("sub-*")):
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                run_analysis(sub_dir.name, ses_dir.name)
    elif args.sub and args.ses:
        run_analysis(args.sub, args.ses)
    else:
        print("Error: Use --sub/--ses OR --all")