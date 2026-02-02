#!/usr/bin/env python3
"""
Sanity-check analysis for sVF task fMRI data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds

# === CONFIG SETUP ===
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
SANITY_CHECK_FIGS_DIR = FIGS_DIR / "fmri_sc"
SANITY_CHECK_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HIGH_PASS_HZ = 0.01
TARGET_ROI = "SomMotB" 

# PSTH Parameters
T_MIN = -15.0
T_MAX = 15.0

def get_events(subject, session):
    """Load events with strict 'depletion switch' logic."""
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*.csv"))
        if candidates:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(f"No CSV for {subject} {session}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    
    # Filter "next" from targets
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    df["onset"] = df["start"] - SCANNER_START_OFFSET
    df["trial_type"] = "all_words"
    
    return df[df["onset"] >= 0]

def extract_signals(bold_path):
    """Extract 400 ROI signals (Schaefer Only - Fast)."""
    print(f"Extracting signals from {bold_path.name}...")
    
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    
    # Robust decoding of labels
    labels = []
    for l in atlas['labels']:
        if hasattr(l, 'decode'):
            labels.append(l.decode())
        else:
            labels.append(str(l))
            
    masker = NiftiLabelsMasker(
        labels_img=atlas['maps'], 
        labels=atlas['labels'], 
        standardize='zscore_sample', 
        high_pass=HIGH_PASS_HZ, 
        t_r=TR, 
        verbose=0,
        memory='nilearn_cache'
    )
    
    clean_strategy = ["motion", "wm_csf", "global_signal"]
    confounds, _ = load_confounds(
        str(bold_path), 
        strategy=clean_strategy, 
        motion="basic", 
        global_signal="basic"
    )
    
    signals = masker.fit_transform(bold_path, confounds=confounds)
    return signals, labels


def compute_trigger_average(signals, labels, events_df, roi_query, condition="all_words"):
    """Compute trigger average for a given ROI and condition."""
    roi_indices = [i for i, label in enumerate(labels) if roi_query.lower() in label.lower()]
    
    if not roi_indices:
        print(f"Error: No parcels found matching '{roi_query}'")
        return None, None
    
    print(f"Found {len(roi_indices)} ROIs matching '{roi_query}'")
    
    roi_signal = np.mean(signals[:, roi_indices], axis=1)
    n_scans = len(roi_signal)

    n_pre = int(np.ceil(abs(T_MIN) / TR))
    n_post = int(np.ceil(T_MAX / TR))

    t_vec = np.arange(-n_pre, n_post + 1) * TR
    
    # Convert onsets to TR indices
    onsets = events_df[events_df["trial_type"] == condition]["onset"].values
    center_indices = np.round(onsets / TR).astype(int)
    
    window_offsets = np.arange(-n_pre, n_post + 1)
    
    epoch_indices = center_indices[:, None] + window_offsets[None, :]
    
    valid_mask = (epoch_indices[:, 0] >= 0) & (epoch_indices[:, -1] < n_scans)
    
    if not np.any(valid_mask):
        return np.array([]), t_vec

    valid_indices = epoch_indices[valid_mask]
    epochs = roi_signal[valid_indices]
    
    return epochs, t_vec

def plot_trigger_average(epochs, time_vec, subject, session, roi_query):
    """
    Plots the mean response with SEM shading and saves to file.
    """
    mean_resp = np.mean(epochs, axis=0)
    sem_resp = np.std(epochs, axis=0) / np.sqrt(len(epochs))
    
    plt.figure(figsize=(10, 6))
    plt.style.use('bmh')
    
    # Added markers to emphasize discrete sampling points
    plt.plot(time_vec, mean_resp, color='#2196F3', linewidth=2.5, marker='o', label=f'{roi_query} Mean')
    plt.fill_between(time_vec, mean_resp - sem_resp, mean_resp + sem_resp, color='#2196F3', alpha=0.2)
    
    plt.axvline(0, color='black', linestyle='--', label='Word Onset')
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    
    plt.title(f"{subject} {session}: Average {roi_query} Response (N={len(epochs)} words)", fontsize=14)
    plt.xlabel("Time relative to Word Onset (s)")
    plt.ylabel("BOLD Z-score")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.ylim(-0.25, 0.4)
    plt.yticks(np.arange(-0.2, 0.5, 0.1))
    
    # Sanitize ROI name for filename
    safe_roi_name = roi_query.replace(" ", "_").replace("/", "-")
    out_path = SANITY_CHECK_FIGS_DIR / f"{subject}_{session}_{safe_roi_name}_word_onset.png"
    
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.close()

def run_analysis(subject, session):
    print(f"\n=== Running Sanity Check ({TARGET_ROI}): {subject} {session} ===")
    
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    pattern = "*svf*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(pattern))
    
    if not bold_files: 
        print(f"No BOLD file found in {func_dir}")
        return

    # 1. Load Data
    try:
        signals, labels = extract_signals(bold_files[0])
        events = get_events(subject, session)
        print(f"Loaded {len(events)} word events.")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # 2. Compute Trigger Average
    epochs, time_vec = compute_trigger_average(signals, labels, events, TARGET_ROI)
    
    if epochs is None or len(epochs) == 0:
        print("No valid epochs created.")
        return

    # 3. Plotting
    plot_trigger_average(epochs, time_vec, subject, session, TARGET_ROI)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help="Subject ID")
    parser.add_argument("--ses", help="Session ID")
    parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions")
    
    args = parser.parse_args()
    
    if args.all:
        sub_dirs = sorted(list(DERIVATIVES_DIR.glob("sub-*")))
        for sub_dir in sub_dirs:
            sub_id = sub_dir.name
            ses_dirs = sorted(list(sub_dir.glob("ses-*")))
            for ses_dir in ses_dirs:
                ses_id = ses_dir.name
                run_analysis(sub_id, ses_id)
    else:
        if not args.sub or not args.ses:
            print("Error: Must provide --sub and --ses OR use --all")
            sys.exit(1)
        run_analysis(args.sub, args.ses)