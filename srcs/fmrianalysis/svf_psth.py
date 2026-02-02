#!/usr/bin/env python3
"""
PSTH Analysis for Switch vs Cluster events.
Focuses on hypothesis-driven ROIs with a wide time window (-15s to 15s).
Generates two plots: one locked to the Switch Word, one locked to the Previous Word.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.stats import zscore
import nibabel as nib
from nilearn import datasets, image
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
PSTH_FIGS_DIR = FIGS_DIR / "fmri_psth"
PSTH_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HIGH_PASS_HZ = 0.01

# PSTH Parameters
T_MIN = -15.0   # Expanded window
T_MAX = 15.0    

# === ROI DEFINITIONS (Schaefer + HO) ===
ROI_GROUPS = {
    "Hippocampus": [
        "Left Hippocampus", "Right Hippocampus" # From Harvard-Oxford
    ],
    # "vlPFC": [
    #     "SalVentAttnA_FrOper", "SalVentAttnB_FrOper", # Frontal Operculum (IFG/vlPFC)
    #     "ContA_FrOper", "ContB_FrOper"
    # ],
    "SomMotB": [
        "SomMotB"
    ],
    "ACC / MCC": [
        "SalVentAttnA_Med", "SalVentAttnB_Med",
        "SalVentAttnA_FrMed", "SalVentAttnB_FrMed"
    ], 
    "dlPFC": [
        "ContA_PFCl", "ContB_PFCl", "ContC_PFCl",
        "ContA_PFCld", "ContB_PFCld", "ContC_PFCld",
        "ContA_PFClp", "ContB_PFClp", "ContC_PFClp"
    ],                            
    "mPFC": [
        "DefaultA_PFCm", "DefaultB_PFCm", "DefaultC_PFCm",
        "DefaultA_PFCd", "DefaultB_PFCd", "DefaultC_PFCd",
        "ContA_PFCmp", "ContB_PFCmp", "ContC_PFCmp"
    ],                         
    "PCC / Precuneus": [
        "DefaultA_PCC", "DefaultB_PCC", "DefaultC_PCC",
        "DefaultA_pCun", "DefaultB_pCun", "DefaultC_pCun"
    ],    
    "PHC": [
        "DefaultA_PHC", "DefaultB_PHC", "DefaultC_PHC"
    ],  
    "Insula": [
        "SalVentAttnA_Ins", "SalVentAttnB_Ins"
    ]                       
}

def get_events(subject, session):
    """Load events with strict 'depletion switch' logic."""
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_transcription.csv"
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*.csv"))
        if candidates:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(f"No CSV for {subject} {session}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)

    # Capture preceding context
    df["preceding_start"] = df["start"].shift(1)
    df["preceding_switch_flag"] = df["switch_flag"].shift(1)
    df["preceding_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    # Filter "next" from targets
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    # Strict Filter: Preceding word must be clustering
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["preceding_switch_flag"] == 1
    prev_was_next = df["preceding_word"] == "next"
    
    invalid_switches = is_switch & (prev_was_switch | prev_was_next)
    if invalid_switches.sum() > 0:
        df = df[~invalid_switches].copy()

    df["trial_type"] = df["switch_flag"].map({1: "Switch", 0: "Cluster"})
    
    # Onset 1: Locked to Switch Word (Current)
    df["onset"] = df["start"] - SCANNER_START_OFFSET
    
    # Onset 2: Locked to Previous Word
    df["prev_onset"] = df["preceding_start"] - SCANNER_START_OFFSET
    
    # Keep events where the *current* word is within scanner time
    return df[df["onset"] >= 0]

def extract_signals(bold_path):
    """
    Separated extraction: Runs masker twice (Schaefer & HO) then concatenates results.
    Avoids image resampling issues.
    """
    print(f"Extracting signals from {bold_path.name}...")
    
    # 1. Fetch Atlases
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    schaefer_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in schaefer['labels']]
    ho_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in ho_sub['labels']]
    
    # 2. Load Confounds (Common for both)
    clean_strategy = ["motion", "wm_csf", "global_signal"]
    confounds, _ = load_confounds(
        str(bold_path), 
        strategy=clean_strategy, 
        motion="basic", 
        global_signal="basic"
    )
    
    # 3. Extract Schaefer (Cortex)
    print("  Extracting cortical signals (Schaefer)...")
    masker_schaefer = NiftiLabelsMasker(
        labels_img=schaefer['maps'], 
        labels=schaefer_labels, 
        standardize='zscore_sample', 
        high_pass=HIGH_PASS_HZ, 
        t_r=TR, 
        verbose=0,
        memory='nilearn_cache'
    )
    signals_schaefer = masker_schaefer.fit_transform(bold_path, confounds=confounds)
    
    # 4. Extract Harvard-Oxford (Subcortex)
    print("  Extracting subcortical signals (Harvard-Oxford)...")
    # Note: This extracts ALL HO regions (21 total). We filter for Hippocampus later by name.
    masker_ho = NiftiLabelsMasker(
        labels_img=ho_sub['maps'], 
        labels=ho_labels, 
        standardize='zscore_sample', 
        high_pass=HIGH_PASS_HZ, 
        t_r=TR, 
        verbose=0,
        memory='nilearn_cache'
    )
    signals_ho = masker_ho.fit_transform(bold_path, confounds=confounds)
    
    # 5. Concatenate
    # shape: (n_timepoints, 400 + 21)
    signals_all = np.hstack([signals_schaefer, signals_ho])
    labels_all = schaefer_labels + ho_labels
    
    return signals_all, labels_all

def aggregate_roi_signals(signals, labels):
    """
    Averages parcels belonging to defined ROI_GROUPS and Re-Z-scores.
    """
    n_scans = signals.shape[0]
    agg_signals = {}
    
    for group_name, keywords in ROI_GROUPS.items():
        indices = []
        for i, label in enumerate(labels):
            if any(k in label for k in keywords):
                indices.append(i)
        
        if indices:
            group_ts = np.mean(signals[:, indices], axis=1)
            agg_signals[group_name] = zscore(group_ts)
        else:
            print(f"Warning: No parcels found for group '{group_name}'")
            agg_signals[group_name] = np.zeros(n_scans)
            
    return pd.DataFrame(agg_signals)

def compute_trigger_average(agg_signals_df, events_df, roi_name, condition, onset_col="onset"):
    """
    Computes trigger average using direct TR slicing (no interpolation).
    """
    if roi_name not in agg_signals_df.columns:
        return np.array([]), np.array([])
    
    signal = agg_signals_df[roi_name].values
    n_scans = len(signal)
    
    # Calculate TR offsets
    n_pre = int(np.ceil(abs(T_MIN) / TR))
    n_post = int(np.ceil(T_MAX / TR))
    
    # Get onsets
    onsets = events_df[events_df["trial_type"] == condition][onset_col].values
    onsets = onsets[~np.isnan(onsets)] 
    
    if len(onsets) == 0:
        return np.array([]), np.array([])
        
    # Convert to TR indices
    center_indices = np.round(onsets / TR).astype(int)
    window_offsets = np.arange(-n_pre, n_post + 1)
    
    # Broadcast to create matrix of indices: (n_events, n_window)
    epoch_indices = center_indices[:, None] + window_offsets[None, :]
    
    # Filter out epochs that go out of bounds
    valid_mask = (epoch_indices[:, 0] >= 0) & (epoch_indices[:, -1] < n_scans)
    
    if not np.any(valid_mask):
        return np.array([]), np.array([])
        
    valid_indices = epoch_indices[valid_mask]
    epochs = signal[valid_indices] 
    
    t_vec = window_offsets * TR
    
    return epochs, t_vec

def plot_psth_grid(agg_signals_df, events_df, subject, session, onset_col="onset", title_suffix="Switch-Locked"):
    """
    Helper function to generate the 2x4 grid plot.
    """
    roi_names = list(ROI_GROUPS.keys())
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    
    # Y-axis config
    y_min, y_max = -0.25, 0.4
    y_ticks = np.arange(-0.2, 0.5, 0.1)
    
    for i, roi_name in enumerate(roi_names):
        ax = axes[i]
        
        # 1. Switch
        s_epochs, t_vec = compute_trigger_average(agg_signals_df, events_df, roi_name, "Switch", onset_col)
        if len(s_epochs) > 0:
            s_mean = np.mean(s_epochs, axis=0)
            s_sem = np.std(s_epochs, axis=0) / np.sqrt(len(s_epochs))
            ax.plot(t_vec, s_mean, color='#D62828', label='Switch', linewidth=2.5)
            ax.fill_between(t_vec, s_mean-s_sem, s_mean+s_sem, color='#D62828', alpha=0.15)
            
        # 2. Cluster
        c_epochs, t_vec_c = compute_trigger_average(agg_signals_df, events_df, roi_name, "Cluster", onset_col)
        if len(c_epochs) > 0:
            c_mean = np.mean(c_epochs, axis=0)
            c_sem = np.std(c_epochs, axis=0) / np.sqrt(len(c_epochs))
            ax.plot(t_vec_c, c_mean, color='#003049', label='Cluster', linewidth=2.5)
            ax.fill_between(t_vec_c, c_mean-c_sem, c_mean+c_sem, color='#003049', alpha=0.15)
            
        # Style
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(roi_name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BOLD Z-score")
        
        # Limits
        ax.set_xlim(T_MIN, T_MAX)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3)
        
        if i == 0: 
            ax.legend(loc='upper left', fontsize='small')

    # Remove extra axes if any
    for j in range(len(roi_names), len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(f"{subject} {session}: {title_suffix}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"{subject}_{session}_psth_{title_suffix.replace(' ', '_').lower()}.png"
    out_path = PSTH_FIGS_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path.name}")

def run_psth_analysis(subject, session):
    print(f"\n=== Running PSTH Analysis: {subject} {session} ===")
    
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    pattern = "*svf*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(pattern))
    
    if not bold_files: 
        print(f"No BOLD file found matching {pattern} in {func_dir}")
        return
    
    try:
        signals, labels = extract_signals(bold_files[0])
        agg_signals_df = aggregate_roi_signals(signals, labels)
    except Exception as e:
        print(f"Error extracting signals: {e}")
        return

    try:
        events = get_events(subject, session)
        print(f"Loaded {len(events)} events")
    except FileNotFoundError:
        print(f"Skipping {subject} {session}: No annotation file found.")
        return
    except Exception as e:
        print(f"Error loading events: {e}")
        return
    
    # Plot 1: Locked to Current Word
    plot_psth_grid(agg_signals_df, events, subject, session, "onset", "Locked to Current Word")
    
    # Plot 2: Locked to Previous Word
    plot_psth_grid(agg_signals_df, events, subject, session, "prev_onset", "Locked to Previous Word")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help="Subject ID")
    parser.add_argument("--ses", help="Session ID")
    parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions")
    
    args = parser.parse_args()
    
    if args.all:
        print("Running batch PSTH analysis for ALL subjects...")
        sub_dirs = sorted(list(DERIVATIVES_DIR.glob("sub-*")))
        for sub_dir in sub_dirs:
            sub_id = sub_dir.name
            ses_dirs = sorted(list(sub_dir.glob("ses-*")))
            for ses_dir in ses_dirs:
                ses_id = ses_dir.name
                run_psth_analysis(sub_id, ses_id)
    else:
        if not args.sub or not args.ses:
            print("Error: Must provide --sub and --ses OR use --all")
            sys.exit(1)
        run_psth_analysis(args.sub, args.ses)