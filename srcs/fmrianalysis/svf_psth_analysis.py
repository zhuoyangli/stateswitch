#!/usr/bin/env python3
"""
PSTH Analysis for Switch vs Cluster events.
Side-by-side plotting: Using verified String Labels for all ROIs.
Features:
  - NO CACHING (Always re-computes signal)
  - PARALLEL PROCESSING (use --n-jobs N)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from joblib import Parallel, delayed  # <--- ADDED FOR PARALLELIZATION

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy

# === CONFIG SETUP ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DATA_DIR, DERIVATIVES_DIR, FIGS_DIR, TR, SUBJECT_IDS
except ImportError:
    print("Error: Could not import 'configs.config'. Ensure your directory structure is correct.")
    sys.exit(1)

# === PATH DEFINITIONS ===
ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
PSTH_FIGS_DIR = FIGS_DIR / "fmri_psth_svf"
PSTH_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# (Removed CACHE_DIR definition)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HIGH_PASS_HZ = 0.01
T_MIN = -15.0   
T_MAX = 15.0    

# === ROI DEFINITIONS ===
ROI_GROUPS = {
    "SomMotB": ["SomMotB"],
    
    # H-O Atlas (Subcortical)
    "Hippocampus": ["Left Hippocampus", "Right Hippocampus"],
    
    "vlPFC": [
        "SalVentAttnA_FrOper", "SalVentAttnB_FrOper", 
        "ContA_FrOper", "ContB_FrOper", "ContA_PFClv", "ContB_PFClv" 
    ],
    
    "ACC / MCC": [
        "SalVentAttnA_Med", "SalVentAttnB_Med",
        "SalVentAttnA_FrMed", "SalVentAttnB_FrMed"
    ],                        
    
    # === VERIFIED STRING LABELS ===
    "left&dorsal mPFC": [
        "17Networks_LH_DefaultA_PFCm_1", "17Networks_LH_DefaultA_PFCm_2",
        "17Networks_LH_DefaultA_PFCm_3", "17Networks_LH_DefaultA_PFCm_4",
        "17Networks_LH_DefaultA_PFCm_5", "17Networks_LH_DefaultA_PFCm_6",
        "17Networks_LH_DefaultB_PFCd_1", "17Networks_LH_DefaultB_PFCd_2",
        "17Networks_LH_DefaultB_PFCd_3", "17Networks_LH_DefaultB_PFCd_4",
        "17Networks_LH_DefaultB_PFCd_5", "17Networks_LH_DefaultB_PFCd_6"
    ],
    
    "rmPFC": [
        "17Networks_RH_DefaultA_PFCm_1", "17Networks_RH_DefaultA_PFCm_2",
        "17Networks_RH_DefaultA_PFCm_3", "17Networks_RH_DefaultA_PFCm_4",
        "17Networks_RH_DefaultA_PFCm_5", "17Networks_RH_DefaultA_PFCm_6"
    ],

    "AG": [
        "17Networks_LH_DefaultA_IPL_1", "17Networks_LH_DefaultA_IPL_2",
        "17Networks_LH_DefaultB_IPL_1", "17Networks_LH_DefaultB_IPL_2",
        "17Networks_LH_DefaultC_IPL_1", 
        "17Networks_RH_DefaultA_IPL_1", "17Networks_RH_DefaultA_IPL_2",
        "17Networks_RH_DefaultC_IPL_1", "17Networks_RH_DefaultC_IPL_2"
    ],

    "PMC": [
        "DefaultA_pCunPCC"
    ],    
    
    "PHC": ["DefaultA_PHC", "DefaultB_PHC", "DefaultC_PHC"],  
    "Insula": ["SalVentAttnA_Ins", "SalVentAttnB_Ins"]
}

# === 1. DATA EXTRACTION (NO CACHE) ===
def get_parcel_data(atlas_label, subject, session, task, data_dir, tr, high_pass=0.01):
    """
    Extracts signal from atlas parcels.
    REMOVED: Cache logic. This now re-computes every time.
    """
    if atlas_label == 'Schaefer400_17Nets':
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    elif atlas_label == 'HarvardOxford_sub':
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    else:
        raise ValueError(f"Unknown atlas label: {atlas_label}")
    
    # Decode labels safely
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    roi_labels = [l for l in all_labels if l != 'Background']
    
    masker = NiftiLabelsMasker(
        labels_img=atlas['maps'], labels=all_labels,
        standardize='zscore_sample', high_pass=high_pass, t_r=tr, verbose=0
    )

    # Directly compute (No check for existing .npz)
    bold_path = DERIVATIVES_DIR / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    
    if not bold_path.exists():
        bold_path = data_dir / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
        if not bold_path.exists():
            raise FileNotFoundError(f"BOLD file not found: {bold_path}")

    # print(f"  [Compute] Extracting {atlas_label} for {subject} {session}...") # Optional: uncomment if you want logs
    
    confounds, sample_mask = load_confounds_strategy(str(bold_path), denoise_strategy="simple", motion="basic", global_signal="basic", wm_csf="basic")
    data = masker.fit_transform(bold_path, confounds=confounds, sample_mask=sample_mask)
    
    parcel_data = {label: d for label, d in zip(roi_labels, data.T)}
    
    return parcel_data

# === 2. EVENT LOADING ===
def get_events(subject, session):
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*wordtimestamps*.csv"))
        if candidates: csv_path = candidates[0]
        else: raise FileNotFoundError(f"No CSV for {subject} {session}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)

    df["preceding_start"] = df["start"].shift(1)
    df["preceding_switch_flag"] = df["switch_flag"].shift(1)
    df["preceding_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["preceding_switch_flag"] == 1
    prev_was_next = df["preceding_word"] == "next"
    
    df = df[~(is_switch & (prev_was_switch | prev_was_next))].copy()
    df["trial_type"] = df["switch_flag"].map({1: "Switch", 0: "Cluster"})
    df["onset"] = df["start"] - SCANNER_START_OFFSET
    df["prev_onset"] = df["preceding_start"] - SCANNER_START_OFFSET
    df["irt"] = df["start"] - df["preceding_start"]
    
    return df[df["onset"] >= 0]

# === 3. AGGREGATION & PLOTTING ===
def aggregate_roi_signals(parcel_data: dict):
    agg_signals = {}
    roi_counts = {}
    
    try:
        n_scans = next(iter(parcel_data.values())).shape[0]
    except StopIteration:
        return pd.DataFrame(), {}

    for group_name, identifiers in ROI_GROUPS.items():
        relevant_series = []
        for identifier in identifiers:
            if identifier in parcel_data:
                 relevant_series.append(parcel_data[identifier])
            else:
                for label, series in parcel_data.items():
                    if identifier in label:
                        relevant_series.append(series)
        
        roi_counts[group_name] = len(relevant_series)
        
        if relevant_series:
            agg_signals[group_name] = np.mean(np.vstack(relevant_series), axis=0)
        else:
            agg_signals[group_name] = np.zeros(n_scans)
            
    return pd.DataFrame(agg_signals), roi_counts

def compute_trigger_average(agg_signals_df, events_df, roi_name, condition, onset_col):
    if roi_name not in agg_signals_df.columns: return np.array([]), np.array([])
    signal = agg_signals_df[roi_name].values
    n_scans = len(signal)
    n_pre, n_post = int(np.ceil(abs(T_MIN)/TR)), int(np.ceil(T_MAX/TR))
    
    onsets = events_df[events_df["trial_type"] == condition][onset_col].dropna().values
    if len(onsets) == 0: return np.array([]), np.array([])
    
    center_indices = np.round(onsets / TR).astype(int)
    window_offsets = np.arange(-n_pre, n_post + 1)
    
    epoch_indices = center_indices[:, None] + window_offsets[None, :]
    valid_mask = np.all((epoch_indices >= 0) & (epoch_indices < n_scans), axis=1)
    
    if not np.any(valid_mask): return np.array([]), np.array([])
    
    return signal[epoch_indices[valid_mask]], window_offsets * TR

def _plot_panel_content(ax, agg_signals_df, events_df, roi_name, onset_col, avg_irt, show_legend=False):
    s_epochs, t_vec = compute_trigger_average(agg_signals_df, events_df, roi_name, "Switch", onset_col)
    if len(s_epochs) > 0:
        s_mean = np.mean(s_epochs, axis=0)
        s_sem = np.std(s_epochs, axis=0) / np.sqrt(len(s_epochs))
        
        ax.plot(t_vec, s_mean, color='#D62828', label='Switch', linewidth=2.5, marker='o', markersize=5)
        ax.fill_between(t_vec, s_mean-s_sem, s_mean+s_sem, color='#D62828', alpha=0.15)
        
    c_epochs, t_vec_c = compute_trigger_average(agg_signals_df, events_df, roi_name, "Cluster", onset_col)
    if len(c_epochs) > 0:
        c_mean = np.mean(c_epochs, axis=0)
        c_sem = np.std(c_epochs, axis=0) / np.sqrt(len(c_epochs))
        
        ax.plot(t_vec_c, c_mean, color='#003049', label='Cluster', linewidth=2.5, marker='o', markersize=5)
        ax.fill_between(t_vec_c, c_mean-c_sem, c_mean+c_sem, color='#003049', alpha=0.15)

    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    
    if avg_irt > 0:
        if onset_col == "onset": 
            ax.axvline(-avg_irt, color='#2a9d8f', linestyle=':', linewidth=2, label='Prev Word')
        elif onset_col == "prev_onset":
            ax.axvline(avg_irt, color='#e76f51', linestyle=':', linewidth=2, label='Next Word')

    ax.set_ylim(-0.6, 0.6)
    ax.grid(True, alpha=0.3)
    if show_legend: ax.legend(loc='upper left', fontsize='small', framealpha=0.9)

def plot_session_psth(agg_signals_df, events_df, roi_counts, subject, session, avg_irt=0):
    part1_rois = ["SomMotB", "Hippocampus", "PHC", "ACC / MCC", "Insula"]
    part2_rois = ["left&dorsal mPFC", "rmPFC", "AG", "vlPFC", "PMC"]
    
    n_rows = len(part1_rois)
    try:
        sub_idx = SUBJECT_IDS.index(subject)
        subj_color = plt.cm.tab10(sub_idx % 10) 
    except (ValueError, NameError): subj_color = 'black'
        
    fig, axes = plt.subplots(n_rows, 5, figsize=(28, 3 * n_rows), sharex=True, sharey=True,
                             gridspec_kw={'width_ratios': [1, 1, 0.2, 1, 1]})
    
    for i in range(n_rows):
        roi1, roi2 = part1_rois[i], part2_rois[i]
        lbl1 = "Hippocampus" if roi1 == "Hippocampus" else f"{roi1} ({roi_counts.get(roi1, 0)})"
        lbl2 = f"{roi2} ({roi_counts.get(roi2, 0)})"

        ax_p1_curr = axes[i, 0]
        _plot_panel_content(ax_p1_curr, agg_signals_df, events_df, roi1, "onset", avg_irt, show_legend=(i==0))
        ax_p1_curr.set_ylabel(lbl1, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
        
        ax_p1_prev = axes[i, 1]
        _plot_panel_content(ax_p1_prev, agg_signals_df, events_df, roi1, "prev_onset", avg_irt)
        
        axes[i, 2].axis('off')

        ax_p2_curr = axes[i, 3]
        _plot_panel_content(ax_p2_curr, agg_signals_df, events_df, roi2, "onset", avg_irt)
        ax_p2_curr.set_ylabel(lbl2, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
        ax_p2_curr.tick_params(axis='y', labelleft=True)

        ax_p2_prev = axes[i, 4]
        _plot_panel_content(ax_p2_prev, agg_signals_df, events_df, roi2, "prev_onset", avg_irt)

        if i == 0:
            ax_p1_curr.set_title("Current Word", fontsize=13, pad=10)
            ax_p1_prev.set_title("Previous Word", fontsize=13, pad=10)
            ax_p2_curr.set_title("Current Word", fontsize=13, pad=10)
            ax_p2_prev.set_title("Previous Word", fontsize=13, pad=10)

    fig.suptitle(f"{subject} {session}: ROI trigger average", fontsize=18, y=0.99, color=subj_color, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.subplots_adjust(wspace=0.1)

    filename = f"{subject}_{session}_psth_trigger_avg.png"
    plt.savefig(PSTH_FIGS_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved plot: {filename}")

# === 4. MAIN PIPELINE ===
def run_psth_analysis(subject, session):
    print(f"Starting: {subject} {session}")
    try:
        # Calls without CACHE_DIR
        data_schaefer = get_parcel_data("Schaefer400_17Nets", subject, session, "svf", DERIVATIVES_DIR, TR)
        data_ho = get_parcel_data("HarvardOxford_sub", subject, session, "svf", DERIVATIVES_DIR, TR)
        
        combined_parcel_data = {**data_schaefer, **data_ho}
        agg_signals_df, roi_counts = aggregate_roi_signals(combined_parcel_data)

        events = get_events(subject, session)
        
        switch_events = events[events["trial_type"] == "Switch"]
        avg_switch_irt = switch_events["irt"].mean() if not switch_events.empty else 0
        
        plot_session_psth(agg_signals_df, events, roi_counts, subject, session, avg_switch_irt)
        print(f"Finished: {subject} {session}")
        
    except FileNotFoundError as e:
        print(f"  [SKIP] {subject} {session}: {e}")
        return
    except Exception as e:
        print(f"  [ERROR] {subject} {session} crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help="Subject ID")
    parser.add_argument("--ses", help="Session ID")
    parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    args = parser.parse_args()
    
    if args.all:
        # 1. Collect all tasks
        tasks = []
        for sub_dir in sorted(list(DERIVATIVES_DIR.glob("sub-*"))):
            for ses_dir in sorted(list(sub_dir.glob("ses-*"))):
                tasks.append((sub_dir.name, ses_dir.name))
        
        print(f"Found {len(tasks)} sessions. Running with {args.n_jobs} parallel jobs...")
        
        # 2. Run in Parallel
        Parallel(n_jobs=args.n_jobs)(
            delayed(run_psth_analysis)(sub, ses) for sub, ses in tasks
        )
            
    elif args.sub and args.ses:
        run_psth_analysis(args.sub, args.ses)
    else:
        print("Error: Provide --sub and --ses OR use --all")
        sys.exit(1)