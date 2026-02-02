#!/usr/bin/env python3
"""
Sanity Check: SomMotB Response (Corrected).
Reuses robust caching and fixes label alignment.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
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
SANITY_FIGS_DIR = FIGS_DIR / "fmri_sc"
SANITY_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Point to the SAME cache as the main analysis
CACHE_DIR = DATA_DIR / "derivatives" / "timeseries" / "nilearn_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HIGH_PASS_HZ = 0.01
T_MIN = -15.0   
T_MAX = 15.0    

ROI_GROUPS = {
    "SomMotB": ["SomMotB"] # Auditory/Speech Motor Network 
}

# === 1. ROBUST DATA LOADER (Copied from svf_psth_analysis.py) ===
def get_parcel_data(subject, session, task="svf"):
    """
    Loads Schaefer400 data using the robust, alignment-fixed logic.
    """
    atlas_label = "Schaefer400_17Nets"
    cache_file = CACHE_DIR / f"{subject}_{session}_task-{task}_atlas-{atlas_label}_desc-clean_timeseries.npz"
    
    # 1. Load Atlas & Filter Background
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    roi_labels = [l for l in all_labels if l != 'Background'] # <--- THE CRITICAL FIX
    
    masker = NiftiLabelsMasker(
        atlas['maps'], labels=all_labels, 
        standardize='zscore_sample', high_pass=HIGH_PASS_HZ, t_r=TR, verbose=0
    )
    
    # 2. Check Cache
    if cache_file.exists():
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            parcel_dict = loaded['parcel_data'].item()
            
            # Key Normalization (Bytes -> String)
            def normalize(k): return str(k).strip("b'").strip("'")
            lookup = {normalize(k): v for k, v in parcel_dict.items()}
            
            # Reconstruct Dict
            clean_data = {}
            for label in roi_labels:
                norm = normalize(label)
                if norm in lookup:
                    clean_data[label] = lookup[norm]
                else:
                    raise KeyError(f"Missing label: {label}")
            
            print(f"  [Cache] Loaded: {cache_file.name}")
            return clean_data
        except Exception as e:
            print(f"  [Cache] Invalid ({e}). Re-computing...")
            cache_file.unlink()

    # 3. Compute Fresh (if needed)
    print(f"  [Compute] Extracting signal...")
    bold_path = DERIVATIVES_DIR / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    
    if not bold_path.exists(): raise FileNotFoundError(f"No BOLD: {bold_path}")

    confounds, sample_mask = load_confounds_strategy(
        str(bold_path), denoise_strategy="simple", motion="basic", wm_csf="basic", global_signal="basic"
    )
    
    data = masker.fit_transform(str(bold_path), confounds=confounds, sample_mask=sample_mask)
    
    parcel_data = {label: d for label, d in zip(roi_labels, data.T)}
    np.savez_compressed(cache_file, parcel_data=parcel_data)
    
    return parcel_data

# === 2. EVENT LOADING ===
def get_events_strict(subject, session):
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    if not csv_path.exists(): return None

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)

    df["preceding_start"] = df["start"].shift(1)
    df["preceding_switch_flag"] = df["switch_flag"].shift(1)
    df["preceding_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    is_switch = df["switch_flag"] == 1
    prev_bad = (df["preceding_switch_flag"] == 1) | (df["preceding_word"] == "next")
    df = df[~(is_switch & prev_bad)].copy()

    df["trial_type"] = df["switch_flag"].map({1: "Switch", 0: "Cluster"})
    df["onset"] = df["start"] - SCANNER_START_OFFSET
    df["prev_onset"] = df["preceding_start"] - SCANNER_START_OFFSET
    df["irt"] = df["start"] - df["preceding_start"]
    
    return df[df["onset"] >= 0]

def get_events_all(subject, session):
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    if not csv_path.exists(): return None

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    df["preceding_start"] = df["start"].shift(1)
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    df["trial_type"] = "All Words"
    df["onset"] = df["start"] - SCANNER_START_OFFSET
    df["prev_onset"] = df["preceding_start"] - SCANNER_START_OFFSET
    
    return df[df["onset"] >= 0]

# === 3. AGGREGATION & PLOTTING ===
def aggregate_roi_signals(parcel_data):
    agg_signals = {}
    
    # Normalize keys just in case
    def normalize(k): return str(k).strip("b'").strip("'")
    clean_data = {normalize(k): v for k, v in parcel_data.items()}
    
    n_scans = next(iter(clean_data.values())).shape[0]

    for group_name, keywords in ROI_GROUPS.items():
        relevant_series = []
        for label, series in clean_data.items():
            if any(k in label for k in keywords):
                relevant_series.append(series)
        
        if relevant_series:
            agg_signals[group_name] = np.mean(np.vstack(relevant_series), axis=0)
        else:
            agg_signals[group_name] = np.zeros(n_scans)
            
    return pd.DataFrame(agg_signals)

def compute_trigger_average(agg_signals_df, events_df, condition, onset_col):
    roi_name = "SomMotB"
    if roi_name not in agg_signals_df.columns: return [], []
    
    signal = agg_signals_df[roi_name].values
    n_scans = len(signal)
    
    n_pre = int(np.ceil(abs(T_MIN) / TR))
    n_post = int(np.ceil(T_MAX / TR))
    
    onsets = events_df[events_df["trial_type"] == condition][onset_col].values
    onsets = onsets[~np.isnan(onsets)]
    if len(onsets) == 0: return [], []
        
    center_indices = np.round(onsets / TR).astype(int)
    window_offsets = np.arange(-n_pre, n_post + 1)
    epoch_indices = center_indices[:, None] + window_offsets[None, :]
    
    valid_mask = np.all((epoch_indices >= 0) & (epoch_indices < n_scans), axis=1)
    if not np.any(valid_mask): return [], []
        
    epochs = signal[epoch_indices[valid_mask, :]]
    t_vec = window_offsets * TR
    
    return epochs, t_vec

def plot_dual_sanity_check(agg_signals, events_strict, events_all, subject, session):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # Calc IRT
    sw_df = events_strict[events_strict["trial_type"] == "Switch"]
    avg_irt = sw_df["irt"].mean() if not sw_df.empty else 0

    # PANEL 1: CURRENT
    ax = axes[0]
    # All Words
    e, t = compute_trigger_average(agg_signals, events_all, "All Words", "onset")
    if len(e) > 0:
        m, s = np.mean(e, axis=0), np.std(e, axis=0)/np.sqrt(len(e))
        ax.plot(t, m, color='gray', ls='--', label='All Words', alpha=0.6)
        ax.fill_between(t, m-s, m+s, color='gray', alpha=0.1)
    
    # Switch/Cluster
    for cond, col in [("Switch", "#D62828"), ("Cluster", "#003049")]:
        e, t = compute_trigger_average(agg_signals, events_strict, cond, "onset")
        if len(e) > 0:
            m, s = np.mean(e, axis=0), np.std(e, axis=0)/np.sqrt(len(e))
            ax.plot(t, m, color=col, marker='o', ms=4, lw=2.5, label=cond)
            ax.fill_between(t, m-s, m+s, color=col, alpha=0.15)
            
    if avg_irt > 0: ax.axvline(-avg_irt, c='#2a9d8f', ls=':', lw=2, label='Prev Word')
    ax.axvline(0, c='black', ls='--'); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("Locked to CURRENT Word"); ax.set_xlabel("Time (s)"); ax.set_ylabel("BOLD Z")
    ax.legend(fontsize='small')

    # PANEL 2: PREVIOUS
    ax = axes[1]
    e, t = compute_trigger_average(agg_signals, events_all, "All Words", "prev_onset")
    if len(e) > 0:
        m, s = np.mean(e, axis=0), np.std(e, axis=0)/np.sqrt(len(e))
        ax.plot(t, m, color='gray', ls='--', label='All Words', alpha=0.6)
        ax.fill_between(t, m-s, m+s, color='gray', alpha=0.1)
        
    for cond, col in [("Switch", "#D62828"), ("Cluster", "#003049")]:
        e, t = compute_trigger_average(agg_signals, events_strict, cond, "prev_onset")
        if len(e) > 0:
            m, s = np.mean(e, axis=0), np.std(e, axis=0)/np.sqrt(len(e))
            ax.plot(t, m, color=col, marker='o', ms=4, lw=2.5, label=f"{cond} Context")
            ax.fill_between(t, m-s, m+s, color=col, alpha=0.15)

    if avg_irt > 0: ax.axvline(avg_irt, c='#e76f51', ls=':', lw=2, label='Next Word')
    ax.axvline(0, c='black', ls='--'); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("Locked to PREVIOUS Word"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize='small')

    fig.suptitle(f"{subject} {session}: SomMotB Response Check")
    plt.tight_layout()
    out_path = SANITY_FIGS_DIR / f"{subject}_{session}_sommot_dual_check.png"
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved: {out_path.name}")

def run_analysis(subject, session):
    print(f"\n=== SomMotB Check: {subject} {session} ===")
    try:
        # Load ROI Data
        parcel_data = get_parcel_data(subject, session)
        agg_signals = aggregate_roi_signals(parcel_data)
        
        # Load Events
        ev_s = get_events_strict(subject, session)
        ev_a = get_events_all(subject, session)
        if ev_s is None or ev_a is None: return
        
        plot_dual_sanity_check(agg_signals, ev_s, ev_a, subject, session)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help="Subject ID")
    parser.add_argument("--ses", help="Session ID")
    parser.add_argument("--all", action="store_true", help="Run all")
    args = parser.parse_args()
    
    if args.all:
        for sub_dir in sorted(list(DERIVATIVES_DIR.glob("sub-*"))):
            for ses_dir in sorted(list(sub_dir.glob("ses-*"))):
                run_analysis(sub_dir.name, ses_dir.name)
    elif args.sub and args.ses:
        run_analysis(args.sub, args.ses)