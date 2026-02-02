#!/usr/bin/env python3
"""
Harvard-Oxford ROI GLM: Switch vs Cluster.
- Atlas: Harvard-Oxford (Cortical & Subcortical).
- Visualization: Bar Plots of T-Statistics.
- Models: Locked to Current Word vs. Locked to Previous Word.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
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
FIG_OUT_DIR = FIGS_DIR / "fmri_ho_bars"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_DIR / "derivatives" / "timeseries"

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HIGH_PASS_HZ = 0.01

# === 1. DATA LOADING ===
def get_ho_data(atlas_type, subject, session, task="svf"):
    """
    Loads HO data. atlas_type should be 'sub' or 'cort'.
    Returns (data_matrix, labels).
    """
    cache_subdir = CACHE_DIR / 'nilearn_cache'
    cache_subdir.mkdir(parents=True, exist_ok=True)
    
    atlas_name = f"HarvardOxford_{atlas_type}"
    if atlas_type == 'sub':
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    else:
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        
    labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    
    masker = NiftiLabelsMasker(
        atlas['maps'], labels=labels, 
        standardize='zscore_sample', high_pass=HIGH_PASS_HZ, t_r=TR, 
        verbose=0, memory=str(cache_subdir)
    )
    
    cache_file = cache_subdir / f"{subject}_{session}_task-{task}_atlas-{atlas_name}_desc-clean_timeseries.npz"
    
    if cache_file.exists():
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            parcel_dict = loaded['parcel_data'].item()
            data_matrix = np.array([parcel_dict[l] for l in labels]).T
            return data_matrix, labels
        except: pass

    bold_path = DERIVATIVES_DIR / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    if not bold_path.exists(): raise FileNotFoundError(f"No BOLD found for {subject} {session}")
    
    confounds, sample_mask = load_confounds_strategy(
        str(bold_path), denoise_strategy="simple", motion="basic", wm_csf="basic", global_signal="basic"
    )
    
    data_matrix = masker.fit_transform(bold_path, confounds=confounds, sample_mask=sample_mask)
    
    parcel_data = {label: d for label, d in zip(labels, data_matrix.T)}
    np.savez_compressed(cache_file, parcel_data=parcel_data)
    
    return data_matrix, labels

# === 2. EVENT GENERATION ===
def get_glm_events(subject, session):
    # Same logic as previous scripts
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_transcription.csv"
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*.csv"))
        csv_path = candidates[0] if candidates else None
        if not csv_path: raise FileNotFoundError

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)
    df["prev_start"] = df["start"].shift(1)
    df["prev_switch"] = df["switch_flag"].shift(1)
    df["prev_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["prev_switch"] == 1
    prev_was_next = df["prev_word"] == "next"
    df = df[~(is_switch & (prev_was_switch | prev_was_next))].copy()
    
    df["condition"] = df["switch_flag"].map({1: "Switch", 0: "Cluster"})
    
    events_curr = pd.DataFrame({
        "onset": df["start"] - SCANNER_START_OFFSET,
        "duration": df["end"] - df["start"],
        "trial_type": df["condition"]
    })
    
    events_prev = pd.DataFrame({
        "onset": df["prev_start"] - SCANNER_START_OFFSET,
        "duration": 0.5,
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

def plot_bars(t_stats, labels, subject, session, atlas_display, model_name):
    print(f"  Plotting Bars: {atlas_display} - {model_name}")
    df = pd.DataFrame({'ROI': labels, 'T_stat': t_stats})
    df = df[df['ROI'] != 'Background'] # Remove background
    df = df.reindex(df.T_stat.abs().sort_values(ascending=False).index) # Sort by magnitude
    
    # Dynamic height
    h = max(6, len(df) * 0.25)
    plt.figure(figsize=(10, h))
    
    colors = ['#D62828' if x > 0 else '#003049' for x in df['T_stat']]
    sns.barplot(data=df, x='T_stat', y='ROI', palette=colors, orient='h')
    
    plt.axvline(1.96, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(-1.96, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.8)
    
    plt.title(f"{subject} {session} ({atlas_display}): {model_name}", fontsize=14)
    plt.xlabel("T-Statistic (Switch > Cluster)")
    
    safe_name = f"{subject}_{session}_{atlas_display.replace(' ', '')}_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(FIG_OUT_DIR / safe_name, dpi=150, bbox_inches='tight')
    plt.close()

def run_analysis(subject, session):
    print(f"\n=== HO ROI GLM: {subject} {session} ===")
    try:
        ev_curr, ev_prev = get_glm_events(subject, session)
        
        # Subcortical
        Y_sub, lab_sub = get_ho_data('sub', subject, session)
        plot_bars(run_glm_contrast(Y_sub, ev_curr), lab_sub, subject, session, "HO_Sub", "Locked to Current Word")
        plot_bars(run_glm_contrast(Y_sub, ev_prev), lab_sub, subject, session, "HO_Sub", "Locked to Previous Word")
        
        # Cortical
        Y_cort, lab_cort = get_ho_data('cort', subject, session)
        plot_bars(run_glm_contrast(Y_cort, ev_curr), lab_cort, subject, session, "HO_Cort", "Locked to Current Word")
        plot_bars(run_glm_contrast(Y_cort, ev_prev), lab_cort, subject, session, "HO_Cort", "Locked to Previous Word")
        
    except Exception as e:
        print(f"  Failed: {e}")

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