#!/usr/bin/env python3
"""
First-level analysis for SVF fMRI data using Schaefer parcellation.
Contrast: Switch (boundary) vs Cluster (non-boundary)

Approach based on Su et al. (2025) Nature Communications:
- Events locked to PRECEDING WORD OFFSET for both Switch and Cluster words
- Boundary period: 6-second window (shifted +4.5s for HRF)
- Non-boundary period: 6-second window (shifted +4.5s for HRF)
- Also extracts PMC and hippocampus time courses locked to events (18 TRs: -2 to +15)
- Group analysis plots mean time courses with SEM shading

ALIGNED WITH PSTH SCRIPT for identical time series extraction.
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
from nilearn.interfaces.fmriprep import load_confounds_strategy
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing
import pickle

# === IMPORT PROJECT CONFIG ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DATA_DIR, DERIVATIVES_DIR, FIGS_DIR, TR
except ImportError:
    print("Error: Could not import 'configs.config'.")
    sys.exit(1)

# === PATH DEFINITIONS ===
ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
GLM_FIGS_DIR = FIGS_DIR / "fmri_glm_svf"
GLM_FIGS_DIR.mkdir(parents=True, exist_ok=True)
GROUP_OUTPUT_DIR = GLM_FIGS_DIR / "group"
GROUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0 
HIGH_PASS_HZ = 0.01
HRF_DELAY = 4.5  # seconds
WINDOW_DURATION = 6.0  # seconds

# Time course extraction parameters (following Su et al.)
TRS_BEFORE = 2  # TRs before event
TRS_AFTER = 15  # TRs after event
TOTAL_TRS = TRS_BEFORE + TRS_AFTER + 1  # 18 TRs total

# Y-axis limits for plots
Y_LIM_SINGLE = (-0.6, 0.6)  # For single-session plots
Y_LIM_GROUP = (-0.15, 0.2)  # For group plots

# Pre-fetch atlases for surface plotting only
print("Pre-fetching surface atlas...")
FSAVERAGE = datasets.fetch_surf_fsaverage('fsaverage6')
print("Surface atlas loaded.")

# === ROI DEFINITIONS (DIRECTLY FROM PSTH SCRIPT) ===
ROI_GROUPS = {
    "PMC": ["DefaultA_pCunPCC"],
    "Hippocampus": ["Left Hippocampus", "Right Hippocampus"],
}


# === DATA EXTRACTION (IDENTICAL TO PSTH SCRIPT) ===
def get_parcel_data(atlas_label, subject, session, task, data_dir, tr, high_pass=0.01):
    """
    Extracts signal from atlas parcels.
    IDENTICAL TO PSTH SCRIPT.
    """
    if atlas_label == 'Schaefer400_17Nets':
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    elif atlas_label == 'HarvardOxford_sub':
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    else:
        raise ValueError(f"Unknown atlas label: {atlas_label}")
    
    # Decode labels safely (IDENTICAL TO PSTH)
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    roi_labels = [l for l in all_labels if l != 'Background']
    
    # NO CACHING - identical to PSTH script
    masker = NiftiLabelsMasker(
        labels_img=atlas['maps'], labels=all_labels,
        standardize='zscore_sample', high_pass=high_pass, t_r=tr, verbose=0
    )

    bold_path = DERIVATIVES_DIR / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    
    if not bold_path.exists():
        bold_path = data_dir / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
        if not bold_path.exists():
            raise FileNotFoundError(f"BOLD file not found: {bold_path}")
    
    confounds, sample_mask = load_confounds_strategy(str(bold_path), denoise_strategy="simple", motion="basic", global_signal="basic", wm_csf="basic")
    data = masker.fit_transform(bold_path, confounds=confounds, sample_mask=sample_mask)
    
    # Return dict keyed by roi_labels (excludes Background) - IDENTICAL TO PSTH
    parcel_data = {label: d for label, d in zip(roi_labels, data.T)}
    
    return parcel_data


def aggregate_roi_signals(parcel_data: dict, roi_groups: dict):
    """
    Aggregate parcel signals into ROI groups.
    IDENTICAL TO PSTH SCRIPT.
    """
    agg_signals = {}
    roi_counts = {}
    
    try:
        n_scans = next(iter(parcel_data.values())).shape[0]
    except StopIteration:
        return pd.DataFrame(), {}

    for group_name, identifiers in roi_groups.items():
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


# === EVENT LOADING (MODIFIED FOR PREVIOUS WORD OFFSET) ===
def get_events(subject, session):
    """
    Load events with PREVIOUS WORD OFFSET timing.
    Based on PSTH script structure but uses 'end' time of preceding word.
    """
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*wordtimestamps*.csv"))
        if candidates: 
            csv_path = candidates[0]
        else: 
            raise FileNotFoundError(f"No CSV for {subject} {session}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)

    # Compute preceding info BEFORE filtering (matches PSTH script)
    df["preceding_start"] = df["start"].shift(1)
    df["preceding_end"] = df["end"].shift(1)  # <-- ADDED: preceding word offset
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
    df["prev_offset"] = df["preceding_end"] - SCANNER_START_OFFSET  # <-- ADDED
    df["irt"] = df["start"] - df["preceding_start"]
    
    return df[df["onset"] >= 0]


def get_switch_and_cluster_events(subject, session):
    """
    Wrapper that returns events using PREVIOUS WORD OFFSET timing.
    """
    df = get_events(subject, session)
    
    print(f"  [{subject} {session}] Loaded {len(df)} events after filtering")
    
    n_switches = (df["trial_type"] == "Switch").sum()
    n_clusters = (df["trial_type"] == "Cluster").sum()
    print(f"  [{subject} {session}] Events: {n_switches} switches, {n_clusters} clusters")
    
    switch_windows = []
    cluster_windows = []
    switch_timepoints = []
    cluster_timepoints = []
    
    for _, row in df.iterrows():
        # Use PREVIOUS WORD OFFSET (prev_offset) instead of prev_onset
        event_time = row["prev_offset"]
        
        if pd.isna(event_time):
            continue
        
        window_start = event_time + HRF_DELAY
        window_end = window_start + WINDOW_DURATION
        
        if row["trial_type"] == "Switch":
            if window_start >= 0:
                switch_windows.append((window_start, window_end))
            if event_time >= TRS_BEFORE * TR:
                switch_timepoints.append(event_time)
        else:
            if window_start >= 0:
                cluster_windows.append((window_start, window_end))
            if event_time >= TRS_BEFORE * TR:
                cluster_timepoints.append(event_time)
    
    print(f"  [{subject} {session}] Switch windows: {len(switch_windows)}, timepoints: {len(switch_timepoints)}")
    print(f"  [{subject} {session}] Cluster windows: {len(cluster_windows)}, timepoints: {len(cluster_timepoints)}")
    
    return switch_windows, cluster_windows, switch_timepoints, cluster_timepoints


# === TIME COURSE EXTRACTION (ALIGNED WITH PSTH SCRIPT) ===
def compute_trigger_average(signal, event_timepoints, trs_before=TRS_BEFORE, trs_after=TRS_AFTER, tr=TR):
    """
    Extract event-locked time courses.
    ALIGNED WITH PSTH SCRIPT's compute_trigger_average logic.
    """
    n_scans = len(signal)
    n_pre = trs_before
    n_post = trs_after
    
    if len(event_timepoints) == 0:
        return np.array([]), np.array([])
    
    onsets = np.array(event_timepoints)
    center_indices = np.round(onsets / tr).astype(int)
    window_offsets = np.arange(-n_pre, n_post + 1)
    
    epoch_indices = center_indices[:, None] + window_offsets[None, :]
    valid_mask = np.all((epoch_indices >= 0) & (epoch_indices < n_scans), axis=1)
    
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    epochs = signal[epoch_indices[valid_mask]]
    time_vec = window_offsets * tr
    
    return epochs, time_vec


def get_mean_activation_in_windows(time_series, windows, tr=TR):
    """Extract mean activation within specified time windows."""
    n_timepoints = len(time_series)
    
    activations = []
    for start_time, end_time in windows:
        start_tr = int(np.floor(start_time / tr))
        end_tr = int(np.ceil(end_time / tr))
        start_tr = max(0, start_tr)
        end_tr = min(n_timepoints, end_tr)
        
        if start_tr < end_tr:
            window_activation = time_series[start_tr:end_tr].mean()
            activations.append(window_activation)
    
    return np.array(activations)


def plot_roi_timecourses(pmc_switch_epochs, pmc_cluster_epochs, 
                          hipp_switch_epochs, hipp_cluster_epochs,
                          time_vec, subject, session, output_path):
    """Plot single-session PMC and hippocampus time courses."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # PMC plot
    ax = axes[0]
    if len(pmc_switch_epochs) > 0:
        pmc_switch_mean = np.mean(pmc_switch_epochs, axis=0)
        pmc_switch_sem = np.std(pmc_switch_epochs, axis=0) / np.sqrt(len(pmc_switch_epochs))
        ax.plot(time_vec, pmc_switch_mean, 'r-', linewidth=2, label='Switch')
        ax.fill_between(time_vec, pmc_switch_mean - pmc_switch_sem, 
                        pmc_switch_mean + pmc_switch_sem, color='red', alpha=0.3)
    
    if len(pmc_cluster_epochs) > 0:
        pmc_cluster_mean = np.mean(pmc_cluster_epochs, axis=0)
        pmc_cluster_sem = np.std(pmc_cluster_epochs, axis=0) / np.sqrt(len(pmc_cluster_epochs))
        ax.plot(time_vec, pmc_cluster_mean, 'gray', linewidth=2, label='Cluster')
        ax.fill_between(time_vec, pmc_cluster_mean - pmc_cluster_sem,
                        pmc_cluster_mean + pmc_cluster_sem, color='gray', alpha=0.3)
    
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.2, color='yellow', label='Analysis window')
    ax.set_xlabel('Time from preceding word offset (s)')
    ax.set_ylabel('Activation (z-scored)')
    ax.set_title('PMC (pCunPCC)')
    ax.legend(loc='upper right')
    ax.set_xlim([time_vec[0], time_vec[-1]])
    ax.set_ylim(Y_LIM_SINGLE)
    
    # Hippocampus plot
    ax = axes[1]
    if len(hipp_switch_epochs) > 0:
        hipp_switch_mean = np.mean(hipp_switch_epochs, axis=0)
        hipp_switch_sem = np.std(hipp_switch_epochs, axis=0) / np.sqrt(len(hipp_switch_epochs))
        ax.plot(time_vec, hipp_switch_mean, 'r-', linewidth=2, label='Switch')
        ax.fill_between(time_vec, hipp_switch_mean - hipp_switch_sem,
                        hipp_switch_mean + hipp_switch_sem, color='red', alpha=0.3)
    
    if len(hipp_cluster_epochs) > 0:
        hipp_cluster_mean = np.mean(hipp_cluster_epochs, axis=0)
        hipp_cluster_sem = np.std(hipp_cluster_epochs, axis=0) / np.sqrt(len(hipp_cluster_epochs))
        ax.plot(time_vec, hipp_cluster_mean, 'gray', linewidth=2, label='Cluster')
        ax.fill_between(time_vec, hipp_cluster_mean - hipp_cluster_sem,
                        hipp_cluster_mean + hipp_cluster_sem, color='gray', alpha=0.3)
    
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.2, color='yellow', label='Analysis window')
    ax.set_xlabel('Time from preceding word offset (s)')
    ax.set_ylabel('Activation (z-scored)')
    ax.set_title('Hippocampus')
    ax.legend(loc='upper right')
    ax.set_xlim([time_vec[0], time_vec[-1]])
    ax.set_ylim(Y_LIM_SINGLE)
    
    fig.suptitle(f'{subject} {session}: Event-locked Time Courses (SVF) - Previous Word Offset', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [{subject} {session}] Saved time course figure to {output_path}")


def plot_contrast_on_fsaverage(t_values, parcel_data_schaefer, subject, session, output_path, contrast_name="Switch > Cluster"):
    """Projects t-values to fsaverage6 and plots 4 views."""
    print(f"  [{subject} {session}] Projecting to fsaverage6 surface...")
    
    # Need to reconstruct the atlas for inverse transform
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    
    masker = NiftiLabelsMasker(labels_img=atlas['maps'], labels=all_labels)
    masker.fit()
    t_map_img = masker.inverse_transform(t_values.reshape(1, -1))
    
    texture_left = surface.vol_to_surf(t_map_img, FSAVERAGE.pial_left)
    texture_right = surface.vol_to_surf(t_map_img, FSAVERAGE.pial_right)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'projection': '3d'})
    
    views = [
        (texture_left, FSAVERAGE.infl_left, FSAVERAGE.sulc_left, 'left', 'lateral'),
        (texture_left, FSAVERAGE.infl_left, FSAVERAGE.sulc_left, 'left', 'medial'),
        (texture_right, FSAVERAGE.infl_right, FSAVERAGE.sulc_right, 'right', 'lateral'),
        (texture_right, FSAVERAGE.infl_right, FSAVERAGE.sulc_right, 'right', 'medial'),
    ]
    
    vmax = np.nanmax(np.abs([texture_left, texture_right]))
    if vmax < 2: vmax = 3 
    
    for ax, (tex, mesh, bg, hemi, view) in zip(axes, views):
        plotting.plot_surf_stat_map(
            mesh, tex, hemi=hemi, bg_map=bg, view=view,
            cmap='cold_hot', threshold=1.96, axes=ax,
            colorbar=False, vmax=vmax, bg_on_data=True, darkness=0.5
        )
        ax.set_title(f"{hemi.upper()} {view.capitalize()}", fontsize=10)
    
    sm = plt.cm.ScalarMappable(cmap='cold_hot', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label('t-statistic')
    
    fig.suptitle(f"{subject} {session}: {contrast_name}", fontsize=14, y=1.05)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [{subject} {session}] Saved figure to {output_path}")


def run_subject_level_analysis(subject, session, task_name="svf"):
    """
    Run single-session analysis and return time courses for group analysis.
    
    ALIGNED WITH PSTH SCRIPT for data extraction.
    """
    print(f"\n=== Processing {subject} {session} (SVF) ===")
    
    # 1. Get events first (to fail fast if no behavioral data)
    try:
        switch_windows, cluster_windows, switch_timepoints, cluster_timepoints = \
            get_switch_and_cluster_events(subject, session)
    except Exception as e:
        print(f"  [{subject} {session}] Error getting events: {e}")
        return None
    
    if len(switch_timepoints) < 2 or len(cluster_timepoints) < 2:
        print(f"  [{subject} {session}] Insufficient events")
        return None
    
    # 2. Extract parcel data IDENTICALLY to PSTH script
    try:
        data_schaefer = get_parcel_data("Schaefer400_17Nets", subject, session, task_name, DERIVATIVES_DIR, TR, HIGH_PASS_HZ)
        data_ho = get_parcel_data("HarvardOxford_sub", subject, session, task_name, DERIVATIVES_DIR, TR, HIGH_PASS_HZ)
        
        combined_parcel_data = {**data_schaefer, **data_ho}
    except Exception as e:
        print(f"  [{subject} {session}] Error extracting parcels: {e}")
        return None
    
    # 3. Aggregate ROI signals IDENTICALLY to PSTH script
    agg_signals_df, roi_counts = aggregate_roi_signals(combined_parcel_data, ROI_GROUPS)
    
    print(f"  [{subject} {session}] ROI counts: {roi_counts}")
    
    # 4. Get PMC and Hippocampus signals
    pmc_signal = agg_signals_df["PMC"].values
    hipp_signal = agg_signals_df["Hippocampus"].values
    
    # 5. Extract event-locked time courses using PSTH-style function
    pmc_switch_epochs, time_vec = compute_trigger_average(pmc_signal, switch_timepoints)
    pmc_cluster_epochs, _ = compute_trigger_average(pmc_signal, cluster_timepoints)
    
    hipp_switch_epochs, _ = compute_trigger_average(hipp_signal, switch_timepoints)
    hipp_cluster_epochs, _ = compute_trigger_average(hipp_signal, cluster_timepoints)
    
    if len(pmc_switch_epochs) == 0 or len(pmc_cluster_epochs) == 0:
        print(f"  [{subject} {session}] No valid epochs extracted")
        return None
    
    # 6. Compute mean time courses for this session
    pmc_switch_tc = np.mean(pmc_switch_epochs, axis=0)
    pmc_cluster_tc = np.mean(pmc_cluster_epochs, axis=0)
    hipp_switch_tc = np.mean(hipp_switch_epochs, axis=0)
    hipp_cluster_tc = np.mean(hipp_cluster_epochs, axis=0)
    
    # 7. Plot single-session time courses
    tc_output_path = GLM_FIGS_DIR / f"{subject}_{session}_svf_roi_timecourses.png"
    plot_roi_timecourses(
        pmc_switch_epochs, pmc_cluster_epochs,
        hipp_switch_epochs, hipp_cluster_epochs,
        time_vec, subject, session, tc_output_path
    )
    
    # 8. Compute whole-brain contrast (for Schaefer parcels only)
    # Get all Schaefer parcel labels in order
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    schaefer_labels = [l for l in all_labels if l != 'Background']
    
    n_rois = len(schaefer_labels)
    t_values = np.zeros(n_rois)
    
    for i, label in enumerate(schaefer_labels):
        if label in data_schaefer:
            signal = data_schaefer[label]
            switch_act = get_mean_activation_in_windows(signal, switch_windows)
            cluster_act = get_mean_activation_in_windows(signal, cluster_windows)
            
            if len(switch_act) > 1 and len(cluster_act) > 1:
                t_stat, _ = stats.ttest_ind(switch_act, cluster_act)
                t_values[i] = t_stat
    
    # 9. Plot whole-brain contrast
    out_file = GLM_FIGS_DIR / f"{subject}_{session}_svf_switch-vs-cluster_fsaverage.png"
    plot_contrast_on_fsaverage(
        t_values, data_schaefer, subject, session, out_file,
        contrast_name="Switch > Cluster"
    )
    
    # Return data for group analysis
    return {
        'subject': subject,
        'session': session,
        'pmc_switch_tc': pmc_switch_tc,
        'pmc_cluster_tc': pmc_cluster_tc,
        'hipp_switch_tc': hipp_switch_tc,
        'hipp_cluster_tc': hipp_cluster_tc,
        'time_vec': time_vec,
        'n_switch_epochs': len(pmc_switch_epochs),
        'n_cluster_epochs': len(pmc_cluster_epochs),
        'n_switch_windows': len(switch_windows),
        'n_cluster_windows': len(cluster_windows),
    }


def plot_group_timecourses(results, output_path, tr=TR):
    """
    Plot group-level time courses with SEM shading (Su et al. Figure 4b style).
    """
    valid_results = [r for r in results if r is not None]
    n_sessions = len(valid_results)
    
    if n_sessions < 2:
        print("Not enough sessions for group analysis")
        return None
    
    print(f"\n=== Group Analysis ({n_sessions} sessions) ===")
    
    # Get time vector from first result
    time_vec = valid_results[0]['time_vec']
    
    # Stack time courses
    pmc_switch_stack = np.array([r['pmc_switch_tc'] for r in valid_results])
    pmc_cluster_stack = np.array([r['pmc_cluster_tc'] for r in valid_results])
    hipp_switch_stack = np.array([r['hipp_switch_tc'] for r in valid_results])
    hipp_cluster_stack = np.array([r['hipp_cluster_tc'] for r in valid_results])
    
    # Compute mean and SEM
    pmc_switch_mean = pmc_switch_stack.mean(axis=0)
    pmc_switch_sem = pmc_switch_stack.std(axis=0) / np.sqrt(n_sessions)
    pmc_cluster_mean = pmc_cluster_stack.mean(axis=0)
    pmc_cluster_sem = pmc_cluster_stack.std(axis=0) / np.sqrt(n_sessions)
    
    hipp_switch_mean = hipp_switch_stack.mean(axis=0)
    hipp_switch_sem = hipp_switch_stack.std(axis=0) / np.sqrt(n_sessions)
    hipp_cluster_mean = hipp_cluster_stack.mean(axis=0)
    hipp_cluster_sem = hipp_cluster_stack.std(axis=0) / np.sqrt(n_sessions)
    
    # Paired t-tests at each time point
    n_timepoints = len(time_vec)
    pmc_pvalues = []
    hipp_pvalues = []
    for t in range(n_timepoints):
        _, p_pmc = stats.ttest_rel(pmc_switch_stack[:, t], pmc_cluster_stack[:, t])
        _, p_hipp = stats.ttest_rel(hipp_switch_stack[:, t], hipp_cluster_stack[:, t])
        pmc_pvalues.append(p_pmc)
        hipp_pvalues.append(p_hipp)
    
    pmc_pvalues = np.array(pmc_pvalues)
    hipp_pvalues = np.array(hipp_pvalues)
    
    # Bonferroni correction
    alpha = 0.05
    bonferroni_alpha = alpha / n_timepoints
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # === PMC Plot ===
    ax = axes[0]
    
    ax.plot(time_vec, pmc_switch_mean, 'r-', linewidth=2, label='Switch', marker='o', markersize=4)
    ax.fill_between(time_vec, 
                    pmc_switch_mean - pmc_switch_sem,
                    pmc_switch_mean + pmc_switch_sem,
                    color='red', alpha=0.3)
    
    ax.plot(time_vec, pmc_cluster_mean, 'gray', linewidth=2, label='Cluster', marker='o', markersize=4)
    ax.fill_between(time_vec,
                    pmc_cluster_mean - pmc_cluster_sem,
                    pmc_cluster_mean + pmc_cluster_sem,
                    color='gray', alpha=0.3)
    
    sig_indices = np.where(pmc_pvalues < bonferroni_alpha)[0]
    if len(sig_indices) > 0:
        y_pos = Y_LIM_GROUP[0] + 0.05 * (Y_LIM_GROUP[1] - Y_LIM_GROUP[0])
        for idx in sig_indices:
            ax.text(time_vec[idx], y_pos, '*', fontsize=14, ha='center', fontweight='bold')
    
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time from preceding word offset (s)', fontsize=12)
    ax.set_ylabel('BOLD (z-scored)', fontsize=12)
    ax.set_title('PMC', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([time_vec[0] - 0.5, time_vec[-1] + 0.5])
    ax.set_ylim(Y_LIM_GROUP)
    
    # === Hippocampus Plot ===
    ax = axes[1]
    
    ax.plot(time_vec, hipp_switch_mean, 'r-', linewidth=2, label='Switch', marker='o', markersize=4)
    ax.fill_between(time_vec,
                    hipp_switch_mean - hipp_switch_sem,
                    hipp_switch_mean + hipp_switch_sem,
                    color='red', alpha=0.3)
    
    ax.plot(time_vec, hipp_cluster_mean, 'gray', linewidth=2, label='Cluster', marker='o', markersize=4)
    ax.fill_between(time_vec,
                    hipp_cluster_mean - hipp_cluster_sem,
                    hipp_cluster_mean + hipp_cluster_sem,
                    color='gray', alpha=0.3)
    
    sig_indices = np.where(hipp_pvalues < bonferroni_alpha)[0]
    if len(sig_indices) > 0:
        y_pos = Y_LIM_GROUP[0] + 0.05 * (Y_LIM_GROUP[1] - Y_LIM_GROUP[0])
        for idx in sig_indices:
            ax.text(time_vec[idx], y_pos, '*', fontsize=14, ha='center', fontweight='bold')
    
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time from preceding word offset (s)', fontsize=12)
    ax.set_ylabel('BOLD (z-scored)', fontsize=12)
    ax.set_title('Hippocampus', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([time_vec[0] - 0.5, time_vec[-1] + 0.5])
    ax.set_ylim(Y_LIM_GROUP)
    
    fig.suptitle(f'Group Event-locked Time Courses - SVF (N={n_sessions})\nLocked to Previous Word Offset | * p<{bonferroni_alpha:.4f} (Bonferroni corrected)', 
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved group time course figure to {output_path}")
    
    # Print statistics
    print(f"\nPMC significant time points (p<{bonferroni_alpha:.4f}):")
    for i, t in enumerate(time_vec):
        if pmc_pvalues[i] < bonferroni_alpha:
            print(f"  t={t:.1f}s: p={pmc_pvalues[i]:.6f}")
    
    print(f"\nHippocampus significant time points (p<{bonferroni_alpha:.4f}):")
    for i, t in enumerate(time_vec):
        if hipp_pvalues[i] < bonferroni_alpha:
            print(f"  t={t:.1f}s: p={hipp_pvalues[i]:.6f}")
    
    total_switches = sum(r['n_switch_epochs'] for r in valid_results)
    total_clusters = sum(r['n_cluster_epochs'] for r in valid_results)
    print(f"\nTotal epochs across all sessions:")
    print(f"  Switch: {total_switches}")
    print(f"  Cluster: {total_clusters}")
    
    return {
        'pmc_switch_mean': pmc_switch_mean,
        'pmc_switch_sem': pmc_switch_sem,
        'pmc_cluster_mean': pmc_cluster_mean,
        'pmc_cluster_sem': pmc_cluster_sem,
        'hipp_switch_mean': hipp_switch_mean,
        'hipp_switch_sem': hipp_switch_sem,
        'hipp_cluster_mean': hipp_cluster_mean,
        'hipp_cluster_sem': hipp_cluster_sem,
        'pmc_pvalues': pmc_pvalues,
        'hipp_pvalues': hipp_pvalues,
        'time_vec': time_vec,
        'n_sessions': n_sessions,
        'total_switches': total_switches,
        'total_clusters': total_clusters,
    }


def get_all_sessions():
    """Collect all subject-session pairs from derivatives directory."""
    sessions = []
    sub_dirs = sorted(list(DERIVATIVES_DIR.glob("sub-*")))
    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        ses_dirs = sorted(list(sub_dir.glob("ses-*")))
        for ses_dir in ses_dirs:
            ses_id = ses_dir.name
            sessions.append((sub_id, ses_id))
    return sessions


def print_summary(results):
    """Print a summary of processing results."""
    valid = [r for r in results if r is not None]
    failed = len(results) - len(valid)
    
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"  ✓ Success: {len(valid)}")
    print(f"  ✗ Failed/Skipped: {failed}")
    
    if valid:
        total_switches = sum(r['n_switch_epochs'] for r in valid)
        total_clusters = sum(r['n_cluster_epochs'] for r in valid)
        print(f"\n  Total epochs across successful sessions:")
        print(f"    Switch: {total_switches}")
        print(f"    Cluster: {total_clusters}")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run switch/cluster analysis for SVF task")
    parser.add_argument("--sub", help="Subject ID (e.g., sub-001)")
    parser.add_argument("--ses", help="Session ID (e.g., ses-01)")
    parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--group_only", action="store_true", 
                        help="Only run group analysis (requires previous --all run)")
    
    args = parser.parse_args()
    
    results_cache_path = GROUP_OUTPUT_DIR / "session_results.pkl"
    
    if args.group_only:
        if results_cache_path.exists():
            with open(results_cache_path, 'rb') as f:
                results = pickle.load(f)
            print(f"Loaded {len(results)} cached results")
        else:
            print("No cached results found. Run with --all first.")
            sys.exit(1)
        
        group_output_path = GROUP_OUTPUT_DIR / "group_roi_timecourses.png"
        group_stats = plot_group_timecourses(results, group_output_path)
        
        if group_stats:
            with open(GROUP_OUTPUT_DIR / "group_stats.pkl", 'wb') as f:
                pickle.dump(group_stats, f)
        
    elif args.all:
        sessions = get_all_sessions()
        n_sessions = len(sessions)
        
        if n_sessions == 0:
            print("No sessions found.")
            sys.exit(1)
        
        n_jobs = args.n_jobs
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        elif n_jobs == -2:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
        
        print(f"\nRunning analysis for {n_sessions} sessions using {n_jobs} workers...")
        
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(run_subject_level_analysis)(sub, ses) for sub, ses in sessions
        )
        
        print_summary(results)
        
        with open(results_cache_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Cached results to {results_cache_path}")
        
        group_output_path = GROUP_OUTPUT_DIR / "group_roi_timecourses.png"
        group_stats = plot_group_timecourses(results, group_output_path)
        
        if group_stats:
            with open(GROUP_OUTPUT_DIR / "group_stats.pkl", 'wb') as f:
                pickle.dump(group_stats, f)
        
    else:
        if not args.sub or not args.ses:
            print("Error: Must provide --sub and --ses OR use --all")
            sys.exit(1)
        
        result = run_subject_level_analysis(args.sub, args.ses)
        if result:
            print(f"\nSuccess: {args.sub} {args.ses}")
        else:
            print(f"\nFailed: {args.sub} {args.ses}")