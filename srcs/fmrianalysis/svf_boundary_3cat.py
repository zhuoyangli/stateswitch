#!/usr/bin/env python3
"""
First-level analysis for SVF fMRI data using Schaefer parcellation.
Contrast: Switch (boundary) vs Cluster (non-boundary) vs Ambiguous

Uses new rating scheme:
- switch_flag = 1: Clustering (non-boundary)
- switch_flag = 2: Switch (boundary)
- switch_flag = 0: Ambiguous/NA

Approach based on Su et al. (2025) Nature Communications:
- 6-second window after word offset (shifted +4.5s for HRF)
- Extracts PMC and hippocampus time courses locked to events (18 TRs: -2 to +15)
- Group analysis plots mean time courses with SEM shading for all 3 conditions
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
ANNOTATIONS_DIR = Path("/home/datasets/stateswitch/rec/svf_ratings/CO")
GLM_FIGS_DIR = FIGS_DIR / "fmri_glm_svf_rated"
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

# Switch flag coding scheme
SWITCH_FLAG_CLUSTER = 1    # Clustering (non-boundary)
SWITCH_FLAG_SWITCH = 2     # Switch (boundary)
SWITCH_FLAG_AMBIGUOUS = 0  # Ambiguous/NA

# Colors for plotting (matching Su et al. style)
COLOR_SWITCH = 'red'
COLOR_CLUSTER = 'gray'
COLOR_AMBIGUOUS = 'orange'

# Pre-fetch atlases
print("Pre-fetching atlases...")
SCHAEFER_ATLAS = datasets.fetch_atlas_schaefer_2018(
    n_rois=400, 
    yeo_networks=17,
    resolution_mm=2
)
FSAVERAGE = datasets.fetch_surf_fsaverage('fsaverage6')
print("Atlases loaded.")


def get_pmc_indices():
    """
    Get indices of PMC parcels in Schaefer 17-network atlas.
    PMC = parcels containing 'pCunPCC' in their label.
    """
    labels = [label.decode() if isinstance(label, bytes) else label 
              for label in SCHAEFER_ATLAS['labels']]
    
    pmc_indices = []
    pmc_labels = []
    for i, label in enumerate(labels):
        if 'pCunPCC' in label:
            pmc_indices.append(i)
            pmc_labels.append(label)
    
    print(f"  PMC ROI: {len(pmc_indices)} parcels")
    for label in pmc_labels:
        print(f"    - {label}")
    
    return pmc_indices

PMC_INDICES = get_pmc_indices()


def get_all_event_types(subject, session):
    """
    Load SVF behavioral data and identify all event types.
    
    Coding scheme:
    - switch_flag = 1: Clustering (non-boundary)
    - switch_flag = 2: Switch (boundary)
    - switch_flag = 0: Ambiguous/NA
    
    Events are defined at word OFFSET (end time).
    
    Returns:
        Dictionary with windows and timepoints for each condition
    """
    # Try to find the Excel file
    xlsx_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestamps_rated.xlsx"
    
    if not xlsx_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*svf*.xlsx"))
        if candidates:
            xlsx_path = candidates[0]
        else:
            raise FileNotFoundError(f"No SVF behavioral file found at: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    
    print(f"  [{subject} {session}] Loaded {len(df)} rows from {xlsx_path.name}")
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Sort by timing
    df = df.sort_values("start").reset_index(drop=True)
    
    # Clean switch_flag
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)
    
    # Print coding distribution
    n_cluster = (df["switch_flag"] == SWITCH_FLAG_CLUSTER).sum()
    n_switch = (df["switch_flag"] == SWITCH_FLAG_SWITCH).sum()
    n_ambiguous = (df["switch_flag"] == SWITCH_FLAG_AMBIGUOUS).sum()
    print(f"  [{subject} {session}] Coding: {n_cluster} cluster, {n_switch} switch, {n_ambiguous} ambiguous")
    
    print(f"  [{subject} {session}] Raw timing range: {df['start'].min():.1f}s - {df['end'].max():.1f}s")
    
    results = {}
    
    # Process each condition
    conditions = {
        'switch': SWITCH_FLAG_SWITCH,
        'cluster': SWITCH_FLAG_CLUSTER,
        'ambiguous': SWITCH_FLAG_AMBIGUOUS
    }
    
    for cond_name, cond_flag in conditions.items():
        cond_df = df[df["switch_flag"] == cond_flag].copy()
        
        windows = []
        timepoints = []
        
        for _, row in cond_df.iterrows():
            # Event time is at word OFFSET (end time), adjusted to scanner time
            event_time = row["end"] - SCANNER_START_OFFSET
            
            # For 6s window analysis: add HRF delay
            window_start = event_time + HRF_DELAY
            window_end = window_start + WINDOW_DURATION
            
            if window_start >= 0:
                windows.append((window_start, window_end))
            
            # For time course extraction
            if event_time >= TRS_BEFORE * TR:
                timepoints.append(event_time)
        
        results[f'{cond_name}_windows'] = windows
        results[f'{cond_name}_timepoints'] = timepoints
        
        print(f"  [{subject} {session}] {cond_name.capitalize()} windows: {len(windows)}, timepoints: {len(timepoints)}")
    
    return results


def extract_roi_time_series(bold_path):
    """Extract time series from Schaefer atlas ROIs."""
    print(f"  Extracting ROI signals from: {bold_path.name}")
    
    masker = NiftiLabelsMasker(
        labels_img=SCHAEFER_ATLAS['maps'],
        labels=SCHAEFER_ATLAS['labels'],
        standardize='zscore_sample', 
        high_pass=HIGH_PASS_HZ,
        t_r=TR,
        memory='nilearn_cache',
        verbose=0
    )
    
    clean_strategy = ["motion", "wm_csf", "global_signal"]
    confounds, sample_mask = load_confounds(
        str(bold_path),
        strategy=clean_strategy,
        motion="basic", 
        global_signal="basic"
    )

    time_series = masker.fit_transform(bold_path, confounds=confounds)
    
    return time_series


def extract_hippocampus_time_series(bold_path):
    """Extract time series from bilateral hippocampus using Harvard-Oxford atlas."""
    print(f"  Extracting hippocampus signals...")
    
    from nilearn.datasets import fetch_atlas_harvard_oxford
    
    ho_atlas = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    masker = NiftiLabelsMasker(
        labels_img=ho_atlas['maps'],
        labels=ho_atlas['labels'],
        standardize='zscore_sample',
        high_pass=HIGH_PASS_HZ,
        t_r=TR,
        memory='nilearn_cache',
        verbose=0
    )
    
    clean_strategy = ["motion", "wm_csf", "global_signal"]
    confounds, sample_mask = load_confounds(
        str(bold_path),
        strategy=clean_strategy,
        motion="basic",
        global_signal="basic"
    )
    
    time_series = masker.fit_transform(bold_path, confounds=confounds)
    
    labels = ho_atlas['labels']
    hipp_indices = [i for i, label in enumerate(labels) 
                   if 'hippocampus' in label.lower()]
    
    if len(hipp_indices) > 0:
        hipp_ts = time_series[:, hipp_indices].mean(axis=1)
        print(f"  Hippocampus ROI: {len(hipp_indices)} regions")
    else:
        print("  WARNING: Hippocampus not found in atlas")
        hipp_ts = np.zeros(time_series.shape[0])
    
    return hipp_ts


def get_mean_activation_in_windows(time_series, windows, tr=TR):
    """Extract mean activation within specified time windows."""
    n_timepoints = time_series.shape[0]
    n_rois = time_series.shape[1] if time_series.ndim > 1 else 1
    
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1, 1)
    
    activations = []
    for start_time, end_time in windows:
        start_tr = int(np.floor(start_time / tr))
        end_tr = int(np.ceil(end_time / tr))
        start_tr = max(0, start_tr)
        end_tr = min(n_timepoints, end_tr)
        
        if start_tr < end_tr:
            window_activation = time_series[start_tr:end_tr, :].mean(axis=0)
            activations.append(window_activation)
    
    if len(activations) == 0:
        return np.array([]).reshape(0, n_rois)
    
    return np.array(activations)


def extract_event_locked_timecourse(time_series, event_timepoints, trs_before=TRS_BEFORE, 
                                     trs_after=TRS_AFTER, tr=TR):
    """
    Extract time courses locked to events.
    """
    n_timepoints = time_series.shape[0]
    total_trs = trs_before + trs_after + 1
    
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1, 1)
    n_rois = time_series.shape[1]
    
    all_timecourses = []
    
    for event_time in event_timepoints:
        event_tr = int(np.round(event_time / tr))
        
        start_tr = event_tr - trs_before
        end_tr = event_tr + trs_after + 1
        
        if start_tr >= 0 and end_tr <= n_timepoints:
            tc = time_series[start_tr:end_tr, :]
            all_timecourses.append(tc)
    
    if len(all_timecourses) == 0:
        return np.zeros((total_trs, n_rois)), np.array([]).reshape(0, total_trs, n_rois)
    
    all_timecourses = np.array(all_timecourses)
    mean_timecourse = all_timecourses.mean(axis=0)
    
    return mean_timecourse, all_timecourses


def plot_roi_timecourses_3conditions(pmc_switch, pmc_cluster, pmc_ambiguous,
                                      hipp_switch, hipp_cluster, hipp_ambiguous,
                                      subject, session, output_path, tr=TR):
    """Plot single-session PMC and hippocampus time courses for all 3 conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    time_axis = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * tr
    
    # PMC plot
    ax = axes[0]
    ax.plot(time_axis, pmc_switch, color=COLOR_SWITCH, linewidth=2, label='Switch')
    ax.plot(time_axis, pmc_cluster, color=COLOR_CLUSTER, linewidth=2, label='Cluster')
    ax.plot(time_axis, pmc_ambiguous, color=COLOR_AMBIGUOUS, linewidth=2, label='Ambiguous')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.2, color='yellow')
    ax.set_xlabel('Time from word offset (s)')
    ax.set_ylabel('Activation (z-scored)')
    ax.set_title('PMC (pCunPCC)')
    ax.legend(loc='upper right')
    ax.set_xlim([time_axis[0], time_axis[-1]])
    
    # Hippocampus plot
    ax = axes[1]
    ax.plot(time_axis, hipp_switch, color=COLOR_SWITCH, linewidth=2, label='Switch')
    ax.plot(time_axis, hipp_cluster, color=COLOR_CLUSTER, linewidth=2, label='Cluster')
    ax.plot(time_axis, hipp_ambiguous, color=COLOR_AMBIGUOUS, linewidth=2, label='Ambiguous')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.2, color='yellow')
    ax.set_xlabel('Time from word offset (s)')
    ax.set_ylabel('Activation (z-scored)')
    ax.set_title('Hippocampus')
    ax.legend(loc='upper right')
    ax.set_xlim([time_axis[0], time_axis[-1]])
    
    fig.suptitle(f'{subject} {session}: Event-locked Time Courses (SVF Rated)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [{subject} {session}] Saved time course figure to {output_path}")


def plot_contrast_on_fsaverage(t_values, subject, session, output_path, contrast_name="Switch > Cluster"):
    """Projects t-values to fsaverage6 and plots 4 views."""
    print(f"  [{subject} {session}] Projecting to fsaverage6 surface...")
    
    masker = NiftiLabelsMasker(labels_img=SCHAEFER_ATLAS['maps'])
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
    
    Returns: dict with time courses and metadata, or None if failed
    """
    print(f"\n=== Processing {subject} {session} (SVF Rated) ===")
    
    # 1. Locate BOLD file
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    search_pattern = f"*{task_name}*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(search_pattern))
    
    if not bold_files:
        print(f"  [{subject} {session}] No BOLD file found")
        return None
    bold_path = bold_files[0]
    
    # 2. Get event windows and timepoints for all conditions
    try:
        events = get_all_event_types(subject, session)
    except FileNotFoundError as e:
        print(f"  [{subject} {session}] {e}")
        return None
    except Exception as e:
        print(f"  [{subject} {session}] Error getting events: {e}")
        return None
    
    # Check minimum events for switch and cluster (ambiguous can have fewer)
    if len(events['switch_windows']) < 2 or len(events['cluster_windows']) < 2:
        print(f"  [{subject} {session}] Insufficient events (switch: {len(events['switch_windows'])}, cluster: {len(events['cluster_windows'])})")
        return None
    
    # 3. Extract ROI time series
    try:
        roi_signals = extract_roi_time_series(bold_path)
    except Exception as e:
        print(f"  [{subject} {session}] Error extracting ROIs: {e}")
        return None
    
    # 4. Extract hippocampus time series
    try:
        hipp_signals = extract_hippocampus_time_series(bold_path)
    except Exception as e:
        print(f"  [{subject} {session}] Warning: Could not extract hippocampus: {e}")
        hipp_signals = np.zeros(roi_signals.shape[0])
    
    # 5. Get PMC time series
    pmc_signals = roi_signals[:, PMC_INDICES].mean(axis=1)
    
    # 6. Extract event-locked time courses for all conditions
    pmc_switch_tc, _ = extract_event_locked_timecourse(pmc_signals, events['switch_timepoints'])
    pmc_cluster_tc, _ = extract_event_locked_timecourse(pmc_signals, events['cluster_timepoints'])
    pmc_ambiguous_tc, _ = extract_event_locked_timecourse(pmc_signals, events['ambiguous_timepoints'])
    
    hipp_switch_tc, _ = extract_event_locked_timecourse(hipp_signals, events['switch_timepoints'])
    hipp_cluster_tc, _ = extract_event_locked_timecourse(hipp_signals, events['cluster_timepoints'])
    hipp_ambiguous_tc, _ = extract_event_locked_timecourse(hipp_signals, events['ambiguous_timepoints'])
    
    # Squeeze to 1D
    pmc_switch_tc = pmc_switch_tc.squeeze()
    pmc_cluster_tc = pmc_cluster_tc.squeeze()
    pmc_ambiguous_tc = pmc_ambiguous_tc.squeeze()
    hipp_switch_tc = hipp_switch_tc.squeeze()
    hipp_cluster_tc = hipp_cluster_tc.squeeze()
    hipp_ambiguous_tc = hipp_ambiguous_tc.squeeze()
    
    # 7. Plot single-session time courses (all 3 conditions)
    tc_output_path = GLM_FIGS_DIR / f"{subject}_{session}_svf-rated_roi_timecourses.png"
    plot_roi_timecourses_3conditions(
        pmc_switch_tc, pmc_cluster_tc, pmc_ambiguous_tc,
        hipp_switch_tc, hipp_cluster_tc, hipp_ambiguous_tc,
        subject, session, tc_output_path
    )
    
    # 8. Compute whole-brain contrast (Switch > Cluster)
    switch_activations = get_mean_activation_in_windows(roi_signals, events['switch_windows'])
    cluster_activations = get_mean_activation_in_windows(roi_signals, events['cluster_windows'])
    
    print(f"  [{subject} {session}] Switch: {switch_activations.shape[0]}, Cluster: {cluster_activations.shape[0]}")
    
    n_rois = roi_signals.shape[1]
    t_values = np.zeros(n_rois)
    
    for i in range(n_rois):
        t_stat, _ = stats.ttest_ind(switch_activations[:, i], cluster_activations[:, i])
        t_values[i] = t_stat
    
    # 9. Plot whole-brain contrast
    out_file = GLM_FIGS_DIR / f"{subject}_{session}_svf-rated_switch-vs-cluster_fsaverage.png"
    plot_contrast_on_fsaverage(
        t_values, subject, session, out_file,
        contrast_name="Switch > Cluster"
    )
    
    # Return data for group analysis (all 3 conditions)
    return {
        'subject': subject,
        'session': session,
        # PMC time courses
        'pmc_switch_tc': pmc_switch_tc,
        'pmc_cluster_tc': pmc_cluster_tc,
        'pmc_ambiguous_tc': pmc_ambiguous_tc,
        # Hippocampus time courses
        'hipp_switch_tc': hipp_switch_tc,
        'hipp_cluster_tc': hipp_cluster_tc,
        'hipp_ambiguous_tc': hipp_ambiguous_tc,
        # Event counts
        'n_switch_windows': len(events['switch_windows']),
        'n_cluster_windows': len(events['cluster_windows']),
        'n_ambiguous_windows': len(events['ambiguous_windows']),
        # Activations
        'switch_activations': switch_activations.mean(axis=0),
        'cluster_activations': cluster_activations.mean(axis=0),
    }


def plot_group_timecourses_3conditions(results, output_path, tr=TR):
    """
    Plot group-level time courses with SEM shading for all 3 conditions.
    Following Su et al. (2025) Figure 4b style.
    """
    valid_results = [r for r in results if r is not None]
    n_sessions = len(valid_results)
    
    if n_sessions < 2:
        print("Not enough sessions for group analysis")
        return None
    
    print(f"\n=== Group Analysis ({n_sessions} sessions) ===")
    
    # Stack time courses for each condition
    pmc_switch_stack = np.array([r['pmc_switch_tc'] for r in valid_results])
    pmc_cluster_stack = np.array([r['pmc_cluster_tc'] for r in valid_results])
    pmc_ambiguous_stack = np.array([r['pmc_ambiguous_tc'] for r in valid_results])
    
    hipp_switch_stack = np.array([r['hipp_switch_tc'] for r in valid_results])
    hipp_cluster_stack = np.array([r['hipp_cluster_tc'] for r in valid_results])
    hipp_ambiguous_stack = np.array([r['hipp_ambiguous_tc'] for r in valid_results])
    
    # Compute mean and SEM for each condition
    def compute_mean_sem(data):
        mean = data.mean(axis=0)
        sem = data.std(axis=0) / np.sqrt(data.shape[0])
        return mean, sem
    
    pmc_switch_mean, pmc_switch_sem = compute_mean_sem(pmc_switch_stack)
    pmc_cluster_mean, pmc_cluster_sem = compute_mean_sem(pmc_cluster_stack)
    pmc_ambiguous_mean, pmc_ambiguous_sem = compute_mean_sem(pmc_ambiguous_stack)
    
    hipp_switch_mean, hipp_switch_sem = compute_mean_sem(hipp_switch_stack)
    hipp_cluster_mean, hipp_cluster_sem = compute_mean_sem(hipp_cluster_stack)
    hipp_ambiguous_mean, hipp_ambiguous_sem = compute_mean_sem(hipp_ambiguous_stack)
    
    # Time axis
    time_axis = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * tr
    
    # Paired t-tests at each time point (Switch vs Cluster)
    pmc_pvalues_switch_vs_cluster = []
    hipp_pvalues_switch_vs_cluster = []
    for t in range(TOTAL_TRS):
        _, p_pmc = stats.ttest_rel(pmc_switch_stack[:, t], pmc_cluster_stack[:, t])
        _, p_hipp = stats.ttest_rel(hipp_switch_stack[:, t], hipp_cluster_stack[:, t])
        pmc_pvalues_switch_vs_cluster.append(p_pmc)
        hipp_pvalues_switch_vs_cluster.append(p_hipp)
    
    pmc_pvalues_switch_vs_cluster = np.array(pmc_pvalues_switch_vs_cluster)
    hipp_pvalues_switch_vs_cluster = np.array(hipp_pvalues_switch_vs_cluster)
    
    # Bonferroni correction
    alpha = 0.05
    bonferroni_alpha = alpha / TOTAL_TRS
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === PMC Plot ===
    ax = axes[0]
    
    # Switch (red)
    ax.plot(time_axis, pmc_switch_mean, color=COLOR_SWITCH, linewidth=2, 
            label='Switch', marker='o', markersize=4)
    ax.fill_between(time_axis, 
                    pmc_switch_mean - pmc_switch_sem,
                    pmc_switch_mean + pmc_switch_sem,
                    color=COLOR_SWITCH, alpha=0.3)
    
    # Ambiguous (orange)
    ax.plot(time_axis, pmc_ambiguous_mean, color=COLOR_AMBIGUOUS, linewidth=2, 
            label='Ambiguous', marker='o', markersize=4)
    ax.fill_between(time_axis,
                    pmc_ambiguous_mean - pmc_ambiguous_sem,
                    pmc_ambiguous_mean + pmc_ambiguous_sem,
                    color=COLOR_AMBIGUOUS, alpha=0.3)
    
    # Cluster (gray)
    ax.plot(time_axis, pmc_cluster_mean, color=COLOR_CLUSTER, linewidth=2, 
            label='Cluster', marker='o', markersize=4)
    ax.fill_between(time_axis,
                    pmc_cluster_mean - pmc_cluster_sem,
                    pmc_cluster_mean + pmc_cluster_sem,
                    color=COLOR_CLUSTER, alpha=0.3)
    
    # Mark significant time points (Switch vs Cluster)
    sig_indices = np.where(pmc_pvalues_switch_vs_cluster < bonferroni_alpha)[0]
    if len(sig_indices) > 0:
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + 0.05 * (y_max - y_min)
        for idx in sig_indices:
            ax.text(time_axis[idx], y_pos, '*', fontsize=14, ha='center', fontweight='bold')
    
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('BOLD (z-scored)', fontsize=12)
    ax.set_title('PMC', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([time_axis[0] - 0.5, time_axis[-1] + 0.5])
    
    # === Hippocampus Plot ===
    ax = axes[1]
    
    # Switch (red)
    ax.plot(time_axis, hipp_switch_mean, color=COLOR_SWITCH, linewidth=2, 
            label='Switch', marker='o', markersize=4)
    ax.fill_between(time_axis,
                    hipp_switch_mean - hipp_switch_sem,
                    hipp_switch_mean + hipp_switch_sem,
                    color=COLOR_SWITCH, alpha=0.3)
    
    # Ambiguous (orange)
    ax.plot(time_axis, hipp_ambiguous_mean, color=COLOR_AMBIGUOUS, linewidth=2, 
            label='Ambiguous', marker='o', markersize=4)
    ax.fill_between(time_axis,
                    hipp_ambiguous_mean - hipp_ambiguous_sem,
                    hipp_ambiguous_mean + hipp_ambiguous_sem,
                    color=COLOR_AMBIGUOUS, alpha=0.3)
    
    # Cluster (gray)
    ax.plot(time_axis, hipp_cluster_mean, color=COLOR_CLUSTER, linewidth=2, 
            label='Cluster', marker='o', markersize=4)
    ax.fill_between(time_axis,
                    hipp_cluster_mean - hipp_cluster_sem,
                    hipp_cluster_mean + hipp_cluster_sem,
                    color=COLOR_CLUSTER, alpha=0.3)
    
    # Mark significant time points (Switch vs Cluster)
    sig_indices = np.where(hipp_pvalues_switch_vs_cluster < bonferroni_alpha)[0]
    if len(sig_indices) > 0:
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + 0.05 * (y_max - y_min)
        for idx in sig_indices:
            ax.text(time_axis[idx], y_pos, '*', fontsize=14, ha='center', fontweight='bold')
    
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('BOLD (z-scored)', fontsize=12)
    ax.set_title('Hippocampus', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([time_axis[0] - 0.5, time_axis[-1] + 0.5])
    
    # Match y-axis limits
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes:
        ax.set_ylim([y_min, y_max])
    
    fig.suptitle(f'Group Event-locked Time Courses - SVF Rated (N={n_sessions})\n* p<{bonferroni_alpha:.4f} Switch vs Cluster (Bonferroni corrected)', 
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved group time course figure to {output_path}")
    
    # Print statistics
    print(f"\nPMC significant time points - Switch vs Cluster (p<{bonferroni_alpha:.4f}):")
    for i, t in enumerate(time_axis):
        if pmc_pvalues_switch_vs_cluster[i] < bonferroni_alpha:
            print(f"  t={t:.1f}s: p={pmc_pvalues_switch_vs_cluster[i]:.6f}")
    
    print(f"\nHippocampus significant time points - Switch vs Cluster (p<{bonferroni_alpha:.4f}):")
    for i, t in enumerate(time_axis):
        if hipp_pvalues_switch_vs_cluster[i] < bonferroni_alpha:
            print(f"  t={t:.1f}s: p={hipp_pvalues_switch_vs_cluster[i]:.6f}")
    
    return {
        # PMC
        'pmc_switch_mean': pmc_switch_mean,
        'pmc_switch_sem': pmc_switch_sem,
        'pmc_cluster_mean': pmc_cluster_mean,
        'pmc_cluster_sem': pmc_cluster_sem,
        'pmc_ambiguous_mean': pmc_ambiguous_mean,
        'pmc_ambiguous_sem': pmc_ambiguous_sem,
        # Hippocampus
        'hipp_switch_mean': hipp_switch_mean,
        'hipp_switch_sem': hipp_switch_sem,
        'hipp_cluster_mean': hipp_cluster_mean,
        'hipp_cluster_sem': hipp_cluster_sem,
        'hipp_ambiguous_mean': hipp_ambiguous_mean,
        'hipp_ambiguous_sem': hipp_ambiguous_sem,
        # P-values
        'pmc_pvalues_switch_vs_cluster': pmc_pvalues_switch_vs_cluster,
        'hipp_pvalues_switch_vs_cluster': hipp_pvalues_switch_vs_cluster,
        # Meta
        'time_axis': time_axis,
        'n_sessions': n_sessions,
    }


def get_all_sessions_from_annotations():
    """Collect all subject-session pairs from annotations directory."""
    sessions = []
    for xlsx_file in sorted(ANNOTATIONS_DIR.glob("sub-*_ses-*_task-svf*.xlsx")):
        parts = xlsx_file.stem.split('_')
        subject = parts[0]
        session = parts[1]
        sessions.append((subject, session))
    return list(set(sessions))


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
        total_switch = sum(r['n_switch_windows'] for r in valid)
        total_cluster = sum(r['n_cluster_windows'] for r in valid)
        total_ambiguous = sum(r['n_ambiguous_windows'] for r in valid)
        print(f"\n  Total events across sessions:")
        print(f"    Switch: {total_switch}")
        print(f"    Cluster: {total_cluster}")
        print(f"    Ambiguous: {total_ambiguous}")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run switch/cluster/ambiguous analysis for SVF task (rated annotations)")
    parser.add_argument("--sub", help="Subject ID (e.g., sub-001)")
    parser.add_argument("--ses", help="Session ID (e.g., ses-05)")
    parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions with annotations")
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
        group_stats = plot_group_timecourses_3conditions(results, group_output_path)
        
        if group_stats:
            with open(GROUP_OUTPUT_DIR / "group_stats.pkl", 'wb') as f:
                pickle.dump(group_stats, f)
        
    elif args.all:
        # Get sessions from annotation files
        sessions = get_all_sessions_from_annotations()
        n_sessions = len(sessions)
        
        if n_sessions == 0:
            print(f"No annotation files found in {ANNOTATIONS_DIR}")
            sys.exit(1)
        
        print(f"Found {n_sessions} sessions with annotations")
        
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
        
        # Cache results
        with open(results_cache_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Cached results to {results_cache_path}")
        
        # Run group analysis
        group_output_path = GROUP_OUTPUT_DIR / "group_roi_timecourses.png"
        group_stats = plot_group_timecourses_3conditions(results, group_output_path)
        
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