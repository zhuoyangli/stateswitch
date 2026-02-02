#!/usr/bin/env python3
"""
Combined Neural Analysis for SVF and AHC Tasks

Analyzes fMRI activity in PMC and Hippocampus at boundaries/switches:
1. SVF Task: Switch vs Cluster words - event-locked time courses + surface maps
2. AHC Task: Boundary vs Non-boundary periods - event-locked time courses + surface maps

Outputs session-level, subject-level, and group-level figures including fsaverage surface plots.

Based on Su et al. (2025) Nature Communications methodology.

Usage:
    python svf_ahc_neural.py --all                      # Run all analyses
    python svf_ahc_neural.py --svf                      # SVF neural analysis only
    python svf_ahc_neural.py --ahc                      # AHC neural analysis only
    python svf_ahc_neural.py --group-only               # Group analysis only (requires cached results)
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import zscore
from joblib import Parallel, delayed
import multiprocessing

# Neuroimaging imports
from nilearn import datasets, image, plotting, surface
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds

# === PROJECT CONFIG ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DATA_DIR, DERIVATIVES_DIR, FIGS_DIR, TR
except ImportError:
    print("Warning: Could not import configs.config, using defaults")
    DATA_DIR = Path("./data")
    DERIVATIVES_DIR = Path("./derivatives")
    FIGS_DIR = Path("./figs")
    TR = 1.5

# === PATH DEFINITIONS ===
SVF_ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
AHC_ANNOTATIONS_DIR = DATA_DIR / "rec/ahc_sentences"

NEURAL_FIGS_DIR = FIGS_DIR / "neural"
NEURAL_FIGS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = NEURAL_FIGS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HIGH_PASS_HZ = 0.01
HRF_DELAY = 4.5  # seconds
WINDOW_DURATION = 6.0  # seconds
MIN_POSSIBILITY_DURATION = 10.0  # seconds (for AHC non-boundary periods)

# Time course extraction parameters (following Su et al.)
TRS_BEFORE = 2
TRS_AFTER = 15
TOTAL_TRS = TRS_BEFORE + TRS_AFTER + 1  # 18 TRs

# Y-axis limits
Y_LIM_SINGLE = (-0.6, 0.6)
Y_LIM_GROUP = (-0.2, 0.25)

# Surface plot parameters
SURFACE_THRESHOLD = 1.5  # t-value threshold for display (used for subject-level)
SURFACE_VMAX = 8.0  # Fixed z-limit for all surface plots
FDR_Q = 0.05  # FDR q-value threshold for group-level

# === STYLE CONSTANTS ===
COLORS = {
    'switch': '#e74c3c',
    'cluster': 'gray',
    'boundary': '#e74c3c',
    'nonboundary': 'gray',
}

LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14

# === PRE-FETCH ATLASES ===
print("Loading atlases...")
SCHAEFER_ATLAS = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
SCHAEFER_LABELS = [label.decode() if isinstance(label, bytes) else str(label) 
                   for label in SCHAEFER_ATLAS['labels']]
FSAVERAGE = datasets.fetch_surf_fsaverage('fsaverage6')
print("Atlases loaded.")


# ============================================================================
# ROI DEFINITIONS
# ============================================================================

def get_pmc_indices():
    """Get indices of PMC parcels (pCunPCC) in Schaefer atlas."""
    pmc_indices = [i for i, label in enumerate(SCHAEFER_LABELS) if 'pCunPCC' in label]
    return pmc_indices


PMC_INDICES = get_pmc_indices()
print(f"PMC ROI: {len(PMC_INDICES)} parcels")


# ============================================================================
# TIME SERIES EXTRACTION
# ============================================================================

def extract_schaefer_time_series(bold_path):
    """Extract time series from Schaefer 400 parcels."""
    masker = NiftiLabelsMasker(
        labels_img=SCHAEFER_ATLAS['maps'],
        labels=SCHAEFER_LABELS,
        standardize='zscore_sample',
        high_pass=HIGH_PASS_HZ,
        t_r=TR,
        memory='nilearn_cache',
        verbose=0
    )
    
    confounds, sample_mask = load_confounds(
        str(bold_path),
        strategy=["motion", "wm_csf", "global_signal"],
        motion="basic",
        global_signal="basic"
    )
    
    time_series = masker.fit_transform(bold_path, confounds=confounds)
    return time_series


def extract_hippocampus_time_series(bold_path):
    """Extract time series from bilateral hippocampus (Harvard-Oxford atlas)."""
    ho_atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    masker = NiftiLabelsMasker(
        labels_img=ho_atlas['maps'],
        labels=ho_atlas['labels'],
        standardize='zscore_sample',
        high_pass=HIGH_PASS_HZ,
        t_r=TR,
        memory='nilearn_cache',
        verbose=0
    )
    
    confounds, sample_mask = load_confounds(
        str(bold_path),
        strategy=["motion", "wm_csf", "global_signal"],
        motion="basic",
        global_signal="basic"
    )
    
    time_series = masker.fit_transform(bold_path, confounds=confounds)
    
    labels = ho_atlas['labels']
    hipp_indices = [i for i, label in enumerate(labels) if 'hippocampus' in label.lower()]
    
    if len(hipp_indices) > 0:
        hipp_ts = time_series[:, hipp_indices].mean(axis=1)
    else:
        print("  WARNING: Hippocampus not found in atlas")
        hipp_ts = np.zeros(time_series.shape[0])
    
    return hipp_ts


def extract_event_locked_timecourse(signal, event_timepoints, trs_before=TRS_BEFORE, 
                                     trs_after=TRS_AFTER, tr=TR):
    """Extract event-locked time courses from a 1D signal."""
    n_scans = len(signal)
    
    if len(event_timepoints) == 0:
        return np.array([]), np.array([])
    
    onsets = np.array(event_timepoints)
    center_indices = np.round(onsets / tr).astype(int)
    window_offsets = np.arange(-trs_before, trs_after + 1)
    
    epoch_indices = center_indices[:, None] + window_offsets[None, :]
    valid_mask = np.all((epoch_indices >= 0) & (epoch_indices < n_scans), axis=1)
    
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    epochs = signal[epoch_indices[valid_mask]]
    time_vec = window_offsets * tr
    
    if len(epochs.shape) == 1:
        epochs = epochs.reshape(1, -1)
    
    return epochs.mean(axis=0), time_vec


def get_mean_activation_in_window(signal, event_timepoints, tr=TR, 
                                   hrf_delay=HRF_DELAY, window_duration=WINDOW_DURATION):
    """Get mean activation in a window after each event (accounting for HRF)."""
    n_scans = len(signal)
    activations = []
    
    for onset in event_timepoints:
        # Window starts at onset + HRF delay
        window_start_sec = onset + hrf_delay
        window_end_sec = window_start_sec + window_duration
        
        # Convert to TR indices
        start_idx = int(np.floor(window_start_sec / tr))
        end_idx = int(np.ceil(window_end_sec / tr))
        
        if start_idx >= 0 and end_idx < n_scans:
            activations.append(np.mean(signal[start_idx:end_idx]))
    
    return np.array(activations)


def get_all_sessions(task):
    """Get list of all (subject, session) pairs for a task."""
    sessions = []
    
    for sub_dir in sorted(DERIVATIVES_DIR.glob("sub-*")):
        subject = sub_dir.name
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            session = ses_dir.name
            
            func_dir = ses_dir / "func"
            bold_pattern = f"*{task}*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
            bold_files = list(func_dir.glob(bold_pattern))
            
            if bold_files:
                sessions.append((subject, session))
    
    return sessions


# ============================================================================
# FDR CORRECTION
# ============================================================================

def fdr_threshold(p_values, q=0.05):
    """
    Compute FDR threshold using Benjamini-Hochberg procedure.
    
    Parameters
    ----------
    p_values : array-like
        Array of p-values
    q : float
        FDR q-value threshold (default: 0.05)
    
    Returns
    -------
    p_threshold : float
        P-value threshold for significance
    fdr_mask : array
        Boolean mask of significant tests
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)
    
    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # BH critical values: (i/m) * q
    bh_critical = q * np.arange(1, n_tests + 1) / n_tests
    
    # Find largest p-value that is less than its BH critical value
    below_threshold = sorted_p <= bh_critical
    
    if np.any(below_threshold):
        max_idx = np.max(np.where(below_threshold)[0])
        p_threshold = sorted_p[max_idx]
    else:
        p_threshold = 0  # No significant results
    
    fdr_mask = p_values <= p_threshold
    
    return p_threshold, fdr_mask


# ============================================================================
# SURFACE PLOTTING
# ============================================================================

def compute_parcel_contrast(schaefer_ts, condition1_timepoints, condition2_timepoints):
    """
    Compute t-values for each Schaefer parcel comparing two conditions.
    
    Returns array of t-values (n_parcels,)
    """
    n_parcels = schaefer_ts.shape[1]
    t_values = np.zeros(n_parcels)
    
    for i in range(n_parcels):
        parcel_signal = schaefer_ts[:, i]
        
        # Get mean activation in window for each condition
        act1 = get_mean_activation_in_window(parcel_signal, condition1_timepoints)
        act2 = get_mean_activation_in_window(parcel_signal, condition2_timepoints)
        
        if len(act1) > 1 and len(act2) > 1:
            t_stat, _ = stats.ttest_ind(act1, act2)
            t_values[i] = t_stat
    
    return t_values


def plot_surface_contrast(t_values, output_path, title="Contrast", threshold=SURFACE_THRESHOLD, vmax=SURFACE_VMAX):
    """
    Project parcel t-values to fsaverage6 surface and plot 4 views.
    """
    print(f"  Projecting to fsaverage6 surface...")
    
    # Create masker for inverse transform
    masker = NiftiLabelsMasker(labels_img=SCHAEFER_ATLAS['maps'])
    masker.fit()
    
    # Inverse transform t-values to volume
    t_map_img = masker.inverse_transform(t_values.reshape(1, -1))
    
    # Project to surface
    texture_left = surface.vol_to_surf(t_map_img, FSAVERAGE['pial_left'])
    texture_right = surface.vol_to_surf(t_map_img, FSAVERAGE['pial_right'])
    
    # Create figure with 4 views
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'projection': '3d'})
    
    views = [
        (texture_left, FSAVERAGE['infl_left'], FSAVERAGE['sulc_left'], 'left', 'lateral'),
        (texture_left, FSAVERAGE['infl_left'], FSAVERAGE['sulc_left'], 'left', 'medial'),
        (texture_right, FSAVERAGE['infl_right'], FSAVERAGE['sulc_right'], 'right', 'lateral'),
        (texture_right, FSAVERAGE['infl_right'], FSAVERAGE['sulc_right'], 'right', 'medial'),
    ]
    
    for ax, (tex, mesh, bg, hemi, view) in zip(axes, views):
        plotting.plot_surf_stat_map(
            mesh, tex, hemi=hemi, bg_map=bg, view=view,
            cmap='coolwarm', threshold=threshold, axes=ax,
            colorbar=False, vmax=vmax, bg_on_data=False,
        )
        ax.set_title(f"{hemi.upper()} {view.capitalize()}", fontsize=10)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1, shrink=0.5)
    cbar.set_label('t-value', fontsize=10)
    
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_subject_surface_contrast(all_results, task_name, condition1_label, condition2_label, 
                                   threshold=SURFACE_THRESHOLD, vmax=SURFACE_VMAX):
    """
    Plot surface contrast for each subject (averaged across sessions).
    """
    print(f"\n  Generating subject-level surface plots for {task_name}...")
    
    # Group sessions by subject
    subject_t_values = defaultdict(list)
    for r in all_results:
        subject = r['subject']
        t_vals = r['parcel_t_values']
        subject_t_values[subject].append(t_vals)
    
    # Plot for each subject
    for subject, session_t_list in sorted(subject_t_values.items()):
        # Average across sessions within subject
        session_stack = np.array(session_t_list)  # (n_sessions, n_parcels)
        subject_mean_t = session_stack.mean(axis=0)  # (n_parcels,)
        n_sessions = len(session_t_list)
        
        # Generate output path
        output_path = NEURAL_FIGS_DIR / f"{subject}_{task_name.lower()}_surface_{condition1_label.lower()}-vs-{condition2_label.lower().replace('-', '')}.png"
        
        # Plot
        title = f"{task_name}: {condition1_label} > {condition2_label}\n{subject} (N={n_sessions} sessions)"
        plot_surface_contrast(subject_mean_t, output_path, title=title, threshold=threshold, vmax=vmax)


def plot_group_surface_contrast(all_results, output_path, title="Group Contrast", vmax=SURFACE_VMAX, q=FDR_Q):
    """
    Compute group-level t-values and plot on surface.
    
    Averages t-values across all sessions, then performs one-sample t-test.
    Threshold is set using FDR correction (Benjamini-Hochberg).
    """
    print(f"  Computing group-level surface contrast (averaging across sessions)...")
    
    # Stack all session t-values
    all_t_values = [r['parcel_t_values'] for r in all_results]
    t_matrix = np.array(all_t_values)  # (n_sessions, n_parcels)
    n_sessions = t_matrix.shape[0]
    n_parcels = t_matrix.shape[1]
    
    print(f"  Performing one-sample t-test across {n_sessions} sessions...")
    
    # One-sample t-test at each parcel across sessions
    group_t = np.zeros(n_parcels)
    group_p = np.zeros(n_parcels)
    
    for i in range(n_parcels):
        if n_sessions > 1:
            t_stat, p_val = stats.ttest_1samp(t_matrix[:, i], 0)
            group_t[i] = t_stat
            group_p[i] = p_val
        else:
            group_t[i] = t_matrix[0, i]
            group_p[i] = 1.0
    
    # FDR correction using Benjamini-Hochberg
    p_threshold, fdr_mask = fdr_threshold(group_p, q=q)
    n_sig_fdr = np.sum(fdr_mask)
    
    # Find t-threshold corresponding to FDR
    if n_sig_fdr > 0:
        t_threshold = np.min(np.abs(group_t[fdr_mask]))
    else:
        # If nothing survives FDR, use a high threshold (nothing will be shown)
        t_threshold = np.max(np.abs(group_t)) + 1
    
    print(f"  FDR-corrected threshold (q < {q}): t = {t_threshold:.3f}, p = {p_threshold:.2e}")
    
    # Report summary statistics
    n_sig_uncorrected = np.sum(group_p < 0.05)
    alpha_bonferroni = 0.05 / n_parcels
    n_sig_bonferroni = np.sum(group_p < alpha_bonferroni)
    print(f"  Significant parcels (p < 0.05 uncorrected): {n_sig_uncorrected}")
    print(f"  Significant parcels (FDR q < {q}): {n_sig_fdr}")
    print(f"  Significant parcels (Bonferroni p < 0.05): {n_sig_bonferroni}")
    
    # Create masker for inverse transform
    masker = NiftiLabelsMasker(labels_img=SCHAEFER_ATLAS['maps'])
    masker.fit()
    
    # Inverse transform
    t_map_img = masker.inverse_transform(group_t.reshape(1, -1))
    
    # Project to surface
    texture_left = surface.vol_to_surf(t_map_img, FSAVERAGE['pial_left'])
    texture_right = surface.vol_to_surf(t_map_img, FSAVERAGE['pial_right'])
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'projection': '3d'})
    
    views = [
        (texture_left, FSAVERAGE['infl_left'], FSAVERAGE['sulc_left'], 'left', 'lateral'),
        (texture_left, FSAVERAGE['infl_left'], FSAVERAGE['sulc_left'], 'left', 'medial'),
        (texture_right, FSAVERAGE['infl_right'], FSAVERAGE['sulc_right'], 'right', 'lateral'),
        (texture_right, FSAVERAGE['infl_right'], FSAVERAGE['sulc_right'], 'right', 'medial'),
    ]
    
    for ax, (tex, mesh, bg, hemi, view) in zip(axes, views):
        plotting.plot_surf_stat_map(
            mesh, tex, hemi=hemi, bg_map=bg, view=view,
            cmap='coolwarm', threshold=t_threshold, axes=ax,
            colorbar=False, vmax=vmax, bg_on_data=False,
        )
        ax.set_title(f"{hemi.upper()} {view.capitalize()}", fontsize=10)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1, shrink=0.5)
    cbar.set_label('t-value', fontsize=10)
    
    fig.suptitle(f"{title} (N={n_sessions} sessions)\nFDR-corrected q < {q}", 
                 fontsize=TITLE_FONTSIZE, fontweight='bold', y=1.05)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return group_t, group_p, t_threshold


# ============================================================================
# SVF NEURAL ANALYSIS
# ============================================================================

def get_svf_events(subject, session):
    """Load SVF events (switch vs cluster) with timing information."""
    csv_path = SVF_ANNOTATIONS_DIR / f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    
    if not csv_path.exists():
        candidates = list(SVF_ANNOTATIONS_DIR.glob(f"{subject}_{session}*wordtimestamps*.csv"))
        if candidates:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(f"No SVF CSV for {subject} {session}")
    
    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)
    
    df["preceding_end"] = df["end"].shift(1)
    df["preceding_switch_flag"] = df["switch_flag"].shift(1)
    df["preceding_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["preceding_switch_flag"] == 1
    prev_was_next = df["preceding_word"] == "next"
    df = df[~(is_switch & (prev_was_switch | prev_was_next))].copy()
    
    df["onset"] = df["start"] - SCANNER_START_OFFSET
    df["prev_offset"] = df["preceding_end"] - SCANNER_START_OFFSET
    
    df = df[df["onset"] >= 0].copy()
    
    switch_timepoints = df[df["switch_flag"] == 1]["prev_offset"].dropna().values
    cluster_timepoints = df[df["switch_flag"] == 0]["prev_offset"].dropna().values
    
    min_time = TRS_BEFORE * TR
    switch_timepoints = switch_timepoints[switch_timepoints >= min_time]
    cluster_timepoints = cluster_timepoints[cluster_timepoints >= min_time]
    
    return switch_timepoints, cluster_timepoints


def run_svf_session(subject, session):
    """Run SVF neural analysis for a single session."""
    print(f"  Processing {subject} {session} (SVF)...")
    
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    bold_pattern = "*svf*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(bold_pattern))
    
    if not bold_files:
        print(f"    No BOLD file found")
        return None
    
    bold_path = bold_files[0]
    
    try:
        switch_timepoints, cluster_timepoints = get_svf_events(subject, session)
    except Exception as e:
        print(f"    Error getting events: {e}")
        return None
    
    if len(switch_timepoints) < 2 or len(cluster_timepoints) < 2:
        print(f"    Insufficient events: {len(switch_timepoints)} switch, {len(cluster_timepoints)} cluster")
        return None
    
    try:
        schaefer_ts = extract_schaefer_time_series(bold_path)
        hipp_ts = extract_hippocampus_time_series(bold_path)
        pmc_ts = schaefer_ts[:, PMC_INDICES].mean(axis=1)
    except Exception as e:
        print(f"    Error extracting ROIs: {e}")
        return None
    
    # Extract event-locked time courses
    pmc_switch_tc, time_vec = extract_event_locked_timecourse(pmc_ts, switch_timepoints)
    pmc_cluster_tc, _ = extract_event_locked_timecourse(pmc_ts, cluster_timepoints)
    hipp_switch_tc, _ = extract_event_locked_timecourse(hipp_ts, switch_timepoints)
    hipp_cluster_tc, _ = extract_event_locked_timecourse(hipp_ts, cluster_timepoints)
    
    if len(pmc_switch_tc) == 0:
        print(f"    No valid epochs extracted")
        return None
    
    # Compute parcel-wise contrast for surface plot
    parcel_t_values = compute_parcel_contrast(schaefer_ts, switch_timepoints, cluster_timepoints)
    
    print(f"    Extracted: {len(switch_timepoints)} switch, {len(cluster_timepoints)} cluster events")
    
    return {
        'subject': subject,
        'session': session,
        'pmc_switch_tc': pmc_switch_tc,
        'pmc_cluster_tc': pmc_cluster_tc,
        'hipp_switch_tc': hipp_switch_tc,
        'hipp_cluster_tc': hipp_cluster_tc,
        'time_vec': time_vec,
        'n_switch': len(switch_timepoints),
        'n_cluster': len(cluster_timepoints),
        'parcel_t_values': parcel_t_values,
    }


def run_svf_neural_analysis(n_jobs=-1):
    """Run full SVF neural analysis."""
    print("\n" + "=" * 60)
    print("SVF NEURAL ANALYSIS: Switch vs Cluster")
    print("=" * 60)
    
    sessions = get_all_sessions('svf')
    print(f"Found {len(sessions)} SVF sessions")
    
    if len(sessions) == 0:
        print("No sessions found!")
        return None
    
    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), len(sessions))
    
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_svf_session)(sub, ses) for sub, ses in sessions
    )
    
    results = [r for r in results if r is not None]
    print(f"\nSuccessfully processed {len(results)} sessions")
    
    if len(results) == 0:
        return None
    
    # Cache results
    cache_path = CACHE_DIR / "svf_session_results.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Cached results to {cache_path}")
    
    # Aggregate by subject (for subject-level plots)
    subject_results = aggregate_by_subject(results)
    
    # Plot subject-level time courses
    for subject, data in subject_results.items():
        plot_neural_subject(subject, data, 'SVF', 'Switch', 'Cluster')
    
    # Plot subject-level surface contrasts
    plot_subject_surface_contrast(results, 'SVF', 'Switch', 'Cluster')
    
    # Plot group-level time courses (averaging across sessions)
    if len(results) >= 2:
        plot_neural_group_sessions(results, 'SVF', 'Switch', 'Cluster')
    
    # Plot group surface contrast
    group_surface_path = NEURAL_FIGS_DIR / "GROUP_svf_surface_switch-vs-cluster.png"
    plot_group_surface_contrast(
        results, group_surface_path, 
        title="SVF: Switch > Cluster"
    )
    
    return subject_results


# ============================================================================
# AHC NEURAL ANALYSIS
# ============================================================================

def get_ahc_events(subject, session):
    """Load AHC events (boundary vs non-boundary) with timing information."""
    xlsx_path = AHC_ANNOTATIONS_DIR / f"{subject}_{session}_task-ahc_desc-sentences.xlsx"
    
    if not xlsx_path.exists():
        candidates = list(AHC_ANNOTATIONS_DIR.glob(f"{subject}_{session}*ahc*.xlsx"))
        if candidates:
            xlsx_path = candidates[0]
        else:
            raise FileNotFoundError(f"No AHC file for {subject} {session}")
    
    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()
    df['Prompt Number'] = df['Prompt Number'].ffill()
    df = df.sort_values(['Prompt Number', 'Start Time']).reset_index(drop=True)
    
    df['Preceding_Possibility'] = df.groupby('Prompt Number')['Possibility Number'].shift(1)
    df['is_boundary'] = (df['Possibility Number'] != df['Preceding_Possibility']) & df['Preceding_Possibility'].notna()
    
    boundary_timepoints = []
    for _, row in df[df['is_boundary']].iterrows():
        onset = row['Start Time'] - SCANNER_START_OFFSET
        if onset >= TRS_BEFORE * TR:
            boundary_timepoints.append(onset)
    
    df['poss_group'] = ((df['Possibility Number'] != df['Possibility Number'].shift(1)) |
                        (df['Prompt Number'] != df['Prompt Number'].shift(1))).cumsum()
    
    nonboundary_timepoints = []
    for group_id, group_df in df.groupby('poss_group'):
        poss_start = group_df['Start Time'].min()
        poss_end = group_df['End Time'].max()
        poss_duration = poss_end - poss_start
        
        if poss_duration >= MIN_POSSIBILITY_DURATION:
            poss_middle = poss_start + (poss_duration / 2)
            onset = poss_middle - SCANNER_START_OFFSET
            if onset >= TRS_BEFORE * TR:
                nonboundary_timepoints.append(onset)
    
    return np.array(boundary_timepoints), np.array(nonboundary_timepoints)


def run_ahc_session(subject, session):
    """Run AHC neural analysis for a single session."""
    print(f"  Processing {subject} {session} (AHC)...")
    
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    bold_pattern = "*ahc*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(bold_pattern))
    
    if not bold_files:
        print(f"    No BOLD file found")
        return None
    
    bold_path = bold_files[0]
    
    try:
        boundary_timepoints, nonboundary_timepoints = get_ahc_events(subject, session)
    except Exception as e:
        print(f"    Error getting events: {e}")
        return None
    
    if len(boundary_timepoints) < 2 or len(nonboundary_timepoints) < 2:
        print(f"    Insufficient events: {len(boundary_timepoints)} boundary, {len(nonboundary_timepoints)} non-boundary")
        return None
    
    try:
        schaefer_ts = extract_schaefer_time_series(bold_path)
        hipp_ts = extract_hippocampus_time_series(bold_path)
        pmc_ts = schaefer_ts[:, PMC_INDICES].mean(axis=1)
    except Exception as e:
        print(f"    Error extracting ROIs: {e}")
        return None
    
    # Extract event-locked time courses
    pmc_boundary_tc, time_vec = extract_event_locked_timecourse(pmc_ts, boundary_timepoints)
    pmc_nonboundary_tc, _ = extract_event_locked_timecourse(pmc_ts, nonboundary_timepoints)
    hipp_boundary_tc, _ = extract_event_locked_timecourse(hipp_ts, boundary_timepoints)
    hipp_nonboundary_tc, _ = extract_event_locked_timecourse(hipp_ts, nonboundary_timepoints)
    
    if len(pmc_boundary_tc) == 0:
        print(f"    No valid epochs extracted")
        return None
    
    # Compute parcel-wise contrast for surface plot
    parcel_t_values = compute_parcel_contrast(schaefer_ts, boundary_timepoints, nonboundary_timepoints)
    
    print(f"    Extracted: {len(boundary_timepoints)} boundary, {len(nonboundary_timepoints)} non-boundary events")
    
    return {
        'subject': subject,
        'session': session,
        'pmc_switch_tc': pmc_boundary_tc,  # Using consistent naming
        'pmc_cluster_tc': pmc_nonboundary_tc,
        'hipp_switch_tc': hipp_boundary_tc,
        'hipp_cluster_tc': hipp_nonboundary_tc,
        'time_vec': time_vec,
        'n_switch': len(boundary_timepoints),
        'n_cluster': len(nonboundary_timepoints),
        'parcel_t_values': parcel_t_values,
    }


def run_ahc_neural_analysis(n_jobs=-1):
    """Run full AHC neural analysis."""
    print("\n" + "=" * 60)
    print("AHC NEURAL ANALYSIS: Boundary vs Non-Boundary")
    print("=" * 60)
    
    sessions = get_all_sessions('ahc')
    print(f"Found {len(sessions)} AHC sessions")
    
    if len(sessions) == 0:
        print("No sessions found!")
        return None
    
    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), len(sessions))
    
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_ahc_session)(sub, ses) for sub, ses in sessions
    )
    
    results = [r for r in results if r is not None]
    print(f"\nSuccessfully processed {len(results)} sessions")
    
    if len(results) == 0:
        return None
    
    # Cache results
    cache_path = CACHE_DIR / "ahc_session_results.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Cached results to {cache_path}")
    
    # Aggregate by subject (for subject-level plots)
    subject_results = aggregate_by_subject(results)
    
    # Plot subject-level time courses
    for subject, data in subject_results.items():
        plot_neural_subject(subject, data, 'AHC', 'Boundary', 'Non-boundary')
    
    # Plot subject-level surface contrasts
    plot_subject_surface_contrast(results, 'AHC', 'Boundary', 'Non-boundary')
    
    # Plot group-level time courses (averaging across sessions)
    if len(results) >= 2:
        plot_neural_group_sessions(results, 'AHC', 'Boundary', 'Non-boundary')
    
    # Plot group surface contrast
    group_surface_path = NEURAL_FIGS_DIR / "GROUP_ahc_surface_boundary-vs-nonboundary.png"
    plot_group_surface_contrast(
        results, group_surface_path,
        title="AHC: Boundary > Non-Boundary"
    )
    
    return subject_results


# ============================================================================
# AGGREGATION AND TIME COURSE PLOTTING
# ============================================================================

def aggregate_by_subject(session_results):
    """Aggregate session results by subject (average across sessions)."""
    subject_data = defaultdict(list)
    
    for r in session_results:
        subject_data[r['subject']].append(r)
    
    aggregated = {}
    
    for subject, sessions in subject_data.items():
        pmc_switch_stack = np.array([s['pmc_switch_tc'] for s in sessions])
        pmc_cluster_stack = np.array([s['pmc_cluster_tc'] for s in sessions])
        hipp_switch_stack = np.array([s['hipp_switch_tc'] for s in sessions])
        hipp_cluster_stack = np.array([s['hipp_cluster_tc'] for s in sessions])
        
        n_sessions = len(sessions)
        
        aggregated[subject] = {
            'pmc_switch_tc': pmc_switch_stack.mean(axis=0),
            'pmc_cluster_tc': pmc_cluster_stack.mean(axis=0),
            'hipp_switch_tc': hipp_switch_stack.mean(axis=0),
            'hipp_cluster_tc': hipp_cluster_stack.mean(axis=0),
            'pmc_switch_sem': pmc_switch_stack.std(axis=0) / np.sqrt(n_sessions) if n_sessions > 1 else np.zeros_like(pmc_switch_stack[0]),
            'pmc_cluster_sem': pmc_cluster_stack.std(axis=0) / np.sqrt(n_sessions) if n_sessions > 1 else np.zeros_like(pmc_cluster_stack[0]),
            'hipp_switch_sem': hipp_switch_stack.std(axis=0) / np.sqrt(n_sessions) if n_sessions > 1 else np.zeros_like(hipp_switch_stack[0]),
            'hipp_cluster_sem': hipp_cluster_stack.std(axis=0) / np.sqrt(n_sessions) if n_sessions > 1 else np.zeros_like(hipp_cluster_stack[0]),
            'time_vec': sessions[0]['time_vec'],
            'n_sessions': n_sessions,
            'total_switch': sum(s['n_switch'] for s in sessions),
            'total_cluster': sum(s['n_cluster'] for s in sessions),
        }
    
    return aggregated


def plot_neural_subject(subject, data, task_name, condition1_label, condition2_label):
    """Plot neural time courses for a single subject."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    time_vec = data['time_vec']
    n_sessions = data['n_sessions']
    
    fig.suptitle(f"{task_name} Neural Analysis: {subject} (N={n_sessions} sessions)",
                 fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    # PMC plot
    ax = axes[0]
    ax.plot(time_vec, data['pmc_switch_tc'], color=COLORS['switch'], linewidth=3, label=condition1_label)
    ax.fill_between(time_vec,
                    data['pmc_switch_tc'] - data['pmc_switch_sem'],
                    data['pmc_switch_tc'] + data['pmc_switch_sem'],
                    color=COLORS['switch'], alpha=0.3)
    
    ax.plot(time_vec, data['pmc_cluster_tc'], color=COLORS['cluster'], linewidth=3, label=condition2_label)
    ax.fill_between(time_vec,
                    data['pmc_cluster_tc'] - data['pmc_cluster_sem'],
                    data['pmc_cluster_tc'] + data['pmc_cluster_sem'],
                    color=COLORS['cluster'], alpha=0.3)
    
    ax.axvline(x=0, color='grey', linestyle='dashed', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.15, color='yellow')
    ax.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FONTSIZE)
    ax.set_title('Posterior Medial Cortex', fontsize=LABEL_FONTSIZE)
    ax.legend(loc='upper right')
    ax.set_xlim([time_vec[0], time_vec[-1]])
    ax.set_ylim(Y_LIM_SINGLE)
    ax.set_yticks([-0.1, 0.0, 0.1, 0.2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Hippocampus plot
    ax = axes[1]
    ax.plot(time_vec, data['hipp_switch_tc'], color=COLORS['switch'], linewidth=3, label=condition1_label)
    ax.fill_between(time_vec,
                    data['hipp_switch_tc'] - data['hipp_switch_sem'],
                    data['hipp_switch_tc'] + data['hipp_switch_sem'],
                    color=COLORS['switch'], alpha=0.3)
    
    ax.plot(time_vec, data['hipp_cluster_tc'], color=COLORS['cluster'], linewidth=3, label=condition2_label)
    ax.fill_between(time_vec,
                    data['hipp_cluster_tc'] - data['hipp_cluster_sem'],
                    data['hipp_cluster_tc'] + data['hipp_cluster_sem'],
                    color=COLORS['cluster'], alpha=0.3)
    
    ax.axvline(x=0, color='grey', linestyle='dashed', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.15, color='yellow')
    ax.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FONTSIZE)
    ax.set_title('Hippocampus', fontsize=LABEL_FONTSIZE)
    ax.legend(loc='upper right')
    ax.set_xlim([time_vec[0], time_vec[-1]])
    ax.set_ylim(Y_LIM_SINGLE)
    ax.set_yticks([-0.1, 0.0, 0.1, 0.2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = NEURAL_FIGS_DIR / f"{subject}_{task_name.lower()}_timecourse.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_neural_group_sessions(session_results, task_name, condition1_label, condition2_label, q=FDR_Q):
    """
    Plot group-level neural time courses with FDR-corrected statistics.
    
    Averages across sessions (not subjects) and uses FDR correction.
    """
    n_sessions = len(session_results)
    
    time_vec = session_results[0]['time_vec']
    
    # Stack all session data
    pmc_switch_stack = np.array([r['pmc_switch_tc'] for r in session_results])
    pmc_cluster_stack = np.array([r['pmc_cluster_tc'] for r in session_results])
    hipp_switch_stack = np.array([r['hipp_switch_tc'] for r in session_results])
    hipp_cluster_stack = np.array([r['hipp_cluster_tc'] for r in session_results])
    
    # Compute group mean and SEM
    pmc_switch_mean = pmc_switch_stack.mean(axis=0)
    pmc_switch_sem = pmc_switch_stack.std(axis=0) / np.sqrt(n_sessions)
    pmc_cluster_mean = pmc_cluster_stack.mean(axis=0)
    pmc_cluster_sem = pmc_cluster_stack.std(axis=0) / np.sqrt(n_sessions)
    
    hipp_switch_mean = hipp_switch_stack.mean(axis=0)
    hipp_switch_sem = hipp_switch_stack.std(axis=0) / np.sqrt(n_sessions)
    hipp_cluster_mean = hipp_cluster_stack.mean(axis=0)
    hipp_cluster_sem = hipp_cluster_stack.std(axis=0) / np.sqrt(n_sessions)
    
    # Paired t-tests at each time point (comparing conditions within sessions)
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
    
    # FDR correction
    pmc_p_threshold, pmc_fdr_mask = fdr_threshold(pmc_pvalues, q=q)
    hipp_p_threshold, hipp_fdr_mask = fdr_threshold(hipp_pvalues, q=q)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    fig.suptitle(f"{task_name} Neural Analysis: Group (N={n_sessions} sessions)\n"
                 f"* FDR-corrected q < {q}",
                 fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    # PMC plot
    ax = axes[0]
    ax.plot(time_vec, pmc_switch_mean, color=COLORS['switch'], linewidth=3, 
            label=condition1_label, marker='o', markersize=4)
    ax.fill_between(time_vec,
                    pmc_switch_mean - pmc_switch_sem,
                    pmc_switch_mean + pmc_switch_sem,
                    color=COLORS['switch'], alpha=0.3)
    
    ax.plot(time_vec, pmc_cluster_mean, color=COLORS['cluster'], linewidth=3,
            label=condition2_label, marker='o', markersize=4)
    ax.fill_between(time_vec,
                    pmc_cluster_mean - pmc_cluster_sem,
                    pmc_cluster_mean + pmc_cluster_sem,
                    color=COLORS['cluster'], alpha=0.3)
    
    # Mark significant time points (FDR-corrected)
    sig_indices = np.where(pmc_pvalues<0.05)[0]
    if len(sig_indices) > 0:
        y_pos = Y_LIM_GROUP[0] + 0.05 * (Y_LIM_GROUP[1] - Y_LIM_GROUP[0])
        for idx in sig_indices:
            ax.text(time_vec[idx], y_pos, '*', fontsize=14, ha='center', fontweight='bold')
    
    ax.axvline(x=0, color='grey', linestyle='dashed', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FONTSIZE)
    ax.set_title('Posterior Medial Cortex', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim([time_vec[0] - 0.5, time_vec[-1] + 0.5])
    ax.set_ylim(Y_LIM_GROUP)
    ax.set_yticks([-0.1, 0.0, 0.1, 0.2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Hippocampus plot
    ax = axes[1]
    ax.plot(time_vec, hipp_switch_mean, color=COLORS['switch'], linewidth=3,
            label=condition1_label, marker='o', markersize=4)
    ax.fill_between(time_vec,
                    hipp_switch_mean - hipp_switch_sem,
                    hipp_switch_mean + hipp_switch_sem,
                    color=COLORS['switch'], alpha=0.3)
    
    ax.plot(time_vec, hipp_cluster_mean, color=COLORS['cluster'], linewidth=3,
            label=condition2_label, marker='o', markersize=4)
    ax.fill_between(time_vec,
                    hipp_cluster_mean - hipp_cluster_sem,
                    hipp_cluster_mean + hipp_cluster_sem,
                    color=COLORS['cluster'], alpha=0.3)
    
    # Mark significant time points (FDR-corrected)
    sig_indices = np.where(hipp_pvalues<0.05)[0]
    if len(sig_indices) > 0:
        y_pos = Y_LIM_GROUP[0] + 0.05 * (Y_LIM_GROUP[1] - Y_LIM_GROUP[0])
        for idx in sig_indices:
            ax.text(time_vec[idx], y_pos, '*', fontsize=14, ha='center', fontweight='bold')
    
    ax.axvline(x=0, color='grey', linestyle='dashed', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('BOLD (z-scored)', fontsize=LABEL_FONTSIZE)
    ax.set_title('Hippocampus', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim([time_vec[0] - 0.5, time_vec[-1] + 0.5])
    ax.set_ylim(Y_LIM_GROUP)
    ax.set_yticks([-0.1, 0.0, 0.1, 0.2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = NEURAL_FIGS_DIR / f"GROUP_{task_name.lower()}_timecourse.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    
    # Print statistics
    print(f"\nPMC significant time points (FDR q < {q}):")
    n_pmc_sig = np.sum(pmc_fdr_mask)
    if n_pmc_sig > 0:
        print(f"  p-threshold: {pmc_p_threshold:.6f}")
        for i, t in enumerate(time_vec):
            if pmc_fdr_mask[i]:
                print(f"  t={t:.1f}s: p={pmc_pvalues[i]:.6f}")
    else:
        print("  No significant time points")
    
    print(f"\nHippocampus significant time points (FDR q < {q}):")
    n_hipp_sig = np.sum(hipp_fdr_mask)
    if n_hipp_sig > 0:
        print(f"  p-threshold: {hipp_p_threshold:.6f}")
        for i, t in enumerate(time_vec):
            if hipp_fdr_mask[i]:
                print(f"  t={t:.1f}s: p={hipp_pvalues[i]:.6f}")
    else:
        print("  No significant time points")
    
    return pmc_pvalues, hipp_pvalues, pmc_fdr_mask, hipp_fdr_mask


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined Neural Analysis for SVF and AHC Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses")
    parser.add_argument("--svf", action="store_true",
                        help="Run SVF neural analysis only")
    parser.add_argument("--ahc", action="store_true",
                        help="Run AHC neural analysis only")
    parser.add_argument("--group-only", action="store_true",
                        help="Run group analysis only (uses cached results)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 for all CPUs)")
    
    args = parser.parse_args()
    
    if not any([args.all, args.svf, args.ahc, args.group_only]):
        args.all = True
    
    print("=" * 60)
    print("COMBINED NEURAL ANALYSIS: SVF & AHC")
    print(f"Output directory: {NEURAL_FIGS_DIR}")
    print("=" * 60)
    
    if args.group_only:
        # Load cached results and run group analysis
        for task, cond1, cond2, contrast_title in [
            ('svf', 'Switching', 'Clustering', 'SVF: Switch > Cluster'), 
            ('ahc', 'Between-explanations', 'Within-explanation', 'AHC: Boundary > Non-Boundary')
        ]:
            cache_path = CACHE_DIR / f"{task}_session_results.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    results = pickle.load(f)
                print(f"\nLoaded {len(results)} cached {task.upper()} results")
                
                # Plot group-level time courses (averaging across sessions with FDR)
                if len(results) >= 2:
                    plot_neural_group_sessions(results, task.upper(), cond1, cond2)
                
                # Plot subject-level surface contrasts
                # plot_subject_surface_contrast(results, task.upper(), cond1, cond2)
                
                # Plot group surface
                group_surface_path = NEURAL_FIGS_DIR / f"GROUP_{task}_surface_{cond1.lower()}-vs-{cond2.lower().replace('-', '')}.png"
                plot_group_surface_contrast(results, group_surface_path, title=contrast_title)
            else:
                print(f"No cached results for {task.upper()}")
    
    else:
        if args.all or args.svf:
            run_svf_neural_analysis(n_jobs=args.n_jobs)
        
        if args.all or args.ahc:
            run_ahc_neural_analysis(n_jobs=args.n_jobs)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {NEURAL_FIGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()