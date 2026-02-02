#!/usr/bin/env python3
"""
First-level analysis for AHC fMRI data using Schaefer parcellation.
Contrast: Across-Possibility Boundaries vs Non-Boundary periods (middle of long possibilities)

Approach based on Su et al. (2025) Nature Communications:
- Boundary period: 6-second window after Across-Possibility transition (shifted +4.5s for HRF)
- Non-boundary period: 6-second window in the middle of long possibilities (≥10s)
- Also extracts PMC and hippocampus time courses locked to boundaries (18 TRs: -2 to +15)
- Group analysis plots mean time courses with SEM shading
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
ANNOTATIONS_DIR = DATA_DIR / "rec/ahc_sentences"
GLM_FIGS_DIR = FIGS_DIR / "fmri_glm_ahc"
GLM_FIGS_DIR.mkdir(parents=True, exist_ok=True)
GROUP_OUTPUT_DIR = GLM_FIGS_DIR / "group"
GROUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0 
HIGH_PASS_HZ = 0.01
HRF_DELAY = 4.5  # seconds
WINDOW_DURATION = 6.0  # seconds
MIN_POSSIBILITY_DURATION = 10.0  # seconds

# Time course extraction parameters (following Su et al.)
TRS_BEFORE = 2  # TRs before boundary
TRS_AFTER = 15  # TRs after boundary
TOTAL_TRS = TRS_BEFORE + TRS_AFTER + 1  # 18 TRs total

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


def get_boundary_and_nonboundary_periods(subject, session):
    """
    Identifies boundary and non-boundary time windows and time points.
    """
    xlsx_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-ahc_desc-sentences.xlsx"
    
    if not xlsx_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*ahc*.xlsx"))
        if candidates:
            xlsx_path = candidates[0]
        else:
            raise FileNotFoundError(f"No AHC behavioral file found at: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()
    df['Prompt Number'] = df['Prompt Number'].ffill()
    df = df.sort_values(['Prompt Number', 'Start Time']).reset_index(drop=True)
    
    print(f"  [{subject} {session}] Loaded {len(df)} rows")
    print(f"  [{subject} {session}] Raw timing range: {df['Start Time'].min():.1f}s - {df['End Time'].max():.1f}s")
    
    # === BOUNDARY PERIODS ===
    df['Preceding_Possibility'] = df.groupby('Prompt Number')['Possibility Number'].shift(1)
    df['is_boundary'] = (df['Possibility Number'] != df['Preceding_Possibility']) & df['Preceding_Possibility'].notna()
    
    boundary_windows = []
    boundary_timepoints = []
    
    for _, row in df[df['is_boundary']].iterrows():
        boundary_onset = row['Start Time'] - SCANNER_START_OFFSET
        
        window_start = boundary_onset + HRF_DELAY
        window_end = window_start + WINDOW_DURATION
        
        if window_start >= 0:
            boundary_windows.append((window_start, window_end))
        
        if boundary_onset >= TRS_BEFORE * TR:
            boundary_timepoints.append(boundary_onset)
    
    # === NON-BOUNDARY PERIODS ===
    df['poss_group'] = ((df['Possibility Number'] != df['Possibility Number'].shift(1)) | 
                        (df['Prompt Number'] != df['Prompt Number'].shift(1))).cumsum()
    
    nonboundary_windows = []
    nonboundary_timepoints = []
    
    for group_id, group_df in df.groupby('poss_group'):
        poss_start = group_df['Start Time'].min()
        poss_end = group_df['End Time'].max()
        poss_duration = poss_end - poss_start
        
        if poss_duration >= MIN_POSSIBILITY_DURATION:
            poss_middle = poss_start + (poss_duration / 2)
            middle_scanner_time = poss_middle - SCANNER_START_OFFSET
            
            window_start = middle_scanner_time + HRF_DELAY - (WINDOW_DURATION / 2)
            window_end = window_start + WINDOW_DURATION
            
            if window_start >= 0:
                nonboundary_windows.append((window_start, window_end))
            
            if middle_scanner_time >= TRS_BEFORE * TR:
                nonboundary_timepoints.append(middle_scanner_time)
    
    print(f"  [{subject} {session}] Boundary windows: {len(boundary_windows)}, timepoints: {len(boundary_timepoints)}")
    print(f"  [{subject} {session}] Non-boundary windows: {len(nonboundary_windows)}, timepoints: {len(nonboundary_timepoints)}")
    
    return boundary_windows, nonboundary_windows, boundary_timepoints, nonboundary_timepoints


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


def plot_roi_timecourses(pmc_boundary, pmc_nonboundary, hipp_boundary, hipp_nonboundary,
                          subject, session, output_path, tr=TR):
    """Plot single-session PMC and hippocampus time courses."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    time_axis = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * tr
    
    # PMC plot
    ax = axes[0]
    ax.plot(time_axis, pmc_boundary, 'r-', linewidth=2, label='Boundary')
    ax.plot(time_axis, pmc_nonboundary, 'gray', linewidth=2, label='Non-boundary')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.2, color='yellow', label='Analysis window')
    ax.set_xlabel('Time from boundary (s)')
    ax.set_ylabel('Activation (z-scored)')
    ax.set_title('PMC (pCunPCC)')
    ax.legend(loc='upper right')
    ax.set_xlim([time_axis[0], time_axis[-1]])
    
    # Hippocampus plot
    ax = axes[1]
    ax.plot(time_axis, hipp_boundary, 'r-', linewidth=2, label='Boundary')
    ax.plot(time_axis, hipp_nonboundary, 'gray', linewidth=2, label='Non-boundary')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.2, color='yellow', label='Analysis window')
    ax.set_xlabel('Time from boundary (s)')
    ax.set_ylabel('Activation (z-scored)')
    ax.set_title('Hippocampus')
    ax.legend(loc='upper right')
    ax.set_xlim([time_axis[0], time_axis[-1]])
    
    fig.suptitle(f'{subject} {session}: Event-locked Time Courses', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [{subject} {session}] Saved time course figure to {output_path}")


def plot_contrast_on_fsaverage(t_values, subject, session, output_path, contrast_name="Boundary > Non-Boundary"):
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


def run_subject_level_analysis(subject, session, task_name="ahc"):
    """
    Run single-session analysis and return time courses for group analysis.
    
    Returns: dict with time courses and metadata, or None if failed
    """
    print(f"\n=== Processing {subject} {session} (AHC) ===")
    
    # 1. Locate BOLD file
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    search_pattern = f"*{task_name}*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(search_pattern))
    
    if not bold_files:
        print(f"  [{subject} {session}] No BOLD file found")
        return None
    bold_path = bold_files[0]
    
    # 2. Get time windows and timepoints
    try:
        boundary_windows, nonboundary_windows, boundary_timepoints, nonboundary_timepoints = \
            get_boundary_and_nonboundary_periods(subject, session)
    except Exception as e:
        print(f"  [{subject} {session}] Error getting windows: {e}")
        return None
    
    if len(boundary_windows) < 2 or len(nonboundary_windows) < 2:
        print(f"  [{subject} {session}] Insufficient windows")
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
    
    # 6. Extract event-locked time courses
    pmc_boundary_tc, _ = extract_event_locked_timecourse(pmc_signals, boundary_timepoints)
    pmc_nonboundary_tc, _ = extract_event_locked_timecourse(pmc_signals, nonboundary_timepoints)
    
    hipp_boundary_tc, _ = extract_event_locked_timecourse(hipp_signals, boundary_timepoints)
    hipp_nonboundary_tc, _ = extract_event_locked_timecourse(hipp_signals, nonboundary_timepoints)
    
    # Squeeze to 1D
    pmc_boundary_tc = pmc_boundary_tc.squeeze()
    pmc_nonboundary_tc = pmc_nonboundary_tc.squeeze()
    hipp_boundary_tc = hipp_boundary_tc.squeeze()
    hipp_nonboundary_tc = hipp_nonboundary_tc.squeeze()
    
    # 7. Plot single-session time courses
    tc_output_path = GLM_FIGS_DIR / f"{subject}_{session}_ahc_roi_timecourses.png"
    plot_roi_timecourses(
        pmc_boundary_tc, pmc_nonboundary_tc,
        hipp_boundary_tc, hipp_nonboundary_tc,
        subject, session, tc_output_path
    )
    
    # 8. Compute whole-brain contrast
    boundary_activations = get_mean_activation_in_windows(roi_signals, boundary_windows)
    nonboundary_activations = get_mean_activation_in_windows(roi_signals, nonboundary_windows)
    
    print(f"  [{subject} {session}] Boundary: {boundary_activations.shape[0]}, Non-boundary: {nonboundary_activations.shape[0]}")
    
    n_rois = roi_signals.shape[1]
    t_values = np.zeros(n_rois)
    
    for i in range(n_rois):
        t_stat, _ = stats.ttest_ind(boundary_activations[:, i], nonboundary_activations[:, i])
        t_values[i] = t_stat
    
    # 9. Plot whole-brain contrast
    out_file = GLM_FIGS_DIR / f"{subject}_{session}_ahc_boundary-vs-nonboundary_fsaverage.png"
    plot_contrast_on_fsaverage(
        t_values, subject, session, out_file,
        contrast_name="Boundary > Non-Boundary"
    )
    
    # Return data for group analysis
    return {
        'subject': subject,
        'session': session,
        'pmc_boundary_tc': pmc_boundary_tc,
        'pmc_nonboundary_tc': pmc_nonboundary_tc,
        'hipp_boundary_tc': hipp_boundary_tc,
        'hipp_nonboundary_tc': hipp_nonboundary_tc,
        'n_boundary_windows': len(boundary_windows),
        'n_nonboundary_windows': len(nonboundary_windows),
        'boundary_activations': boundary_activations.mean(axis=0),  # Mean across windows
        'nonboundary_activations': nonboundary_activations.mean(axis=0),
    }


def plot_group_timecourses(results, output_path, tr=TR):
    """
    Plot group-level time courses with SEM shading (Su et al. Figure 4b style).
    
    Args:
        results: list of dicts from run_subject_level_analysis
    """
    # Filter valid results
    valid_results = [r for r in results if r is not None]
    n_sessions = len(valid_results)
    
    if n_sessions < 2:
        print("Not enough sessions for group analysis")
        return
    
    print(f"\n=== Group Analysis ({n_sessions} sessions) ===")
    
    # Stack time courses across sessions
    pmc_boundary_stack = np.array([r['pmc_boundary_tc'] for r in valid_results])
    pmc_nonboundary_stack = np.array([r['pmc_nonboundary_tc'] for r in valid_results])
    hipp_boundary_stack = np.array([r['hipp_boundary_tc'] for r in valid_results])
    hipp_nonboundary_stack = np.array([r['hipp_nonboundary_tc'] for r in valid_results])
    
    # Compute mean and SEM
    pmc_boundary_mean = pmc_boundary_stack.mean(axis=0)
    pmc_boundary_sem = pmc_boundary_stack.std(axis=0) / np.sqrt(n_sessions)
    pmc_nonboundary_mean = pmc_nonboundary_stack.mean(axis=0)
    pmc_nonboundary_sem = pmc_nonboundary_stack.std(axis=0) / np.sqrt(n_sessions)
    
    hipp_boundary_mean = hipp_boundary_stack.mean(axis=0)
    hipp_boundary_sem = hipp_boundary_stack.std(axis=0) / np.sqrt(n_sessions)
    hipp_nonboundary_mean = hipp_nonboundary_stack.mean(axis=0)
    hipp_nonboundary_sem = hipp_nonboundary_stack.std(axis=0) / np.sqrt(n_sessions)
    
    # Time axis
    time_axis = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * tr
    
    # Perform t-tests at each time point
    pmc_pvalues = []
    hipp_pvalues = []
    for t in range(TOTAL_TRS):
        _, p_pmc = stats.ttest_rel(pmc_boundary_stack[:, t], pmc_nonboundary_stack[:, t])
        _, p_hipp = stats.ttest_rel(hipp_boundary_stack[:, t], hipp_nonboundary_stack[:, t])
        pmc_pvalues.append(p_pmc)
        hipp_pvalues.append(p_hipp)
    
    pmc_pvalues = np.array(pmc_pvalues)
    hipp_pvalues = np.array(hipp_pvalues)
    
    # Bonferroni correction
    alpha = 0.05
    bonferroni_alpha = alpha / TOTAL_TRS
    
    # Create figure (Su et al. style)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # === PMC Plot ===
    ax = axes[0]
    
    # Boundary condition (red)
    ax.plot(time_axis, pmc_boundary_mean, 'r-', linewidth=2, label='Boundary', marker='o', markersize=4)
    ax.fill_between(time_axis, 
                    pmc_boundary_mean - pmc_boundary_sem,
                    pmc_boundary_mean + pmc_boundary_sem,
                    color='red', alpha=0.3)
    
    # Non-boundary condition (gray)
    ax.plot(time_axis, pmc_nonboundary_mean, 'gray', linewidth=2, label='Non-boundary', marker='o', markersize=4)
    ax.fill_between(time_axis,
                    pmc_nonboundary_mean - pmc_nonboundary_sem,
                    pmc_nonboundary_mean + pmc_nonboundary_sem,
                    color='gray', alpha=0.3)
    
    # Mark significant time points
    sig_times = time_axis[pmc_pvalues < bonferroni_alpha]
    if len(sig_times) > 0:
        y_pos = ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for t in sig_times:
            ax.text(t, y_pos, '*', fontsize=14, ha='center', fontweight='bold')
    
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('BOLD (z-scored)', fontsize=12)
    ax.set_title('PMC', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([time_axis[0] - 0.5, time_axis[-1] + 0.5])
    
    # === Hippocampus Plot ===
    ax = axes[1]
    
    # Boundary condition (red)
    ax.plot(time_axis, hipp_boundary_mean, 'r-', linewidth=2, label='Boundary', marker='o', markersize=4)
    ax.fill_between(time_axis,
                    hipp_boundary_mean - hipp_boundary_sem,
                    hipp_boundary_mean + hipp_boundary_sem,
                    color='red', alpha=0.3)
    
    # Non-boundary condition (gray)
    ax.plot(time_axis, hipp_nonboundary_mean, 'gray', linewidth=2, label='Non-boundary', marker='o', markersize=4)
    ax.fill_between(time_axis,
                    hipp_nonboundary_mean - hipp_nonboundary_sem,
                    hipp_nonboundary_mean + hipp_nonboundary_sem,
                    color='gray', alpha=0.3)
    
    # Mark significant time points
    sig_times = time_axis[hipp_pvalues < bonferroni_alpha]
    if len(sig_times) > 0:
        y_pos = ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for t in sig_times:
            ax.text(t, y_pos, '*', fontsize=14, ha='center', fontweight='bold')
    
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('BOLD (z-scored)', fontsize=12)
    ax.set_title('Hippocampus', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([time_axis[0] - 0.5, time_axis[-1] + 0.5])
    
    # Adjust y-axis to be same for both plots
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes:
        ax.set_ylim([y_min, y_max])
    
    fig.suptitle(f'Group Event-locked Time Courses (N={n_sessions})\n* p<{bonferroni_alpha:.4f} (Bonferroni corrected)', 
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved group time course figure to {output_path}")
    
    # Print statistics
    print(f"\nPMC significant time points (p<{bonferroni_alpha:.4f}):")
    for i, t in enumerate(time_axis):
        if pmc_pvalues[i] < bonferroni_alpha:
            print(f"  t={t:.1f}s: p={pmc_pvalues[i]:.6f}")
    
    print(f"\nHippocampus significant time points (p<{bonferroni_alpha:.4f}):")
    for i, t in enumerate(time_axis):
        if hipp_pvalues[i] < bonferroni_alpha:
            print(f"  t={t:.1f}s: p={hipp_pvalues[i]:.6f}")
    
    return {
        'pmc_boundary_mean': pmc_boundary_mean,
        'pmc_boundary_sem': pmc_boundary_sem,
        'pmc_nonboundary_mean': pmc_nonboundary_mean,
        'pmc_nonboundary_sem': pmc_nonboundary_sem,
        'hipp_boundary_mean': hipp_boundary_mean,
        'hipp_boundary_sem': hipp_boundary_sem,
        'hipp_nonboundary_mean': hipp_nonboundary_mean,
        'hipp_nonboundary_sem': hipp_nonboundary_sem,
        'pmc_pvalues': pmc_pvalues,
        'hipp_pvalues': hipp_pvalues,
        'time_axis': time_axis,
        'n_sessions': n_sessions,
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
    failed = [r for r in results if r is None]
    
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"  ✓ Success: {len(valid)}")
    print(f"  ✗ Failed/Skipped: {len(failed)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run boundary analysis for AHC task")
    parser.add_argument("--sub", help="Subject ID (e.g., sub-001)")
    parser.add_argument("--ses", help="Session ID (e.g., ses-05)")
    parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--group_only", action="store_true", 
                        help="Only run group analysis (requires previous --all run with saved results)")
    
    args = parser.parse_args()
    
    results_cache_path = GROUP_OUTPUT_DIR / "session_results.pkl"
    
    if args.group_only:
        # Load cached results
        if results_cache_path.exists():
            with open(results_cache_path, 'rb') as f:
                results = pickle.load(f)
            print(f"Loaded {len(results)} cached results")
        else:
            print("No cached results found. Run with --all first.")
            sys.exit(1)
        
        # Run group analysis
        group_output_path = GROUP_OUTPUT_DIR / "group_roi_timecourses.png"
        group_stats = plot_group_timecourses(results, group_output_path)
        
        # Save group stats
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
        
        # Cache results for later group analysis
        with open(results_cache_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Cached results to {results_cache_path}")
        
        # Run group analysis
        group_output_path = GROUP_OUTPUT_DIR / "group_roi_timecourses.png"
        group_stats = plot_group_timecourses(results, group_output_path)
        
        # Save group stats
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