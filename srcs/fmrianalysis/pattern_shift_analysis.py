#!/usr/bin/env python3
"""
Pattern Shift Analysis for SVF and AHC Tasks

For each TR, computes cosine similarity between:
- Average of multi-voxel patterns from preceding 5 TRs
- Average of multi-voxel patterns from following 5 TRs

Extracts event-locked pattern shift time courses for all 400 Schaefer parcels,
then compares conditions (Switch vs Cluster for SVF, Boundary vs Non-boundary for AHC).

Outputs:
- Distribution of max |t-values| across parcels
- Heatmap of all parcels x time
- Top 20 parcel time courses
- Network-level (17 Yeo) time courses

Usage:
    python pattern_shift_analysis.py --svf --subject sub-003
    python pattern_shift_analysis.py --ahc --subject sub-003
    python pattern_shift_analysis.py --all
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.stats import zscore
from joblib import Parallel, delayed
import multiprocessing

# Neuroimaging imports
from nilearn import datasets, plotting, surface
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib

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

PATTERN_SHIFT_FIGS_DIR = FIGS_DIR / "pattern_shift"
PATTERN_SHIFT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = PATTERN_SHIFT_FIGS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0

# Pattern shift parameters
WINDOW_SIZE = 5  # Number of TRs before and after for pattern averaging

# Event-locked time course parameters (following Su et al.)
TRS_BEFORE = 2  # 3 seconds before
TRS_AFTER = 15  # 22.5 seconds after
TOTAL_TRS = TRS_BEFORE + TRS_AFTER + 1  # 18 TRs

# T-value computation window (matching svf_ahc_neural.py)
# Average signal in HRF-adjusted window, then compute t-value
HRF_DELAY = 4.5  # seconds (same as svf_ahc_neural.py)
WINDOW_DURATION = 6.0  # seconds (same as svf_ahc_neural.py)
# Window is HRF_DELAY to HRF_DELAY + WINDOW_DURATION (4.5s to 10.5s post-event)

# Minimum events for analysis
MIN_EVENTS = 3

# Group-level analysis parameters
FDR_Q = 0.05  # FDR q-value threshold
SURFACE_VMAX = 8.0  # Fixed vmax for group surface plots

# === STYLE CONSTANTS ===
COLORS = {
    'switch': '#e74c3c',
    'cluster': '#3498db',
    'boundary': '#e74c3c',
    'nonboundary': '#3498db',
}

LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14

# Y-axis limits for pattern shift plots
Y_LIM = (0.3, 0.8)

# === PRE-FETCH ATLASES ===
print("Loading Schaefer atlas...")
SCHAEFER_ATLAS = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
# Skip the first 'Background' label - atlas has 401 labels, we want 400 parcels
SCHAEFER_LABELS = [label.decode() if isinstance(label, bytes) else str(label)
                   for label in SCHAEFER_ATLAS['labels'][1:]]  # Skip background
FSAVERAGE = datasets.fetch_surf_fsaverage('fsaverage6')
print(f"Atlas loaded. {len(SCHAEFER_LABELS)} parcel labels.")

# === YEO 17 NETWORK MAPPING ===
YEO_17_NETWORKS = [
    'VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB',
    'SalVentAttnA', 'SalVentAttnB', 'LimbicA', 'LimbicB', 'ContA', 'ContB', 'ContC',
    'DefaultA', 'DefaultB', 'DefaultC', 'TempPar'
]

def get_network_for_parcel(label):
    """Extract network name from Schaefer parcel label."""
    # Label format: '17Networks_LH_DefaultA_PFCd_1'
    parts = label.split('_')
    if len(parts) >= 3:
        return parts[2]  # Network name
    return 'Unknown'

PARCEL_NETWORKS = [get_network_for_parcel(label) for label in SCHAEFER_LABELS]


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
# PATTERN SHIFT COMPUTATION
# ============================================================================

def extract_voxelwise_parcel_data(bold_path, atlas_img):
    """
    Extract voxel-wise time series for each parcel.
    """
    print(f"    Loading BOLD data...")
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()
    n_timepoints = bold_data.shape[3]
    print(f"    BOLD shape: {bold_data.shape}")

    print(f"    Loading atlas...")
    atlas_data = nib.load(atlas_img).get_fdata().astype(int)

    parcel_labels = np.unique(atlas_data)
    parcel_labels = parcel_labels[parcel_labels != 0]
    n_parcels = len(parcel_labels)
    print(f"    Extracting voxel patterns from {n_parcels} parcels...")

    parcel_voxels = {}

    for i, parcel_idx in enumerate(parcel_labels):
        if (i + 1) % 100 == 0:
            print(f"      Progress: {i + 1}/{n_parcels} parcels...")

        parcel_mask = atlas_data == parcel_idx
        voxel_coords = np.where(parcel_mask)
        voxel_timeseries = bold_data[voxel_coords[0], voxel_coords[1], voxel_coords[2], :]
        voxel_timeseries = voxel_timeseries.T  # (n_timepoints, n_voxels)
        voxel_timeseries = zscore(voxel_timeseries, axis=0, nan_policy='omit')
        voxel_timeseries = np.nan_to_num(voxel_timeseries)
        parcel_voxels[int(parcel_idx)] = voxel_timeseries

    print(f"    Extraction complete.")
    return parcel_voxels, n_timepoints


def compute_pattern_shift_timeseries(parcel_voxels, n_timepoints, window_size=WINDOW_SIZE):
    """
    Compute pattern shift time series for each parcel.
    Pattern shift = 1 - cosine_similarity between pre and post windows.
    """
    n_parcels = len(parcel_voxels)
    pattern_shifts = np.full((n_parcels, n_timepoints), np.nan)

    print(f"    Computing pattern shifts for {n_parcels} parcels...")
    for i, (parcel_idx, voxel_ts) in enumerate(parcel_voxels.items()):
        if (i + 1) % 100 == 0:
            print(f"      Progress: {i + 1}/{n_parcels} parcels...")

        arr_idx = i  # Use enumerate index directly

        for t in range(window_size, n_timepoints - window_size):
            pre_patterns = voxel_ts[t - window_size:t, :]
            pre_mean = np.mean(pre_patterns, axis=0)

            post_patterns = voxel_ts[t + 1:t + 1 + window_size, :]
            post_mean = np.mean(post_patterns, axis=0)

            if np.std(pre_mean) > 0 and np.std(post_mean) > 0:
                cos_sim = 1 - cosine(pre_mean, post_mean)
                pattern_shifts[arr_idx, t] = 1 - cos_sim
            else:
                pattern_shifts[arr_idx, t] = np.nan

    print(f"    Pattern shift computation complete.")
    return pattern_shifts


def extract_event_locked_pattern_shifts(pattern_shifts, event_timepoints,
                                         trs_before=TRS_BEFORE, trs_after=TRS_AFTER, tr=TR):
    """
    Extract event-locked pattern shift time courses.

    Returns
    -------
    event_locked : ndarray
        Shape (n_events, n_parcels, n_timepoints)
    time_vec : ndarray
        Time vector in seconds relative to event
    """
    n_parcels, n_total_trs = pattern_shifts.shape
    event_trs = np.round(np.array(event_timepoints) / tr).astype(int)

    window_offsets = np.arange(-trs_before, trs_after + 1)
    n_window_trs = len(window_offsets)

    valid_events = []
    for event_tr in event_trs:
        start_tr = event_tr - trs_before
        end_tr = event_tr + trs_after + 1
        if start_tr >= 0 and end_tr <= n_total_trs:
            valid_events.append(event_tr)

    if len(valid_events) == 0:
        return np.array([]), window_offsets * tr

    event_locked = np.zeros((len(valid_events), n_parcels, n_window_trs))

    for i, event_tr in enumerate(valid_events):
        for j in range(n_parcels):
            event_locked[i, j, :] = pattern_shifts[j, event_tr - trs_before:event_tr + trs_after + 1]

    time_vec = window_offsets * tr
    return event_locked, time_vec


# ============================================================================
# EVENT LOADING
# ============================================================================

def get_svf_events(subject, session):
    """Load SVF events (switch vs cluster) with timing information.

    Time zero = prev_offset = end time of the preceding word.
    """
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

    # Use prev_offset (end of preceding word) as event onset
    df["prev_offset"] = df["preceding_end"] - SCANNER_START_OFFSET

    df = df[df["prev_offset"] >= 0].copy()

    switch_timepoints = df[df["switch_flag"] == 1]["prev_offset"].dropna().values
    cluster_timepoints = df[df["switch_flag"] == 0]["prev_offset"].dropna().values

    min_time = (WINDOW_SIZE + TRS_BEFORE) * TR
    switch_timepoints = switch_timepoints[switch_timepoints >= min_time]
    cluster_timepoints = cluster_timepoints[cluster_timepoints >= min_time]

    return switch_timepoints, cluster_timepoints


def get_ahc_events(subject, session):
    """Load AHC events (boundary vs non-boundary) with timing information.

    Boundary: end time of the last sentence in the previous possibility.
    Non-boundary: middle of long possibility periods (>10s).
    """
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

    # Create possibility groups
    df['poss_group'] = ((df['Possibility Number'] != df['Possibility Number'].shift(1)) |
                        (df['Prompt Number'] != df['Prompt Number'].shift(1))).cumsum()

    min_time = (WINDOW_SIZE + TRS_BEFORE) * TR

    # Boundary: end time of the last sentence in the previous possibility
    boundary_timepoints = []
    poss_groups = df.groupby('poss_group')

    for group_id, group_df in poss_groups:
        # Get the end time of this possibility group (last sentence's end time)
        poss_end_time = group_df['End Time'].max()

        # Check if there's a next possibility group (i.e., this is not the last one)
        next_group_id = group_id + 1
        if next_group_id in poss_groups.groups:
            # The boundary is the end of this possibility (before the next one starts)
            onset = poss_end_time - SCANNER_START_OFFSET
            if onset >= min_time:
                boundary_timepoints.append(onset)

    # Non-boundary: middle of long possibility periods (>10s)
    MIN_POSSIBILITY_DURATION = 10.0
    nonboundary_timepoints = []
    for group_id, group_df in poss_groups:
        poss_start = group_df['Start Time'].min()
        poss_end = group_df['End Time'].max()
        poss_duration = poss_end - poss_start

        if poss_duration >= MIN_POSSIBILITY_DURATION:
            poss_middle = poss_start + (poss_duration / 2)
            onset = poss_middle - SCANNER_START_OFFSET
            if onset >= min_time:
                nonboundary_timepoints.append(onset)

    return np.array(boundary_timepoints), np.array(nonboundary_timepoints)


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
# SESSION PROCESSING
# ============================================================================

def run_session(subject, session, task='svf'):
    """Run pattern shift analysis for a single session."""
    print(f"  Processing {subject} {session} ({task.upper()} pattern shift)...")

    func_dir = DERIVATIVES_DIR / subject / session / "func"
    bold_pattern = f"*{task}*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(bold_pattern))

    if not bold_files:
        print(f"    No BOLD file found")
        return None

    bold_path = bold_files[0]

    try:
        if task == 'svf':
            cond1_timepoints, cond2_timepoints = get_svf_events(subject, session)
        else:
            cond1_timepoints, cond2_timepoints = get_ahc_events(subject, session)
    except Exception as e:
        print(f"    Error getting events: {e}")
        return None

    if len(cond1_timepoints) < MIN_EVENTS or len(cond2_timepoints) < MIN_EVENTS:
        print(f"    Insufficient events: {len(cond1_timepoints)} cond1, {len(cond2_timepoints)} cond2")
        return None

    try:
        parcel_voxels, n_timepoints = extract_voxelwise_parcel_data(
            bold_path, SCHAEFER_ATLAS['maps']
        )
        pattern_shifts = compute_pattern_shift_timeseries(parcel_voxels, n_timepoints)

        # Extract event-locked time courses
        cond1_locked, time_vec = extract_event_locked_pattern_shifts(pattern_shifts, cond1_timepoints)
        cond2_locked, _ = extract_event_locked_pattern_shifts(pattern_shifts, cond2_timepoints)

    except Exception as e:
        print(f"    Error computing pattern shifts: {e}")
        import traceback
        traceback.print_exc()
        return None

    if len(cond1_locked) == 0 or len(cond2_locked) == 0:
        print(f"    No valid pattern shifts extracted")
        return None

    # Average across events within each condition
    cond1_mean = np.nanmean(cond1_locked, axis=0)  # (n_parcels, n_timepoints)
    cond2_mean = np.nanmean(cond2_locked, axis=0)

    print(f"    Extracted: {len(cond1_locked)} cond1, {len(cond2_locked)} cond2 events")

    return {
        'subject': subject,
        'session': session,
        'task': task,
        'cond1_locked': cond1_locked,  # (n_events, n_parcels, n_timepoints)
        'cond2_locked': cond2_locked,
        'cond1_mean': cond1_mean,  # (n_parcels, n_timepoints)
        'cond2_mean': cond2_mean,
        'time_vec': time_vec,
        'n_cond1': len(cond1_locked),
        'n_cond2': len(cond2_locked),
    }


# ============================================================================
# ANALYSIS AND STATISTICS
# ============================================================================

def compute_parcel_statistics(results, cond1_label, cond2_label):
    """
    Compute statistics for each parcel across sessions.

    Uses average signal in HRF window (4.5-10.5s) for t-value computation,
    matching the approach in svf_ahc_neural.py.

    Returns
    -------
    stats_dict : dict with keys:
        - t_values_timecourse: (n_parcels, n_timepoints) t-values at each timepoint
        - p_values_timecourse: (n_parcels, n_timepoints) p-values at each timepoint
        - t_values: (n_parcels,) t-values based on window-averaged signal
        - p_values: (n_parcels,) p-values based on window-averaged signal
    """
    n_sessions = len(results)
    time_vec = results[0]['time_vec']
    n_timepoints = len(time_vec)
    n_parcels = results[0]['cond1_mean'].shape[0]

    # Stack session data: (n_sessions, n_parcels, n_timepoints)
    cond1_stack = np.array([r['cond1_mean'] for r in results])
    cond2_stack = np.array([r['cond2_mean'] for r in results])

    # Find indices for HRF window (4.5s to 10.5s)
    window_start = HRF_DELAY
    window_end = HRF_DELAY + WINDOW_DURATION
    t_start_idx = np.searchsorted(time_vec, window_start)
    t_end_idx = np.searchsorted(time_vec, window_end, side='right')

    # Compute t-values based on average signal in HRF window
    t_values = np.zeros(n_parcels)
    p_values = np.ones(n_parcels)

    for p in range(n_parcels):
        # Average signal in HRF window for each session
        c1_window_avg = np.nanmean(cond1_stack[:, p, t_start_idx:t_end_idx], axis=1)  # (n_sessions,)
        c2_window_avg = np.nanmean(cond2_stack[:, p, t_start_idx:t_end_idx], axis=1)  # (n_sessions,)

        # Remove NaN values
        c1_valid = c1_window_avg[~np.isnan(c1_window_avg)]
        c2_valid = c2_window_avg[~np.isnan(c2_window_avg)]

        if len(c1_valid) > 1 and len(c2_valid) > 1:
            t_stat, p_val = stats.ttest_ind(c1_valid, c2_valid)
            t_values[p] = t_stat
            p_values[p] = p_val

    # Also compute t-values at each timepoint (for reference/plotting)
    t_values_timecourse = np.zeros((n_parcels, n_timepoints))
    p_values_timecourse = np.ones((n_parcels, n_timepoints))

    for p in range(n_parcels):
        for t in range(n_timepoints):
            c1 = cond1_stack[:, p, t]
            c2 = cond2_stack[:, p, t]
            c1 = c1[~np.isnan(c1)]
            c2 = c2[~np.isnan(c2)]

            if len(c1) > 1 and len(c2) > 1:
                t_stat, p_val = stats.ttest_ind(c1, c2)
                t_values_timecourse[p, t] = t_stat
                p_values_timecourse[p, t] = p_val

    return {
        't_values_timecourse': t_values_timecourse,
        'p_values_timecourse': p_values_timecourse,
        't_values': t_values,
        'p_values': p_values,
        'time_vec': time_vec,
        'cond1_stack': cond1_stack,
        'cond2_stack': cond2_stack,
    }


def compute_network_timecourses(results):
    """
    Average pattern shifts within each Yeo network.

    Returns
    -------
    network_data : dict with keys for each network
        Each contains: cond1_mean, cond2_mean, cond1_sem, cond2_sem
    """
    time_vec = results[0]['time_vec']
    n_timepoints = len(time_vec)
    n_sessions = len(results)

    # Stack all sessions
    cond1_stack = np.array([r['cond1_mean'] for r in results])  # (n_sessions, n_parcels, n_timepoints)
    cond2_stack = np.array([r['cond2_mean'] for r in results])

    network_data = {}

    for network in YEO_17_NETWORKS:
        # Find parcels in this network
        parcel_idx = [i for i, n in enumerate(PARCEL_NETWORKS) if n == network]

        if len(parcel_idx) == 0:
            continue

        # Average across parcels in network, then across sessions
        cond1_net = cond1_stack[:, parcel_idx, :].mean(axis=1)  # (n_sessions, n_timepoints)
        cond2_net = cond2_stack[:, parcel_idx, :].mean(axis=1)

        network_data[network] = {
            'cond1_mean': np.nanmean(cond1_net, axis=0),
            'cond2_mean': np.nanmean(cond2_net, axis=0),
            'cond1_sem': stats.sem(cond1_net, axis=0, nan_policy='omit'),
            'cond2_sem': stats.sem(cond2_net, axis=0, nan_policy='omit'),
            'n_parcels': len(parcel_idx),
        }

    return network_data


# ============================================================================
# PLOTTING
# ============================================================================

def plot_t_distribution(parcel_stats, output_path, task_name, cond1_label, cond2_label):
    """Plot distribution of t-values (from window-averaged signal) across parcels."""
    t_values = parcel_stats['t_values']

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(t_values, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.median(t_values), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(t_values):.2f}')
    ax.axvline(x=np.percentile(t_values, 95), color='orange', linestyle='--', linewidth=2,
               label=f'95th %ile: {np.percentile(t_values, 95):.2f}')

    ax.set_xlabel(f't-value (avg signal in {HRF_DELAY}-{HRF_DELAY + WINDOW_DURATION}s)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Number of parcels', fontsize=LABEL_FONTSIZE)
    ax.set_title(f'{task_name}: Distribution of t-value per Parcel\n({cond1_label} > {cond2_label})',
                 fontsize=TITLE_FONTSIZE)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_surface_map(t_values, output_path, title="Pattern Shift"):
    """Plot t-values on fsaverage6 surface (4 views, no thresholding)."""
    print(f"  Plotting surface map...")

    masker = NiftiLabelsMasker(labels_img=SCHAEFER_ATLAS['maps'])
    masker.fit()

    t_map_img = masker.inverse_transform(t_values.reshape(1, -1))

    texture_left = surface.vol_to_surf(t_map_img, FSAVERAGE['pial_left'])
    texture_right = surface.vol_to_surf(t_map_img, FSAVERAGE['pial_right'])

    vmax = np.max(np.abs(t_values[~np.isnan(t_values)]))

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
            cmap='coolwarm', threshold=None, axes=ax,
            colorbar=False, vmax=vmax, bg_on_data=False,
        )
        ax.set_title(f"{hemi.upper()} {view.capitalize()}", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1, shrink=0.5)
    cbar.set_label('t-value', fontsize=10)

    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold', y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_top_parcels(results, parcel_stats, output_path, task_name, cond1_label, cond2_label, n_top=20):
    """Plot time courses for top N parcels by t-value (from window-averaged signal)."""
    t_values = parcel_stats['t_values']
    time_vec = parcel_stats['time_vec']
    cond1_stack = parcel_stats['cond1_stack']  # (n_sessions, n_parcels, n_timepoints)
    cond2_stack = parcel_stats['cond2_stack']

    # Get top parcels by t-value (highest positive first)
    top_idx = np.argsort(t_values)[::-1][:n_top]

    n_cols = 4
    n_rows = int(np.ceil(n_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, parcel_idx in enumerate(top_idx):
        ax = axes[i]

        # Mean and SEM across sessions
        cond1_mean = np.nanmean(cond1_stack[:, parcel_idx, :], axis=0)
        cond2_mean = np.nanmean(cond2_stack[:, parcel_idx, :], axis=0)
        cond1_sem = stats.sem(cond1_stack[:, parcel_idx, :], axis=0, nan_policy='omit')
        cond2_sem = stats.sem(cond2_stack[:, parcel_idx, :], axis=0, nan_policy='omit')

        ax.plot(time_vec, cond1_mean, color=COLORS['switch'], linewidth=2, label=cond1_label)
        ax.fill_between(time_vec, cond1_mean - cond1_sem, cond1_mean + cond1_sem,
                        color=COLORS['switch'], alpha=0.3)

        ax.plot(time_vec, cond2_mean, color=COLORS['cluster'], linewidth=2, label=cond2_label)
        ax.fill_between(time_vec, cond2_mean - cond2_sem, cond2_mean + cond2_sem,
                        color=COLORS['cluster'], alpha=0.3)

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        # Shade the HRF window used for t-value computation
        ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.1, color='yellow')

        # Title with parcel name and t-value
        label = SCHAEFER_LABELS[parcel_idx]
        parts = label.split('_')
        short_label = f"{parts[1]}_{parts[2]}_{parts[3]}" if len(parts) >= 4 else label
        ax.set_title(f"{short_label}\nt={t_values[parcel_idx]:.2f}", fontsize=9)

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time (s)', fontsize=10)
        if i % n_cols == 0:
            ax.set_ylabel('Pattern shift', fontsize=10)

        ax.set_ylim(Y_LIM)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused axes
    for i in range(n_top, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{task_name}: Top {n_top} Parcels by t-value ({cond1_label} > {cond2_label}, avg in {HRF_DELAY}-{HRF_DELAY + WINDOW_DURATION}s)',
                 fontsize=TITLE_FONTSIZE, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def compute_t_values_in_window(cond1_locked, cond2_locked, time_vec):
    """
    Compute t-values based on average signal in HRF-adjusted window.

    For each parcel:
    1. Average the signal within the HRF window (4.5s to 10.5s post-event) for each event
    2. Compute t-value comparing these averaged values between conditions

    Returns
    -------
    t_values : ndarray
        Shape (n_parcels,) - t-value for each parcel based on window-averaged signal
    """
    n_events_c1, n_parcels, n_timepoints = cond1_locked.shape
    n_events_c2 = cond2_locked.shape[0]

    # Find indices for HRF window (4.5s to 10.5s)
    window_start = HRF_DELAY
    window_end = HRF_DELAY + WINDOW_DURATION
    t_start_idx = np.searchsorted(time_vec, window_start)
    t_end_idx = np.searchsorted(time_vec, window_end, side='right')

    # Compute t-values based on average signal in window
    t_values = np.zeros(n_parcels)

    for p in range(n_parcels):
        # Average signal in HRF window for each event
        c1_window_avg = np.nanmean(cond1_locked[:, p, t_start_idx:t_end_idx], axis=1)  # (n_events_c1,)
        c2_window_avg = np.nanmean(cond2_locked[:, p, t_start_idx:t_end_idx], axis=1)  # (n_events_c2,)

        # Remove NaN values
        c1_window_avg = c1_window_avg[~np.isnan(c1_window_avg)]
        c2_window_avg = c2_window_avg[~np.isnan(c2_window_avg)]

        if len(c1_window_avg) > 1 and len(c2_window_avg) > 1:
            t_stat, _ = stats.ttest_ind(c1_window_avg, c2_window_avg)
            t_values[p] = t_stat

    return t_values


def plot_session_top_parcels(result, output_path, task_name, cond1_label, cond2_label, n_top=20):
    """Plot time courses for top N parcels for a single session with SEM shading."""
    cond1_locked = result['cond1_locked']  # (n_events, n_parcels, n_timepoints)
    cond2_locked = result['cond2_locked']
    cond1_mean = result['cond1_mean']  # (n_parcels, n_timepoints)
    cond2_mean = result['cond2_mean']
    time_vec = result['time_vec']
    subject = result['subject']
    session = result['session']

    # Compute SEM across events
    cond1_sem = stats.sem(cond1_locked, axis=0, nan_policy='omit')  # (n_parcels, n_timepoints)
    cond2_sem = stats.sem(cond2_locked, axis=0, nan_policy='omit')

    # Compute t-values based on average signal in HRF window
    t_values = compute_t_values_in_window(cond1_locked, cond2_locked, time_vec)

    # Get top parcels by t-value (highest positive first)
    top_idx = np.argsort(t_values)[::-1][:n_top]

    n_cols = 4
    n_rows = int(np.ceil(n_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, parcel_idx in enumerate(top_idx):
        ax = axes[i]

        # Plot with SEM shading
        ax.plot(time_vec, cond1_mean[parcel_idx, :], color=COLORS['switch'], linewidth=2, label=cond1_label)
        ax.fill_between(time_vec,
                        cond1_mean[parcel_idx, :] - cond1_sem[parcel_idx, :],
                        cond1_mean[parcel_idx, :] + cond1_sem[parcel_idx, :],
                        color=COLORS['switch'], alpha=0.3)

        ax.plot(time_vec, cond2_mean[parcel_idx, :], color=COLORS['cluster'], linewidth=2, label=cond2_label)
        ax.fill_between(time_vec,
                        cond2_mean[parcel_idx, :] - cond2_sem[parcel_idx, :],
                        cond2_mean[parcel_idx, :] + cond2_sem[parcel_idx, :],
                        color=COLORS['cluster'], alpha=0.3)

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        # Shade the HRF window used for t-value computation
        ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.1, color='yellow')

        label = SCHAEFER_LABELS[parcel_idx]
        parts = label.split('_')
        short_label = f"{parts[1]}_{parts[2]}_{parts[3]}" if len(parts) >= 4 else label
        ax.set_title(f"{short_label}\nt={t_values[parcel_idx]:.2f}", fontsize=9)

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time (s)', fontsize=10)
        if i % n_cols == 0:
            ax.set_ylabel('Pattern shift', fontsize=10)

        ax.set_ylim(Y_LIM)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    for i in range(n_top, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{task_name}: {subject} {session} - Top {n_top} Parcels ({cond1_label} > {cond2_label}, avg in {HRF_DELAY}-{HRF_DELAY + WINDOW_DURATION}s)',
                 fontsize=TITLE_FONTSIZE, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_session_network_timecourses(result, output_path, task_name, cond1_label, cond2_label):
    """Plot network time courses for a single session with SEM shading."""
    cond1_locked = result['cond1_locked']  # (n_events, n_parcels, n_timepoints)
    cond2_locked = result['cond2_locked']
    time_vec = result['time_vec']
    subject = result['subject']
    session = result['session']

    n_networks = len(YEO_17_NETWORKS)
    n_cols = 4
    n_rows = int(np.ceil(n_networks / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    plot_idx = 0
    for network in YEO_17_NETWORKS:
        parcel_idx = [i for i, n in enumerate(PARCEL_NETWORKS) if n == network]

        if len(parcel_idx) == 0:
            continue

        ax = axes[plot_idx]

        # Average across parcels in network for each event, then compute mean/SEM across events
        cond1_net_events = cond1_locked[:, parcel_idx, :].mean(axis=1)  # (n_events, n_timepoints)
        cond2_net_events = cond2_locked[:, parcel_idx, :].mean(axis=1)

        cond1_mean = np.nanmean(cond1_net_events, axis=0)
        cond2_mean = np.nanmean(cond2_net_events, axis=0)
        cond1_sem = stats.sem(cond1_net_events, axis=0, nan_policy='omit')
        cond2_sem = stats.sem(cond2_net_events, axis=0, nan_policy='omit')

        ax.plot(time_vec, cond1_mean, color=COLORS['switch'], linewidth=2, label=cond1_label)
        ax.fill_between(time_vec, cond1_mean - cond1_sem, cond1_mean + cond1_sem,
                        color=COLORS['switch'], alpha=0.3)

        ax.plot(time_vec, cond2_mean, color=COLORS['cluster'], linewidth=2, label=cond2_label)
        ax.fill_between(time_vec, cond2_mean - cond2_sem, cond2_mean + cond2_sem,
                        color=COLORS['cluster'], alpha=0.3)

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        ax.set_title(f"{network} (n={len(parcel_idx)})", fontsize=10)

        if plot_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time (s)', fontsize=10)
        if plot_idx % n_cols == 0:
            ax.set_ylabel('Pattern shift', fontsize=10)

        ax.set_ylim(Y_LIM)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{task_name}: {subject} {session} - Network-Level Pattern Shifts',
                 fontsize=TITLE_FONTSIZE, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_session_surface(result, output_path, task_name, cond1_label, cond2_label):
    """Plot surface map of t-values for a single session (avg signal in HRF window)."""
    cond1_locked = result['cond1_locked']  # (n_events, n_parcels, n_timepoints)
    cond2_locked = result['cond2_locked']
    time_vec = result['time_vec']
    subject = result['subject']
    session = result['session']

    # Compute t-values based on average signal in HRF window
    t_values = compute_t_values_in_window(cond1_locked, cond2_locked, time_vec)

    plot_surface_map(t_values, output_path,
                     title=f"{task_name}: {subject} {session}\n{cond1_label} > {cond2_label} (avg in {HRF_DELAY}-{HRF_DELAY + WINDOW_DURATION}s)")


def plot_network_timecourses(network_data, time_vec, output_path, task_name, cond1_label, cond2_label):
    """Plot time courses for each Yeo 17 network."""
    n_networks = len(network_data)
    n_cols = 4
    n_rows = int(np.ceil(n_networks / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, (network, data) in enumerate(network_data.items()):
        ax = axes[i]

        ax.plot(time_vec, data['cond1_mean'], color=COLORS['switch'], linewidth=2, label=cond1_label)
        ax.fill_between(time_vec,
                        data['cond1_mean'] - data['cond1_sem'],
                        data['cond1_mean'] + data['cond1_sem'],
                        color=COLORS['switch'], alpha=0.3)

        ax.plot(time_vec, data['cond2_mean'], color=COLORS['cluster'], linewidth=2, label=cond2_label)
        ax.fill_between(time_vec,
                        data['cond2_mean'] - data['cond2_sem'],
                        data['cond2_mean'] + data['cond2_sem'],
                        color=COLORS['cluster'], alpha=0.3)

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        ax.set_title(f"{network} (n={data['n_parcels']})", fontsize=10)

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time (s)', fontsize=10)
        if i % n_cols == 0:
            ax.set_ylabel('Pattern shift', fontsize=10)

        ax.set_ylim(Y_LIM)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused axes
    for i in range(n_networks, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{task_name}: Network-Level Pattern Shifts\n({cond1_label} vs {cond2_label})',
                 fontsize=TITLE_FONTSIZE, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# GROUP-LEVEL ANALYSIS
# ============================================================================

def plot_group_surface_contrast(results, output_path, task_name, cond1_label, cond2_label,
                                 vmax=SURFACE_VMAX):
    """
    Compute group-level t-values and plot on surface.

    For each session, computes t-value based on average signal in HRF window.
    Then performs one-sample t-test across sessions.
    No threshold is applied (all values shown).
    """
    print(f"  Computing group-level surface contrast...")

    n_sessions = len(results)
    time_vec = results[0]['time_vec']

    # Compute t-values for each session
    session_t_values = []
    for r in results:
        t_vals = compute_t_values_in_window(r['cond1_locked'], r['cond2_locked'], time_vec)
        session_t_values.append(t_vals)

    t_matrix = np.array(session_t_values)  # (n_sessions, n_parcels)
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

    # Use no threshold (show all values)
    t_threshold = None

    # Report summary statistics
    n_sig_uncorrected = np.sum(group_p < 0.05)
    print(f"  Significant parcels (p < 0.05 uncorrected): {n_sig_uncorrected}")

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

    fig.suptitle(f"{task_name}: {cond1_label} > {cond2_label} (N={n_sessions} sessions)",
                 fontsize=TITLE_FONTSIZE, fontweight='bold', y=1.05)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")

    return group_t, group_p, t_threshold


def plot_group_network_timecourses(results, output_path, task_name, cond1_label, cond2_label, q=FDR_Q):
    """
    Plot group-level network time courses with FDR-corrected statistics.

    Averages across sessions and uses paired t-tests at each timepoint.
    """
    n_sessions = len(results)
    time_vec = results[0]['time_vec']
    n_timepoints = len(time_vec)

    n_networks = len(YEO_17_NETWORKS)
    n_cols = 4
    n_rows = int(np.ceil(n_networks / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    plot_idx = 0
    for network in YEO_17_NETWORKS:
        parcel_idx = [i for i, n in enumerate(PARCEL_NETWORKS) if n == network]

        if len(parcel_idx) == 0:
            continue

        ax = axes[plot_idx]

        # For each session, average across parcels in network, then average across events
        cond1_session_means = []  # (n_sessions, n_timepoints)
        cond2_session_means = []

        for r in results:
            # r['cond1_mean'] is (n_parcels, n_timepoints), already averaged across events
            cond1_net = r['cond1_mean'][parcel_idx, :].mean(axis=0)  # (n_timepoints,)
            cond2_net = r['cond2_mean'][parcel_idx, :].mean(axis=0)
            cond1_session_means.append(cond1_net)
            cond2_session_means.append(cond2_net)

        cond1_stack = np.array(cond1_session_means)  # (n_sessions, n_timepoints)
        cond2_stack = np.array(cond2_session_means)

        # Compute group mean and SEM
        cond1_mean = np.nanmean(cond1_stack, axis=0)
        cond1_sem = np.nanstd(cond1_stack, axis=0) / np.sqrt(n_sessions)
        cond2_mean = np.nanmean(cond2_stack, axis=0)
        cond2_sem = np.nanstd(cond2_stack, axis=0) / np.sqrt(n_sessions)

        # Paired t-tests at each time point
        pvalues = []
        for t in range(n_timepoints):
            _, p = stats.ttest_rel(cond1_stack[:, t], cond2_stack[:, t])
            pvalues.append(p)
        pvalues = np.array(pvalues)

        # Plot
        ax.plot(time_vec, cond1_mean, color=COLORS['switch'], linewidth=2,
                label=cond1_label, marker='o', markersize=3)
        ax.fill_between(time_vec, cond1_mean - cond1_sem, cond1_mean + cond1_sem,
                        color=COLORS['switch'], alpha=0.3)

        ax.plot(time_vec, cond2_mean, color=COLORS['cluster'], linewidth=2,
                label=cond2_label, marker='o', markersize=3)
        ax.fill_between(time_vec, cond2_mean - cond2_sem, cond2_mean + cond2_sem,
                        color=COLORS['cluster'], alpha=0.3)

        # Mark significant time points (uncorrected p < 0.05)
        sig_indices = np.where(pvalues < 0.05)[0]
        if len(sig_indices) > 0:
            y_pos = Y_LIM[0] + 0.05 * (Y_LIM[1] - Y_LIM[0])
            for idx in sig_indices:
                ax.text(time_vec[idx], y_pos, '*', fontsize=10, ha='center', fontweight='bold')

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax.axvspan(HRF_DELAY, HRF_DELAY + WINDOW_DURATION, alpha=0.1, color='yellow')

        ax.set_title(f"{network} (n={len(parcel_idx)})", fontsize=10)

        if plot_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time (s)', fontsize=10)
        if plot_idx % n_cols == 0:
            ax.set_ylabel('Pattern shift', fontsize=10)

        ax.set_ylim(Y_LIM)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{task_name}: Group Network-Level Pattern Shifts (N={n_sessions} sessions)\n'
                 f'{cond1_label} vs {cond2_label}, * p < 0.05 uncorrected',
                 fontsize=TITLE_FONTSIZE, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def run_pattern_shift_analysis(task='svf', n_jobs=1, subject_filter=None):
    """Run pattern shift analysis for a task."""
    if task == 'svf':
        cond1_label, cond2_label = 'Switch', 'Cluster'
    else:
        cond1_label, cond2_label = 'Boundary', 'Non-boundary'

    print("\n" + "=" * 60)
    print(f"{task.upper()} PATTERN SHIFT ANALYSIS: {cond1_label} vs {cond2_label}")
    print("=" * 60)

    sessions = get_all_sessions(task)

    if subject_filter:
        sessions = [(sub, ses) for sub, ses in sessions if sub == subject_filter]

    print(f"Found {len(sessions)} {task.upper()} sessions")

    if len(sessions) == 0:
        print("No sessions found!")
        return None

    # Process sessions
    if n_jobs == 1:
        results = [run_session(sub, ses, task=task) for sub, ses in sessions]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(run_session)(sub, ses, task) for sub, ses in sessions
        )

    results = [r for r in results if r is not None]
    print(f"\nSuccessfully processed {len(results)} sessions")

    if len(results) == 0:
        return None

    # Cache results
    suffix = f"_{subject_filter}" if subject_filter else ""
    cache_path = CACHE_DIR / f"{task}_pattern_shift_results{suffix}.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Cached results to {cache_path}")

    # Generate per-session plots
    print("\nGenerating per-session plots...")

    for r in results:
        subject = r['subject']
        session = r['session']
        prefix = f"{subject}_{session}_{task}"

        # 1. Top 20 parcels for this session
        top_path = PATTERN_SHIFT_FIGS_DIR / f"{prefix}_top20_parcels.png"
        plot_session_top_parcels(r, top_path, task.upper(), cond1_label, cond2_label, n_top=20)

        # 2. Network time courses for this session
        network_path = PATTERN_SHIFT_FIGS_DIR / f"{prefix}_network_timecourses.png"
        plot_session_network_timecourses(r, network_path, task.upper(), cond1_label, cond2_label)

        # 3. Surface map for this session
        surface_path = PATTERN_SHIFT_FIGS_DIR / f"{prefix}_surface.png"
        plot_session_surface(r, surface_path, task.upper(), cond1_label, cond2_label)

    # Generate group-level plots
    if len(results) > 1:
        print("\nGenerating group-level plots...")
        prefix = f"{subject_filter}_" if subject_filter else ""

        # 1. T-value distribution
        parcel_stats = compute_parcel_statistics(results, cond1_label, cond2_label)
        dist_path = PATTERN_SHIFT_FIGS_DIR / f"{prefix}GROUP_{task}_t_distribution.png"
        plot_t_distribution(parcel_stats, dist_path, task.upper(), cond1_label, cond2_label)

        # 2. Group surface map
        surface_path = PATTERN_SHIFT_FIGS_DIR / f"{prefix}GROUP_{task}_surface.png"
        plot_group_surface_contrast(results, surface_path, task.upper(), cond1_label, cond2_label)

        # 3. Group network time courses
        network_path = PATTERN_SHIFT_FIGS_DIR / f"{prefix}GROUP_{task}_network_timecourses.png"
        plot_group_network_timecourses(results, network_path, task.upper(), cond1_label, cond2_label)

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_group_analysis_from_cache(task, subject_filter=None):
    """Run group-level analysis using cached session results."""
    if task == 'svf':
        cond1_label, cond2_label = 'Switch', 'Cluster'
    else:
        cond1_label, cond2_label = 'Boundary', 'Non-boundary'

    suffix = f"_{subject_filter}" if subject_filter else ""
    cache_path = CACHE_DIR / f"{task}_pattern_shift_results{suffix}.pkl"

    if not cache_path.exists():
        print(f"No cached results found at {cache_path}")
        return None

    print(f"\nLoading cached results from {cache_path}...")
    with open(cache_path, 'rb') as f:
        results = pickle.load(f)

    print(f"Loaded {len(results)} cached {task.upper()} sessions")

    if len(results) < 2:
        print("Need at least 2 sessions for group analysis")
        return None

    print("\nGenerating group-level plots...")
    prefix = f"{subject_filter}_" if subject_filter else ""

    # 1. T-value distribution
    parcel_stats = compute_parcel_statistics(results, cond1_label, cond2_label)
    dist_path = PATTERN_SHIFT_FIGS_DIR / f"{prefix}GROUP_{task}_t_distribution.png"
    plot_t_distribution(parcel_stats, dist_path, task.upper(), cond1_label, cond2_label)

    # 2. Group surface map
    surface_path = PATTERN_SHIFT_FIGS_DIR / f"{prefix}GROUP_{task}_surface.png"
    plot_group_surface_contrast(results, surface_path, task.upper(), cond1_label, cond2_label)

    # 3. Group network time courses
    network_path = PATTERN_SHIFT_FIGS_DIR / f"{prefix}GROUP_{task}_network_timecourses.png"
    plot_group_network_timecourses(results, network_path, task.upper(), cond1_label, cond2_label)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pattern Shift Analysis for SVF and AHC Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--all", action="store_true",
                        help="Run all analyses")
    parser.add_argument("--svf", action="store_true",
                        help="Run SVF pattern shift analysis only")
    parser.add_argument("--ahc", action="store_true",
                        help="Run AHC pattern shift analysis only")
    parser.add_argument("--group-only", action="store_true",
                        help="Run group analysis only using cached results")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs (default: 1 for memory efficiency)")
    parser.add_argument("--subject", type=str, default=None,
                        help="Run analysis for a single subject (e.g., sub-003)")

    args = parser.parse_args()

    if not any([args.all, args.svf, args.ahc, args.group_only]):
        args.all = True

    print("=" * 60)
    print("PATTERN SHIFT ANALYSIS: SVF & AHC")
    print(f"Window size: {WINDOW_SIZE} TRs ({WINDOW_SIZE * TR:.1f}s)")
    print(f"Event window: -{TRS_BEFORE} to +{TRS_AFTER} TRs")
    if args.subject:
        print(f"Subject filter: {args.subject}")
    if args.group_only:
        print("Mode: GROUP-ONLY (using cached results)")
    print(f"Output directory: {PATTERN_SHIFT_FIGS_DIR}")
    print("=" * 60)

    if args.group_only:
        # Load cached results and run group analysis only
        if args.all or args.svf:
            run_group_analysis_from_cache(task='svf', subject_filter=args.subject)
        if args.all or args.ahc:
            run_group_analysis_from_cache(task='ahc', subject_filter=args.subject)
    else:
        # Run full analysis
        if args.all or args.svf:
            run_pattern_shift_analysis(task='svf', n_jobs=args.n_jobs, subject_filter=args.subject)

        if args.all or args.ahc:
            run_pattern_shift_analysis(task='ahc', n_jobs=args.n_jobs, subject_filter=args.subject)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {PATTERN_SHIFT_FIGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
