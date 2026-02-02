#!/usr/bin/env python3
"""
First-level GLM analysis for fMRI data using Schaefer parcellation.
Runs a contrast analysis for Switching > Clustering.

Strict filtering:
- Switch: Only keep the FIRST switch in a sequence of consecutive switches
- Cluster: Only keep clustering words where the NEXT word is also a clustering word

Supports parallel processing across multiple sessions.
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
from nilearn.glm.first_level import make_first_level_design_matrix
import statsmodels.api as sm
from joblib import Parallel, delayed
import multiprocessing

# === IMPORT PROJECT CONFIG ===
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
GLM_FIGS_DIR = FIGS_DIR / "fmri_glm"
GLM_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0 
HIGH_PASS_HZ = 0.01

# Pre-fetch atlases to avoid race conditions during parallel processing
print("Pre-fetching atlases...")
SCHAEFER_ATLAS = datasets.fetch_atlas_schaefer_2018(
    n_rois=400, 
    yeo_networks=17,
    resolution_mm=2
)
FSAVERAGE = datasets.fetch_surf_fsaverage('fsaverage6')
print("Atlases loaded.")


def get_events_dataframe(subject, session):
    """
    Loads behavioral CSV and formats it for nilearn design matrix.
    
    Strict filtering:
    - Switch: Only keep the FIRST switch in a sequence of consecutive switches
             (i.e., exclude switches where the preceding word was also a switch)
    - Cluster: Only keep clustering words where the NEXT word is also a clustering word
              (i.e., exclude the last word of each cluster)
    """
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_transcription.csv"
    
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*.csv"))
        if candidates:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(f"No behavioral file found at: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # 1. Ensure chronological order
    df = df.sort_values("start").reset_index(drop=True)
    
    # 2. Clean switch_flag (fill NaNs with 0, ensure int)
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)
    
    # 3. Filter out "next" words
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    df = df.reset_index(drop=True)
    
    # 4. Capture Preceding and Following Info (for strict filtering)
    df["preceding_switch_flag"] = df["switch_flag"].shift(1)
    df["following_switch_flag"] = df["switch_flag"].shift(-1)
    
    # 5. Drop the first item of the session (no preceding verbal response)
    df = df.iloc[1:].copy()
    
    # 6. Apply STRICT filtering
    
    # --- SWITCH FILTER ---
    # A switch is valid ONLY if it's the FIRST in a sequence of switches
    # i.e., the preceding word was NOT a switch (was a cluster, switch_flag=0)
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["preceding_switch_flag"] == 1
    
    # Valid switch = is a switch AND preceding was NOT a switch
    valid_switch = is_switch & (~prev_was_switch)
    
    # Invalid switches (not the first in sequence)
    invalid_switch_mask = is_switch & prev_was_switch
    n_invalid_switches = invalid_switch_mask.sum()
    
    # --- CLUSTER FILTER ---
    # A cluster word is valid ONLY if the NEXT word is also a cluster word
    # i.e., exclude the last word of each cluster (which is followed by a switch or end)
    is_cluster = df["switch_flag"] == 0
    next_is_cluster = df["following_switch_flag"] == 0
    
    # Valid cluster = is a cluster AND next word is also a cluster
    valid_cluster = is_cluster & (next_is_cluster == True)
    
    # Invalid clusters (last word of cluster, followed by switch or NaN)
    invalid_cluster_mask = is_cluster & ((next_is_cluster == False) | df["following_switch_flag"].isna())
    n_invalid_clusters = invalid_cluster_mask.sum()
    
    # 7. Create filtered dataframe
    # Keep only valid switches and valid clusters
    valid_events_mask = valid_switch | valid_cluster
    
    print(f"  [{subject} {session}] Original: {is_switch.sum()} switches, {is_cluster.sum()} clusters")
    print(f"  [{subject} {session}] Excluded: {n_invalid_switches} switches (not first in sequence), "
          f"{n_invalid_clusters} clusters (last in cluster)")
    
    df_filtered = df[valid_events_mask].copy()
    
    print(f"  [{subject} {session}] Remaining: {valid_switch.sum()} switches, {valid_cluster.sum()} clusters")
    
    # 8. Map Conditions
    condition_map = {1: "Switch", 0: "Cluster"}
    df_filtered["trial_type"] = df_filtered["switch_flag"].map(condition_map)
    
    # 9. Calculate scanner-aligned onset (at word offset)
    df_filtered["onset"] = df_filtered["end"] - SCANNER_START_OFFSET
    df_filtered["duration"] = 0.0  # Impulse events
    
    # 10. Remove events before scanner start
    df_filtered = df_filtered[df_filtered["onset"] >= 0]
    
    # Final count
    n_switch_final = (df_filtered["trial_type"] == "Switch").sum()
    n_cluster_final = (df_filtered["trial_type"] == "Cluster").sum()
    print(f"  [{subject} {session}] Final (after timing filter): {n_switch_final} switches, {n_cluster_final} clusters")
    
    return df_filtered[["onset", "duration", "trial_type"]]


def extract_roi_time_series(bold_path):
    """Extract time series from Schaefer atlas ROIs using pre-fetched atlas"""
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


def plot_contrast_on_fsaverage(t_map_img, subject, session, output_path):
    """
    Projects volumetric t-map to fsaverage6 and plots 4 views (L/R x Lat/Med).
    Uses pre-fetched fsaverage surfaces.
    """
    # Project volume to surface
    texture_left = surface.vol_to_surf(t_map_img, FSAVERAGE.pial_left)
    texture_right = surface.vol_to_surf(t_map_img, FSAVERAGE.pial_right)
    
    # Setup Figure (1x4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'projection': '3d'})
    
    # Define the 4 views
    views = [
        (texture_left, FSAVERAGE.infl_left, FSAVERAGE.sulc_left, 'left', 'lateral'),
        (texture_left, FSAVERAGE.infl_left, FSAVERAGE.sulc_left, 'left', 'medial'),
        (texture_right, FSAVERAGE.infl_right, FSAVERAGE.sulc_right, 'right', 'lateral'),
        (texture_right, FSAVERAGE.infl_right, FSAVERAGE.sulc_right, 'right', 'medial'),
    ]
    
    # Common plotting parameters
    vmax = np.nanmax(np.abs([texture_left, texture_right]))
    if vmax < 2: vmax = 3 
    
    for ax, (tex, mesh, bg, hemi, view) in zip(axes, views):
        plotting.plot_surf_stat_map(
            mesh, tex, 
            hemi=hemi, 
            bg_map=bg,
            view=view,
            cmap='cold_hot', 
            threshold=1.96,
            axes=ax,
            colorbar=False, 
            vmax=vmax,
            bg_on_data=True,
            darkness=0.5
        )
        ax.set_title(f"{hemi.upper()} {view.capitalize()}", fontsize=10)
    
    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap='cold_hot', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label('t-statistic')
    
    fig.suptitle(f"{subject} {session}: Switching > Clustering (Strict Filter)", fontsize=14, y=1.05)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [{subject} {session}] Saved figure to {output_path}")


def run_subject_level_glm(subject, session, task_name="svf"):
    """
    Main function to run ROI-based GLM.
    
    Returns: tuple (subject, session, status, message)
    """
    print(f"\n=== Processing {subject} {session} ===")
    
    # 1. Locate BOLD file with specific MNI152NLin6Asym_res-2 space
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    
    search_pattern = f"*{task_name}*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(search_pattern))
    
    if not bold_files:
        msg = f"No BOLD file found matching '{search_pattern}' in {func_dir}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "skipped", msg)
    bold_path = bold_files[0]
    
    # 2. Get Events FIRST (fail fast if no valid events)
    try:
        events_df = get_events_dataframe(subject, session)
    except FileNotFoundError as e:
        msg = f"No annotation file found: {e}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "skipped", msg)
    except Exception as e:
        msg = f"Error loading events: {e}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "error", msg)
    
    # Check if we have enough events
    n_switches = (events_df["trial_type"] == "Switch").sum()
    n_clusters = (events_df["trial_type"] == "Cluster").sum()
    
    if n_switches < 2 or n_clusters < 2:
        msg = f"Insufficient events after strict filtering (switches: {n_switches}, clusters: {n_clusters})"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "skipped", msg)
    
    # 3. Extract Data
    try:
        roi_signals = extract_roi_time_series(bold_path)
    except Exception as e:
        msg = f"Error extracting ROIs: {e}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "error", msg)

    n_scans = roi_signals.shape[0]
    frame_times = np.arange(n_scans) * TR
    
    # 4. Create Design Matrix
    try:
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events_df,
            hrf_model='spm',
            drift_model=None
        )
    except Exception as e:
        msg = f"Error creating design matrix: {e}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "error", msg)
    
    if "Switch" not in design_matrix.columns or "Cluster" not in design_matrix.columns:
        msg = f"Missing conditions. Found: {list(design_matrix.columns)}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "error", msg)

    # 5. Fit GLM
    print(f"  [{subject} {session}] Fitting GLM on 400 ROIs...")
    t_values = []
    contrast_vector = np.zeros(design_matrix.shape[1])
    
    try:
        contrast_vector[design_matrix.columns.get_loc("Switch")] = 1
        contrast_vector[design_matrix.columns.get_loc("Cluster")] = -1
    except KeyError as e:
        msg = f"Error setting up contrast: {e}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "error", msg)
    
    X = design_matrix.values
    for i in range(roi_signals.shape[1]):
        Y = roi_signals[:, i]
        model = sm.OLS(Y, X).fit()
        t_test = model.t_test(contrast_vector)
        t_values.append(t_test.tvalue.item())
        
    t_values = np.array(t_values)
    
    # 6. Visualization on fsaverage
    masker = NiftiLabelsMasker(labels_img=SCHAEFER_ATLAS['maps'])
    masker.fit()
    t_map_img = masker.inverse_transform(t_values.reshape(1, -1))
    
    out_file = GLM_FIGS_DIR / f"{subject}_{session}_contrast_strict_fsaverage.png"
    plot_contrast_on_fsaverage(t_map_img, subject, session, out_file)
    
    return (subject, session, "success", f"Output: {out_file}")


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
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    success = [r for r in results if r[2] == "success"]
    skipped = [r for r in results if r[2] == "skipped"]
    errors = [r for r in results if r[2] == "error"]
    
    print(f"\nTotal sessions processed: {len(results)}")
    print(f"  ✓ Success: {len(success)}")
    print(f"  ○ Skipped: {len(skipped)}")
    print(f"  ✗ Errors:  {len(errors)}")
    
    if skipped:
        print("\nSkipped sessions:")
        for sub, ses, _, msg in skipped:
            print(f"  - {sub} {ses}: {msg}")
    
    if errors:
        print("\nFailed sessions:")
        for sub, ses, _, msg in errors:
            print(f"  - {sub} {ses}: {msg}")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ROI GLM with strict event filtering (parallelized)")
    parser.add_argument("--sub", help="Subject ID (e.g., sub-008)")
    parser.add_argument("--ses", help="Session ID (e.g., ses-01)")
    parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions found in derivatives")
    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=-1, 
        help="Number of parallel jobs. -1 uses all CPUs, -2 uses all but one. Default: -1"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="loky",
        choices=["loky", "multiprocessing", "threading"],
        help="Joblib backend. Default: loky"
    )
    
    args = parser.parse_args()
    
    if args.all:
        sessions = get_all_sessions()
        n_sessions = len(sessions)
        
        if n_sessions == 0:
            print("No sessions found in derivatives directory.")
            sys.exit(1)
        
        # Determine number of jobs
        n_jobs = args.n_jobs
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        elif n_jobs == -2:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
        
        print(f"\nRunning parallel analysis for {n_sessions} sessions using {n_jobs} workers...")
        print(f"Backend: {args.backend}")
        print("-" * 60)
        
        # Run in parallel
        results = Parallel(n_jobs=n_jobs, backend=args.backend, verbose=10)(
            delayed(run_subject_level_glm)(sub, ses) for sub, ses in sessions
        )
        
        # Print summary
        print_summary(results)
        
    else:
        if not args.sub or not args.ses:
            print("Error: Must provide --sub and --ses OR use --all")
            sys.exit(1)
        result = run_subject_level_glm(args.sub, args.ses)
        print(f"\nResult: {result[2]} - {result[3]}")