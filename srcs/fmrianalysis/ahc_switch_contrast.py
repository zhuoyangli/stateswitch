#!/usr/bin/env python3
"""
First-level GLM analysis for fMRI data using Schaefer parcellation.
Runs a contrast analysis for Across-Possibility > Within-Possibility transitions (AHC task).
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
ANNOTATIONS_DIR = DATA_DIR / "rec/ahc_sentences"
GLM_FIGS_DIR = FIGS_DIR / "fmri_glm_ahc"
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
    Loads AHC behavioral data (Excel) and formats it for nilearn design matrix.
    
    Contrast logic:
    - Within-Possibility: Consecutive sentences with same Possibility Number
    - Across-Possibility: Consecutive sentences with different Possibility Number
    
    Events are modeled as impulses at sentence OFFSET (End Time).
    Transitions across prompts are excluded.
    """
    # Try to find the Excel file
    xlsx_path = ANNOTATIONS_DIR / f"{subject}_{session}_task-ahc_desc-sentences.xlsx"
    
    if not xlsx_path.exists():
        # Try alternative patterns
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*ahc*.xlsx"))
        if candidates:
            xlsx_path = candidates[0]
        else:
            raise FileNotFoundError(f"No AHC behavioral file found at: {xlsx_path}")

    # Load Excel file
    df = pd.read_excel(xlsx_path)
    
    # DEBUG: Print initial load info
    print(f"  [{subject} {session}] Loaded {len(df)} rows from {xlsx_path.name}")
    
    # Standardize column names (handle potential variations)
    df.columns = df.columns.str.strip()
    
    # Expected columns: Prompt Number, Segment Number, Text, Start Time, End Time, Possibility Number
    required_cols = ['Prompt Number', 'Start Time', 'End Time', 'Possibility Number']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # DEBUG: Print timing range
    print(f"  [{subject} {session}] Raw timing range: {df['Start Time'].min():.1f}s - {df['End Time'].max():.1f}s")

    df['Prompt Number'] = df['Prompt Number'].ffill()
    
    # 1. Sort by timing to ensure chronological order
    df = df.sort_values(['Prompt Number', 'Start Time']).reset_index(drop=True)
    
    # 2. Compute transition type within each prompt
    # Look at preceding sentence's Possibility Number (within same prompt only)
    df['Preceding_Possibility'] = df.groupby('Prompt Number')['Possibility Number'].shift(1)
    
    # 3. Determine if this is a switch (across-possibility) or cluster (within-possibility)
    # First sentence of each prompt has no preceding -> will be NaN
    df['is_switch'] = df['Possibility Number'] != df['Preceding_Possibility']
    
    # 4. Drop first sentence of each prompt (no transition to classify)
    n_before_drop = len(df)
    df = df.dropna(subset=['Preceding_Possibility']).copy()
    
    # DEBUG: Print drop info
    print(f"  [{subject} {session}] Dropped {n_before_drop - len(df)} first-sentence-of-prompt events")
    
    # 5. Map to condition labels
    df['trial_type'] = df['is_switch'].map({True: 'Across', False: 'Within'})
    
    # 6. Calculate scanner-aligned onset at sentence OFFSET (End Time)
    df['onset'] = df['End Time'] - SCANNER_START_OFFSET
    
    # 7. Model as impulse events (duration = 0)
    df['duration'] = 0.0
    
    # DEBUG: Check events before scanner start
    n_negative = (df['onset'] < 0).sum()
    if n_negative > 0:
        print(f"  [{subject} {session}] WARNING: {n_negative}/{len(df)} events have onset < 0 (before scanner start)")
    
    # 8. Remove events before scanner start
    df = df[df['onset'] >= 0].copy()
    
    # Print summary
    n_within = (df['trial_type'] == 'Within').sum()
    n_across = (df['trial_type'] == 'Across').sum()
    print(f"  [{subject} {session}] Events: {n_within} Within-Possibility, {n_across} Across-Possibility")
    
    return df[['onset', 'duration', 'trial_type']]


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
    roi_labels = [label for label in SCHAEFER_ATLAS['labels']]
    
    return time_series, roi_labels


def plot_contrast_on_fsaverage(t_map_img, subject, session, output_path, contrast_name="Across > Within"):
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
            threshold=1.96,  # p < 0.05 uncorrected
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
    
    fig.suptitle(f"{subject} {session}: {contrast_name}", fontsize=14, y=1.05)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [{subject} {session}] Saved figure to {output_path}")


def run_subject_level_glm(subject, session, task_name="ahc"):
    """
    Main function to run ROI-based GLM for AHC task.
    Contrast: Across-Possibility > Within-Possibility
    
    Returns: tuple (subject, session, status, message)
    """
    print(f"\n=== Processing {subject} {session} (AHC) ===")
    
    # 1. Locate BOLD file with specific MNI152NLin6Asym_res-2 space
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    
    search_pattern = f"*{task_name}*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(search_pattern))
    
    if not bold_files:
        msg = f"No BOLD file found matching '{search_pattern}' in {func_dir}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "skipped", msg)
    bold_path = bold_files[0]
    
    # 2. Extract Data
    try:
        roi_signals, roi_labels = extract_roi_time_series(bold_path)
    except Exception as e:
        msg = f"Error extracting ROIs: {e}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "error", msg)

    n_scans = roi_signals.shape[0]
    frame_times = np.arange(n_scans) * TR
    
    # 3. Get Events
    try:
        events_df = get_events_dataframe(subject, session)
        print(f"  [{subject} {session}] Loaded {len(events_df)} valid transition events")
    except FileNotFoundError as e:
        msg = f"No annotation file found: {e}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "skipped", msg)
    except Exception as e:
        msg = f"Error loading events: {e}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "error", msg)
    
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
    
    # Check for required conditions
    if "Across" not in design_matrix.columns or "Within" not in design_matrix.columns:
        msg = f"Missing conditions. Found: {list(design_matrix.columns)}"
        print(f"  [{subject} {session}] {msg}")
        return (subject, session, "error", msg)

    # 5. Fit GLM
    print(f"  [{subject} {session}] Fitting GLM on 400 ROIs...")
    t_values = []
    contrast_vector = np.zeros(design_matrix.shape[1])
    
    # Contrast: Across > Within (analogous to Switch > Cluster)
    try:
        contrast_vector[design_matrix.columns.get_loc("Across")] = 1
        contrast_vector[design_matrix.columns.get_loc("Within")] = -1
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
    
    out_file = GLM_FIGS_DIR / f"{subject}_{session}_ahc_across-vs-within_fsaverage.png"
    plot_contrast_on_fsaverage(
        t_map_img, subject, session, out_file, 
        contrast_name="Across-Possibility > Within-Possibility"
    )
    
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
    parser = argparse.ArgumentParser(description="Run ROI GLM for AHC task (parallelized)")
    parser.add_argument("--sub", help="Subject ID (e.g., sub-001)")
    parser.add_argument("--ses", help="Session ID (e.g., ses-05)")
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