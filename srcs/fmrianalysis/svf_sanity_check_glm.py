# #!/usr/bin/env python3
# """
# Sanity-Check GLM Analysis: Speech > Baseline.
# Runs a First-Level GLM on volumetric data and projects T-stats to fsaverage surface.
# """

# import sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import argparse
# import matplotlib.pyplot as plt
# from nilearn import datasets, surface, plotting
# from nilearn.glm.first_level import FirstLevelModel
# from nilearn.interfaces.fmriprep import load_confounds

# # === CONFIG SETUP ===
# current_file = Path(__file__).resolve()
# project_root = current_file.parent.parent
# sys.path.insert(0, str(project_root))

# try:
#     from configs.config import DATA_DIR, DERIVATIVES_DIR, FIGS_DIR, TR
# except ImportError:
#     print("Error: Could not import 'configs.config'. Ensure your directory structure is correct.")
#     sys.exit(1)

# # === PATH DEFINITIONS ===
# ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
# SANITY_FIGS_DIR = FIGS_DIR / "fmri_sanity_glm"
# SANITY_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# # === CONSTANTS ===
# SCANNER_START_OFFSET = 12.0
# HIGH_PASS_HZ = 0.01
# SMOOTHING_FWHM = 6.0 # Standard smoothing for voxel-wise GLM

# def get_events(subject, session):
#     """
#     Creates a simple event file for the GLM: "Speech" vs Baseline.
#     """
#     csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_transcription.csv"
#     if not csv_path.exists():
#         candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*.csv"))
#         if candidates:
#             csv_path = candidates[0]
#         else:
#             raise FileNotFoundError(f"No CSV for {subject} {session}")

#     df = pd.read_csv(csv_path)
#     df = df.sort_values("start").reset_index(drop=True)
    
#     # Filter "next" (we only want the actual word production)
#     df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
#     # Create simple event dataframe for nilearn
#     events = pd.DataFrame()
#     events["onset"] = df["start"] - SCANNER_START_OFFSET
#     events["duration"] = df["end"] - df["start"]
#     events["trial_type"] = "speech"
    
#     # Keep only valid events
#     events = events[events["onset"] >= 0]
    
#     return events

# def plot_surface_grid(t_map_img, subject, session):
#     """
#     Projects T-map to fsaverage6 and plots 1x4 grid.
#     """
#     print("  Projecting to fsaverage6 surface...")
#     fsaverage = datasets.fetch_surf_fsaverage('fsaverage6')
    
#     # Project Volume -> Surface
#     texture_left = surface.vol_to_surf(t_map_img, fsaverage.pial_left)
#     texture_right = surface.vol_to_surf(t_map_img, fsaverage.pial_right)
    
#     # Determine scale (robust max)
#     # vmax = np.nanpercentile(np.abs(np.concatenate([texture_left, texture_right])), 99.9)
#     # if vmax < 4: vmax = 4 # Minimum threshold for good visualization
#     vmax=10
    
#     # Plotting
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})
    
#     views = [
#         (texture_left, fsaverage.infl_left, fsaverage.sulc_left, 'left', 'lateral'),
#         (texture_left, fsaverage.infl_left, fsaverage.sulc_left, 'left', 'medial'),
#         (texture_right, fsaverage.infl_right, fsaverage.sulc_right, 'right', 'lateral'),
#         (texture_right, fsaverage.infl_right, fsaverage.sulc_right, 'right', 'medial'),
#     ]
    
#     for ax, (tex, mesh, bg, hemi, view) in zip(axes, views):
#         plotting.plot_surf_stat_map(
#             mesh, tex, 
#             hemi=hemi, 
#             bg_map=bg,
#             view=view,
#             cmap='cold_hot', 
#             threshold=3.0, # T-threshold (approx p<0.001 uncorrected)
#             vmax=vmax,
#             bg_on_data=True,
#             darkness=0.5,
#             axes=ax,
#             colorbar=False
#         )
#         ax.set_title(f"{hemi.upper()} {view.capitalize()}", fontsize=14)
        
#     # Add Colorbar
#     sm = plt.cm.ScalarMappable(cmap='cold_hot', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
#     cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
#     cbar.set_label('T-statistic')
    
#     fig.suptitle(f"{subject} {session}: Speech > Baseline (Sanity Check)", fontsize=18, y=1.05)
    
#     out_path = SANITY_FIGS_DIR / f"{subject}_{session}_glm_speech_surf.png"
#     plt.savefig(out_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"  Saved figure to {out_path}")

# def run_sanity_glm(subject, session):
#     print(f"\n=== Running Sanity GLM: {subject} {session} ===")
    
#     # 1. Locate BOLD
#     func_dir = DERIVATIVES_DIR / subject / session / "func"
#     pattern = "*svf*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
#     bold_files = list(func_dir.glob(pattern))
    
#     if not bold_files:
#         print(f"No BOLD file found in {func_dir}")
#         return
#     bold_path = str(bold_files[0])
    
#     # 2. Get Events
#     try:
#         events = get_events(subject, session)
#         print(f"  Loaded {len(events)} speech events.")
#     except Exception as e:
#         print(f"  Error loading events: {e}")
#         return

#     # 3. Load Confounds
#     try:
#         clean_strategy = ["motion", "wm_csf", "global_signal"]
#         confounds, _ = load_confounds(
#             bold_path,
#             strategy=clean_strategy,
#             motion="basic", 
#             global_signal="basic"
#         )
#     except Exception as e:
#         print(f"  Error loading confounds: {e}")
#         return

#     # 4. Fit GLM
#     print("  Fitting FirstLevelModel (Voxel-wise)...")
#     glm = FirstLevelModel(
#         t_r=TR,
#         noise_model='ar1',
#         standardize=True,
#         hrf_model='spm',
#         drift_model=None,
#         high_pass=HIGH_PASS_HZ,
#         smoothing_fwhm=SMOOTHING_FWHM,
#         mask_img=None,
#         verbose=1
#     )
    
#     glm.fit(bold_path, events=events, confounds=confounds)
    
#     # 5. Compute Contrast
#     print("  Computing contrast 'speech'...")
#     # Since we only have one condition "speech", the contrast vector is just [1] on "speech" column.
#     # We can pass the condition string directly.
#     t_map = glm.compute_contrast('speech', stat_type='t')
    
#     # 6. Plot
#     plot_surface_grid(t_map, subject, session)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--sub", help="Subject ID")
#     parser.add_argument("--ses", help="Session ID")
#     parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions")
    
#     args = parser.parse_args()
    
#     if args.all:
#         sub_dirs = sorted(list(DERIVATIVES_DIR.glob("sub-*")))
#         for sub_dir in sub_dirs:
#             sub_id = sub_dir.name
#             ses_dirs = sorted(list(sub_dir.glob("ses-*")))
#             for ses_dir in ses_dirs:
#                 ses_id = ses_dir.name
#                 run_sanity_glm(sub_id, ses_id)
#     else:
#         if not args.sub or not args.ses:
#             print("Error: Must provide --sub and --ses OR use --all")
#             sys.exit(1)
#         run_sanity_glm(args.sub, args.ses)

#!/usr/bin/env python3
"""
Sanity-Check GLM Analysis: Speech > Baseline.
Uses nilearn.glm.first_level.run_glm on PARCEL data (Schaefer 400) for speed.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from nilearn import datasets, surface, plotting, image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.contrasts import compute_contrast
from nilearn.interfaces.fmriprep import load_confounds

# === CONFIG SETUP ===
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
SANITY_FIGS_DIR = FIGS_DIR / "fmri_sanity_glm"
SANITY_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
SCANNER_START_OFFSET = 12.0
HIGH_PASS_HZ = 0.01

def get_events(subject, session):
    """Creates a simple event file for the GLM: 'Speech' vs Baseline."""
    csv_path = ANNOTATIONS_DIR / f"{subject}_{session}_transcription.csv"
    if not csv_path.exists():
        candidates = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*.csv"))
        if candidates:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(f"No CSV for {subject} {session}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    
    # Filter "next"
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    events = pd.DataFrame()
    events["onset"] = df["start"] - SCANNER_START_OFFSET
    events["duration"] = df["end"] - df["start"]
    events["trial_type"] = "speech"
    
    return events[events["onset"] >= 0]

def plot_surface_grid(t_map_img, subject, session):
    """Projects T-map to fsaverage6 and plots 1x4 grid."""
    print("  Projecting to fsaverage6 surface...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage6')
    
    texture_left = surface.vol_to_surf(t_map_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(t_map_img, fsaverage.pial_right)
    
    vmax = 10
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})
    
    views = [
        (texture_left, fsaverage.infl_left, fsaverage.sulc_left, 'left', 'lateral'),
        (texture_left, fsaverage.infl_left, fsaverage.sulc_left, 'left', 'medial'),
        (texture_right, fsaverage.infl_right, fsaverage.sulc_right, 'right', 'lateral'),
        (texture_right, fsaverage.infl_right, fsaverage.sulc_right, 'right', 'medial'),
    ]
    
    for ax, (tex, mesh, bg, hemi, view) in zip(axes, views):
        plotting.plot_surf_stat_map(
            mesh, tex, 
            hemi=hemi, 
            bg_map=bg,
            view=view,
            cmap='seismic', 
            # threshold=1.96, # Z/T threshold
            vmax=vmax,
            bg_on_data=True,
            darkness=None,
            axes=ax,
            colorbar=False
        )
        ax.set_title(f"{hemi.upper()} {view.capitalize()}", fontsize=14)
        
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label('T-statistic')
    
    fig.suptitle(f"{subject} {session}: Speech > Baseline (Parcel-wise GLM)", fontsize=18, y=1.05)
    
    out_path = SANITY_FIGS_DIR / f"{subject}_{session}_glm_speech_surf.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure to {out_path}")

def run_sanity_glm(subject, session):
    print(f"\n=== Running Sanity GLM (ROI-based): {subject} {session} ===")
    
    # 1. Locate BOLD
    func_dir = DERIVATIVES_DIR / subject / session / "func"
    pattern = "*svf*space-MNI152NLin6Asym_res-2*desc-preproc_bold.nii.gz"
    bold_files = list(func_dir.glob(pattern))
    
    if not bold_files:
        print(f"No BOLD file found in {func_dir}")
        return
    bold_path = str(bold_files[0])
    
    # 2. Extract ROI Signals (Schaefer 400)
    print("  Extracting ROI signals (Schaefer 400)...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    masker = NiftiLabelsMasker(
        labels_img=atlas['maps'],
        standardize='zscore_sample', # Normalize Y
        memory='nilearn_cache',
        verbose=0
    )
    Y = masker.fit_transform(bold_path) 
    
    # 3. Load Confounds & Events
    try:
        clean_strategy = ["motion", "wm_csf", "global_signal"]
        confounds, _ = load_confounds(bold_path, strategy=clean_strategy, motion="basic", global_signal="basic")
        events = get_events(subject, session)
        print(f"  Loaded {len(events)} speech events.")
    except Exception as e:
        print(f"  Error loading data: {e}")
        return

    # 4. Create Design Matrix
    n_scans = Y.shape[0]
    frame_times = np.arange(n_scans) * TR
    
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model='spm',
        drift_model='cosine', # High-pass filtering via DCT regressors in X
        high_pass=HIGH_PASS_HZ,
        add_regs=confounds, # Add motion/physio regressors here
    )
    
    # 5. Run GLM (Mass Univariate on 400 parcels)
    print("  Fitting GLM on 400 parcels...")
    labels, estimates = run_glm(Y, design_matrix.values, noise_model='ar1')
    
    # 6. Compute Contrast
    print("  Computing contrast 'speech'...")
    # Find the column index for "speech"
    contrast_vector = np.zeros(design_matrix.shape[1])
    if "speech" in design_matrix.columns:
        contrast_vector[design_matrix.columns.get_loc("speech")] = 1
    else:
        print("Error: 'speech' condition not found in design matrix.")
        return

    contrast = compute_contrast(labels, estimates, contrast_vector, stat_type='t')
    t_values = contrast.stat() # Extract T-values array (400,)
    
    # 7. Map back to Brain Image
    # masker.inverse_transform expects 2D array (n_samples, n_features)
    t_map_img = masker.inverse_transform(t_values.reshape(1, -1))
    
    # 8. Plot
    plot_surface_grid(t_map_img, subject, session)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help="Subject ID")
    parser.add_argument("--ses", help="Session ID")
    parser.add_argument("--all", action="store_true", help="Run for all subjects/sessions")
    
    args = parser.parse_args()
    
    if args.all:
        sub_dirs = sorted(list(DERIVATIVES_DIR.glob("sub-*")))
        for sub_dir in sub_dirs:
            sub_id = sub_dir.name
            ses_dirs = sorted(list(sub_dir.glob("ses-*")))
            for ses_dir in ses_dirs:
                ses_id = ses_dir.name
                run_sanity_glm(sub_id, ses_id)
    else:
        if not args.sub or not args.ses:
            print("Error: Must provide --sub and --ses OR use --all")
            sys.exit(1)
        run_sanity_glm(args.sub, args.ses)