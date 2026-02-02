#!/usr/bin/env python3
"""
02_visualize_results.py
Generates:
  1. Standardized Stat Maps (Volumetric & Surface) with fixed colorbars.
  2. Violin Plots of voxel distributions for specific ROIs with fixed Y-limits.
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from nilearn import plotting, datasets, surface, image

# === CONFIGURATION ===
# 1. VISUALIZATION LIMITS (Crucial for comparison)
STAT_MAP_VMAX = 8.0        # Max Z-score for Red/Blue maps
VIOLIN_YLIM = (-8, 12)     # Y-axis limit for distribution plots

# 2. ROIs TO ANALYZE (Schaefer Atlas Labels)
# Any label containing these strings will be included in the violin plots
ROI_KEYWORDS = ["DefaultA_pCun", "DefaultA_PCC"] 

# === PATHS ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DERIVATIVES_DIR, FIGS_DIR
except ImportError:
    print("Warning: Config not found, using default paths.")
    DERIVATIVES_DIR = Path("./derivatives")
    FIGS_DIR = Path("./figures")

INPUT_DIR = DERIVATIVES_DIR / "contrast_maps_unsmoothed"
OUT_DIR_MAPS = FIGS_DIR / "stat_maps_final"
OUT_DIR_VIOLINS = FIGS_DIR / "roi_violins"

OUT_DIR_MAPS.mkdir(parents=True, exist_ok=True)
OUT_DIR_VIOLINS.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PART 1: STAT MAP GENERATOR
# =============================================================================
def generate_stat_maps(z_map_path):
    """
    Creates Volumetric and Surface plots for a single Z-map.
    Uses fixed VMAX for consistency.
    """
    fname = z_map_path.name
    # Expected format: sub-001_ses-01_model-prev_desc-unsmoothed_zmap.nii.gz
    parts = fname.split("_")
    sub = parts[0]
    ses = parts[1]
    model = [p for p in parts if "model-" in p][0].replace("model-", "")
    
    title = f"{sub} {ses}: {model} word (Z-Map)"
    
    # Load Image
    img = image.load_img(str(z_map_path))
    
    # 1. Find Peak (for centering)
    data = img.get_fdata()
    if np.nanmax(data) > 0:
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        cut_coords = image.coord_transform(max_idx[0], max_idx[1], max_idx[2], img.affine)
    else:
        cut_coords = (0, -60, 20) # Default to Precuneus if empty

    # --- A. Volumetric Plot ---
    fig = plt.figure(figsize=(12, 6))
    plotting.plot_stat_map(
        img, 
        bg_img=datasets.load_mni152_template(),
        threshold=1.96,      # p < 0.05
        vmax=STAT_MAP_VMAX,  # <--- FIXED SCALE
        display_mode='ortho', 
        cut_coords=cut_coords,
        draw_cross=True,      
        cmap='cold_hot',
        figure=fig,
        title=title
    )
    vol_name = f"{sub}_{ses}_model-{model}_vol.png"
    plt.savefig(OUT_DIR_MAPS / vol_name, dpi=150, bbox_inches='tight')
    plt.close()

    # --- B. Surface Plot ---
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    texture_l = surface.vol_to_surf(img, fsaverage.pial_left)
    texture_r = surface.vol_to_surf(img, fsaverage.pial_right)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
    
    # Plot Medial Views (Best for Precuneus/PCC)
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, texture_l, hemi='left', view='medial',
        cmap='cold_hot', threshold=1.96, vmax=STAT_MAP_VMAX,
        bg_map=fsaverage.sulc_left, bg_on_data=True, axes=axes[0], colorbar=False
    )
    axes[0].set_title("Left Medial")

    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture_r, hemi='right', view='medial',
        cmap='cold_hot', threshold=1.96, vmax=STAT_MAP_VMAX,
        bg_map=fsaverage.sulc_right, bg_on_data=True, axes=axes[1], colorbar=True
    )
    axes[1].set_title("Right Medial")
    
    fig.suptitle(title, fontsize=16)
    surf_name = f"{sub}_{ses}_model-{model}_surf.png"
    plt.savefig(OUT_DIR_MAPS / surf_name, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# PART 2: VIOLIN PLOT DATA EXTRACTION
# =============================================================================
def extract_roi_data(z_map_path, atlas_img_resampled, target_parcels):
    """
    Extracts raw voxel values for specified ROIs from a single Z-map.
    """
    records = []
    fname = z_map_path.name
    sub = fname.split("_")[0]
    ses = fname.split("_")[1]
    model = [p for p in fname.split("_") if "model-" in p][0].replace("model-", "")

    # Load Z-map
    z_img = image.load_img(str(z_map_path))
    z_data = z_img.get_fdata()
    atlas_data = atlas_img_resampled.get_fdata()

    for parcel in target_parcels:
        # Mask: where atlas matches parcel ID
        mask = (atlas_data == parcel['id'])
        
        # Extract values
        voxels = z_data[mask]
        voxels = voxels[voxels != 0] # clean padding
        
        # Sub-sample if huge (optional, keeps plotting fast)
        if len(voxels) > 500:
            voxels = np.random.choice(voxels, 500, replace=False)

        for v in voxels:
            records.append({
                "Subject": sub,
                "Session": ses,
                "Model": model,
                "ROI": parcel['label'],
                "Z_Score": v
            })
            
    return records

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    # 1. Find all Z-maps
    z_maps = sorted(list(INPUT_DIR.glob("*.nii.gz")))
    if not z_maps:
        print(f"Error: No maps found in {INPUT_DIR}")
        sys.exit(1)
        
    print(f"Found {len(z_maps)} maps. Starting visualization...")

    # --- STEP 1: GENERATE STAT MAPS (Parallel) ---
    print("  [1/3] Generating Stat Maps...")
    Parallel(n_jobs=args.n_jobs)(
        delayed(generate_stat_maps)(f) for f in z_maps
    )

    # --- STEP 2: PREPARE ATLAS FOR EXTRACTION ---
    print("  [2/3] Preparing Atlas for ROI Extraction...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    
    # Identify target parcels
    targets = []
    for i, lbl in enumerate(all_labels):
        # Clean label name for plotting
        short_lbl = "_".join(lbl.split("_")[2:]) # e.g. "DefaultA_pCun_1"
        if any(k in lbl for k in ROI_KEYWORDS):
            targets.append({"id": i+1, "label": short_lbl, "full": lbl})
            
    print(f"        Found {len(targets)} target ROIs: {[t['label'] for t in targets]}")
    
    # Resample atlas ONCE to match the first Z-map (assumption: all maps are same grid)
    ref_img = image.load_img(str(z_maps[0]))
    atlas_resampled = image.resample_to_img(atlas['maps'], ref_img, interpolation='nearest')

    # --- STEP 3: EXTRACT & PLOT VIOLINS ---
    print("  [3/3] Extracting Voxel Data & Plotting...")
    
    # Extract in parallel chunks
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(extract_roi_data)(f, atlas_resampled, targets) for f in z_maps
    )
    
    # Flatten list of lists
    all_records = [item for sublist in results for item in sublist]
    df = pd.DataFrame(all_records)
    
    if df.empty:
        print("Warning: No ROI data extracted.")
        sys.exit()

    # Create Summary Plots (One per ROI, per Model)
    models = df["Model"].unique()
    rois = df["ROI"].unique()
    
    for model in models:
        for roi in rois:
            subset = df[(df["Model"] == model) & (df["ROI"] == roi)]
            
            plt.figure(figsize=(max(6, len(subset["Session"].unique()) * 0.8), 6))
            
            # THE VIOLIN PLOT
            sns.violinplot(
                data=subset, x="Session", y="Z_Score", hue="Subject",
                palette="mako", linewidth=1, inner="box", cut=0
            )
            
            # Formatting
            plt.axhline(0, color="black", linestyle="-", alpha=0.3)
            plt.axhline(1.96, color="red", linestyle=":", alpha=0.5, label="p<0.05")
            
            plt.ylim(VIOLIN_YLIM) # <--- FIXED Y-LIMIT
            plt.title(f"Voxel Distribution: {roi} ({model} word)", fontsize=14)
            plt.xlabel("")
            plt.ylabel("Z-Score")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            out_name = f"violin_{model}_{roi}.png"
            plt.savefig(OUT_DIR_VIOLINS / out_name, dpi=300)
            plt.close()
            
    print(f"Done! Output saved to:\n  - Maps: {OUT_DIR_MAPS}\n  - Violins: {OUT_DIR_VIOLINS}")