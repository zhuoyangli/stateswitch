#!/usr/bin/env python3
"""
First-level GLM analysis for fMRI data using Schaefer parcellation.
"""

import os
from pathlib import Path
import sys
import numpy as np
import argparse
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import make_first_level_design_matrix
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from nilearn import datasets, plotting, image, surface
import matplotlib.pyplot as plt
import nibabel as nib

# add parent directory to path for imports
current_dir = os.getcwd()
sys.path.insert(0, str(Path(current_dir).parent))
from configs.config import DERIVATIVES_DIR, TR, FIGS_DIR
from utils import generate_events_dataframe


def extract_roi_time_series(bold_path, n_rois=400, yeo_networks=17):
    """Extract time series from Schaefer atlas ROIs"""
    # Load Schaefer atlas
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, 
        yeo_networks=yeo_networks,
        resolution_mm=2
    )
    
    # Create masker for ROI extraction
    masker = NiftiLabelsMasker(
        labels_img=schaefer_atlas['maps'],
        labels=schaefer_atlas['labels'],
        standardize='zscore_sample',
        memory='nilearn_cache',
        verbose=1
    )
    
    # Load confounds
    confounds, sample_mask = load_confounds(
        bold_path,
        strategy=["high_pass", "motion", "wm_csf", "global_signal"]
    )

    # Extract time series
    bold_img = image.load_img(bold_path)
    time_series = masker.fit_transform(bold_img, confounds=confounds)
    
    # Remove background label (first entry)
    roi_labels = schaefer_atlas['labels'][1:]  # Skip background
    print(len(schaefer_atlas['labels']))
    return time_series, roi_labels, schaefer_atlas


def run_glm_analysis(time_series, events_df, contrast):
    """Run GLM analysis and compute contrast"""
    # generate events dataframe and construct design matrix
    n_trs, n_rois = time_series.shape
    frame_times = np.arange(n_trs) * TR
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events_df,
        hrf_model='glover',
        high_pass=0.01
    )
    print(f'Design matrix shape: {design_matrix.shape}')
    
    # fit GLM and compute contrast
    t_values = []
    p_values = []

    for roi in range(time_series.shape[1]):
        # Fit model
        model = sm.OLS(time_series[:, roi], design_matrix).fit()
        
        # Test contrast
        t_stat = model.t_test(contrast)
        t_values.append(t_stat.tvalue[0][0])
        p_values.append(t_stat.pvalue)

    t_values = np.array(t_values)
    p_values = np.array(p_values)
    
    return t_values, p_values, design_matrix


def create_surface_plots(t_values, p_values, schaefer_atlas, subject_id, session_id, task, run):
    """Create surface plots with FDR correction"""
    # Load the atlas image from the file path
    atlas_img = nib.load(schaefer_atlas.maps)
    atlas_data = atlas_img.get_fdata()
    
    # Apply FDR correction
    rejected, q_values, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=0.05, method='fdr_bh'
    )

    # Find the t-value threshold that corresponds to FDR q = 0.05
    if np.any(rejected):
        significant_t_values = np.abs(t_values[rejected])
        t_threshold_fdr = np.min(significant_t_values)
        print(f"FDR q < 0.05 corresponds to |t| > {t_threshold_fdr:.3f}")
    else:
        t_threshold_fdr = np.max(np.abs(t_values)) + 1
        print("No ROIs survive FDR correction at q < 0.05")

    # Create the t-value map
    t_map_data = np.zeros_like(atlas_data)
    for roi_idx in range(len(t_values)):
        roi_mask = atlas_data == roi_idx + 1
        t_map_data[roi_mask] = t_values[roi_idx]

    t_map_img = image.new_img_like(atlas_img, t_map_data)

    # Fetch fsaverage surface meshes
    fsaverage = datasets.fetch_surf_fsaverage()

    # Project volume data to surface
    texture_left = surface.vol_to_surf(t_map_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(t_map_img, fsaverage.pial_right)

    # Create 1x4 subplot figure with adjusted spacing
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})
    fig.suptitle(f'{subject_id}_{session_id}_{task}_run-{run}: T-values thresholded at FDR q < 0.05 (|t| > {t_threshold_fdr:.2f})', fontsize=16)

    # Left lateral
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, texture_left,
        hemi='left', view='lateral',
        title='Left Lateral', colorbar=False,
        threshold=t_threshold_fdr,
        axes=axes[0],
        bg_map=fsaverage.sulc_left,
        cmap='RdBu_r',
        symmetric_cbar=True,
        bg_on_data=True,
        darkness=None
    )

    # Left medial
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, texture_left,
        hemi='left', view='medial',
        title='Left Medial', colorbar=False,
        threshold=t_threshold_fdr,
        axes=axes[1],
        bg_map=fsaverage.sulc_left,
        cmap='RdBu_r',
        symmetric_cbar=True,
        bg_on_data=True,
        darkness=None
    )

    # Right lateral
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture_right,
        hemi='right', view='lateral',
        title='Right Lateral', colorbar=False,
        threshold=t_threshold_fdr,
        axes=axes[2],
        bg_map=fsaverage.sulc_right,
        cmap='RdBu_r',
        symmetric_cbar=True,
        bg_on_data=True,
        darkness=None
    )

    # Right medial
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture_right,
        hemi='right', view='medial',
        title='Right Medial', colorbar=True,
        threshold=t_threshold_fdr,
        axes=axes[3],
        bg_map=fsaverage.sulc_right,
        cmap='RdBu_r',
        symmetric_cbar=True,
        bg_on_data=True,
        darkness=None
    )

    # Adjust layout to make room for colorbar
    plt.subplots_adjust(right=0.85)  # Leave space on the right for colorbar
    
    return fig


def main(subject_id, session_id, task, run_num):
    """Main analysis pipeline"""
    # Build BOLD path
    BOLD_PATH = str(DERIVATIVES_DIR / subject_id / session_id / 'func' / 
                    f'{subject_id}_{session_id}_task-{task + str(run_num)}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')
    
    # Extract time series
    time_series, roi_labels, schaefer_atlas = extract_roi_time_series(BOLD_PATH)
    print(f'Extracted time series shape: {time_series.shape}')
    
    # Generate events and contrast
    events_df, contrast = generate_events_dataframe(task, run_num)
    
    # Run GLM
    t_values, p_values, design_matrix = run_glm_analysis(time_series, events_df, contrast)
    
    # Find significant ROIs
    sig_rois_idx = np.where(p_values < 0.001)[0]
    sig_rois = [roi_labels[i] for i in sig_rois_idx]
    print(f"Significant ROIs: {sig_rois}")
    
    # Create surface plots
    fig = create_surface_plots(t_values, p_values, schaefer_atlas, subject_id, session_id, task, run_num)
    
    # Save figure
    output_dir = FIGS_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{subject_id}_{session_id}_task-{task}_run-{run_num}_surface_FDR05.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-001)')
    parser.add_argument('--session', type=str, required=True, help='Session ID (e.g., ses-01)')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., langloc)')
    parser.add_argument('--run', type=int, required=True, help='Run number (e.g., 1)')
    
    args = parser.parse_args()
    
    main(args.subject, args.session, args.task, args.run)