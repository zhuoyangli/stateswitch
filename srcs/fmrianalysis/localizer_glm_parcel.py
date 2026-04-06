#!/usr/bin/env python3
"""
Parcel-wise GLM analysis for localizer tasks using Schaefer parcellation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import datasets, image, surface
from nilearn.maskers import NiftiLabelsMasker
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import nibabel as nib

from configs.config import DERIVATIVES_DIR, TR, FIGS_DIR
from plotting_config import PLOT_PARAMS, FIGURE_PARAMS, COLORBAR_PARAMS, LAYOUT_PARAMS


def generate_events_dataframe(task, run_num):
    """Generate events for specified task and run"""
    if task == 'langloc':
        contrast = 'intact - degraded'
        conditions_run1 = [[1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 1, 0]]

        conditions_run2 = [[0, 1, 1, 0],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 0, 1]]

        conditions = conditions_run1 if run_num == 1 else conditions_run2

        events = []
        current_time = 15.0  # Skip initial fixation

        for loop_conditions in conditions:
            for is_intact in loop_conditions:
                condition = 'intact' if is_intact else 'degraded'
                events.append({
                    'onset': current_time,
                    'duration': 18.0,
                    'trial_type': condition
                })
                current_time += 18.0
            current_time += 15.0  # Inter-loop fixation

        return pd.DataFrame(events), contrast

    elif task == 'mdloc':
        contrast = 'hard - easy'
        conditions_run1 = [[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1],
                           [1, 1, 0, 0]]

        conditions_run2 = [[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1]]

        conditions = conditions_run1 if run_num == 1 else conditions_run2

        events = []
        current_time = 15.0  # Skip initial fixation

        for iloop, loop_conditions in enumerate(conditions):
            for itrial, is_hard in enumerate(loop_conditions):
                condition = 'hard' if is_hard else 'easy'

                if run_num == 1 and iloop == 0 and itrial == 0:
                    duration = 5.0
                else:
                    duration = 9.0
                events.append({
                    'onset': current_time,
                    'duration': duration,
                    'trial_type': condition
                })
                current_time += duration
            current_time += 15.0  # Inter-loop fixation

        return pd.DataFrame(events), contrast

    elif task == 'tomloc':
        contrast = 'belief - photo'
        conditions_run1 = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
        conditions_run2 = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]

        conditions = conditions_run1 if run_num == 1 else conditions_run2

        events = []
        current_time = 12.0  # Skip initial fixation

        for is_belief in conditions:
            condition = 'belief' if is_belief else 'photo'
            events.append({
                'onset': current_time,
                'duration': 16.5,
                'trial_type': condition
            })
            current_time += 16.5
            current_time += 12.0  # Inter-trial interval (fixation)

        return pd.DataFrame(events), contrast


def extract_roi_time_series(bold_path, n_rois=400, yeo_networks=17):
    """Extract time series from Schaefer atlas ROIs"""
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=2
    )
    masker = NiftiLabelsMasker(
        labels_img=schaefer_atlas['maps'],
        standardize='zscore_sample',
        memory='nilearn_cache',
        verbose=1
    )
    bold_img = image.load_img(bold_path)
    time_series = masker.fit_transform(bold_img)
    return time_series, schaefer_atlas


def compute_contrast_glm(subject, session, task, run, events_df, contrast, fsaverage='fsaverage6'):
    """Compute parcel-wise GLM and contrast, return z-map NIfTI"""
    task_name = task + str(run)
    bold_path = str(DERIVATIVES_DIR / subject / session / 'func' /
                    f'{subject}_{session}_task-{task_name}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')

    time_series, schaefer_atlas = extract_roi_time_series(bold_path)
    n_trs, n_parcels = time_series.shape

    # Wrap parcel time series in a pseudo-NIfTI so FirstLevelModel can operate on it
    parcel_nii = nib.Nifti1Image(
        time_series.T[:, np.newaxis, np.newaxis, :],  # (n_parcels, 1, 1, n_trs)
        affine=np.eye(4)
    )
    mask_nii = nib.Nifti1Image(np.ones((n_parcels, 1, 1), dtype=np.int8), affine=np.eye(4))

    frame_times = np.arange(n_trs) * TR
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model='glover + derivative',
        drift_model='cosine',
        high_pass=0.01
    )

    glm = FirstLevelModel(mask_img=mask_nii)
    glm.fit(parcel_nii, design_matrices=design_matrix)
    contrast_map = glm.compute_contrast(contrast, stat_type='t', output_type='z_score')

    # Map z-scores back to atlas volume
    z_values = contrast_map.get_fdata()[:, 0, 0]
    atlas_img = nib.load(schaefer_atlas.maps)
    atlas_data = atlas_img.get_fdata()
    z_map_data = np.zeros_like(atlas_data)
    for roi_idx, z in enumerate(z_values):
        z_map_data[atlas_data == roi_idx + 1] = z
    z_map_img = image.new_img_like(atlas_img, z_map_data)

    return z_map_img


def create_masked_colormap(cmap_name, vmin, vmax, threshold, gray_color):
    """Create a colormap with sub-threshold values masked in gray"""
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, 256))
    n_colors = len(colors)
    range_val = vmax - vmin
    threshold_idx_pos = int((threshold - vmin) / range_val * n_colors)
    threshold_idx_neg = int((-threshold - vmin) / range_val * n_colors)
    colors[threshold_idx_neg:threshold_idx_pos] = gray_color
    return ListedColormap(colors)


def plot_surface_results(subject, session, task, run, contrast, contrast_results, fsaverage='fsaverage6'):
    """Plot surface results with 4 views and masked colorbar"""
    fsavg = datasets.fetch_surf_fsaverage(fsaverage)

    texture_left = surface.vol_to_surf(contrast_results, fsavg.pial_left)
    texture_right = surface.vol_to_surf(contrast_results, fsavg.pial_right)

    fig, axes = plt.subplots(1, 4, figsize=FIGURE_PARAMS['figsize'],
                             subplot_kw={'projection': '3d'})

    plot_configs = [
        ('left',  'lateral', fsavg.infl_left,  fsavg.sulc_left,  texture_left,  'Left Lateral'),
        ('left',  'medial',  fsavg.infl_left,  fsavg.sulc_left,  texture_left,  'Left Medial'),
        ('right', 'lateral', fsavg.infl_right, fsavg.sulc_right, texture_right, 'Right Lateral'),
        ('right', 'medial',  fsavg.infl_right, fsavg.sulc_right, texture_right, 'Right Medial'),
    ]

    for ax, (hemi, view, mesh, bg_map, texture, title) in zip(axes, plot_configs):
        plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=texture,
            hemi=hemi,
            view=view,
            bg_map=bg_map,
            axes=ax,
            title=title,
            **PLOT_PARAMS
        )

    plt.subplots_adjust(**LAYOUT_PARAMS)

    masked_cmap = create_masked_colormap(
        PLOT_PARAMS['cmap'],
        -PLOT_PARAMS['vmax'],
        PLOT_PARAMS['vmax'],
        PLOT_PARAMS['threshold'],
        COLORBAR_PARAMS['gray_color']
    )

    cbar_ax = fig.add_axes(COLORBAR_PARAMS['position'])
    sm = plt.cm.ScalarMappable(
        cmap=masked_cmap,
        norm=plt.Normalize(vmin=-PLOT_PARAMS['vmax'], vmax=PLOT_PARAMS['vmax'])
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(COLORBAR_PARAMS['label'], fontsize=COLORBAR_PARAMS['label_size'])
    cbar.ax.tick_params(labelsize=COLORBAR_PARAMS['tick_size'])

    threshold = PLOT_PARAMS['threshold']
    cbar.ax.axhline(y=threshold, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    cbar.ax.axhline(y=-threshold, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    fig.suptitle(f'[{subject}] Task: {task}, contrast: {contrast}', fontsize=16)

    out_dir = FIGS_DIR / f'{task}_schaefer'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{subject}_{session}_task-{task}_run-{run}_contrast_map.png'
    plt.savefig(out_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight',
                facecolor=FIGURE_PARAMS['facecolor'],
                edgecolor=FIGURE_PARAMS['edgecolor'])
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parcel-wise GLM analysis for localizer tasks")
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-001)')
    parser.add_argument('--session', type=str, required=True, help='Session ID (e.g., ses-01)')
    parser.add_argument('--task', type=str, required=True, choices=['langloc', 'mdloc', 'tomloc'], help='Task name')
    parser.add_argument('--run', type=int, required=True, choices=[1, 2], help='Run number (1 or 2)')
    parser.add_argument('--fsaverage', type=str, default='fsaverage6', help='fsaverage template (default: fsaverage6)')
    args = parser.parse_args()

    events_df, contrast = generate_events_dataframe(args.task, args.run)

    contrast_results = compute_contrast_glm(
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run,
        events_df=events_df,
        contrast=contrast,
        fsaverage=args.fsaverage,
    )

    plot_surface_results(
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run,
        contrast=contrast,
        contrast_results=contrast_results,
        fsaverage=args.fsaverage,
    )
