"""
GLM analysis for localizer tasks (language: intact-degraded, multiple demands: hard-easy, theory of mind: belief-photo)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn import datasets, plotting
from nilearn.plotting import plot_surf_stat_map
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_surface_image
from configs.config import DERIVATIVES_DIR, TR
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
        current_time = 15.0 # Skip initial fixation
        
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
            current_time += 15.0 # Inter-loop fixation
        
        return pd.DataFrame(events), contrast
    
    elif task == 'tomloc':
        contrast = 'belief - photo'
        conditions_run1 = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
        conditions_run2 = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]

        conditions = conditions_run1 if run_num == 1 else conditions_run2

        events = []
        current_time = 12.0 # Skip initial fixation

        for is_belief in conditions:
            condition = 'belief' if is_belief else 'photo'
            events.append({
                'onset': current_time,
                'duration': 16.5,
                'trial_type': condition
            })
            current_time += 16.5
            current_time += 12.0 # Inter-trial interval (fixation)

        return pd.DataFrame(events), contrast

def compute_contrast_glm(subject, session, task, run, events_df, contrast, fsaverage):
    """Compute GLM and contrast"""
    task_name = task + str(run)
    surface_image = load_surface_image(subject, session, task_name, DERIVATIVES_DIR, fsaverage)
    # Fit GLM
    frame_times = np.arange(0, surface_image.shape[1] * TR, TR)
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model='glover + derivative',
        drift_model='cosine',
        high_pass=0.01
    )
    glm = FirstLevelModel()
    glm = glm.fit(surface_image, design_matrices=design_matrix)

    # Compute contrast
    contrast_results = glm.compute_contrast(
        contrast, 
        stat_type='t', 
        output_type='z_score'
    )
    
    return contrast_results

def create_masked_colormap(cmap_name, vmin, vmax, threshold, gray_color):
    """Create a colormap with sub-threshold values masked in gray"""
    # Get the original colormap
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, 256))
    
    # Calculate threshold indices
    n_colors = len(colors)
    range_val = vmax - vmin
    threshold_idx_pos = int((threshold - vmin) / range_val * n_colors)
    threshold_idx_neg = int((-threshold - vmin) / range_val * n_colors)
    
    # Set sub-threshold colors to gray
    colors[threshold_idx_neg:threshold_idx_pos] = gray_color
    
    return ListedColormap(colors)

def plot_surface_results(subject, task, contrast, contrast_results, fsaverage, output_path=None):
    """Plot surface results with 4 views and masked colorbar"""
    fsavg = datasets.fetch_surf_fsaverage(fsaverage)
    
    # Create figure with 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_PARAMS['figsize'], 
                             subplot_kw={'projection': '3d'})
    
    # Plot configurations for each view
    plot_configs = [
        ('left', 'lateral', fsavg.infl_left, fsavg.sulc_left, 'Left Lateral'),
        ('left', 'medial', fsavg.infl_left, fsavg.sulc_left, 'Left Medial'),
        ('right', 'lateral', fsavg.infl_right, fsavg.sulc_right, 'Right Lateral'),
        ('right', 'medial', fsavg.infl_right, fsavg.sulc_right, 'Right Medial')
    ]
    
    # Plot each view
    for ax, (hemi, view, mesh, bg_map, title) in zip(axes, plot_configs):
        plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=contrast_results,
            hemi=hemi,
            view=view,
            bg_map=bg_map,
            axes=ax,
            title=title,
            **PLOT_PARAMS
        )
    
    # Adjust layout
    plt.subplots_adjust(**LAYOUT_PARAMS)
    
    # Create masked colormap
    masked_cmap = create_masked_colormap(
        PLOT_PARAMS['cmap'],
        -PLOT_PARAMS['vmax'],
        PLOT_PARAMS['vmax'],
        PLOT_PARAMS['threshold'],
        COLORBAR_PARAMS['gray_color']
    )
    
    # Add colorbar with masked colormap
    cbar_ax = fig.add_axes(COLORBAR_PARAMS['position'])
    sm = plt.cm.ScalarMappable(
        cmap=masked_cmap,
        norm=plt.Normalize(vmin=-PLOT_PARAMS['vmax'], vmax=PLOT_PARAMS['vmax'])
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(COLORBAR_PARAMS['label'], fontsize=COLORBAR_PARAMS['label_size'])
    cbar.ax.tick_params(labelsize=COLORBAR_PARAMS['tick_size'])
    
    # Add threshold indicators
    threshold = PLOT_PARAMS['threshold']
    cbar.ax.axhline(y=threshold, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    cbar.ax.axhline(y=-threshold, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add overall title
    fig.suptitle(f'[{subject}] Task: {task}, contrast: {contrast}', fontsize=16)
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight', 
                facecolor=FIGURE_PARAMS['facecolor'], 
                edgecolor=FIGURE_PARAMS['edgecolor'])
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GLM analysis for localizer tasks")
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-001)')
    parser.add_argument('--session', type=str, required=True, help='Session ID (e.g., ses-01)')
    parser.add_argument('--task', type=str, required=True, choices=['langloc', 'mdloc', 'tomloc'], help='Task name')
    parser.add_argument('--run', type=int, required=True, choices=[1, 2], help='Run number (1 or 2)')
    parser.add_argument('--fsaverage', type=str, default='fsaverage6', help='fsaverage template (default: fsaverage6)')
    parser.add_argument('--output', type=str, default=None, help='Output path for the figure')
    args = parser.parse_args()

    # Generate events dataframe
    events_df, contrast = generate_events_dataframe(args.task, args.run)

    # Compute contrast
    contrast_results = compute_contrast_glm(
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run,
        events_df=events_df,
        contrast=contrast,
        fsaverage=args.fsaverage
    )

    # Plot results
    plot_surface_results(
        subject=args.subject,
        task=args.task,
        contrast=contrast,
        contrast_results=contrast_results,
        fsaverage=args.fsaverage,
        output_path=args.output
    )
