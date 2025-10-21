"""
Utility functions for fMRI analysis
"""
import os
import numpy as np
from pathlib import Path
import pandas as pd
from nilearn.surface import load_surf_data


## Localizers functions
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
        current_time = 15.0 # Skip initial fixation
        
        for iloop, loop_conditions in enumerate(conditions):
            for itrial, is_hard in enumerate(loop_conditions):
                condition = 'hard' if is_hard else 'easy'
                
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
            current_time += 12 # Response + Inter-trial fixation

        return pd.DataFrame(events), contrast

def load_surface_data(subject, session, task, hemi, data_dir, fsaverage='fsaverage6'):
    """
    Load fMRIPrep surface data for a subject
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-001')
    session : str
        Session ID (e.g., 'ses-01')
    task : str
        Task name (e.g., 'rest')
    hemi : str
        Hemisphere ('L' or 'R')
    data_dir : str or Path
        Path to fMRIPrep derivatives
        
    Returns
    -------
    data : numpy array
        Time series data (n_timepoints, n_vertices)
    """
    filename = f"{subject}_{session}_task-{task}_hemi-{hemi}_space-{fsaverage}_bold.func.gii"
    filepath = Path(data_dir) / subject / session / "func" / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = load_surf_data(filepath)
    
    return data

def load_surface_image(subject, session, task, data_dir, fsaverage='fsaverage6'):
    """
    Load fMRIPrep surface data and create a SurfaceImage object
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-001')
    session : str
        Session ID (e.g., 'ses-01')
    task : str
        Task name (e.g., 'rest')
    data_dir : str or Path
        Path to fMRIPrep derivatives
    fsaverage : str
        Name of the fsaverage template (default: 'fsaverage6')
        
    Returns
    -------
    surface_image : SurfaceImage
        Nilearn SurfaceImage object containing left and right hemisphere data
    """
    from nilearn import datasets
    from nilearn.surface import SurfaceImage
    
    # Load left hemisphere data
    left_data = load_surface_data(subject, session, task, 'L', data_dir, fsaverage=fsaverage)
    
    # Load right hemisphere data
    right_data = load_surface_data(subject, session, task, 'R', data_dir, fsaverage=fsaverage)

    # Fetch fsaverage surfaces
    fsavg = datasets.fetch_surf_fsaverage(fsaverage)
    
    # Create SurfaceImage
    surface_image = SurfaceImage(
        mesh={'left': fsavg.infl_left, 'right': fsavg.infl_right},
        data={'left': left_data, 'right': right_data}
    )
    
    return surface_image