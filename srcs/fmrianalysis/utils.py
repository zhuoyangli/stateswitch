"""
Utility functions for fMRI analysis
"""
import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd
from nilearn import datasets
from nilearn.surface import load_surf_data
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy

# === CONFIG SETUP ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DATA_DIR, CACHE_DIR, DERIVATIVES_DIR, FIGS_DIR, TR
except ImportError:
    print("Error: Could not import 'configs.config'. Ensure your directory structure is correct.")
    sys.exit(1)

HIGH_PASS_HZ = 0.01

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

def get_parcel_data(
    atlas_label: str, 
    subject: str, 
    session: str, 
    task: str,
    data_dir: Path, 
    cache_dir: Path,
    high_pass: float = 0.01
) -> dict:
    """
    Load and cache atlas parcel data.
    """
    # 1. Fix Cache Filename: Include the actual atlas_label to avoid collisions
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_subdir = cache_dir / 'nilearn_cache'
    cache_subdir.mkdir(exist_ok=True)
    
    cache_file = cache_subdir / f"{subject}_{session}_task-{task}_atlas-{atlas_label}_desc-clean_timeseries.npz"
    
    # 2. Check Cache
    if cache_file.exists():
        print(f"Loading cached parcel data from: {cache_file.name}...")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            # Ensure we return a standard dict, not a numpy wrapper
            return loaded['parcel_data'].item()
        except Exception as e:
            print(f"Cache load failed ({e}), re-computing...")

    # 3. Load Atlas
    if atlas_label == 'Schaefer400_17Nets':
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=400, yeo_networks=17, resolution_mm=2
        )
    elif atlas_label == 'HarvardOxford_sub':
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    elif atlas_label == 'HarvardOxford_cort':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    else:
        raise ValueError(f"Unknown atlas label: {atlas_label}")
    
    # Handle label decoding safely
    labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    
    # 4. Define Masker
    # Note: 'memory' caches the masking step itself, separate from your manual npz cache
    masker = NiftiLabelsMasker(
        labels_img=atlas['maps'], 
        labels=labels,
        standardize='zscore_sample', 
        high_pass=high_pass, 
        t_r=TR, 
        verbose=0, 
        memory=str(cache_subdir)
    )
    
    bold_path = data_dir / subject / session / "func" / f"{subject}_{session}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD file not found: {bold_path}")

    print(f"Extracting signal for {subject} {session} using {atlas_label}...")
    confounds, sample_mask = load_confounds_strategy(
        str(bold_path), 
        denoise_strategy="simple"
    )
    
    # 6. Fit and Transform
    # Pass sample_mask to handle volumes dropped by scrubbing (if any)
    data = masker.fit_transform(bold_path, confounds=confounds, sample_mask=sample_mask)
    
    # 7. Structure and Save
    # Create dict mapping label -> timeseries (Time x 1)
    parcel_data = {label: d for label, d in zip(labels, data.T)}
    
    np.savez_compressed(cache_file, parcel_data=parcel_data)
    
    return parcel_data