from pathlib import Path
import sys
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm import compute_contrast
from matplotlib import pyplot as plt
from localizers_contrast import generate_events_dataframe
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests  # For p-value correction

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DERIVATIVES_DIR, TR

def extract_schaefer_roi_time_series(data_dir, subject, session, taskname, run_num, n_rois, yeo_networks):
    data_filename = f'sub-{subject}_ses-{session}_task-{taskname}{str(run_num)}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
    data_path = Path(data_dir) / f'sub-{subject}' / f'ses-{session}' / 'func' / data_filename
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    
    # Load the data and extract time series
    bold_img = image.load_img(data_path)
    schaefer_atlas = fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=yeo_networks, resolution_mm=2)
    atlas_filename = schaefer_atlas['maps']
    labels = schaefer_atlas['labels']

    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        labels=labels,
        standardize="zscore_sample",
        memory="nilearn_cache",
        verbose=1,
    )

    confounds, sample_mask = load_confounds(str(data_path), strategy=["high_pass", "motion", "wm_csf", "global_signal"], motion='basic')

    time_series = masker.fit_transform(bold_img, confounds=confounds)
    
    return time_series, labels[1:]

def compute_roi_contrast(subject, session, taskname, run_num):
    """Compute GLM and contrast for extracted ROI time series"""
    # Generate events
    events_df, contrast = generate_events_dataframe(taskname, run_num)

    # Extract time series
    time_series, roi_labels = extract_schaefer_roi_time_series(
        DERIVATIVES_DIR, subject, session, taskname, run_num, n_rois=200, yeo_networks=17
    )

    # Generate the GLM design matrix
    frame_times = np.arange(0, time_series.shape[0] * TR, TR)
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model='glover + derivative',
        drift_model='cosine',
        high_pass=0.01  # This may be adjusted based on your requirements
    )

    # Fit GLM
    labels, results = run_glm(time_series, design_matrix)

    # Compute contrast and get the t-statistic
    t_values = compute_contrast(contrast, stat_type='t', output_type='stat')

    # Calculate p-values from t-values
    n_samples = time_series.shape[0]
    n_params = design_matrix.shape[1]
    df = n_samples - n_params

    # Calculate p-values
    p_values = 1 - t_dist.cdf(np.abs(t_values), df)  # Two-tailed p-values

    # Correct p-values for multiple comparisons (e.g., using Benjamini-Hochberg)
    corrected_p_values = multipletests(p_values, method='fdr_by')[1]  # fdr_bh for FDR control

    return t_values, corrected_p_values, labels

def plot_contrast_results(t_values, corrected_p_values, labels):
    """Plotting the computed contrast results for the selected ROIs with p-values"""
    plt.figure(figsize=(10, 6))
    
    # Plot t-values
    plt.bar(range(len(labels)), t_values, color='blue', alpha=0.7)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel('T-value')
    plt.title('T-value Results for ROIs')
    plt.tight_layout()
    
    # Print corrected p-values
    for idx, label in enumerate(labels):
        print(f"{label}: T-value = {t_values[idx]:.3f}, Corrected p-value = {corrected_p_values[idx]:.3f}")
    
    plt.show()

def plot_roi_time_series_with_events(subject, session, taskname, run_num, roi_keyword):
    """Plot ROI time series and highlights events"""
    events_df, contrast = generate_events_dataframe(taskname, run_num)
    time_series, labels = extract_schaefer_roi_time_series(
        DERIVATIVES_DIR, subject, session, taskname, run_num, n_rois=200, yeo_networks=17
    )

    # Find all ROIs matching the keyword
    roi_indices = [idx - 1 for idx, label in enumerate(labels) if roi_keyword.lower() in label.lower()]
    if not roi_indices:
        raise ValueError(f"No ROIs found matching keyword: {roi_keyword}")

    # Plot time series for each matching ROI
    for roi_idx in roi_indices:
        plt.figure(figsize=(15, 5))
        plt.plot(time_series[:, roi_idx], label=f'ROI: {labels[roi_idx]}', color='blue')

        # Overlay events
        for _, event in events_df.iterrows():
            onset = int(event['onset'] / TR)
            duration = int(event['duration'] / TR)
            plt.axvspan(onset, onset + duration, color='red' if event['trial_type'] in ['intact', 'hard', 'belief'] else 'green', alpha=0.3)

        plt.title(f'Subject: {subject}, Session: {session}, Task: {taskname}, Run: {run_num}, ROI: {labels[roi_idx]}')
        plt.xlabel('Time (TRs)')
        plt.ylabel('Z-scored Signal')
        plt.xlim(0, time_series.shape[0])
        plt.legend()
        plt.show()

    # Compute contrast for the selected ROIs
    t_values, corrected_p_values, labels = compute_roi_contrast(subject, session, taskname, run_num)
    plot_contrast_results(t_values, corrected_p_values, labels)

if __name__ == "__main__":
    # Example usage
    plot_roi_time_series_with_events('008', '04', 'tomloc', 1, 'TempPar')