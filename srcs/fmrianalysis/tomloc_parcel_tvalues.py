"""
ToM localizer: per-parcel t-values, histograms, and top parcels.

For each subject × run, fit a Schaefer-parcel GLM for the belief-photo
contrast, plot a histogram of t-values, and print the 20 parcels with the
highest positive t-values.

Output:
  figs/tomloc_schaefer/parcel_tvalue_histograms.png

Usage:
    python srcs/fmrianalysis/tomloc_parcel_tvalues.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, SUBJECT_IDS

# subject -> (session, run1_exists, run2_exists) — auto-discovered
TOMLOC_SESSIONS = {
    'sub-001': 'ses-04',
    'sub-003': 'ses-08',
    'sub-004': 'ses-05',
    'sub-006': 'ses-04',
    'sub-007': 'ses-04',
    'sub-008': 'ses-04',
    'sub-009': 'ses-04',
}


def generate_tomloc_events(run_num):
    conditions_run1 = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
    conditions_run2 = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
    conditions = conditions_run1 if run_num == 1 else conditions_run2

    events, current_time = [], 12.0
    for is_belief in conditions:
        events.append({
            'onset': current_time,
            'duration': 16.5,
            'trial_type': 'belief' if is_belief else 'photo',
        })
        current_time += 16.5 + 12.0
    return pd.DataFrame(events)


def compute_parcel_tvalues(subject, session, run):
    """Return (t_values array shape (400,), labels list) for belief-photo contrast."""
    task_name = f'tomloc{run}'
    bold_path = (DERIVATIVES_DIR / subject / session / 'func' /
                 f'{subject}_{session}_task-{task_name}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')

    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)

    # Parcel labels (filter Background if present)
    raw_labels = schaefer['labels']
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in raw_labels]
    labels = [l for l in all_labels if l != 'Background']

    masker = NiftiLabelsMasker(
        labels_img=schaefer['maps'],
        standardize='zscore_sample',
        memory='nilearn_cache',
        verbose=0,
    )
    time_series = masker.fit_transform(str(bold_path))  # (T, n_parcels)
    n_trs, n_parcels = time_series.shape

    # Wrap in pseudo-NIfTI for FirstLevelModel
    parcel_nii = nib.Nifti1Image(
        time_series.T[:, np.newaxis, np.newaxis, :], affine=np.eye(4)
    )
    mask_nii = nib.Nifti1Image(np.ones((n_parcels, 1, 1), dtype=np.int8), affine=np.eye(4))

    events_df = generate_tomloc_events(run)
    frame_times = np.arange(n_trs) * TR
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model='glover + derivative',
        drift_model='cosine',
        high_pass=0.01,
    )

    glm = FirstLevelModel(mask_img=mask_nii)
    glm.fit(parcel_nii, design_matrices=design_matrix)
    t_map = glm.compute_contrast('belief - photo', stat_type='t', output_type='stat')
    t_values = t_map.get_fdata()[:, 0, 0]  # (n_parcels,)

    return t_values, labels


def main():
    results = []
    for subject, session in TOMLOC_SESSIONS.items():
        for run in (1, 2):
            bold_path = (DERIVATIVES_DIR / subject / session / 'func' /
                         f'{subject}_{session}_task-tomloc{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')
            if not bold_path.exists():
                print(f"SKIP {subject} {session} run-{run}: BOLD not found")
                continue
            print(f"Processing {subject} {session} run-{run}...")
            try:
                t_values, labels = compute_parcel_tvalues(subject, session, run)
                results.append((subject, session, run, t_values, labels))
            except Exception as e:
                print(f"  ERROR: {e}")

    if not results:
        print("No results.")
        return

    # Figure: 7 rows × 2 cols (one row per subject, one col per run)
    n_subjects = len(TOMLOC_SESSIONS)
    fig, axes = plt.subplots(n_subjects, 2, figsize=(12, 3.5 * n_subjects),
                              sharex=False, sharey=False)
    fig.suptitle('ToM Localizer — t-values across Schaefer 400 parcels (belief − photo)',
                 fontsize=13, fontweight='bold')

    # Build lookup for subject row
    subject_order = list(TOMLOC_SESSIONS.keys())

    for subject, session, run, t_values, labels in results:
        row = subject_order.index(subject)
        col = run - 1
        ax = axes[row, col]

        ax.hist(t_values, bins=40, color='steelblue', edgecolor='none', alpha=0.85)
        ax.axvline(0, color='k', lw=0.8, linestyle='--', alpha=0.6)
        ax.set_title(f'{subject} {session}  run-{run}', fontsize=9)
        ax.set_xlabel('t-value', fontsize=8)
        ax.set_ylabel('parcels', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide any unused panels (e.g. if a run is missing)
    plotted = {(s, r) for s, _, r, _, _ in results}
    for subject, session in TOMLOC_SESSIONS.items():
        row = subject_order.index(subject)
        for run in (1, 2):
            if (subject, run) not in plotted:
                axes[row, run - 1].set_visible(False)

    plt.tight_layout()

    out_dir = FIGS_DIR / 'tomloc_schaefer'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'parcel_tvalue_histograms.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {out_path}")

    # -------------------------------------------------------------------------
    # Print and save top-20 parcel rankings
    # -------------------------------------------------------------------------
    txt_path = out_dir / 'top20_parcels.txt'
    lines = ['Top 20 Schaefer parcels by t-value (belief − photo)\n']
    for subject, session, run, t_values, labels in results:
        header = f"\n{'=' * 66}\n{subject}  {session}  run-{run}\n{'=' * 66}"
        lines.append(header)
        top20_idx = np.argsort(t_values)[::-1][:20]
        for rank, idx in enumerate(top20_idx, 1):
            lines.append(f"  {rank:2d}. {labels[idx]:<60s}  t = {t_values[idx]:.3f}")

    report = '\n'.join(lines)
    print(report)
    txt_path.write_text(report)
    print(f"\nSaved → {txt_path}")


if __name__ == '__main__':
    main()
