"""
ToM Localizer — Trial-onset-locked BOLD time courses.

For each ROI (rTPJ, PMC, AG, EVC), plots mean ± SEM BOLD response locked
to trial onset, separately for belief and photo conditions.  A small brain
surface map showing the ROI's location is placed to the left of each row.

Pipeline:
  - Per subject × run: extract parcel time series via get_parcel_data
  - Average across ROI parcels
  - Average trials within subject × run, then average across runs per subject
  - Group mean ± SEM computed across subjects

Window:  -6 s to +21 s from trial onset (TR=1.5 s: -4 to +14 TRs)
Markers:
  dashed line at t=0        → trial onset
  dashed line at t=12 s     → story offset / question onset
  dotted line at t=16.5 s   → trial end

Output:
  figs/tomloc_schaefer/trial_locked_timecourses.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel.freesurfer as fs
from nilearn import datasets, plotting

from configs.config import DERIVATIVES_DIR, CACHE_DIR, FIGS_DIR, TR
from configs.schaefer_rois import POSTERIOR_MEDIAL, ANGULAR_GYRUS, EARLY_VISUAL, RIGHT_TPJ
from fmrianalysis.utils import get_parcel_data

# ============================================================================
# CONFIG
# ============================================================================

TOMLOC_SESSIONS = {
    'sub-001': 'ses-04',
    'sub-003': 'ses-08',
    'sub-004': 'ses-05',
    'sub-006': 'ses-04',
    'sub-007': 'ses-04',
    'sub-008': 'ses-04',
    'sub-009': 'ses-04',
}

# 1 = belief, 0 = photo
CONDITIONS_BY_RUN = {
    1: [1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    2: [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
}

# Trial onsets: initial 12 s fixation, then 10 trials × 28.5 s cycle
TRIAL_ONSETS_S = [12.0 + i * 28.5 for i in range(10)]
STIMULUS_DURATION_S = 16.5   # trial end
STORY_OFFSET_S      = 12.0   # story offset / question onset

TRS_BEFORE = 4   # = 6.0 s
TRS_AFTER  = 14  # = 21.0 s
TIME_VEC   = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR  # -6.0 … +21.0 s

ROI_SPECS = [
    ('rtpj', 'Right TPJ',
     RIGHT_TPJ.get('right_labels', [])),
    ('pmc',  'Posterior Medial Cortex',
     POSTERIOR_MEDIAL.get('left_labels', []) + POSTERIOR_MEDIAL.get('right_labels', [])),
    ('ag',   'Angular Gyrus',
     ANGULAR_GYRUS.get('left_labels', [])   + ANGULAR_GYRUS.get('right_labels', [])),
    ('evc',  'Early Visual Cortex',
     EARLY_VISUAL.get('left_labels', [])    + EARLY_VISUAL.get('right_labels', [])),
]

COND_COLORS = {'belief': '#d62728', 'photo': '#1f77b4'}
OUTPUT_DIR = FIGS_DIR / 'tomloc_schaefer'

# Schaefer fsaverage6 annotation files
ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT  = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT  = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'

# Per-ROI brain surface map specs (hemi, view, which annot file, which labels)
BRAIN_MAP_SPECS = [
    # rTPJ — right lateral
    {'hemi': 'right', 'view': 'lateral', 'annot': RH_ANNOT,
     'roi_labels': set(RIGHT_TPJ.get('right_labels', []))},
    # PMC (pCunPCC) — right medial
    {'hemi': 'right', 'view': 'medial',  'annot': RH_ANNOT,
     'roi_labels': set(POSTERIOR_MEDIAL.get('right_labels', []))},
    # AG (IPL) — right lateral
    {'hemi': 'right', 'view': 'lateral', 'annot': RH_ANNOT,
     'roi_labels': set(ANGULAR_GYRUS.get('right_labels', []))},
    # EVC (striate/calcarine) — left medial
    {'hemi': 'left',  'view': 'medial',  'annot': LH_ANNOT,
     'roi_labels': set(EARLY_VISUAL.get('left_labels', []))},
]


# ============================================================================
# HELPERS
# ============================================================================

def avg_labels(parcel_data, labels):
    """Average time series across parcel labels present in parcel_data."""
    ts_list = [parcel_data[l] for l in labels if l in parcel_data]
    if not ts_list:
        return None
    return np.mean(np.stack(ts_list, axis=0), axis=0)


def extract_onset_locked(signal, onset_trs):
    """
    Extract baseline-corrected epochs around each onset index.

    Baseline = mean of the TRS_BEFORE pre-onset timepoints.

    Returns (n_valid_trials, WINDOW_LEN) or None.
    """
    n = len(signal)
    epochs = []
    for tr_idx in onset_trs:
        start = tr_idx - TRS_BEFORE
        end   = tr_idx + TRS_AFTER + 1
        if start >= 0 and end <= n:
            epoch = signal[start:end].astype(float)
            epochs.append(epoch)
    return np.stack(epochs) if epochs else None   # (n_trials, WINDOW_LEN)


def load_roi_texture(annot_path, roi_labels):
    """
    Create a per-vertex texture array: 1.0 for ROI vertices, 0.0 elsewhere.

    roi_labels : set of full parcel name strings matching the annot file.
    """
    labels_arr, _ctab, names = fs.read_annot(str(annot_path))
    names = [n.decode() if hasattr(n, 'decode') else n for n in names]

    texture = np.zeros(len(labels_arr), dtype=float)
    for name_idx, name in enumerate(names):
        if name in roi_labels:
            texture[labels_arr == name_idx] = 1.0
    return texture


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_subject_means():
    """
    Returns subj_means : dict
        subject → condition → roi_key → np.ndarray (WINDOW_LEN,) or None
    """
    subj_means = {}

    for subject, session in TOMLOC_SESSIONS.items():
        print(f"\n{'─' * 40}")
        print(f"{subject}  {session}")
        subj_means[subject] = {cond: {k: [] for k, _, _ in ROI_SPECS}
                               for cond in ('belief', 'photo')}

        for run_num, conditions in CONDITIONS_BY_RUN.items():
            task = f'tomloc{run_num}'
            bold_path = (DERIVATIVES_DIR / subject / session / 'func' /
                         f'{subject}_{session}_task-{task}'
                         f'_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')
            if not bold_path.exists():
                print(f"  run-{run_num}: BOLD not found, skipping")
                continue

            try:
                parcel_data = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
            except Exception as e:
                print(f"  run-{run_num}: {e}")
                continue

            # Trial onset indices for each condition
            onset_trs = {
                'belief': [round(TRIAL_ONSETS_S[i] / TR)
                           for i, c in enumerate(conditions) if c == 1],
                'photo':  [round(TRIAL_ONSETS_S[i] / TR)
                           for i, c in enumerate(conditions) if c == 0],
            }

            for roi_key, roi_name, labels in ROI_SPECS:
                ts = avg_labels(parcel_data, labels)
                if ts is None:
                    print(f"    WARNING: no parcels for {roi_key}")
                    continue
                for cond, trs in onset_trs.items():
                    epochs = extract_onset_locked(ts, trs)
                    if epochs is not None:
                        subj_means[subject][cond][roi_key].append(epochs.mean(axis=0))

            print(f"  run-{run_num}: extracted "
                  f"{sum(c == 1 for c in conditions)} belief, "
                  f"{sum(c == 0 for c in conditions)} photo trials")

        # Collapse across runs: mean of run-level means per subject
        for cond in ('belief', 'photo'):
            for roi_key, _, _ in ROI_SPECS:
                runs = subj_means[subject][cond][roi_key]
                subj_means[subject][cond][roi_key] = (
                    np.stack(runs).mean(axis=0) if runs else None
                )

    return subj_means


def compute_group_stats(subj_means):
    """Group mean ± SEM across subjects."""
    group = {}
    for cond in ('belief', 'photo'):
        group[cond] = {}
        for roi_key, _, _ in ROI_SPECS:
            valid = [subj_means[s][cond][roi_key]
                     for s in TOMLOC_SESSIONS
                     if subj_means[s][cond][roi_key] is not None]
            if valid:
                arr = np.stack(valid)   # (n_subj, WINDOW_LEN)
                group[cond][roi_key] = {
                    'mean': arr.mean(axis=0),
                    'sem':  arr.std(axis=0) / np.sqrt(len(arr)),
                    'n':    len(arr),
                    'subj': arr,
                }
            else:
                group[cond][roi_key] = None
    return group


# ============================================================================
# FIGURE
# ============================================================================

def make_figure(group):
    n_rois = len(ROI_SPECS)

    # Fetch fsaverage6 surfaces once
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    # ------------------------------------------------------------------
    # Compute global y limits across all ROIs (same for every subplot)
    # ------------------------------------------------------------------
    all_vals = []
    for roi_key, _, _ in ROI_SPECS:
        for cond in ('belief', 'photo'):
            g = group[cond][roi_key]
            if g is not None:
                all_vals.extend((g['mean'] - g['sem']).tolist())
                all_vals.extend((g['mean'] + g['sem']).tolist())
    val_range = max(all_vals) - min(all_vals)
    y_lim = (min(all_vals) - 0.07 * val_range,
             max(all_vals) + 0.07 * val_range)

    # ------------------------------------------------------------------
    # Layout: 2 columns — [brain surface (3D)] | [timecourse (2D)]
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 3.2 * n_rois))
    gs = gridspec.GridSpec(
        n_rois, 2,
        width_ratios=[1, 2.5],
        hspace=0.5, wspace=0.02,
        left=0.04, right=0.98, top=0.92, bottom=0.06,
    )

    fig.suptitle(
        'ToM Localizer — Trial-onset-locked BOLD (belief vs. photo)\n'
        'Mean ± SEM across subjects',
        fontsize=11, fontweight='bold',
    )

    tc_axes = []

    for i, ((roi_key, roi_name, _), brain_spec) in enumerate(
            zip(ROI_SPECS, BRAIN_MAP_SPECS)):

        # ── Brain surface ──────────────────────────────────────────────
        brain_ax = fig.add_subplot(gs[i, 0], projection='3d')

        texture = load_roi_texture(brain_spec['annot'], brain_spec['roi_labels'])
        mesh = fsavg.infl_right if brain_spec['hemi'] == 'right' else fsavg.infl_left
        bg   = fsavg.sulc_right if brain_spec['hemi'] == 'right' else fsavg.sulc_left

        plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=texture,
            hemi=brain_spec['hemi'],
            view=brain_spec['view'],
            bg_map=bg,
            axes=brain_ax,
            colorbar=False,
            cmap='autumn_r',
            vmin=0.0,
            vmax=1.0,
            threshold=0.5,
            bg_on_data=True,
            darkness=0.5,
        )
        brain_ax.set_title(roi_name, fontsize=9, fontweight='bold', pad=-10)

        # ── Timecourse ─────────────────────────────────────────────────
        ax = fig.add_subplot(gs[i, 1])
        tc_axes.append(ax)

        for cond in ('belief', 'photo'):
            g = group[cond][roi_key]
            if g is None:
                continue
            color = COND_COLORS[cond]
            label = f"{cond.capitalize()} (N={g['n']})" if i == 0 else None
            ax.plot(TIME_VEC, g['mean'], color=color, lw=2.5, label=label)
            ax.fill_between(TIME_VEC,
                            g['mean'] - g['sem'],
                            g['mean'] + g['sem'],
                            color=color, alpha=0.2)

        # Reference lines
        ax.axvline(0,                   color='grey', ls='--', lw=1.0)
        ax.axvline(STORY_OFFSET_S,      color='grey', ls='--', lw=1.0)
        ax.axvline(STIMULUS_DURATION_S, color='grey', ls=':',  lw=1.0)
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)

        # Label vertical lines on first row only
        if i == 0:
            ytop = y_lim[1]
            ax.text(0 + 0.25,                   ytop, 'story onset',
                    fontsize=7, color='dimgray', va='top', ha='left')
            ax.text(STORY_OFFSET_S + 0.25,      ytop, 'story offset /\nquestion onset',
                    fontsize=7, color='dimgray', va='top', ha='left')
            ax.text(STIMULUS_DURATION_S + 0.25, ytop, 'trial end',
                    fontsize=7, color='dimgray', va='top', ha='left')

        ax.set_ylabel('BOLD (z)', fontsize=9)
        ax.set_ylim(y_lim)
        ax.set_xlim(TIME_VEC[0], TIME_VEC[-1])
        ax.spines[['top', 'right']].set_visible(False)

        if i == 0:
            ax.legend(loc='lower left', fontsize=8, frameon=False, ncol=2)

    tc_axes[-1].set_xlabel('Time from trial onset (s)', fontsize=10)
    tc_axes[-1].set_xticks(np.arange(-6, 22, 3))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / 'trial_locked_timecourses.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    subj_means = collect_subject_means()
    group = compute_group_stats(subj_means)

    print('\nGroup N per condition/ROI:')
    for cond in ('belief', 'photo'):
        for roi_key, roi_name, _ in ROI_SPECS:
            g = group[cond][roi_key]
            n = g['n'] if g else 0
            print(f'  {cond:7s}  {roi_key:6s}: N={n}')

    make_figure(group)


if __name__ == '__main__':
    main()
