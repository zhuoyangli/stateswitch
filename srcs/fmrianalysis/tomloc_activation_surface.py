"""
ToM Localizer — Vertex-wise PMC activation surface maps, per subject.

For each PMC parcel, computes the mean baseline-corrected BOLD amplitude
over the full trial window (0 → 16.5 s) averaged across trials.  The
activation value is projected onto the cortical surface and plotted with
everything outside PMC masked (NaN).

Figure layout per subject:
  Rows: belief (LH medial), belief (RH medial),
        photo  (LH medial), photo  (RH medial)
  Cols: Run 1 | Run 2 | Run 1+2 combined

Output:
  figs/tomloc_schaefer/pmc_surface/<subject>_pmc_surface.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import nibabel.freesurfer as fs
from nilearn import datasets, plotting

from configs.config import DERIVATIVES_DIR, CACHE_DIR, FIGS_DIR, TR
from configs.schaefer_rois import POSTERIOR_MEDIAL
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

CONDITIONS_BY_RUN = {
    1: [1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    2: [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
}

TRIAL_ONSETS_S      = [12.0 + i * 28.5 for i in range(10)]
STIMULUS_DURATION_S = 16.5   # trial end relative to onset
STORY_OFFSET_S      = 12.0

TRS_BEFORE = 4                        # baseline window before onset
TRS_AFTER  = 14
# TRs that fall within 0 → STIMULUS_DURATION_S relative to onset
WINDOW_START_IDX = TRS_BEFORE         # TR index 0 in epoch = onset
WINDOW_END_IDX   = TRS_BEFORE + int(STIMULUS_DURATION_S / TR) + 1

PMC_LEFT_LABELS  = POSTERIOR_MEDIAL['left_labels']   # LH parcels
PMC_RIGHT_LABELS = POSTERIOR_MEDIAL['right_labels']  # RH parcels
PMC_ALL_LABELS   = PMC_LEFT_LABELS + PMC_RIGHT_LABELS

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT  = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT  = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'

OUTPUT_DIR = FIGS_DIR / 'tomloc_schaefer' / 'pmc_surface'

CMAP = 'RdBu_r'


# ============================================================================
# HELPERS
# ============================================================================

def extract_onset_locked(signal, onset_trs):
    """
    Baseline-corrected epochs: (n_trials, WINDOW_LEN).
    Baseline = mean of TRS_BEFORE pre-onset TRs.
    Returns None if no valid trials.
    """
    n = len(signal)
    epochs = []
    for tr_idx in onset_trs:
        start = tr_idx - TRS_BEFORE
        end   = tr_idx + TRS_AFTER + 1
        if start >= 0 and end <= n:
            epoch = signal[start:end].astype(float)
            baseline = epoch[:TRS_BEFORE].mean()
            epochs.append(epoch - baseline)
    return np.stack(epochs) if epochs else None


def parcel_activation(epochs):
    """
    Mean BOLD amplitude over the trial window, averaged across trials.
    Returns scalar or NaN.
    """
    if epochs is None:
        return np.nan
    trial_means = epochs[:, WINDOW_START_IDX:WINDOW_END_IDX].mean(axis=1)
    return float(trial_means.mean())


def build_texture(annot_path, label_to_value):
    """
    Build per-vertex texture: NaN everywhere, except vertices belonging to
    parcels listed in label_to_value get that parcel's float value.

    label_to_value : dict  parcel_name → float activation
    """
    labels_arr, _ctab, names = fs.read_annot(str(annot_path))
    names = [n.decode() if hasattr(n, 'decode') else n for n in names]
    texture = np.full(len(labels_arr), np.nan, dtype=float)
    for name_idx, name in enumerate(names):
        if name in label_to_value:
            texture[labels_arr == name_idx] = label_to_value[name]
    return texture


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_run_parcel_activations(subject, session):
    """
    Returns:
        run_acts : dict  run_num → condition → parcel_label → float (or nan)
        Returns None for runs whose BOLD file is missing.
    """
    run_acts = {}
    for run_num, conditions in CONDITIONS_BY_RUN.items():
        task = f'tomloc{run_num}'
        bold_path = (DERIVATIVES_DIR / subject / session / 'func' /
                     f'{subject}_{session}_task-{task}'
                     f'_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')
        if not bold_path.exists():
            print(f"  run-{run_num}: BOLD not found, skipping")
            run_acts[run_num] = None
            continue

        try:
            parcel_data = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
        except Exception as e:
            print(f"  run-{run_num}: {e}")
            run_acts[run_num] = None
            continue

        onset_trs = {
            'belief': [round(TRIAL_ONSETS_S[i] / TR)
                       for i, c in enumerate(conditions) if c == 1],
            'photo':  [round(TRIAL_ONSETS_S[i] / TR)
                       for i, c in enumerate(conditions) if c == 0],
        }

        run_acts[run_num] = {cond: {} for cond in ('belief', 'photo')}
        for label in PMC_ALL_LABELS:
            ts = parcel_data.get(label)
            for cond, trs in onset_trs.items():
                if ts is None:
                    run_acts[run_num][cond][label] = np.nan
                else:
                    epochs = extract_onset_locked(ts, trs)
                    run_acts[run_num][cond][label] = parcel_activation(epochs)

        print(f"  run-{run_num}: {sum(c==1 for c in conditions)} belief, "
              f"{sum(c==0 for c in conditions)} photo trials")

    return run_acts


def combined_activations(run_acts):
    """
    Pool epochs from both runs before averaging.
    Returns dict  condition → parcel_label → float (or nan).
    """
    # We need raw epochs per run to pool properly; here we weight by trial counts.
    # Since CONDITIONS_BY_RUN has fixed trial counts per condition per run, we
    # can just average the per-run activation values weighted equally.
    combined = {cond: {} for cond in ('belief', 'photo')}
    for cond in ('belief', 'photo'):
        for label in PMC_ALL_LABELS:
            vals = []
            for run_num in (1, 2):
                rd = run_acts.get(run_num)
                if rd is not None:
                    v = rd[cond].get(label, np.nan)
                    if not np.isnan(v):
                        vals.append(v)
            combined[cond][label] = float(np.mean(vals)) if vals else np.nan
    return combined


# ============================================================================
# FIGURE
# ============================================================================

def make_surface_figure(subject, run_acts, fsavg):
    # Build per-column label→value maps
    # col_data[col_idx][cond] = {label: float}
    col_data = []
    for run_num in (1, 2):
        rd = run_acts.get(run_num)
        if rd is None:
            col_data.append(None)
        else:
            col_data.append(rd)
    col_data.append(combined_activations(run_acts))  # combined

    col_titles = ['Run 1', 'Run 2', 'Run 1 + 2 combined']

    # Determine shared symmetric color scale across all non-NaN values
    all_vals = []
    for col in col_data:
        if col is None:
            continue
        for cond in ('belief', 'photo'):
            for label in PMC_ALL_LABELS:
                v = col[cond].get(label, np.nan)
                if not np.isnan(v):
                    all_vals.append(v)
    if not all_vals:
        print(f"  {subject}: no data, skipping")
        return
    vmax = max(abs(min(all_vals)), abs(max(all_vals)))
    vmin = -vmax

    # ── Layout ────────────────────────────────────────────────────────────
    # 4 rows: [belief-LH, belief-RH, photo-LH, photo-RH]
    # 3 cols: [run-1, run-2, combined]
    # Extra right column for colorbar
    n_rows, n_cols = 4, 3
    fig = plt.figure(figsize=(13, 12))
    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1, 1, 1, 0.06],
        hspace=0.05, wspace=0.02,
        left=0.06, right=0.97, top=0.90, bottom=0.03,
    )

    session = TOMLOC_SESSIONS[subject]
    fig.suptitle(
        f'ToM Localizer — PMC activation (belief vs. photo): {subject} ({session})\n'
        'Mean BOLD over trial window [0 → 16.5 s], averaged across trials',
        fontsize=11, fontweight='bold',
    )

    # Column titles
    for col_idx, title in enumerate(col_titles):
        fig.text(
            0.08 + col_idx * 0.295, 0.916,
            title, ha='center', va='bottom', fontsize=10, fontweight='bold',
        )

    # Row labels
    row_labels = [
        'Belief\nLH medial', 'Belief\nRH medial',
        'Photo\nLH medial',  'Photo\nRH medial',
    ]
    for row_idx, rl in enumerate(row_labels):
        fig.text(
            0.005, 0.83 - row_idx * 0.21,
            rl, ha='left', va='center', fontsize=9, fontweight='bold',
            rotation=0,
        )

    # Hemisphere/view specs per row
    row_specs = [
        # (cond, hemi, view, annot, labels_subset)
        ('belief', 'left',  'medial', LH_ANNOT, PMC_LEFT_LABELS),
        ('belief', 'right', 'medial', RH_ANNOT, PMC_RIGHT_LABELS),
        ('photo',  'left',  'medial', LH_ANNOT, PMC_LEFT_LABELS),
        ('photo',  'right', 'medial', RH_ANNOT, PMC_RIGHT_LABELS),
    ]

    for row_idx, (cond, hemi, view, annot, hemi_labels) in enumerate(row_specs):
        mesh = fsavg.infl_left  if hemi == 'left' else fsavg.infl_right
        bg   = fsavg.sulc_left  if hemi == 'left' else fsavg.sulc_right

        for col_idx, col in enumerate(col_data):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d')

            if col is None:
                ax.set_visible(False)
                continue

            # Build vertex texture for this hemisphere
            label_to_value = {
                label: col[cond].get(label, np.nan)
                for label in hemi_labels
            }
            texture = build_texture(annot, label_to_value)

            # threshold set just below |vmin| so NaN-equivalent 0s in the
            # background are not shown; NaN vertices are naturally invisible
            plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=texture,
                hemi=hemi,
                view=view,
                bg_map=bg,
                axes=ax,
                colorbar=False,
                cmap=CMAP,
                vmin=vmin,
                vmax=vmax,
                threshold=None,
                bg_on_data=True,
                darkness=0.5,
            )

    # ── Colorbar ──────────────────────────────────────────────────────────
    cbar_ax = fig.add_subplot(gs[:, -1])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cbar_ax)
    cb.set_label('Mean BOLD (z, baseline-corrected)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f'{subject}_pmc_surface.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    for subject, session in TOMLOC_SESSIONS.items():
        print(f"\n{'─' * 50}")
        print(f"{subject}  {session}")
        run_acts = collect_run_parcel_activations(subject, session)
        make_surface_figure(subject, run_acts, fsavg)

    print('\nDone.')


if __name__ == '__main__':
    main()
