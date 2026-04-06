"""
ToM Localizer — Subject-specific trial-onset-locked BOLD time courses
                + vertex-wise ROI activation surface maps.

For each subject, one figure with n_rois rows and 5 columns:
  Col 0 : ROI illustration (red patch on brain surface)
  Col 1 : Run-1  mean ± SEM timecourse (across trials)
  Col 2 : Run-2  mean ± SEM timecourse (across trials)
  Col 3 : Run-1 + Run-2 combined timecourse
  Col 4 : Belief − Photo activation map (combined runs, vertex-wise within ROI)

The surface column shows the mean baseline-corrected BOLD amplitude over the
full trial window (0 → 16.5 s), collapsed across trials, then differenced
(belief − photo).  Vertices outside the ROI are masked (NaN → transparent).

Window:  −6 s to +21 s from trial onset (TR=1.5 s)

Usage:
  # Scan data and print contrast distribution (to pick SURF_VMAX):
  uv run python tomloc_subject_timecourses.py --scan

  # Generate figures (uses SURF_VMAX defined below):
  uv run python tomloc_subject_timecourses.py

Output:
  figs/tomloc_schaefer/subject_timecourses/<subject>_trial_locked_timecourses.png
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
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

CONDITIONS_BY_RUN = {
    1: [1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    2: [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
}

TRIAL_ONSETS_S      = [12.0 + i * 28.5 for i in range(10)]
STIMULUS_DURATION_S = 16.5
STORY_OFFSET_S      = 12.0

TRS_BEFORE = 4
TRS_AFTER  = 14
TIME_VEC   = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR  # −6 … +21 s

# Epoch indices spanning [0, STIMULUS_DURATION_S] relative to onset
SURF_WIN_START = TRS_BEFORE                              # index of t=0
SURF_WIN_END   = TRS_BEFORE + round(STIMULUS_DURATION_S / TR) + 1  # exclusive

# ROI timecourse specs (labels averaged for the BOLD time-series plots)
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

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT  = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT  = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'

# Brain surface specs: one entry per ROI row.
# roi_labels  → used for the red-patch illustration (col 0).
# surf_labels → same labels, used for the activation surface map (col 4).
BRAIN_MAP_SPECS = [
    {'hemi': 'right', 'view': 'lateral', 'annot': RH_ANNOT,
     'roi_labels':  set(RIGHT_TPJ.get('right_labels', [])),
     'surf_labels':     RIGHT_TPJ.get('right_labels', [])},
    {'hemi': 'right', 'view': 'medial',  'annot': RH_ANNOT,
     'roi_labels':  set(POSTERIOR_MEDIAL.get('right_labels', [])),
     'surf_labels':     POSTERIOR_MEDIAL.get('right_labels', [])},
    {'hemi': 'right', 'view': 'lateral', 'annot': RH_ANNOT,
     'roi_labels':  set(ANGULAR_GYRUS.get('right_labels', [])),
     'surf_labels':     ANGULAR_GYRUS.get('right_labels', [])},
    {'hemi': 'left',  'view': 'medial',  'annot': LH_ANNOT,
     'roi_labels':  set(EARLY_VISUAL.get('left_labels', [])),
     'surf_labels':     EARLY_VISUAL.get('left_labels', [])},
]

# All individual parcel labels that appear in any surface column
ALL_SURF_LABELS = [
    label
    for spec in BRAIN_MAP_SPECS
    for label in spec['surf_labels']
]

# Universal colorbar limit for the belief−photo surface column (symmetric ±).
# Set to None to use per-ROI per-subject auto-scaling (useful during exploration).
# Run with --scan to print the empirical distribution and choose a value.
SURF_VMAX = 1.0

OUTPUT_DIR = FIGS_DIR / 'tomloc_schaefer' / 'subject_timecourses'


# ============================================================================
# HELPERS
# ============================================================================

def avg_labels(parcel_data, labels):
    ts_list = [parcel_data[l] for l in labels if l in parcel_data]
    if not ts_list:
        return None
    return np.mean(np.stack(ts_list, axis=0), axis=0)


def extract_onset_locked(signal, onset_trs):
    """
    Baseline-corrected epochs (n_trials, WINDOW_LEN).
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


def trial_stats(epochs):
    """Mean ± SEM across trials → (mean, sem, n)."""
    mean = epochs.mean(axis=0)
    sem  = epochs.std(axis=0) / np.sqrt(len(epochs))
    return mean, sem, len(epochs)


def parcel_activation(ts, onset_trs):
    """
    Scalar: mean baseline-corrected BOLD over [0, STIMULUS_DURATION_S],
    averaged across trials.  Returns NaN if no valid data.
    """
    epochs = extract_onset_locked(ts, onset_trs)
    if epochs is None:
        return np.nan
    return float(epochs[:, SURF_WIN_START:SURF_WIN_END].mean())


def build_texture(annot_path, label_to_value):
    """
    Per-vertex texture: NaN everywhere except vertices belonging to parcels
    in label_to_value, which receive that parcel's float value.
    """
    labels_arr, _ctab, names = fs.read_annot(str(annot_path))
    names = [n.decode() if hasattr(n, 'decode') else n for n in names]
    texture = np.full(len(labels_arr), np.nan, dtype=float)
    for name_idx, name in enumerate(names):
        if name in label_to_value:
            texture[labels_arr == name_idx] = label_to_value[name]
    return texture


def load_roi_texture(annot_path, roi_labels):
    """Binary texture (1 = ROI, 0 = elsewhere) for the red-patch illustration."""
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

def collect_subject_data(subject, session):
    """
    Returns
    -------
    run_epochs : dict
        run_num → roi_key → condition → ndarray (n_trials, WINDOW_LEN) or None
    run_surf_acts : dict
        run_num → condition → {label: float}   (per-parcel scalar activation)
        run_num → None  if BOLD missing
    """
    run_epochs    = {}
    run_surf_acts = {}

    for run_num, conditions in CONDITIONS_BY_RUN.items():
        task = f'tomloc{run_num}'
        bold_path = (DERIVATIVES_DIR / subject / session / 'func' /
                     f'{subject}_{session}_task-{task}'
                     f'_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')
        if not bold_path.exists():
            print(f"  run-{run_num}: BOLD not found, skipping")
            run_epochs[run_num]    = None
            run_surf_acts[run_num] = None
            continue

        try:
            parcel_data = get_parcel_data(subject, session, task, atlas='Schaefer400_17Nets')
        except Exception as e:
            print(f"  run-{run_num}: {e}")
            run_epochs[run_num]    = None
            run_surf_acts[run_num] = None
            continue

        onset_trs = {
            'belief': [round(TRIAL_ONSETS_S[i] / TR)
                       for i, c in enumerate(conditions) if c == 1],
            'photo':  [round(TRIAL_ONSETS_S[i] / TR)
                       for i, c in enumerate(conditions) if c == 0],
        }

        # ── ROI-averaged timecourse epochs ──────────────────────────────
        run_epochs[run_num] = {}
        for roi_key, _, labels in ROI_SPECS:
            ts = avg_labels(parcel_data, labels)
            run_epochs[run_num][roi_key] = {}
            if ts is None:
                print(f"    WARNING: no parcels for {roi_key}")
                for cond in ('belief', 'photo'):
                    run_epochs[run_num][roi_key][cond] = None
                continue
            for cond, trs in onset_trs.items():
                run_epochs[run_num][roi_key][cond] = extract_onset_locked(ts, trs)

        # ── Per-parcel scalar activations for the surface column ─────────
        run_surf_acts[run_num] = {cond: {} for cond in ('belief', 'photo')}
        for label in ALL_SURF_LABELS:
            ts = parcel_data.get(label)
            for cond, trs in onset_trs.items():
                if ts is None:
                    run_surf_acts[run_num][cond][label] = np.nan
                else:
                    run_surf_acts[run_num][cond][label] = parcel_activation(ts, trs)

        print(f"  run-{run_num}: {sum(c==1 for c in conditions)} belief, "
              f"{sum(c==0 for c in conditions)} photo trials")

    return run_epochs, run_surf_acts


def combined_surf_acts(run_surf_acts):
    """
    Average per-parcel activations across available runs.
    Returns  condition → {label: float}
    """
    result = {cond: {} for cond in ('belief', 'photo')}
    for cond in ('belief', 'photo'):
        for label in ALL_SURF_LABELS:
            vals = []
            for run_num in (1, 2):
                rd = run_surf_acts.get(run_num)
                if rd is not None:
                    v = rd[cond].get(label, np.nan)
                    if not np.isnan(v):
                        vals.append(v)
            result[cond][label] = float(np.mean(vals)) if vals else np.nan
    return result


# ============================================================================
# FIGURE
# ============================================================================

def make_subject_figure(subject, run_epochs, run_surf_acts, fsavg):
    n_rois = len(ROI_SPECS)

    # ── Timecourse stats (run-1, run-2, combined) ──────────────────────
    tc_col_data = []
    for run_num in (1, 2):
        col = {}
        rd = run_epochs.get(run_num)
        for cond in ('belief', 'photo'):
            col[cond] = {}
            for roi_key, _, _ in ROI_SPECS:
                if rd is None or rd[roi_key][cond] is None:
                    col[cond][roi_key] = None
                else:
                    col[cond][roi_key] = trial_stats(rd[roi_key][cond])
        tc_col_data.append(col)

    col_combined_tc = {}
    for cond in ('belief', 'photo'):
        col_combined_tc[cond] = {}
        for roi_key, _, _ in ROI_SPECS:
            parts = []
            for run_num in (1, 2):
                rd = run_epochs.get(run_num)
                if rd is not None and rd[roi_key][cond] is not None:
                    parts.append(rd[roi_key][cond])
            if parts:
                pooled = np.concatenate(parts, axis=0)
                col_combined_tc[cond][roi_key] = trial_stats(pooled)
            else:
                col_combined_tc[cond][roi_key] = None
    tc_col_data.append(col_combined_tc)

    # ── Surface contrast: belief − photo (combined runs) ──────────────
    surf_acts = combined_surf_acts(run_surf_acts)
    # Per ROI: {label: contrast_value}
    surf_contrasts = {}
    for spec, (roi_key, _, _) in zip(BRAIN_MAP_SPECS, ROI_SPECS):
        contrast = {}
        for label in spec['surf_labels']:
            b = surf_acts['belief'].get(label, np.nan)
            p = surf_acts['photo'].get(label, np.nan)
            contrast[label] = b - p if not (np.isnan(b) or np.isnan(p)) else np.nan
        surf_contrasts[roi_key] = contrast

    # ── Shared y-limits for timecourse panels ─────────────────────────
    all_vals = []
    for col in tc_col_data:
        for cond in ('belief', 'photo'):
            for roi_key, _, _ in ROI_SPECS:
                stats = col[cond][roi_key]
                if stats is not None:
                    m, s, _ = stats
                    all_vals.extend((m - s).tolist())
                    all_vals.extend((m + s).tolist())
    if not all_vals:
        print(f"  {subject}: no data, skipping figure")
        return
    val_range = max(all_vals) - min(all_vals)
    y_lim = (min(all_vals) - 0.07 * val_range,
             max(all_vals) + 0.07 * val_range)

    # ── Layout ────────────────────────────────────────────────────────
    # 5 cols: [illus | run-1 TC | run-2 TC | combined TC | surface map]
    fig = plt.figure(figsize=(20, 3.2 * n_rois))
    gs = gridspec.GridSpec(
        n_rois, 5,
        width_ratios=[1, 2, 2, 2, 1.2],
        hspace=0.5, wspace=0.25,
        left=0.03, right=0.98, top=0.90, bottom=0.07,
    )

    session = TOMLOC_SESSIONS[subject]
    fig.suptitle(
        f'ToM Localizer — Trial-onset-locked BOLD: {subject} ({session})\n'
        'Mean ± SEM across trials',
        fontsize=11, fontweight='bold',
    )

    tc_titles   = ['Run 1', 'Run 2', 'Run 1 + 2 combined']
    surf_title  = 'Belief − Photo\n(combined runs)'

    for col_idx, title in enumerate(tc_titles):
        fig.text(0.22 + col_idx * 0.205, 0.915,
                 title, ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    fig.text(0.935, 0.915, surf_title,
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    for i, ((roi_key, roi_name, _), brain_spec) in enumerate(
            zip(ROI_SPECS, BRAIN_MAP_SPECS)):

        mesh = fsavg.infl_right if brain_spec['hemi'] == 'right' else fsavg.infl_left
        bg   = fsavg.sulc_right if brain_spec['hemi'] == 'right' else fsavg.sulc_left

        # ── Col 0: ROI illustration ──────────────────────────────────
        ax_illus = fig.add_subplot(gs[i, 0], projection='3d')
        roi_tex  = load_roi_texture(brain_spec['annot'], brain_spec['roi_labels'])
        plotting.plot_surf_stat_map(
            surf_mesh=mesh, stat_map=roi_tex,
            hemi=brain_spec['hemi'], view=brain_spec['view'],
            bg_map=bg, axes=ax_illus, colorbar=False,
            cmap='autumn_r', vmin=0.0, vmax=1.0, threshold=0.5,
            bg_on_data=True, darkness=0.5,
        )
        ax_illus.set_title(roi_name, fontsize=9, fontweight='bold', pad=-10)

        # ── Cols 1–3: timecourse panels ──────────────────────────────
        for col_idx, col in enumerate(tc_col_data):
            ax = fig.add_subplot(gs[i, col_idx + 1])

            for cond in ('belief', 'photo'):
                stats = col[cond][roi_key]
                if stats is None:
                    continue
                m, s, n = stats
                color = COND_COLORS[cond]
                label = f"{cond.capitalize()} (n={n})" if i == 0 else None
                ax.plot(TIME_VEC, m, color=color, lw=2, label=label)
                ax.fill_between(TIME_VEC, m - s, m + s, color=color, alpha=0.2)

            ax.axvline(0,                   color='grey', ls='--', lw=1.0)
            ax.axvline(STORY_OFFSET_S,      color='grey', ls='--', lw=1.0)
            ax.axvline(STIMULUS_DURATION_S, color='grey', ls=':',  lw=1.0)
            ax.axhline(0, color='k', lw=0.5, alpha=0.3)

            if i == 0:
                ytop = y_lim[1]
                ax.text(0.25,                   ytop, 'story onset',
                        fontsize=6, color='dimgray', va='top', ha='left')
                ax.text(STORY_OFFSET_S + 0.25,  ytop, 'story offset /\nquestion onset',
                        fontsize=6, color='dimgray', va='top', ha='left')
                ax.text(STIMULUS_DURATION_S + 0.25, ytop, 'trial end',
                        fontsize=6, color='dimgray', va='top', ha='left')

            if col_idx == 0:
                ax.set_ylabel('BOLD (z)', fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.set_ylim(y_lim)
            ax.set_xlim(TIME_VEC[0], TIME_VEC[-1])
            ax.spines[['top', 'right']].set_visible(False)

            if i == n_rois - 1:
                ax.set_xlabel('Time from trial onset (s)', fontsize=9)
                ax.set_xticks(np.arange(-6, 22, 3))
            else:
                ax.set_xticklabels([])

            if i == 0:
                ax.legend(loc='lower left', fontsize=7, frameon=False)

        # ── Col 4: belief − photo surface map ────────────────────────
        ax_surf = fig.add_subplot(gs[i, 4], projection='3d')
        contrast = surf_contrasts[roi_key]
        surf_vals = [v for v in contrast.values() if not np.isnan(v)]

        if surf_vals:
            if SURF_VMAX is not None:
                vmax = SURF_VMAX
            else:
                vmax = max(abs(min(surf_vals)), abs(max(surf_vals)))
                vmax = vmax if vmax > 0 else 1.0
            surf_tex = build_texture(brain_spec['annot'], contrast)
            plotting.plot_surf_stat_map(
                surf_mesh=mesh, stat_map=surf_tex,
                hemi=brain_spec['hemi'], view=brain_spec['view'],
                bg_map=bg, axes=ax_surf,
                colorbar=True,
                cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                threshold=None,
                bg_on_data=True, darkness=0.5,
            )
        else:
            ax_surf.set_visible(False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f'{subject}_trial_locked_timecourses.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")


# ============================================================================
# SCAN MODE — print contrast distribution to choose SURF_VMAX
# ============================================================================

def scan_contrast_distribution():
    """
    Collect all belief−photo contrast values across subjects and ROIs,
    print percentile table, and suggest a SURF_VMAX value.
    """
    all_contrasts = {roi_key: [] for roi_key, _, _ in ROI_SPECS}

    for subject, session in TOMLOC_SESSIONS.items():
        print(f"  loading {subject} …")
        _, run_surf_acts = collect_subject_data(subject, session)
        surf_acts = combined_surf_acts(run_surf_acts)

        for spec, (roi_key, _, _) in zip(BRAIN_MAP_SPECS, ROI_SPECS):
            for label in spec['surf_labels']:
                b = surf_acts['belief'].get(label, np.nan)
                p = surf_acts['photo'].get(label, np.nan)
                if not (np.isnan(b) or np.isnan(p)):
                    all_contrasts[roi_key].append(b - p)

    print('\n' + '=' * 56)
    print(f"{'ROI':8s}  {'n':>4}  {'min':>7}  {'p5':>7}  {'p25':>7}  "
          f"{'med':>7}  {'p75':>7}  {'p95':>7}  {'max':>7}")
    print('-' * 56)

    all_vals = []
    for roi_key, roi_name, _ in ROI_SPECS:
        vals = np.array(all_contrasts[roi_key])
        all_vals.extend(vals.tolist())
        if len(vals) == 0:
            print(f"{roi_key:8s}  {'—':>4}")
            continue
        p = np.percentile(vals, [5, 25, 50, 75, 95])
        print(f"{roi_key:8s}  {len(vals):>4d}  "
              f"{vals.min():>7.3f}  {p[0]:>7.3f}  {p[1]:>7.3f}  "
              f"{p[2]:>7.3f}  {p[3]:>7.3f}  {p[4]:>7.3f}  {vals.max():>7.3f}")

    print('=' * 56)
    all_vals = np.array(all_vals)
    p95_abs  = np.percentile(np.abs(all_vals), 95)
    p99_abs  = np.percentile(np.abs(all_vals), 99)
    abs_max  = np.abs(all_vals).max()
    print(f"\nAll ROIs combined (n={len(all_vals)}):")
    print(f"  |contrast| p95 = {p95_abs:.3f}")
    print(f"  |contrast| p99 = {p99_abs:.3f}")
    print(f"  |contrast| max = {abs_max:.3f}")
    print(f"\nSuggested SURF_VMAX values:")
    print(f"  conservative (p95):  {p95_abs:.2f}")
    print(f"  liberal     (p99):  {p99_abs:.2f}")
    print(f"  (set SURF_VMAX in the CONFIG section above)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys
    if '--scan' in sys.argv:
        print('Scanning contrast distribution across all subjects …\n')
        scan_contrast_distribution()
        return

    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    for subject, session in TOMLOC_SESSIONS.items():
        print(f"\n{'─' * 50}")
        print(f"{subject}  {session}")
        run_epochs, run_surf_acts = collect_subject_data(subject, session)
        make_subject_figure(subject, run_epochs, run_surf_acts, fsavg)

    print('\nDone.')


if __name__ == '__main__':
    main()
