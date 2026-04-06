"""
Silence-Locked Control Analysis for Task Boundaries

Locks ROI time courses to long silent gaps (≥ 10 s) in the audio recording,
excluding silences within 30 s of any actual trial offset (boundary event).

If the silence-locked signal resembles the boundary-locked signal in task_boundary.py,
that would suggest the boundary effect could be driven by silence itself. A flat
signal here argues against that confound.

Layout: 6 rows (5 ROIs + audio envelope) × 2 columns (SVF, AHC).
Time window: -30 to +60 s from silence onset.

Usage:
    uv run srcs/fmrianalysis/silence_control.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# === CONFIG ===
from configs.config import DATA_DIR, FIGS_DIR

# Import everything we need from task_boundary
from fmrianalysis.task_boundary import (
    extract_roi_timeseries,
    extract_event_locked,
    parse_all_trial_boundaries,
    find_psychopy_csv,
    discover_sessions,
    load_audio_envelope,
    extract_audio_event_locked,
    find_audio_wav,
    AUDIO_SCAN_OFFSET,
    AUDIO_TARGET_SR,
    AUDIO_PRE,
    AUDIO_POST,
    TRS_BEFORE,
    TRS_AFTER,
    TR,
    ROI_SPEC,
    SUBJECT_IDS,
    SUBJECT_COLORS,
    NEXT_TRIAL_ONSET,
    TITLE_FS,
    LABEL_FS,
)

SVF_ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
AHC_ANNOTATIONS_DIR = DATA_DIR / "rec/ahc_sentences"
OUTPUT_DIR = FIGS_DIR / 'task_boundary'

MIN_SILENCE_DURATION = 10.0   # seconds
EXCLUSION_RADIUS = 30.0       # seconds around any real boundary


# ============================================================================
# SILENCE EXTRACTION
# ============================================================================

def _find_svf_annotation(subject, session):
    """Find SVF annotation CSV for a given subject/session, if it exists."""
    pattern = f"{subject}_{session}_task-svf_desc-wordtimestampswithswitch.csv"
    path = SVF_ANNOTATIONS_DIR / pattern
    return path if path.exists() else None


def _find_ahc_annotation(subject, session):
    """Find AHC annotation XLSX for a given subject/session, if it exists."""
    pattern = f"{subject}_{session}_task-ahc_desc-sentences.xlsx"
    path = AHC_ANNOTATIONS_DIR / pattern
    return path if path.exists() else None


def extract_svf_silences(csv_path):
    """Return list of (silence_onset_scan_relative, silence_duration) for long silences.

    Silence onset is the end of the previous word, in scan-relative seconds.
    Excludes 'next' words.
    """
    df = pd.read_csv(csv_path)
    # Exclude 'next' words
    df = df[~df['word_clean'].str.lower().str.strip().str.startswith('next')].copy()
    df = df.sort_values('start').reset_index(drop=True)

    silences = []
    for i in range(1, len(df)):
        gap_start_wav = df['end'].iloc[i - 1]
        gap_end_wav = df['start'].iloc[i]
        duration = gap_end_wav - gap_start_wav
        if duration >= MIN_SILENCE_DURATION:
            onset_scan = gap_start_wav - AUDIO_SCAN_OFFSET
            silences.append((onset_scan, duration))
    return silences


def extract_ahc_silences(xlsx_path):
    """Return list of (silence_onset_scan_relative, silence_duration) for long silences.

    Forward-fills Prompt Number, filters to valid Possibility Number rows, sorts by Start Time.
    Silence onset is the end of the previous sentence, in scan-relative seconds.
    """
    df = pd.read_excel(xlsx_path)
    df['Prompt Number'] = df['Prompt Number'].ffill()
    df = df[df['Possibility Number'].notna()].copy()
    df = df.sort_values('Start Time').reset_index(drop=True)

    silences = []
    for i in range(1, len(df)):
        gap_start_wav = df['End Time'].iloc[i - 1]
        gap_end_wav = df['Start Time'].iloc[i]
        duration = gap_end_wav - gap_start_wav
        if duration >= MIN_SILENCE_DURATION:
            onset_scan = gap_start_wav - AUDIO_SCAN_OFFSET
            silences.append((onset_scan, duration))
    return silences


def exclude_near_boundaries(silences, boundary_times, radius=EXCLUSION_RADIUS):
    """Drop any silence onset within `radius` seconds of a real boundary.

    Parameters
    ----------
    silences : list of (onset, duration)
    boundary_times : list of float (scan-relative seconds)

    Returns list of (onset, duration).
    """
    boundaries = np.array(boundary_times)
    kept = []
    for onset, dur in silences:
        if boundaries.size == 0 or np.all(np.abs(onset - boundaries) >= radius):
            kept.append((onset, dur))
    return kept


def filter_silences_by_audio(silences, envelope, sr, threshold=0.0):
    """Keep only silences where the max z-scored audio envelope during the
    window is below `threshold`.

    Uses max (not mean) so that any second of above-average amplitude in the
    window causes the silence to be rejected — even if most of the window is
    quiet. threshold=0 means every sample in the window must be below the
    session average amplitude.
    """
    kept = []
    for onset_scan, dur in silences:
        wav_onset = onset_scan + AUDIO_SCAN_OFFSET
        i_start = int(round(wav_onset * sr))
        i_end = int(round((wav_onset + dur) * sr))
        i_end = min(i_end, len(envelope))
        if i_start < 0 or i_start >= i_end:
            continue
        max_amp = envelope[i_start:i_end].max()
        if max_amp < threshold:
            kept.append((onset_scan, dur))
    return kept


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time_vec = np.arange(-TRS_BEFORE, TRS_AFTER + 1) * TR
    n_audio_pre = int(AUDIO_PRE * AUDIO_TARGET_SR)
    n_audio_post = int(AUDIO_POST * AUDIO_TARGET_SR)
    audio_time_vec = np.linspace(-AUDIO_PRE, AUDIO_POST, n_audio_pre + n_audio_post + 1)

    roi_keys = [k for k, _ in ROI_SPEC]
    task_keys = ('svf', 'ahc')

    # Group-level: task -> roi -> list of (subject, mean_epoch)
    group_data        = {t: {k: [] for k in roi_keys} for t in task_keys}
    group_offset_data = {t: {k: [] for k in roi_keys} for t in task_keys}
    group_audio        = {t: [] for t in task_keys}
    group_offset_audio = {t: [] for t in task_keys}
    # Track silence event counts per subject per task (for figure annotation)
    group_n_events = {t: [] for t in task_keys}  # list of per-subject event counts

    for subject in SUBJECT_IDS:
        print(f"\n{'='*60}")
        print(f"  {subject}")
        print(f"{'='*60}")

        fmri_hits = discover_sessions(subject)
        if not fmri_hits:
            print(f"  No SVF/AHC fMRI data found, skipping")
            continue

        # Accumulate silence epochs across sessions per task/roi
        subj_epochs        = {t: {k: [] for k in roi_keys} for t in task_keys}
        subj_offset_epochs = {t: {k: [] for k in roi_keys} for t in task_keys}
        subj_audio_epochs        = {t: [] for t in task_keys}
        subj_offset_audio_epochs = {t: [] for t in task_keys}

        for session, task in sorted(fmri_hits):
            # --- Find annotation file ---
            if task == 'svf':
                ann_path = _find_svf_annotation(subject, session)
                if ann_path is None:
                    print(f"  {session} svf: no annotation CSV, skipping")
                    continue
                raw_silences = extract_svf_silences(ann_path)
            else:
                ann_path = _find_ahc_annotation(subject, session)
                if ann_path is None:
                    print(f"  {session} ahc: no annotation XLSX, skipping")
                    continue
                raw_silences = extract_ahc_silences(ann_path)

            # --- Load all trial boundaries (onsets + offsets) for exclusion ---
            csv_path = find_psychopy_csv(subject, session, task)
            boundary_times = parse_all_trial_boundaries(csv_path, task) if csv_path else []

            silences = exclude_near_boundaries(raw_silences, boundary_times)
            n_after_boundary = len(silences)

            # --- Load audio envelope early (needed for amplitude filter) ---
            env, env_sr = None, None
            wav_path = find_audio_wav(subject, session, task)
            if wav_path is not None:
                try:
                    env, env_sr = load_audio_envelope(wav_path)
                except Exception as e:
                    print(f"  {session} {task}: audio load failed: {e}")

            # --- Filter by audio amplitude (keep only genuinely quiet windows) ---
            if env is not None:
                silences = filter_silences_by_audio(silences, env, env_sr, threshold=0.0)

            print(f"  {session} {task}: {len(raw_silences)} raw → "
                  f"{n_after_boundary} after boundary exclusion → "
                  f"{len(silences)} after audio filter")

            if not silences:
                continue

            onset_times  = [s[0]           for s in silences]
            offset_times = [s[0] + s[1]    for s in silences]

            # --- Load ROI time series ---
            roi_ts = extract_roi_timeseries(subject, session, task)
            if roi_ts is None:
                print(f"  {session} {task}: no cached parcel data, skipping")
                continue

            for k in roi_keys:
                ep = extract_event_locked(roi_ts[k], onset_times)
                if ep is not None:
                    subj_epochs[task][k].append(ep)
                ep_off = extract_event_locked(roi_ts[k], offset_times)
                if ep_off is not None:
                    subj_offset_epochs[task][k].append(ep_off)

            # --- Audio epochs (envelope already loaded above) ---
            if env is not None:
                wav_onset_times  = [t + AUDIO_SCAN_OFFSET for t in onset_times]
                wav_offset_times = [t + AUDIO_SCAN_OFFSET for t in offset_times]
                audio_ep, _ = extract_audio_event_locked(env, env_sr, wav_onset_times)
                if audio_ep is not None:
                    subj_audio_epochs[task].append(audio_ep)
                    print(f"  {session} {task}: {audio_ep.shape[0]} audio epochs")
                audio_ep_off, _ = extract_audio_event_locked(env, env_sr, wav_offset_times)
                if audio_ep_off is not None:
                    subj_offset_audio_epochs[task].append(audio_ep_off)

        # Compute per-subject mean for each task/roi and add to group
        for task in task_keys:
            has_data = all(len(subj_epochs[task][k]) > 0 for k in roi_keys)
            if not has_data:
                continue
            n_subj_events = None
            for k in roi_keys:
                stacked = np.vstack(subj_epochs[task][k])
                n = stacked.shape[0]
                group_data[task][k].append((subject, stacked.mean(axis=0)))
                if k == roi_keys[0]:
                    n_subj_events = n
            group_n_events[task].append(n_subj_events)

            for k in roi_keys:
                if subj_offset_epochs[task][k]:
                    stacked = np.vstack(subj_offset_epochs[task][k])
                    group_offset_data[task][k].append((subject, stacked.mean(axis=0)))

            if subj_audio_epochs[task]:
                stacked_audio = np.vstack(subj_audio_epochs[task])
                group_audio[task].append((subject, stacked_audio.mean(axis=0)))

            if subj_offset_audio_epochs[task]:
                stacked_audio = np.vstack(subj_offset_audio_epochs[task])
                group_offset_audio[task].append((subject, stacked_audio.mean(axis=0)))

    # =====================================================================
    # GROUP FIGURE: 6 rows × 4 cols (SVF onset, AHC onset, SVF offset, AHC offset)
    # Audio row inserted after EAC (row 3), EVC moves to row 5.
    # =====================================================================
    print(f"\n{'='*60}")
    print("  Generating group figure")
    print(f"{'='*60}")

    # Row layout: ROIs in ROI_SPEC order, then audio envelope last
    n_rois = len(ROI_SPEC)
    ROI_ROWS = list(range(n_rois))
    AUDIO_ROW = n_rois
    n_grid_rows = n_rois + 1

    # Column definitions: (task, lock_type, group_dict, group_audio_dict, xlabel)
    COL_DEFS = [
        ('svf', 'onset',  group_data,        group_audio,        'Semantic Fluency\n— silence onset'),
        ('ahc', 'onset',  group_data,        group_audio,        'Explanation Generation\n— silence onset'),
        ('svf', 'offset', group_offset_data, group_offset_audio, 'Semantic Fluency\n— silence offset'),
        ('ahc', 'offset', group_offset_data, group_offset_audio, 'Explanation Generation\n— silence offset'),
    ]
    n_cols = len(COL_DEFS)

    fig, axes = plt.subplots(
        n_grid_rows, n_cols,
        figsize=(5 * n_cols, 3 * n_grid_rows),
        sharex=True,
    )
    fig.suptitle(
        f"Silence-Locked Control  |  gaps ≥{MIN_SILENCE_DURATION:.0f} s, "
        f"±{EXCLUSION_RADIUS:.0f} s from boundaries excluded",
        fontsize=TITLE_FS, fontweight='bold', y=1.01,
    )

    bold_axes = []
    audio_axes = []

    for col_idx, (task, lock_type, g_data, g_audio, col_label) in enumerate(COL_DEFS):
        counts = group_n_events[task]
        total_events = sum(counts)
        n_subj = len(counts)
        axes[0, col_idx].set_title(
            f"{col_label}\n(n_events={total_events} across {n_subj} subjects)",
            fontsize=LABEL_FS, fontweight='bold',
        )

        # --- ROI rows ---
        for roi_idx, (roi_key, roi_title) in enumerate(ROI_SPEC):
            grid_row = ROI_ROWS[roi_idx]
            ax = axes[grid_row, col_idx]
            bold_axes.append(ax)
            subj_means = g_data[task][roi_key]

            if subj_means:
                for subj, mean_epoch in subj_means:
                    ci = SUBJECT_IDS.index(subj) if subj in SUBJECT_IDS else 0
                    ax.plot(time_vec, mean_epoch,
                            color=SUBJECT_COLORS[ci % len(SUBJECT_COLORS)],
                            lw=1, alpha=0.5)

                data_arr = np.array([m for _, m in subj_means])
                gmean = data_arr.mean(axis=0)
                gsem = data_arr.std(axis=0) / np.sqrt(len(data_arr))
                ax.plot(time_vec, gmean, color='k', lw=2.5)
                ax.fill_between(time_vec, gmean - gsem, gmean + gsem,
                                color='k', alpha=0.2)

            ax.axvline(0, color='grey', ls='--', lw=1)
            ax.axhline(0, color='k', ls='-', alpha=0.3)
            ax.set_xlim(-30, 60)
            ax.spines[['top', 'right']].set_visible(False)

            if col_idx == 0:
                ax.set_ylabel(roi_title, fontsize=LABEL_FS - 1)
            if grid_row == n_grid_rows - 1:
                ax.set_xlabel(f'Time from silence {lock_type} (s)', fontsize=LABEL_FS - 1)

        # --- Audio row ---
        ax_audio = axes[AUDIO_ROW, col_idx]
        audio_axes.append(ax_audio)
        subj_audio = g_audio[task]

        if subj_audio:
            for subj, mean_epoch in subj_audio:
                ci = SUBJECT_IDS.index(subj) if subj in SUBJECT_IDS else 0
                ax_audio.plot(audio_time_vec, mean_epoch,
                              color=SUBJECT_COLORS[ci % len(SUBJECT_COLORS)],
                              lw=1, alpha=0.5)

            data_arr = np.array([m for _, m in subj_audio])
            gmean = data_arr.mean(axis=0)
            gsem = data_arr.std(axis=0) / np.sqrt(len(data_arr))
            ax_audio.plot(audio_time_vec, gmean, color='k', lw=2.5)
            ax_audio.fill_between(audio_time_vec, gmean - gsem, gmean + gsem,
                                  color='k', alpha=0.2)

        ax_audio.axvline(0, color='grey', ls='--', lw=1)
        ax_audio.axhline(0, color='k', ls='-', alpha=0.3)
        ax_audio.set_xlim(-30, 60)
        ax_audio.spines[['top', 'right']].set_visible(False)

        if col_idx == 0:
            ax_audio.set_ylabel('Audio\nEnvelope', fontsize=LABEL_FS - 1)

    # Shared y-limits: all BOLD axes together, all audio axes together
    bold_data = [m for t in task_keys
                 for gd in (group_data, group_offset_data)
                 for _, m in gd[t][roi_keys[0]]]
    bold_ylim = max(np.nanpercentile(np.abs(np.concatenate(bold_data)), 99), 0.5) if bold_data else 1.5
    for ax in bold_axes:
        ax.set_ylim(-bold_ylim, bold_ylim)

    audio_data = [m for t in task_keys
                  for ga in (group_audio, group_offset_audio)
                  for _, m in ga[t]]
    audio_ylim = max(np.nanpercentile(np.abs(np.concatenate(audio_data)), 99), 0.5) if audio_data else 1.5
    for ax in audio_axes:
        ax.set_ylim(-audio_ylim, audio_ylim)

    fig.tight_layout()
    suffix = f"_gap{MIN_SILENCE_DURATION:.0f}s" if MIN_SILENCE_DURATION != 10.0 else ""
    out = OUTPUT_DIR / f'silence_control{suffix}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-gap', type=float, default=None,
                        help='Override MIN_SILENCE_DURATION (seconds)')
    parser.add_argument('--exclusion-radius', type=float, default=None,
                        help='Override EXCLUSION_RADIUS (seconds)')
    args = parser.parse_args()
    if args.min_gap is not None:
        MIN_SILENCE_DURATION = args.min_gap
    if args.exclusion_radius is not None:
        EXCLUSION_RADIUS = args.exclusion_radius
    main()
