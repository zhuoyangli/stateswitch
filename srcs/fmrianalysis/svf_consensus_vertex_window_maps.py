"""
Stage 1 (compute) — per-subject whole-brain peri-boundary mean epochs for the
SVF (word generation) task, split by switching/clustering rater-consensus level.

Motivation: the ROI time courses mix in neural responses to neighbouring words,
and because switch words follow a much longer pause than cluster words the
conditions are locked to different effective event times, which makes the
overlaid curves hard to compare. A whole-brain surface map of a single peri-word
window is far less sensitive to that, and locking to the word ONSET (default)
puts every consensus level on the same clock.

Consensus levels come from the aggregated per-word rating file
    figs/behavior/svf_transition_irr/svf_transition_word_scores.csv
  --by class (default): clustering (k<=1) / ambiguous (k=2-5) / switching (k>=6)
  --by votes           : one level per k_switch_votes (0..7)

For each subject x consensus level: load both hemispheres whole-brain, highpass +
z-score over time per vertex, extract peri-word epochs, average over events (and
over sessions), and store the mean epoch (n_window, V) per hemisphere. Units are
z-scored BOLD over time (NOT re-z-scored across vertices), so window magnitudes
stay comparable. Plotting is done separately by
svf_consensus_vertex_window_maps_plot.py under the MNE env.

Usage:
    uv run python srcs/fmrianalysis/svf_consensus_vertex_window_maps.py
    uv run python srcs/fmrianalysis/svf_consensus_vertex_window_maps.py --by votes
    uv run python srcs/fmrianalysis/svf_consensus_vertex_window_maps.py --align offset --force
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import zscore as sp_zscore

from configs.config import DERIVATIVES_DIR, TR, ANALYSIS_CACHE_DIR, FIGS_DIR
from fmrianalysis.utils import load_surface_data, highpass_filter

WORD_SCORES = FIGS_DIR / 'behavior' / 'svf_transition_irr' / 'svf_transition_word_scores.csv'
CACHE_DIR = ANALYSIS_CACHE_DIR / 'svf_consensus_vertex_window_maps'

SCANNER_START_OFFSET = 12.0
TRS_BEFORE = 6    # -9 s   (word events are densely packed; a long pre-window
TRS_AFTER = 14    # +21 s   would drop most early words to the out-of-range mask)

CLASS_LEVELS = ['clustering', 'ambiguous', 'switching']
VOTE_LEVELS = [f'k{k}' for k in range(8)]


def levels_for(mode):
    return CLASS_LEVELS if mode == 'class' else VOTE_LEVELS


def label_series(df, mode):
    if mode == 'class':
        return df['consensus_class']
    return 'k' + df['k_switch_votes'].astype(int).astype(str)


# ---------------------------------------------------------------------------
# Preprocess + epoch  (verbatim recipe from boundary_vertex_window_maps.py)
# ---------------------------------------------------------------------------

def _preprocess(subject, session, task, hemi_letter):
    """Load one hemisphere, highpass + z-score over time. Returns (T, V) or None."""
    try:
        ts = load_surface_data(
            subject, session, task, hemi_letter, DERIVATIVES_DIR
        ).astype(np.float64).T
    except FileNotFoundError:
        return None
    ts = highpass_filter(ts)
    ts = sp_zscore(ts, axis=0, nan_policy='omit')
    return np.nan_to_num(ts, nan=0.0)


def _mean_epoch(ts, boundary_times):
    """Mean peri-boundary epoch. Returns (n_window, V) and n_valid_events."""
    n_t = ts.shape[0]
    epochs = []
    for t_sec in boundary_times:
        center = int(round(t_sec / TR))
        s = center - TRS_BEFORE
        e = center + TRS_AFTER + 1
        if s < 0 or e > n_t:
            continue
        epochs.append(ts[s:e, :])
    if not epochs:
        return None, 0
    return np.array(epochs).mean(axis=0), len(epochs)


# ---------------------------------------------------------------------------
# Consensus events
# ---------------------------------------------------------------------------

def consensus_event_times(words_ses, mode, align):
    """{level_key: [scan-relative event times]} for one (subject, session).

    align='onset' -> lock to the word's own onset (equalises timing across
    consensus levels); align='offset' -> preceding word offset within category.
    """
    df = words_ses.sort_values(['category', 'start']).reset_index(drop=True)
    if align == 'offset':
        df['lock'] = df.groupby('category')['end'].shift(1)
    else:
        df['lock'] = df['start']
    df['event_t'] = df['lock'] - SCANNER_START_OFFSET
    df = df[df['event_t'].notna() & (df['event_t'] >= 0)]
    lab = label_series(df, mode)
    return {lvl: df.loc[lab == lvl, 'event_t'].to_numpy() for lvl in levels_for(mode)}


def compute_subject_level(subject, sessions_ses, mode, align):
    """Average mean epochs across sessions, per consensus level.

    Returns {level: dict(epoch_L, epoch_R, time_sec, n_events, ...)}.
    """
    per_run = {lvl: {'L': [], 'R': []} for lvl in levels_for(mode)}
    n_events = {lvl: 0 for lvl in levels_for(mode)}

    for session, words_ses in sessions_ses:
        ts_l = _preprocess(subject, session, 'svf', 'L')
        ts_r = _preprocess(subject, session, 'svf', 'R')
        if ts_l is None or ts_r is None:
            print(f"    {session}: surface missing, skipping")
            continue
        events = consensus_event_times(words_ses, mode, align)
        for lvl, times in events.items():
            if len(times) == 0:
                continue
            ep_l, n_l = _mean_epoch(ts_l, times)
            ep_r, n_r = _mean_epoch(ts_r, times)
            if ep_l is None or ep_r is None:
                continue
            per_run[lvl]['L'].append(ep_l)
            per_run[lvl]['R'].append(ep_r)
            n_events[lvl] += n_l
        print(f"    {session}: " +
              ", ".join(f"{lvl}={n_events[lvl]}" for lvl in levels_for(mode)))

    out = {}
    for lvl in levels_for(mode):
        if not per_run[lvl]['L']:
            continue
        epoch_L = np.mean(per_run[lvl]['L'], axis=0)
        epoch_R = np.mean(per_run[lvl]['R'], axis=0)
        time_sec = (np.arange(epoch_L.shape[0]) - TRS_BEFORE) * TR
        out[lvl] = dict(epoch_L=epoch_L, epoch_R=epoch_R, time_sec=time_sec,
                        n_events=n_events[lvl], trs_before=TRS_BEFORE,
                        trs_after=TRS_AFTER)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--by', choices=['class', 'votes'], default='class')
    ap.add_argument('--align', choices=['onset', 'offset'], default='onset',
                    help="onset (default; equalises timing) or previous-word offset")
    ap.add_argument('--subjects', nargs='+', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    words = pd.read_csv(WORD_SCORES)
    subjects = args.subjects or sorted(words['subject'].unique())

    for subject in subjects:
        w_sub = words[words['subject'] == subject]
        sessions_ses = [(ses, g) for ses, g in w_sub.groupby('session')]
        print(f"\n{subject}  ({len(sessions_ses)} sessions, by={args.by}, align={args.align})")

        # skip if all level caches already present
        tag = f'by-{args.by}_align-{args.align}'
        expected = [CACHE_DIR / f'{subject}_{lvl}_{tag}.npz' for lvl in levels_for(args.by)]
        if all(p.exists() for p in expected) and not args.force:
            print("  cached, skipping")
            continue

        results = compute_subject_level(subject, sessions_ses, args.by, args.align)
        for lvl, res in results.items():
            out = CACHE_DIR / f'{subject}_{lvl}_{tag}.npz'
            np.savez_compressed(out, **res)
            print(f"  saved {out.name}  (L{res['epoch_L'].shape}, n_events={res['n_events']})")


if __name__ == '__main__':
    main()
