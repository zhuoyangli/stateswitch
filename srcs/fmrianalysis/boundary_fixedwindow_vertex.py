"""
Stage 1 (compute) — fixed-window (NOT GLM) per-vertex boundary activation.

Model-free analogue of boundary_glm_vertex.py. Instead of fitting an HRF GLM, for
each boundary we take the mean of the peri-boundary BOLD signal in a fixed post-
boundary window (default 4.5-10.5 s, i.e. TR offsets +3..+7 with TR=1.5), average
over events and runs, and store the per-vertex mean as the activation map. Done
separately for boundary ONSET- and OFFSET-locked events.

Event definitions and preprocessing are identical to boundary_vertex_window_maps
(highpass + z-score over time per vertex; same _condition_runs), so these maps are
directly comparable to the peri-boundary epoch analysis — this is just that epoch
collapsed to a single scalar per vertex over the 4.5-10.5 s window.

Four conditions (same subset as boundary_glm_vertex; movieWithin excluded because
its onset and offset coincide):
  wordGen      — SVF between-trial boundaries (word generation)
  explGen      — AHC between-trial boundaries (explanation generation)
  movieBetween — filmfest between-movie boundaries (encoding)
  movieRecall  — filmfest recall between-movie boundaries

Per-subject vertex maps (mean window z per vertex) are cached to .npz alongside a
group npz holding the across-subject mean map and a one-sample t across subjects'
per-subject maps. Plotting is done by boundary_fixedwindow_vertex_plot.py and
boundary_fixedwindow_parcel_map.py under the MNE env.

Usage:
    uv run python srcs/fmrianalysis/boundary_fixedwindow_vertex.py
    uv run python srcs/fmrianalysis/boundary_fixedwindow_vertex.py \
        --subjects sub-003 --conditions movieBetween
    uv run python srcs/fmrianalysis/boundary_fixedwindow_vertex.py \
        --window 4.5 10.5 --force
"""
import argparse

import numpy as np
from scipy import stats

from configs.config import TR, ANALYSIS_CACHE_DIR, SUBJECT_IDS
from fmrianalysis.boundary_vertex_window_maps import _condition_runs, _preprocess

CACHE_DIR = ANALYSIS_CACHE_DIR / 'boundary_fixedwindow_vertex'

# Same four conditions as boundary_glm_vertex (movieWithin excluded — onset and
# offset coincide, so onset/offset activation maps would be identical).
CONDITIONS = ['wordGen', 'explGen', 'movieBetween', 'movieRecall']
REGRESSORS = ['onset', 'offset']
ALIGN_FOR = {'onset': 'onset', 'offset': 'offset'}

DEFAULT_SUBJECTS = list(SUBJECT_IDS)

# Fixed post-boundary averaging window (seconds), inclusive of both TR endpoints.
WINDOW_START_SEC = 4.5
WINDOW_END_SEC = 10.5


def _window_trs(start_sec, end_sec):
    """Integer TR offsets covered by [start_sec, end_sec] post-boundary, inclusive."""
    k0 = int(round(start_sec / TR))
    k1 = int(round(end_sec / TR))
    return k0, k1


def _mean_window(ts, boundary_times, k0, k1):
    """Mean signal over TR offsets k0..k1 (inclusive) after each boundary.

    Returns (V,) mean over valid events and the number of valid events."""
    n_t = ts.shape[0]
    vals = []
    for t_sec in boundary_times:
        center = int(round(t_sec / TR))
        s = center + k0
        e = center + k1 + 1
        if s < 0 or e > n_t:
            continue
        vals.append(ts[s:e, :].mean(axis=0))
    if not vals:
        return None, 0
    return np.array(vals).mean(axis=0), len(vals)


def compute_subject_condition(subject, condition, k0, k1):
    """Per-vertex fixed-window activation for onset & offset. Returns dict or None."""
    out = {}
    n_counts = {}
    for reg in REGRESSORS:
        per_run_L, per_run_R = [], []
        n_events = 0
        for session, task, times in _condition_runs(
                subject, condition, ALIGN_FOR[reg]):
            if not times:
                continue
            ts_l = _preprocess(subject, session, task, 'L')
            ts_r = _preprocess(subject, session, task, 'R')
            if ts_l is None or ts_r is None:
                print(f"    {session} {task} [{reg}]: surface missing, skipping")
                continue
            w_l, n_l = _mean_window(ts_l, times, k0, k1)
            w_r, n_r = _mean_window(ts_r, times, k0, k1)
            if w_l is None or w_r is None:
                print(f"    {session} {task} [{reg}]: no valid windows, skipping")
                continue
            print(f"    {session} {task} [{reg}]: {n_l} events")
            per_run_L.append(w_l)
            per_run_R.append(w_r)
            n_events += n_l
        if not per_run_L:
            return None
        out[f'{reg}_L'] = np.mean(per_run_L, axis=0)
        out[f'{reg}_R'] = np.mean(per_run_R, axis=0)
        n_counts[reg] = n_events

    out['n_onset'] = n_counts['onset']
    out['n_offset'] = n_counts['offset']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS)
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument('--window', type=float, nargs=2,
                    default=[WINDOW_START_SEC, WINDOW_END_SEC],
                    metavar=('START_SEC', 'END_SEC'),
                    help='post-boundary averaging window in seconds '
                         f'(default {WINDOW_START_SEC} {WINDOW_END_SEC})')
    ap.add_argument('--force', action='store_true',
                    help='recompute even if cache exists')
    args = ap.parse_args()

    k0, k1 = _window_trs(args.window[0], args.window[1])
    print(f'Window {args.window[0]}-{args.window[1]} s -> TR offsets +{k0}..+{k1} '
          f'({k1 - k0 + 1} TRs)')
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    group = {c: {k: [] for k in ('onset_L', 'onset_R', 'offset_L', 'offset_R')}
             for c in args.conditions}

    for subject in args.subjects:
        for condition in args.conditions:
            out = CACHE_DIR / f'{subject}_{condition}.npz'
            if out.exists() and not args.force:
                print(f'{subject} {condition}: cached, skipping')
                result = dict(np.load(out))
            else:
                print(f'\n{subject} {condition}')
                result = compute_subject_condition(subject, condition, k0, k1)
                if result is None:
                    print(f'  no data for {subject} {condition}')
                    continue
                result['window_start'] = args.window[0]
                result['window_end'] = args.window[1]
                np.savez_compressed(out, **result)
                print(f'  saved {out.name}  '
                      f'(onset n={result["n_onset"]}, offset n={result["n_offset"]})')
            for k in ('onset_L', 'onset_R', 'offset_L', 'offset_R'):
                group[condition][k].append(result[k])

    # Group maps: across-subject mean window activation (descriptive) plus a
    # one-sample t-test across subjects' per-subject maps (second-level inference).
    for condition in args.conditions:
        if not group[condition]['onset_L']:
            continue
        gout = CACHE_DIR / f'group_{condition}.npz'
        n_sub = len(group[condition]['onset_L'])
        payload = {}
        for k, v in group[condition].items():
            arr = np.array(v)                       # (n_subjects, V)
            payload[k] = arr.mean(axis=0)           # mean activation map
            if n_sub >= 2:
                t, _ = stats.ttest_1samp(arr, popmean=0.0, axis=0)
                payload[f'{k}_t'] = np.nan_to_num(t, nan=0.0)   # one-sample t
        payload['n_subjects'] = n_sub
        payload['df'] = n_sub - 1
        payload['window_start'] = args.window[0]
        payload['window_end'] = args.window[1]
        np.savez_compressed(gout, **payload)
        print(f'group {condition}: saved {gout.name} '
              f'(N={n_sub}, df={n_sub - 1})')


if __name__ == '__main__':
    main()
