#!/usr/bin/env python3
"""
Batch extract and cache parcel time series for any task.

Auto-detects which subjects/sessions have BOLD data for the given task
by scanning the derivatives directory.

Extracts:
  - Schaefer 400 (17 Networks) cortical parcels
  - Harvard-Oxford subcortical ROIs

Cached .npz files are saved to {CACHE_DIR}/nilearn_cache/ and will be
automatically reused by any downstream call to get_parcel_data().

Usage:
    uv run python srcs/fmrianalysis/extract_task_parcels.py --task filmfest1
    uv run python srcs/fmrianalysis/extract_task_parcels.py --task filmfest1 filmfest2
    uv run python srcs/fmrianalysis/extract_task_parcels.py --task svf --n_jobs 12
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

# === CONFIG SETUP ===
from configs.config import DERIVATIVES_DIR, CACHE_DIR
from fmrianalysis.utils import get_parcel_data

ATLASES = ['Schaefer400_17Nets', 'HarvardOxford_sub']


def find_subjects_for_task(task):
    """Scan derivatives directory to find all subject-session pairs with BOLD data for a task."""
    pattern = f"sub-*/ses-*/func/sub-*_ses-*_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    bold_files = sorted(DERIVATIVES_DIR.glob(pattern))

    pairs = []
    for f in bold_files:
        parts = f.name.split('_')
        subject = parts[0]
        session = parts[1]
        pairs.append((subject, session))

    return pairs


def extract_one(subject, session, task, atlas):
    """Extract parcel time series for one subject/session/task/atlas combo."""
    try:
        parcel_dict = get_parcel_data(subject, session, task, atlas=atlas)
        labels = [l for l in parcel_dict if l != 'Background']
        n_parcels = len(labels)
        n_trs = len(next(iter(parcel_dict.values())))
        print(f"  {subject} {session} {task} {atlas} -> {n_parcels} parcels x {n_trs} TRs")
        return True
    except FileNotFoundError as e:
        print(f"  {subject} {session} {task} {atlas} -> SKIPPED: {e}")
        return False
    except Exception as e:
        print(f"  {subject} {session} {task} {atlas} -> ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache parcel time series (Schaefer 400 + Harvard-Oxford subcortical)."
    )
    parser.add_argument(
        '--task', nargs='+', required=True,
        help="Task name(s) to extract (e.g. filmfest1 filmfest2 svf)",
    )
    parser.add_argument(
        '--n_jobs', type=int, default=1,
        help="Number of parallel jobs (default: 1, sequential)",
    )
    args = parser.parse_args()

    # Collect all (subject, session, task, atlas) jobs
    jobs = []
    for task in args.task:
        pairs = find_subjects_for_task(task)
        if not pairs:
            print(f"WARNING: No BOLD files found for task '{task}' in {DERIVATIVES_DIR}")
            continue
        for subject, session in pairs:
            for atlas in ATLASES:
                jobs.append((subject, session, task, atlas))

    if not jobs:
        print("Nothing to extract. Check that BOLD files exist in derivatives.")
        sys.exit(1)

    print(f"Extracting parcel time series: {len(jobs)} jobs, n_jobs={args.n_jobs}\n")

    Parallel(n_jobs=args.n_jobs)(
        delayed(extract_one)(subject, session, task, atlas)
        for subject, session, task, atlas in jobs
    )

    print("\nDone. Cached files are in:", CACHE_DIR / 'parcels')


if __name__ == '__main__':
    main()
