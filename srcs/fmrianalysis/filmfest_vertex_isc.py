"""
Filmfest Inter-Subject Correlation (ISC) Analysis

Computes vertex-wise leave-one-out ISC on fsaverage6 surface for each movie clip,
with high-pass filtering (0.01 Hz) and z-scoring.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore
from nilearn import datasets, plotting
import matplotlib.pyplot as plt

# === CONFIG SETUP ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, FILMFEST_SUBJECTS
from fmrianalysis.utils import load_surface_data, highpass_filter, mss_to_seconds

# --- Constants ---
ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
OUTPUT_DIR = FIGS_DIR / 'filmfest_isc'

MOVIE_INFO = [
    {'id': 1,  'file': 'FilmFest_01_CMIYC_Segments.xlsx',          'task': 'filmfest1', 'name': 'CMIYC'},
    {'id': 2,  'file': 'FilmFest_02_The_Record_Segments.xlsx',     'task': 'filmfest1', 'name': 'The Record'},
    {'id': 3,  'file': 'FilmFest_03_The_Boyfriend_Segments.xlsx',  'task': 'filmfest1', 'name': 'The Boyfriend'},
    {'id': 4,  'file': 'FilmFest_04_The_Shoe_Segments.xlsx',       'task': 'filmfest1', 'name': 'The Shoe'},
    {'id': 5,  'file': 'FilmFest_05_Keith_Reynolds_Segments.xlsx', 'task': 'filmfest1', 'name': 'Keith Reynolds'},
    {'id': 6,  'file': 'FilmFest_06_The_Rock_Segments.xlsx',       'task': 'filmfest2', 'name': 'The Rock'},
    {'id': 7,  'file': 'FilmFest_07_The_Prisoner_Segments.xlsx',   'task': 'filmfest2', 'name': 'The Prisoner'},
    {'id': 8,  'file': 'FilmFest_08_The_Black_Hole_Segments.xlsx', 'task': 'filmfest2', 'name': 'The Black Hole'},
    {'id': 9,  'file': 'FilmFest_09_Post-it_Love_Segments.xlsx',   'task': 'filmfest2', 'name': 'Post-it Love'},
    {'id': 10, 'file': 'FilmFest_10_Bus_Stop_Segments.xlsx',       'task': 'filmfest2', 'name': 'Bus Stop'},
]

HIGH_PASS_HZ = 0.01


def get_movie_tr_range(movie):
    """Return (start_tr, end_tr) for a movie based on annotation timestamps."""
    df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
    start_sec = mss_to_seconds(df['SEG-C Start Time (m.ss)'].min())
    end_sec = mss_to_seconds(df['SEG-C End Time (m.ss)'].max())
    start_tr = int(np.floor(start_sec / TR))
    end_tr = int(np.ceil(end_sec / TR))
    return start_tr, end_tr


def compute_isc(data_list):
    """Leave-one-out ISC per subject.

    Parameters
    ----------
    data_list : list of arrays, each (n_timepoints, n_vertices)

    Returns
    -------
    isc_per_subject : array (n_subjects, n_vertices)
    """
    n_subjects = len(data_list)
    all_data = np.stack(data_list)  # (S, T, V)

    isc_per_subject = np.zeros((n_subjects, all_data.shape[2]))

    for i in range(n_subjects):
        subj = all_data[i]
        others_mean = np.delete(all_data, i, axis=0).mean(axis=0)

        # Demean
        s = subj - subj.mean(axis=0, keepdims=True)
        o = others_mean - others_mean.mean(axis=0, keepdims=True)

        # Pearson r per vertex (vectorized)
        num = (s * o).sum(axis=0)
        den = np.sqrt((s ** 2).sum(axis=0) * (o ** 2).sum(axis=0))
        den = np.where(den == 0, 1, den)
        isc_per_subject[i] = num / den

    return isc_per_subject


def plot_isc_surface(isc_left, isc_right, movie_name, output_path):
    """Plot mean ISC on fsaverage6 inflated surface (4 views).

    Parameters
    ----------
    isc_left : array (n_vertices,)
    isc_right : array (n_vertices,)
    """
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage6')

    vmax = 0.5

    views_spec = [
        ('left',  'lateral',  isc_left),
        ('right', 'lateral',  isc_right),
        ('left',  'medial',   isc_left),
        ('right', 'medial',   isc_right),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        subplot_kw={'projection': '3d'},
    )
    fig.suptitle(f'Inter-Subject Correlation — {movie_name}', fontsize=16, y=0.95)

    for idx, (hemi, view, data) in enumerate(views_spec):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        mesh = fsaverage[f'infl_{hemi}']
        bg = fsaverage[f'sulc_{hemi}']
        plotting.plot_surf_stat_map(
            mesh, data,
            hemi=hemi,
            view=view,
            bg_map=bg,
            colorbar=(col == 1),
            vmax=vmax,
            threshold=0.05,
            cmap='inferno',
            axes=ax,
            figure=fig,
        )
        ax.set_title(f'{hemi.capitalize()} {view}', fontsize=12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {output_path}')


def load_all_runs():
    """Pre-load all run data to avoid redundant I/O.

    Returns dict: run_data[task][subject][hemi] = array (n_timepoints, n_vertices)
    """
    run_data = {}
    for task in ('filmfest1', 'filmfest2'):
        run_data[task] = {}
        for subject, session in FILMFEST_SUBJECTS.items():
            run_data[task][subject] = {}
            print(f'  Loading {subject} {session} {task} ...')
            for hemi in ('L', 'R'):
                ts = load_surface_data(
                    subject, session, task, hemi,
                    data_dir=DERIVATIVES_DIR,
                )
                run_data[task][subject][hemi] = ts.astype(np.float64).T  # (T, V)
    return run_data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading all run data ...')
    run_data = load_all_runs()

    for movie in MOVIE_INFO:
        print(f'\n=== Movie {movie["id"]:02d}: {movie["name"]} ===')

        start_tr, end_tr = get_movie_tr_range(movie)
        n_trs = end_tr - start_tr
        print(f'  TR range: {start_tr}–{end_tr} ({n_trs} TRs, '
              f'{n_trs * TR:.1f} s)')

        left_data, right_data = [], []

        for subject in FILMFEST_SUBJECTS:
            for hemi, data_list in [('L', left_data), ('R', right_data)]:
                ts = run_data[movie['task']][subject][hemi]
                ts = ts[start_tr:end_tr].copy()
                ts = highpass_filter(ts)
                ts = zscore(ts, axis=0, nan_policy='omit')
                ts = np.nan_to_num(ts, nan=0.0)
                data_list.append(ts)

        print('  Computing ISC ...')
        isc_left = compute_isc(left_data).mean(axis=0)   # average across subjects
        isc_right = compute_isc(right_data).mean(axis=0)

        out_path = (
            OUTPUT_DIR
            / f'isc_movie{movie["id"]:02d}_{movie["name"].replace(" ", "_")}.png'
        )
        plot_isc_surface(isc_left, isc_right, movie['name'], out_path)


if __name__ == '__main__':
    main()
